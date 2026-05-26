use steel_registry::{blocks::block_state_ext::BlockStateExt, vanilla_blocks};
use steel_utils::{BlockPos, Direction, SectionPos};

use super::{
    CachedLightBlock, LIGHT_BLOCKED, LightAxisDirection, LightCacheLayout, LightDirectionSet,
    LightLayer, LightLayerWriteCache, LightQueueFlags, LightSectionReadCache, LightWorkset,
    MAX_LIGHT_LEVEL, PackedLightPropagationQueues, PackedLightQueueEntry, get_light_block_into,
    get_light_opacity, light_occlusion_shape,
};

/// Error returned when a block-light propagation context is built from mismatched caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLightPropagationContextError {
    /// Block-light propagation requires a block light write cache.
    WrongLayer {
        /// Layer supplied by the write cache.
        layer: LightLayer,
    },
    /// Section and light caches were built from different cache layouts.
    LayoutMismatch {
        /// Layout used by the section cache.
        section_layout: LightCacheLayout,
        /// Layout used by the light cache.
        light_layout: LightCacheLayout,
    },
}

/// Sections whose visible block-light data changed during a scoped update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLightUpdateResult {
    /// Light sections that should be reported to the world/chunk update layer.
    pub updated_sections: Vec<SectionPos>,
}

/// Runs ScalableLux-style block-light propagation for changed blocks in a scoped workset.
///
/// This is the block-light equivalent of ScalableLux `propagateBlockChanges`
/// plus `updateVisible`: it assumes the caller already created a cache window
/// around the affected chunk and will deliver returned section updates to the
/// world/chunk notification layer.
pub fn propagate_block_light_changes(
    workset: &LightWorkset,
    positions: impl IntoIterator<Item = BlockPos>,
) -> Result<BlockLightUpdateResult, BlockLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        BlockLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    for position in positions {
                        context.check_block(position);
                    }
                    context.perform_light_decrease();
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(BlockLightUpdateResult { updated_sections })
            })
        })
    })
}

/// ScalableLux-style block-light propagation over scoped Steel light caches.
///
/// This keeps the queue algorithm close to ScalableLux while avoiding long-lived
/// references into chunks: the caller owns the scoped section and light caches,
/// and this context only borrows them for one propagation pass.
pub struct BlockLightPropagationContext<'a, 'sections, 'light> {
    layout: LightCacheLayout,
    sections: &'a LightSectionReadCache<'sections>,
    light: &'a mut LightLayerWriteCache<'light>,
    queues: &'a mut PackedLightPropagationQueues,
}

impl<'a, 'sections, 'light> BlockLightPropagationContext<'a, 'sections, 'light> {
    /// Creates a block-light propagation context from matching scoped caches.
    pub fn new(
        sections: &'a LightSectionReadCache<'sections>,
        light: &'a mut LightLayerWriteCache<'light>,
        queues: &'a mut PackedLightPropagationQueues,
    ) -> Result<Self, BlockLightPropagationContextError> {
        if light.layer() != LightLayer::Block {
            return Err(BlockLightPropagationContextError::WrongLayer {
                layer: light.layer(),
            });
        }

        if sections.layout() != light.layout() {
            return Err(BlockLightPropagationContextError::LayoutMismatch {
                section_layout: sections.layout(),
                light_layout: light.layout(),
            });
        }

        Ok(Self {
            layout: light.layout(),
            sections,
            light,
            queues,
        })
    }

    /// Handles one block-light source/opacity change, matching ScalableLux `checkBlock`.
    ///
    /// Returns false when the changed block is outside this cache window.
    pub fn check_block(&mut self, block_pos: BlockPos) -> bool {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return false;
        };

        let current_level = self.light.get_updating(cached_block);
        let block_state = self.sections.get_block_state(cached_block);
        let emitted_level = block_state.get_light_emission() & MAX_LIGHT_LEVEL;

        self.light.set(cached_block, emitted_level);
        if emitted_level != 0 {
            self.enqueue_increase(
                block_pos,
                emitted_level,
                LightDirectionSet::all(),
                Self::shape_flags(block_state),
            );
        }

        self.enqueue_decrease(
            block_pos,
            current_level,
            LightDirectionSet::all(),
            LightQueueFlags::EMPTY,
        );
        true
    }

    /// Calculates the block-light value that should exist at `block_pos`.
    ///
    /// Returns `None` when the position is outside this cache window.
    #[must_use]
    pub fn calculate_light_value(&self, block_pos: BlockPos, expect: u8) -> Option<u8> {
        let cached_block = self.layout.cached_block(block_pos)?;
        let center_state = self.sections.get_block_state(cached_block);
        let mut level = center_state.get_light_emission() & MAX_LIGHT_LEVEL;

        if level >= MAX_LIGHT_LEVEL - 1 || level > expect {
            return Some(level);
        }

        let opacity = get_light_opacity(center_state);
        if opacity >= MAX_LIGHT_LEVEL {
            return Some(level);
        }

        for axis_direction in LightAxisDirection::ALL {
            let neighbor_pos = Self::offset(block_pos, axis_direction);
            let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                continue;
            };
            let neighbor_level = self.light.get_updating(neighbor_block);
            if neighbor_level.saturating_sub(1) <= level {
                continue;
            }

            let neighbor_state = self.sections.get_block_state(neighbor_block);
            let direction_from_neighbor = axis_direction.opposite().direction();
            if get_light_block_into(
                neighbor_state,
                center_state,
                direction_from_neighbor,
                opacity,
            ) == LIGHT_BLOCKED
            {
                continue;
            }

            level = level.max(neighbor_level.saturating_sub(opacity));
            if level > expect {
                return Some(level);
            }
        }

        Some(level)
    }

    /// Performs queued ScalableLux block-light decreases, then re-propagates increases.
    pub fn perform_light_decrease(&mut self) {
        while let Some(entry) = self.queues.dequeue_decrease() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            let source_state = if entry.has_sided_transparent_blocks() {
                Some(self.sections.get_block_state(source_block))
            } else {
                None
            };

            for axis_direction in entry.directions().directions() {
                let neighbor_pos = Self::offset(source_block.block_pos, axis_direction);
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    continue;
                };
                if !self.light.has_non_null_updating(neighbor_block) {
                    continue;
                }
                let current_level = self.light.get_updating(neighbor_block);
                if current_level == 0 {
                    continue;
                }

                let neighbor_state = self.sections.get_block_state(neighbor_block);
                let Some((target_level, flags)) = Self::target_level(
                    entry.level(),
                    source_state,
                    neighbor_state,
                    axis_direction.direction(),
                    true,
                ) else {
                    continue;
                };

                if current_level > target_level {
                    self.enqueue_increase(
                        neighbor_pos,
                        current_level,
                        LightDirectionSet::all(),
                        flags.with(LightQueueFlags::RECHECK_LEVEL),
                    );
                    continue;
                }

                let emitted_light = neighbor_state.get_light_emission() & MAX_LIGHT_LEVEL;
                if emitted_light != 0 {
                    self.enqueue_increase(
                        neighbor_pos,
                        emitted_light,
                        LightDirectionSet::all(),
                        flags.with(LightQueueFlags::WRITE_LEVEL),
                    );
                }

                self.light.set(neighbor_block, 0);
                if target_level > 0 {
                    self.enqueue_decrease(
                        neighbor_pos,
                        target_level,
                        LightDirectionSet::all_except_opposite(axis_direction),
                        flags,
                    );
                }
            }
        }

        self.perform_light_increase();
    }

    /// Performs queued ScalableLux block-light increases.
    pub fn perform_light_increase(&mut self) {
        while let Some(entry) = self.queues.dequeue_increase() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            if entry.should_recheck_level() {
                if self.light.get_updating(source_block) != entry.level() {
                    continue;
                }
            } else if entry.should_write_level() {
                self.light.set(source_block, entry.level());
            }

            let source_state = if entry.has_sided_transparent_blocks() {
                Some(self.sections.get_block_state(source_block))
            } else {
                None
            };

            for axis_direction in entry.directions().directions() {
                let neighbor_pos = Self::offset(source_block.block_pos, axis_direction);
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    continue;
                };
                if !self.light.has_non_null_updating(neighbor_block) {
                    continue;
                }
                let current_level = self.light.get_updating(neighbor_block);
                if current_level >= entry.level().saturating_sub(1) {
                    continue;
                }

                let neighbor_state = self.sections.get_block_state(neighbor_block);
                let Some((target_level, flags)) = Self::target_level(
                    entry.level(),
                    source_state,
                    neighbor_state,
                    axis_direction.direction(),
                    false,
                ) else {
                    continue;
                };
                if target_level <= current_level {
                    continue;
                }

                self.light.set(neighbor_block, target_level);
                if target_level > 1 {
                    self.enqueue_increase(
                        neighbor_pos,
                        target_level,
                        LightDirectionSet::all_except_opposite(axis_direction),
                        flags,
                    );
                }
            }
        }
    }

    fn cached_block_from_entry(&self, entry: PackedLightQueueEntry) -> Option<CachedLightBlock> {
        self.layout.cached_block_from_packed(entry.block_pos())
    }

    fn enqueue_decrease(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return;
        };
        self.queues
            .enqueue_decrease(PackedLightQueueEntry::from_parts(
                packed_pos, level, directions, flags,
            ));
    }

    fn enqueue_increase(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return;
        };
        self.queues
            .enqueue_increase(PackedLightQueueEntry::from_parts(
                packed_pos, level, directions, flags,
            ));
    }

    fn target_level(
        propagated_level: u8,
        source_state: Option<steel_utils::BlockStateId>,
        target_state: steel_utils::BlockStateId,
        direction: Direction,
        saturating: bool,
    ) -> Option<(u8, LightQueueFlags)> {
        let source_state = source_state.unwrap_or_else(Self::air);
        let opacity = get_light_block_into(
            source_state,
            target_state,
            direction,
            get_light_opacity(target_state),
        );
        if opacity == LIGHT_BLOCKED {
            return None;
        }

        let target_level = if saturating {
            propagated_level.saturating_sub(opacity)
        } else if opacity >= propagated_level {
            return None;
        } else {
            propagated_level - opacity
        };

        Some((target_level, Self::shape_flags(target_state)))
    }

    fn shape_flags(block_state: steel_utils::BlockStateId) -> LightQueueFlags {
        if light_occlusion_shape(block_state).is_empty() {
            LightQueueFlags::EMPTY
        } else {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        }
    }

    fn offset(block_pos: BlockPos, direction: LightAxisDirection) -> BlockPos {
        let (dx, dy, dz) = direction.offset();
        block_pos.offset(dx, dy, dz)
    }

    fn air() -> steel_utils::BlockStateId {
        vanilla_blocks::AIR.default_state()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use steel_registry::{
        blocks::properties::{BlockStateProperties, SlabType},
        test_support::init_test_registry,
        vanilla_blocks,
    };
    use steel_utils::{ChunkPos, types::UpdateFlags};

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::chunk::{
        chunk_access::{ChunkAccess, ChunkStatus},
        chunk_holder::ChunkHolder,
        light::{LightCacheSetupRadius, LightSectionRange, LightWorkset},
        proto_chunk::ProtoChunk,
        section::{ChunkSection, Sections},
    };

    fn init_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn range() -> LightSectionRange {
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        range
    }

    fn holder_with_section(pos: ChunkPos, section: ChunkSection) -> Arc<ChunkHolder> {
        let sections = Sections::from_owned(vec![section].into_boxed_slice());
        let proto = ProtoChunk::new(sections, pos, 0, 16, Weak::new());
        let holder = Arc::new(ChunkHolder::new(pos, 0, 0, 16));
        holder.insert_chunk(ChunkAccess::Proto(proto), ChunkStatus::Light);
        holder
    }

    fn set_block_nibble_non_null(holder: &ChunkHolder, section_y: i32) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let mut light = chunk.light_mut();
        let Some(nibble) = light.block.nibble_mut(section_y) else {
            panic!("test nibble should be inside light range");
        };
        nibble.set_non_null();
    }

    fn set_visible_block_light(
        holder: &ChunkHolder,
        section_y: i32,
        x: usize,
        y: usize,
        z: usize,
        level: u8,
    ) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let mut light = chunk.light_mut();
        let Some(nibble) = light.block.nibble_mut(section_y) else {
            panic!("test nibble should be inside light range");
        };
        nibble.set_non_null();
        nibble.set(x, y, z, level);
        assert!(nibble.update_visible());
    }

    #[test]
    fn context_requires_block_layer() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        set_block_nibble_non_null(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let result =
                        BlockLightPropagationContext::new(section_cache, light_cache, &mut queues);

                    assert_eq!(
                        result.err(),
                        Some(BlockLightPropagationContextError::WrongLayer {
                            layer: LightLayer::Sky,
                        })
                    );
                });
            });
        });
    }

    #[test]
    fn block_light_check_propagates_emission_through_air() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_nibble_non_null(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let Ok(mut context) =
                        BlockLightPropagationContext::new(section_cache, light_cache, &mut queues)
                    else {
                        panic!("matching block caches should build a propagation context");
                    };

                    assert!(context.check_block(source_pos));
                    context.perform_light_decrease();

                    let Some(source) = layout.cached_block(source_pos) else {
                        panic!("source should be cached");
                    };
                    let Some(east) = layout.cached_block(BlockPos::new(2, 1, 1)) else {
                        panic!("east neighbor should be cached");
                    };
                    let Some(two_east) = layout.cached_block(BlockPos::new(3, 1, 1)) else {
                        panic!("second east neighbor should be cached");
                    };

                    assert_eq!(context.light.get_updating(source), 15);
                    assert_eq!(context.light.get_updating(east), 14);
                    assert_eq!(context.light.get_updating(two_east), 13);
                });
            });
        });
    }

    #[test]
    fn block_light_runner_publishes_visible_updates() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(1, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
        let holder = holder_with_section(center, section);
        set_block_nibble_non_null(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_block_light_changes(&workset, [source_pos]) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(result.updated_sections, vec![SectionPos::new(0, 0, 0)]);

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.block.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(source) = layout.cached_block(source_pos) else {
            panic!("source should be cached");
        };
        let Some(east) = layout.cached_block(BlockPos::new(2, 1, 1)) else {
            panic!("east neighbor should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(source.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(east.local_index), 14);
    }

    #[test]
    fn block_light_runner_repropagates_after_opacity_decrease() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let source_pos = BlockPos::new(0, 1, 1);
        let opened_pos = BlockPos::new(1, 1, 1);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(0, 1, 1, vanilla_blocks::LIGHT.default_state());
        section.set_block_state(1, 1, 1, vanilla_blocks::STONE.default_state());
        let holder = holder_with_section(center, section);
        set_visible_block_light(&holder, 0, 0, 1, 1, 15);
        let layout = LightCacheLayout::new(center, range());

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        assert_eq!(
            chunk.set_block_state(
                opened_pos,
                vanilla_blocks::AIR.default_state(),
                UpdateFlags::UPDATE_NONE,
            ),
            Some(vanilla_blocks::STONE.default_state())
        );
        drop(chunk);

        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_block_light_changes(&workset, [opened_pos]) else {
            panic!("matching block caches should run block light updates");
        };

        assert_eq!(result.updated_sections, vec![SectionPos::new(0, 0, 0)]);

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.block.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(source) = layout.cached_block(source_pos) else {
            panic!("source should be cached");
        };
        let Some(opened) = layout.cached_block(opened_pos) else {
            panic!("opened block should be cached");
        };
        let Some(east) = layout.cached_block(BlockPos::new(2, 1, 1)) else {
            panic!("east neighbor should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(source.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(opened.local_index), 14);
        assert_eq!(nibble.get_visible_at_index(east.local_index), 13);
    }

    #[test]
    fn block_light_calculation_respects_occluding_faces() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        let bottom_slab = vanilla_blocks::STONE_SLAB
            .default_state()
            .set_value(&BlockStateProperties::SLAB_TYPE, SlabType::Bottom);
        section.set_block_state(1, 1, 1, bottom_slab);
        let holder = holder_with_section(center, section);
        set_block_nibble_non_null(&holder, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let Ok(context) =
                        BlockLightPropagationContext::new(section_cache, light_cache, &mut queues)
                    else {
                        panic!("matching block caches should build a propagation context");
                    };
                    let Some(below) = layout.cached_block(BlockPos::new(1, 0, 1)) else {
                        panic!("below neighbor should be cached");
                    };
                    assert!(context.light.set(below, 15));

                    assert_eq!(
                        context.calculate_light_value(BlockPos::new(1, 1, 1), 0),
                        Some(0)
                    );
                });
            });
        });
    }
}

use steel_registry::{blocks::block_state_ext::BlockStateExt, vanilla_blocks};
use steel_utils::{BlockPos, ChunkPos, Direction, SectionPos};

use super::{
    CachedLightBlock, LIGHT_BLOCKED, LightAxisDirection, LightCacheLayout, LightDirectionSet,
    LightLayer, LightLayerWriteCache, LightQueueFlags, LightSectionReadCache, LightWorkset,
    MAX_LIGHT_LEVEL, PackedLightPropagationQueues, PackedLightQueueEntry, get_light_block_into,
    get_light_opacity, light_occlusion_shape,
};

/// Error returned when a sky-light propagation context is built from mismatched caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyLightPropagationContextError {
    /// Sky-light propagation requires a sky light write cache.
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
    /// The workset does not contain its center chunk.
    MissingCenterChunk {
        /// Missing center chunk position.
        chunk_pos: ChunkPos,
    },
}

/// Sections whose visible sky-light data changed during a scoped update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkyLightUpdateResult {
    /// Light sections that should be reported to the world/chunk update layer.
    pub updated_sections: Vec<SectionPos>,
}

/// Seeds and propagates sky light for the center chunk without edge checks.
///
/// This matches ScalableLux `SkyStarLightEngine.lightChunk` for the
/// no-edge-check path used during ordered relighting: sky nibbles around
/// non-empty sections are initialized, full skylight is propagated down through
/// transparent columns, initialized neighbor levels are pulled inward, and
/// dirty visible nibbles are published.
pub fn propagate_sky_light_chunk_without_edge_checks(
    workset: &LightWorkset,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = chunk_cache.layout();
        let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        };
        if chunk_cache.chunk(center_slot).is_none() {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        }

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    context.initialize_unlit_chunk_nibbles(layout.center_chunk());
                    context.light_chunk_without_edge_checks(layout.center_chunk());
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(SkyLightUpdateResult { updated_sections })
            })
        })
    })
}

/// ScalableLux-style sky-light propagation over scoped Steel light caches.
pub struct SkyLightPropagationContext<'a, 'sections, 'light> {
    layout: LightCacheLayout,
    sections: &'a LightSectionReadCache<'sections>,
    light: &'a mut LightLayerWriteCache<'light>,
    queues: &'a mut PackedLightPropagationQueues,
    null_section_checked: Vec<bool>,
}

impl<'a, 'sections, 'light> SkyLightPropagationContext<'a, 'sections, 'light> {
    /// Creates a sky-light propagation context from matching scoped caches.
    pub fn new(
        sections: &'a LightSectionReadCache<'sections>,
        light: &'a mut LightLayerWriteCache<'light>,
        queues: &'a mut PackedLightPropagationQueues,
    ) -> Result<Self, SkyLightPropagationContextError> {
        if light.layer() != LightLayer::Sky {
            return Err(SkyLightPropagationContextError::WrongLayer {
                layer: light.layer(),
            });
        }

        if sections.layout() != light.layout() {
            return Err(SkyLightPropagationContextError::LayoutMismatch {
                section_layout: sections.layout(),
                light_layout: light.layout(),
            });
        }

        let layout = light.layout();
        let section_count = layout.range().section_count();

        Ok(Self {
            layout,
            sections,
            light,
            queues,
            null_section_checked: vec![false; section_count],
        })
    }

    /// Initializes the sky nibbles required around non-empty center sections.
    pub fn initialize_unlit_chunk_nibbles(&mut self, chunk_pos: ChunkPos) {
        for section_y in (self.layout.range().min_chunk_section_y()
            ..self.layout.range().max_chunk_section_y_exclusive())
            .rev()
        {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.sections.has_non_empty_section(section_pos) {
                continue;
            }

            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    let extrude = (offset_x | offset_z) != 0;
                    for offset_y in (-1..=1).rev() {
                        self.init_nibble(
                            SectionPos::new(
                                chunk_pos.0.x + offset_x,
                                section_y + offset_y,
                                chunk_pos.0.y + offset_z,
                            ),
                            extrude,
                        );
                    }
                }
            }
        }
    }

    /// Runs the no-edge-check sky chunk path.
    pub fn light_chunk_without_edge_checks(&mut self, chunk_pos: ChunkPos) {
        let min_section = self.layout.range().min_chunk_section_y();
        let mut highest_non_empty_section = self.layout.range().max_chunk_section_y_exclusive() - 1;

        loop {
            let section_pos =
                SectionPos::new(chunk_pos.0.x, highest_non_empty_section, chunk_pos.0.y);
            if highest_non_empty_section != min_section - 1
                && self.sections.has_non_empty_section(section_pos)
            {
                break;
            }

            self.check_null_section(chunk_pos, highest_non_empty_section, false);
            self.propagate_full_empty_section_edges(chunk_pos, highest_non_empty_section);

            if highest_non_empty_section == min_section - 1 {
                highest_non_empty_section -= 1;
                break;
            }
            highest_non_empty_section -= 1;
        }

        if highest_non_empty_section >= min_section {
            let min_x = chunk_pos.0.x << 4;
            let max_x = min_x | 15;
            let min_z = chunk_pos.0.y << 4;
            let max_z = min_z | 15;
            let start_y = (highest_non_empty_section << 4) | 15;
            for z in min_z..=max_z {
                for x in min_x..=max_x {
                    self.try_propagate_skylight(x, start_y + 1, z, false);
                }
            }
        }

        for section_y in (self.layout.range().min_section_y()..=highest_non_empty_section).rev() {
            self.check_null_section(chunk_pos, section_y, false);
        }
        self.propagate_neighbor_levels(
            chunk_pos,
            self.layout.range().min_section_y(),
            highest_non_empty_section,
        );
        self.perform_light_increase();
    }

    fn init_nibble(&mut self, section_pos: SectionPos, extrude: bool) {
        if self.layout.section_slot(section_pos).is_none()
            || !self.light.has_cached_section(section_pos)
        {
            return;
        }
        if !self.light.is_section_null_updating(section_pos) {
            return;
        }

        let mut highest_non_empty_section = self.layout.range().min_section_y() - 1;
        for section_y in (self.layout.range().min_chunk_section_y()
            ..self.layout.range().max_chunk_section_y_exclusive())
            .rev()
        {
            let candidate = SectionPos::new(section_pos.x(), section_y, section_pos.z());
            if self.sections.has_non_empty_section(candidate) {
                highest_non_empty_section = section_y;
                break;
            }
        }

        if section_pos.y() > highest_non_empty_section {
            self.light.set_section_non_null(section_pos);
            self.light.fill_section(section_pos, MAX_LIGHT_LEVEL);
        } else if extrude {
            self.light
                .extrude_lower_from_first_section_above(section_pos);
        } else {
            self.light.set_section_non_null(section_pos);
        }
    }

    fn check_null_section(
        &mut self,
        chunk_pos: ChunkPos,
        section_y: i32,
        extrude_initialized: bool,
    ) -> bool {
        let Some(section_index) = self.layout.range().section_index(section_y) else {
            return false;
        };
        if self.null_section_checked[section_index] {
            return false;
        }
        self.null_section_checked[section_index] = true;

        let mut need_init_neighbors = false;
        'search: for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                let section_pos = SectionPos::new(
                    chunk_pos.0.x + offset_x,
                    section_y,
                    chunk_pos.0.y + offset_z,
                );
                if self.light.has_non_null_section(section_pos) {
                    need_init_neighbors = true;
                    break 'search;
                }
            }
        }

        if need_init_neighbors {
            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    self.init_nibble(
                        SectionPos::new(
                            chunk_pos.0.x + offset_x,
                            section_y,
                            chunk_pos.0.y + offset_z,
                        ),
                        (offset_x | offset_z) == 0 && extrude_initialized,
                    );
                }
            }
        }

        need_init_neighbors
    }

    fn propagate_full_empty_section_edges(&mut self, chunk_pos: ChunkPos, section_y: i32) {
        for direction in LightAxisDirection::HORIZONTAL {
            let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
            let neighbor_section_pos = SectionPos::new(
                chunk_pos.0.x + neighbor_offset_x,
                section_y,
                chunk_pos.0.y + neighbor_offset_z,
            );
            if !self.light.has_non_null_section(neighbor_section_pos) {
                continue;
            }

            let (increment_x, increment_z, start_x, start_z) =
                Self::current_edge_scan(chunk_pos, direction);
            let directions = LightDirectionSet::only(direction);
            let min_y = section_y << 4;
            let max_y = min_y | 15;
            for y in min_y..=max_y {
                let mut x = start_x;
                let mut z = start_z;
                for _ in 0..16 {
                    self.enqueue_increase(
                        x,
                        y,
                        z,
                        MAX_LIGHT_LEVEL,
                        directions,
                        LightQueueFlags::EMPTY,
                    );
                    x += increment_x;
                    z += increment_z;
                }
            }
        }
    }

    fn try_propagate_skylight(
        &mut self,
        x: i32,
        mut y: i32,
        z: i32,
        extrude_initialized: bool,
    ) -> i32 {
        if self.get_light_level_extruded(BlockPos::new(x, y + 1, z)) != MAX_LIGHT_LEVEL {
            return y;
        }

        self.check_null_section(
            ChunkPos::new(
                SectionPos::block_to_section_coord(x),
                SectionPos::block_to_section_coord(z),
            ),
            SectionPos::block_to_section_coord(y),
            extrude_initialized,
        );

        let mut above_state = self.block_state(BlockPos::new(x, y + 1, z));
        while y >= (self.layout.range().min_section_y() << 4) {
            if (y & 15) == 15 {
                self.check_null_section(
                    ChunkPos::new(
                        SectionPos::block_to_section_coord(x),
                        SectionPos::block_to_section_coord(z),
                    ),
                    SectionPos::block_to_section_coord(y),
                    extrude_initialized,
                );
            }

            let current_pos = BlockPos::new(x, y, z);
            let current_state = self.block_state(current_pos);
            let opacity = current_state.get_light_dampening();
            if get_light_block_into(above_state, current_state, Direction::Down, opacity)
                == LIGHT_BLOCKED
                || opacity > 0
            {
                break;
            }

            let section_pos = SectionPos::from_block_pos(current_pos);
            if !self.light.has_non_null_section(section_pos) {
                y &= !15;
                above_state = Self::air();
            } else {
                let Some(cached_block) = self.layout.cached_block(current_pos) else {
                    break;
                };
                self.light.set(cached_block, MAX_LIGHT_LEVEL);
                self.enqueue_increase(
                    x,
                    y,
                    z,
                    MAX_LIGHT_LEVEL,
                    LightDirectionSet::all_except(LightAxisDirection::PositiveY),
                    Self::shape_flags(current_state),
                );
                above_state = current_state;
            }

            y -= 1;
        }

        y
    }

    fn get_light_level_extruded(&self, block_pos: BlockPos) -> u8 {
        let mut section_y = SectionPos::block_to_section_coord(block_pos.y());
        let section_x = SectionPos::block_to_section_coord(block_pos.x());
        let section_z = SectionPos::block_to_section_coord(block_pos.z());

        if let Some(cached_block) = self.layout.cached_block(block_pos)
            && self
                .light
                .has_non_null_section(SectionPos::new(section_x, section_y, section_z))
        {
            return self.light.get_updating(cached_block);
        }

        loop {
            section_y += 1;
            if section_y >= self.layout.range().max_section_y_exclusive() {
                return MAX_LIGHT_LEVEL;
            }

            let section_pos = SectionPos::new(section_x, section_y, section_z);
            if !self.light.has_non_null_section(section_pos) {
                continue;
            }
            let block_above = BlockPos::new(block_pos.x(), section_y << 4, block_pos.z());
            let Some(cached_block) = self.layout.cached_block(block_above) else {
                continue;
            };
            return self.light.get_updating(cached_block);
        }
    }

    fn propagate_neighbor_levels(
        &mut self,
        chunk_pos: ChunkPos,
        from_section: i32,
        to_section: i32,
    ) {
        for section_y in (from_section..=to_section).rev() {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.light.has_cached_section(section_pos) {
                continue;
            }

            for direction in LightAxisDirection::HORIZONTAL {
                self.propagate_neighbor_level_section(chunk_pos, section_y, direction);
            }
        }
    }

    fn propagate_neighbor_level_section(
        &mut self,
        chunk_pos: ChunkPos,
        section_y: i32,
        direction: LightAxisDirection,
    ) {
        let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
        let neighbor_section_pos = SectionPos::new(
            chunk_pos.0.x + neighbor_offset_x,
            section_y,
            chunk_pos.0.y + neighbor_offset_z,
        );
        if !self
            .light
            .is_section_initialized_updating(neighbor_section_pos)
        {
            return;
        }

        let (increment_x, increment_z, start_x, start_z) =
            Self::neighbor_edge_scan(chunk_pos, direction);
        let directions = LightDirectionSet::only(direction.opposite());
        let flags = LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);

        let min_y = section_y << 4;
        let max_y = min_y | 15;
        for y in min_y..=max_y {
            let mut x = start_x;
            let mut z = start_z;
            for _ in 0..16 {
                let source_pos = BlockPos::new(x, y, z);
                let Some(source_block) = self.layout.cached_block(source_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };
                let level = self.light.get_updating(source_block);
                if level > 1 {
                    self.enqueue_increase(x, y, z, level, directions, flags);
                }
                x += increment_x;
                z += increment_z;
            }
        }
    }

    fn perform_light_increase(&mut self) {
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
                ) else {
                    continue;
                };
                if target_level <= current_level {
                    continue;
                }

                self.light.set(neighbor_block, target_level);
                if target_level > 1 {
                    self.enqueue_increase(
                        neighbor_pos.x(),
                        neighbor_pos.y(),
                        neighbor_pos.z(),
                        target_level,
                        LightDirectionSet::all_except_opposite(axis_direction),
                        flags,
                    );
                }
            }
        }
    }

    fn target_level(
        propagated_level: u8,
        source_state: Option<steel_utils::BlockStateId>,
        target_state: steel_utils::BlockStateId,
        direction: Direction,
    ) -> Option<(u8, LightQueueFlags)> {
        let source_state = source_state.unwrap_or_else(Self::air);
        let opacity = get_light_block_into(
            source_state,
            target_state,
            direction,
            get_light_opacity(target_state),
        );
        if opacity == LIGHT_BLOCKED || opacity >= propagated_level {
            return None;
        }

        Some((propagated_level - opacity, Self::shape_flags(target_state)))
    }

    fn cached_block_from_entry(&self, entry: PackedLightQueueEntry) -> Option<CachedLightBlock> {
        self.layout.cached_block_from_packed(entry.block_pos())
    }

    fn enqueue_increase(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) {
        let Some(packed_pos) = self.layout.encode_block_pos(BlockPos::new(x, y, z)) else {
            return;
        };
        self.queues
            .enqueue_increase(PackedLightQueueEntry::from_parts(
                packed_pos, level, directions, flags,
            ));
    }

    fn block_state(&self, block_pos: BlockPos) -> steel_utils::BlockStateId {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return Self::air();
        };
        self.sections.get_block_state(cached_block)
    }

    fn current_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                chunk_pos.0.x << 4
            } else {
                (chunk_pos.0.x << 4) | 15
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            chunk_pos.0.y << 4
        } else {
            (chunk_pos.0.y << 4) | 15
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
    }

    fn neighbor_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                (chunk_pos.0.x << 4) - 1
            } else {
                (chunk_pos.0.x << 4) + 16
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            (chunk_pos.0.y << 4) - 1
        } else {
            (chunk_pos.0.y << 4) + 16
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
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

    use steel_registry::{test_support::init_test_registry, vanilla_blocks};

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

    #[test]
    fn context_requires_sky_layer() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
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
                    let result =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues);

                    assert_eq!(
                        result.err(),
                        Some(SkyLightPropagationContextError::WrongLayer {
                            layer: LightLayer::Block,
                        })
                    );
                });
            });
        });
    }

    #[test]
    fn sky_light_chunk_without_edge_checks_propagates_down_air_column() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 0, 1, vanilla_blocks::STONE.default_state());
        let holder = holder_with_section(center, section);
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

        let Ok(result) = propagate_sky_light_chunk_without_edge_checks(&workset) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(top_air) = layout.cached_block(BlockPos::new(1, 15, 1)) else {
            panic!("top air block should be cached");
        };
        let Some(lower_air) = layout.cached_block(BlockPos::new(1, 1, 1)) else {
            panic!("lower air block should be cached");
        };
        let Some(stone) = layout.cached_block(BlockPos::new(1, 0, 1)) else {
            panic!("stone block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(top_air.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(lower_air.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(stone.local_index), 0);
    }

    #[test]
    fn sky_light_chunk_requires_center_chunk() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |_| None,
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing chunks");
        };

        assert_eq!(
            propagate_sky_light_chunk_without_edge_checks(&workset).err(),
            Some(SkyLightPropagationContextError::MissingCenterChunk { chunk_pos: center })
        );
    }
}

use std::sync::Arc;

use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use steel_registry::{REGISTRY, vanilla_blocks};
use steel_utils::{BlockStateId, ChunkPos};

use crate::chunk::{
    chunk_access::{ChunkAccess, ChunkStatus},
    chunk_holder::ChunkHolder,
    section::ChunkSection,
};

use super::{
    CachedLightBlock, CachedLightChunk, ChunkLightData, ChunkLightLayerStorage,
    LightCacheChunkScope, LightCacheLayout, LightCacheSetupRadius, LightChunkSlotArray, LightLayer,
    LightNibbleArray, LightSectionSlotArray,
};

/// Error returned when a scoped light cache cannot acquire required chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightWorksetSetupError {
    /// A chunk inside ScalableLux's required 1-radius cache was unavailable.
    MissingRequiredChunk {
        /// Missing chunk position.
        chunk_pos: ChunkPos,
    },
}

/// Scoped chunk access for one ScalableLux-style lighting operation.
///
/// Unlike ScalableLux's Java engine, Steel does not cache long-lived references
/// directly into chunk internals. This workset owns temporary `ChunkHolder`
/// references so chunk unload/save accounting still observes active lighting
/// users through the existing holder `Arc` lifecycle, then locks chunk data
/// only inside scoped cache closures.
pub struct LightWorkset {
    layout: LightCacheLayout,
    chunks: LightChunkSlotArray<Arc<ChunkHolder>>,
}

impl LightWorkset {
    /// Creates a scoped cache window by scanning chunks in ScalableLux setup order.
    pub fn setup(
        layout: LightCacheLayout,
        radius: LightCacheSetupRadius,
        relaxed: bool,
        mut chunk_for_lighting: impl FnMut(ChunkPos) -> Option<Arc<ChunkHolder>>,
        mut can_use_chunk: impl FnMut(&ChunkAccess) -> bool,
    ) -> Result<Self, LightWorksetSetupError> {
        let mut chunks = LightChunkSlotArray::new();

        for cached_chunk in layout.setup_chunks(radius) {
            let Some(holder) = Self::try_get_chunk(cached_chunk, relaxed, &mut chunk_for_lighting)?
            else {
                continue;
            };

            let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
                continue;
            };
            if !can_use_chunk(&chunk) {
                continue;
            }
            drop(chunk);

            chunks.insert(cached_chunk, holder);
        }

        Ok(Self { layout, chunks })
    }

    /// Returns this workset's cache layout.
    #[must_use]
    pub const fn layout(&self) -> LightCacheLayout {
        self.layout
    }

    /// Returns the holder for a cached chunk slot.
    #[must_use]
    pub fn chunk_holder(&self, cached_chunk: CachedLightChunk) -> Option<&Arc<ChunkHolder>> {
        self.chunks.get(cached_chunk)
    }

    /// Builds a chunk-read cache for the duration of `f`.
    ///
    /// Chunk locks are acquired in cache-slot order and released before this
    /// method returns. The workset keeps holder `Arc`s alive, while this cache
    /// keeps the guarded chunk data stable during the lighting operation.
    pub fn with_chunk_read_cache<R>(&self, f: impl FnOnce(&LightChunkReadCache<'_>) -> R) -> R {
        let mut chunks = LightChunkSlotArray::new();

        for chunk_slot in 0..self.chunks.slot_count() {
            let Some(holder) = self.chunks.get_slot(chunk_slot) else {
                continue;
            };
            let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
                continue;
            };
            chunks.insert_slot(chunk_slot, chunk);
        }

        let cache = LightChunkReadCache {
            layout: self.layout,
            chunks,
        };
        f(&cache)
    }

    fn try_get_chunk(
        cached_chunk: CachedLightChunk,
        relaxed: bool,
        chunk_for_lighting: &mut impl FnMut(ChunkPos) -> Option<Arc<ChunkHolder>>,
    ) -> Result<Option<Arc<ChunkHolder>>, LightWorksetSetupError> {
        let required = !relaxed && cached_chunk.scope == LightCacheChunkScope::Inner;
        let holder = chunk_for_lighting(cached_chunk.chunk_pos)
            .filter(|holder| holder.try_chunk(ChunkStatus::Empty).is_some());

        if holder.is_none() && required {
            return Err(LightWorksetSetupError::MissingRequiredChunk {
                chunk_pos: cached_chunk.chunk_pos,
            });
        }

        Ok(holder)
    }
}

/// Flat cached chunk reads for one scoped lighting operation.
pub struct LightChunkReadCache<'a> {
    layout: LightCacheLayout,
    chunks: LightChunkSlotArray<RwLockReadGuard<'a, ChunkAccess>>,
}

impl LightChunkReadCache<'_> {
    /// Returns this read cache's layout.
    #[must_use]
    pub const fn layout(&self) -> LightCacheLayout {
        self.layout
    }

    /// Returns the cached chunk for a chunk slot.
    #[must_use]
    pub fn chunk(&self, cached_chunk: CachedLightChunk) -> Option<&ChunkAccess> {
        self.chunks.get(cached_chunk).map(|chunk| &**chunk)
    }

    /// Builds a section-read cache for the duration of `f`.
    ///
    /// Section locks are acquired in cache-slot order and released before this
    /// method returns. This keeps ScalableLux-style flat section indexing
    /// without storing self-referential borrows inside [`LightWorkset`].
    pub fn with_section_read_cache<R>(&self, f: impl FnOnce(&LightSectionReadCache<'_>) -> R) -> R {
        let mut sections = LightSectionSlotArray::new(self.layout);

        for chunk_slot in 0..self.chunks.slot_count() {
            let Some(chunk_guard) = self.chunks.get_slot(chunk_slot) else {
                continue;
            };
            let Some(chunk_pos) = self.layout.chunk_pos_for_slot(chunk_slot) else {
                continue;
            };
            let Some(section_slots) = self.layout.inner_light_section_slots_for_chunk(chunk_pos)
            else {
                continue;
            };

            let chunk_sections = chunk_guard.sections();
            for cached_section in section_slots {
                let Some(section_index) = self
                    .layout
                    .range()
                    .chunk_section_index(cached_section.section_pos.y())
                else {
                    continue;
                };
                let Some(section) = chunk_sections.sections.get(section_index) else {
                    continue;
                };
                sections.insert(cached_section, section.read());
            }
        }

        let cache = LightSectionReadCache {
            layout: self.layout,
            sections,
        };
        f(&cache)
    }

    /// Builds a layer-specific light write cache for the duration of `f`.
    ///
    /// Light locks are acquired in chunk cache-slot order and released before
    /// this method returns. Callers that also need section reads should build
    /// the section cache first, then this light cache, matching chunk mutation
    /// lock ordering.
    pub fn with_light_write_cache<R>(
        &self,
        layer: LightLayer,
        f: impl FnOnce(&mut LightLayerWriteCache<'_>) -> R,
    ) -> R {
        let mut chunks = LightChunkSlotArray::new();

        for chunk_slot in 0..self.chunks.slot_count() {
            let Some(chunk_guard) = self.chunks.get_slot(chunk_slot) else {
                continue;
            };
            chunks.insert_slot(chunk_slot, chunk_guard.light_mut());
        }

        let nibbles = Self::build_nibble_cache(self.layout, layer, &chunks);
        let mut cache = LightLayerWriteCache {
            layout: self.layout,
            layer,
            chunks,
            nibbles,
        };
        f(&mut cache)
    }

    fn build_nibble_cache(
        layout: LightCacheLayout,
        layer: LightLayer,
        chunks: &LightChunkSlotArray<RwLockWriteGuard<'_, ChunkLightData>>,
    ) -> LightSectionSlotArray<LightNibbleCacheEntry> {
        let mut nibbles = LightSectionSlotArray::new(layout);

        for chunk_slot in 0..chunks.slot_count() {
            let Some(light_data) = chunks.get_slot(chunk_slot) else {
                continue;
            };
            let Some(chunk_pos) = layout.chunk_pos_for_slot(chunk_slot) else {
                continue;
            };
            let Some(section_slots) = layout.inner_light_section_slots_for_chunk(chunk_pos) else {
                continue;
            };

            let layer_storage = LightLayerWriteCache::layer_storage(light_data, layer);
            for cached_section in section_slots {
                let Some(nibble_index) = layer_storage
                    .range()
                    .section_index(cached_section.section_pos.y())
                else {
                    continue;
                };
                if layer_storage.nibbles().get(nibble_index).is_none() {
                    continue;
                }

                nibbles.insert(
                    cached_section,
                    LightNibbleCacheEntry {
                        chunk_slot,
                        nibble_index,
                    },
                );
            }
        }

        nibbles
    }
}

/// Flat cached chunk-section reads for ScalableLux-style block-state access.
pub struct LightSectionReadCache<'a> {
    layout: LightCacheLayout,
    sections: LightSectionSlotArray<RwLockReadGuard<'a, ChunkSection>>,
}

impl LightSectionReadCache<'_> {
    /// Returns this read cache's layout.
    #[must_use]
    pub const fn layout(&self) -> LightCacheLayout {
        self.layout
    }

    /// Returns the block state for a cached light block, or air for missing sections.
    #[must_use]
    pub fn get_block_state(&self, cached_block: CachedLightBlock) -> BlockStateId {
        let Some(section) = self.sections.get_slot(cached_block.section_slot) else {
            return Self::air();
        };

        if section.is_empty() {
            return Self::air();
        }

        section.states.get_at_index(cached_block.local_index)
    }

    fn air() -> BlockStateId {
        REGISTRY.blocks.get_base_state_id(&vanilla_blocks::AIR)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LightNibbleCacheEntry {
    chunk_slot: usize,
    nibble_index: usize,
}

/// Flat cached light-nibble writes for one layer and one scoped lighting operation.
pub struct LightLayerWriteCache<'a> {
    layout: LightCacheLayout,
    layer: LightLayer,
    chunks: LightChunkSlotArray<RwLockWriteGuard<'a, ChunkLightData>>,
    nibbles: LightSectionSlotArray<LightNibbleCacheEntry>,
}

impl LightLayerWriteCache<'_> {
    /// Returns this write cache's layout.
    #[must_use]
    pub const fn layout(&self) -> LightCacheLayout {
        self.layout
    }

    /// Returns this write cache's light layer.
    #[must_use]
    pub const fn layer(&self) -> LightLayer {
        self.layer
    }

    /// Returns an updating light value for a cached light block.
    #[must_use]
    pub fn get_updating(&self, cached_block: CachedLightBlock) -> u8 {
        self.get_updating_at_section_index(cached_block.section_slot, cached_block.local_index)
    }

    /// Returns true when a cached block has a non-null updating nibble.
    #[must_use]
    pub fn has_non_null_updating(&self, cached_block: CachedLightBlock) -> bool {
        self.nibble(cached_block.section_slot)
            .is_some_and(|nibble| !nibble.is_null_updating())
    }

    /// Returns an updating light value for a section slot and local nibble index.
    #[must_use]
    pub fn get_updating_at_section_index(&self, section_slot: usize, local_index: usize) -> u8 {
        let Some(nibble) = self.nibble(section_slot) else {
            return 0;
        };
        nibble.get_updating_at_index(local_index)
    }

    /// Sets an updating light value for a cached light block.
    ///
    /// Returns false when no writable non-null nibble was cached for the block.
    pub fn set(&mut self, cached_block: CachedLightBlock, level: u8) -> bool {
        self.set_at_section_index(cached_block.section_slot, cached_block.local_index, level)
    }

    /// Sets an updating light value for a section slot and local nibble index.
    ///
    /// Returns false when no writable non-null nibble was cached for the slot.
    pub fn set_at_section_index(
        &mut self,
        section_slot: usize,
        local_index: usize,
        level: u8,
    ) -> bool {
        let Some(nibble) = self.nibble_mut(section_slot) else {
            return false;
        };
        if nibble.is_null_updating() {
            return false;
        }

        nibble.set_at_index(local_index, level);
        true
    }

    /// Publishes every dirty cached nibble and returns the number updated.
    pub fn update_visible(&mut self) -> usize {
        let mut updated = 0;

        for section_slot in 0..self.nibbles.slot_count() {
            let Some(nibble) = self.nibble_mut(section_slot) else {
                continue;
            };
            if nibble.update_visible() {
                updated += 1;
            }
        }

        updated
    }

    fn nibble(&self, section_slot: usize) -> Option<&LightNibbleArray> {
        let entry = self.nibbles.get_slot(section_slot)?;
        let light_data = self.chunks.get_slot(entry.chunk_slot)?;
        Self::layer_storage(light_data, self.layer)
            .nibbles()
            .get(entry.nibble_index)
    }

    fn nibble_mut(&mut self, section_slot: usize) -> Option<&mut LightNibbleArray> {
        let entry = *self.nibbles.get_slot(section_slot)?;
        let layer = self.layer;
        let light_data = self.chunks.get_mut_slot(entry.chunk_slot)?;
        Self::layer_storage_mut(light_data, layer)
            .nibbles_mut()
            .get_mut(entry.nibble_index)
    }

    fn layer_storage(light_data: &ChunkLightData, layer: LightLayer) -> &ChunkLightLayerStorage {
        match layer {
            LightLayer::Sky => &light_data.sky,
            LightLayer::Block => &light_data.block,
        }
    }

    fn layer_storage_mut(
        light_data: &mut ChunkLightData,
        layer: LightLayer,
    ) -> &mut ChunkLightLayerStorage {
        match layer {
            LightLayer::Sky => &mut light_data.sky,
            LightLayer::Block => &mut light_data.block,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::BlockPos;

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::chunk::{
        chunk_access::ChunkAccess,
        proto_chunk::ProtoChunk,
        section::{ChunkSection, Sections},
    };

    fn init_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn range() -> super::super::LightSectionRange {
        let Ok(range) = super::super::LightSectionRange::from_world_height(0, 16) else {
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

    fn set_nibble_non_null(holder: &ChunkHolder, layer: LightLayer, section_y: i32) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let mut light = chunk.light_mut();
        let storage = match layer {
            LightLayer::Sky => &mut light.sky,
            LightLayer::Block => &mut light.block,
        };
        let Some(nibble) = storage.nibble_mut(section_y) else {
            panic!("test nibble should be inside light range");
        };
        nibble.set_non_null();
    }

    #[test]
    fn workset_pins_cached_chunk_holder_until_dropped() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        let layout = LightCacheLayout::new(center, range());

        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing optional chunks");
        };

        let Some(cached_center) = layout.cached_chunk(center) else {
            panic!("center chunk should be inside the cache");
        };
        assert!(workset.chunk_holder(cached_center).is_some());
        assert_eq!(Arc::strong_count(&holder), 2);

        drop(workset);
        assert_eq!(Arc::strong_count(&holder), 1);
    }

    #[test]
    fn workset_reports_missing_required_inner_chunk() {
        init_tests();
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range());

        let result = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            false,
            |_| None,
            |_| true,
        );

        assert_eq!(
            result.err(),
            Some(LightWorksetSetupError::MissingRequiredChunk {
                chunk_pos: ChunkPos::new(-1, -1),
            })
        );
    }

    #[test]
    fn section_read_cache_uses_scalable_lux_local_indices() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        let stone = vanilla_blocks::STONE.default_state();
        section.set_block_state(1, 2, 3, stone);
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

        let Some(cached_block) = layout.cached_block(BlockPos::new(1, 2, 3)) else {
            panic!("test block should be inside light cache");
        };
        let read_state = workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                section_cache.get_block_state(cached_block)
            })
        });

        assert_eq!(read_state, stone);
    }

    #[test]
    fn light_write_cache_reads_writes_and_publishes_cached_nibbles() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        set_nibble_non_null(&holder, LightLayer::Block, 0);
        set_nibble_non_null(&holder, LightLayer::Sky, 0);
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
        let Some(cached_block) = layout.cached_block(BlockPos::new(1, 2, 3)) else {
            panic!("test block should be inside light cache");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                assert_eq!(light_cache.layout(), layout);
                assert_eq!(light_cache.layer(), LightLayer::Block);
                assert_eq!(light_cache.get_updating(cached_block), 0);
                assert!(light_cache.set(cached_block, 12));
                assert_eq!(light_cache.get_updating(cached_block), 12);
                assert_eq!(light_cache.update_visible(), 1);
                assert_eq!(light_cache.update_visible(), 0);
            });

            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                assert_eq!(light_cache.get_updating(cached_block), 0);
                assert!(light_cache.set(cached_block, 7));
                assert_eq!(light_cache.get_updating(cached_block), 7);
            });
        });

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should still be available");
        };
        let light = chunk.light();
        let Some(block_nibble) = light.block.nibble(0) else {
            panic!("block nibble should be present");
        };
        let Some(sky_nibble) = light.sky.nibble(0) else {
            panic!("sky nibble should be present");
        };

        assert_eq!(
            block_nibble.get_visible_at_index(cached_block.local_index),
            12
        );
        assert_eq!(
            sky_nibble.get_updating_at_index(cached_block.local_index),
            7
        );
        assert_eq!(sky_nibble.get_visible_at_index(cached_block.local_index), 0);
    }

    #[test]
    fn light_write_cache_does_not_initialize_null_nibbles() {
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
        let Some(cached_block) = layout.cached_block(BlockPos::new(1, 2, 3)) else {
            panic!("test block should be inside light cache");
        };

        let wrote = workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                assert_eq!(light_cache.get_updating(cached_block), 0);
                light_cache.set(cached_block, 12)
            })
        });

        assert!(!wrote);
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should still be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.block.nibble(0) else {
            panic!("block nibble should be present");
        };
        assert!(nibble.is_null_updating());
        assert_eq!(nibble.get_updating_at_index(cached_block.local_index), 0);
    }

    #[test]
    fn light_write_cache_returns_zero_for_missing_cached_chunks() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        set_nibble_non_null(&holder, LightLayer::Block, 0);
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
        let Some(missing_neighbor_block) = layout.cached_block(BlockPos::new(16, 2, 3)) else {
            panic!("neighbor block should be inside light cache");
        };

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                assert_eq!(light_cache.get_updating(missing_neighbor_block), 0);
                assert!(!light_cache.set(missing_neighbor_block, 12));
            });
        });
    }
}

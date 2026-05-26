use std::sync::Arc;

use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use steel_registry::{REGISTRY, vanilla_blocks};
use steel_utils::{BlockStateId, ChunkPos, SectionPos};

use crate::chunk::{
    chunk_access::{ChunkAccess, ChunkStatus},
    chunk_holder::ChunkHolder,
    section::ChunkSection,
};

use super::{
    CachedLightBlock, CachedLightChunk, ChunkLightData, ChunkLightLayerStorage,
    LightCacheChunkScope, LightCacheLayout, LightCacheSetupRadius, LightChunkSlotArray, LightLayer,
    LightNibbleArray, LightSectionSlotArray, LightUpdateNotificationCache,
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
    chunks: LightChunkSlotArray<LightWorksetChunk>,
}

struct LightWorksetChunk {
    holder: Arc<ChunkHolder>,
    section_readable: bool,
    light_writable: bool,
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
        Self::setup_with_scopes(
            layout,
            radius,
            relaxed,
            &mut chunk_for_lighting,
            |_, _, chunk| {
                let usable = can_use_chunk(chunk);
                (usable, usable)
            },
        )
    }

    /// Creates a scoped cache window with separate section-read and light-write admission.
    pub fn setup_with_scopes(
        layout: LightCacheLayout,
        radius: LightCacheSetupRadius,
        relaxed: bool,
        mut chunk_for_lighting: impl FnMut(ChunkPos) -> Option<Arc<ChunkHolder>>,
        mut can_use_chunk: impl FnMut(CachedLightChunk, &ChunkHolder, &ChunkAccess) -> (bool, bool),
    ) -> Result<Self, LightWorksetSetupError> {
        let mut chunks = LightChunkSlotArray::new();

        for cached_chunk in layout.setup_chunks(radius) {
            let Some(holder) =
                Self::try_get_holder(cached_chunk, relaxed, &mut chunk_for_lighting)?
            else {
                continue;
            };

            let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
                continue;
            };
            let (use_sections, use_light) = can_use_chunk(cached_chunk, &holder, &chunk);
            if !use_sections && !use_light {
                continue;
            }
            drop(chunk);

            chunks.insert(
                cached_chunk,
                LightWorksetChunk {
                    holder,
                    section_readable: use_sections,
                    light_writable: use_light,
                },
            );
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
        self.chunks.get(cached_chunk).map(|chunk| &chunk.holder)
    }

    /// Builds a chunk-read cache for the duration of `f`.
    ///
    /// Chunk locks are acquired in cache-slot order and released before this
    /// method returns. The workset keeps holder `Arc`s alive, while this cache
    /// keeps the guarded chunk data stable during the lighting operation.
    pub fn with_chunk_read_cache<R>(&self, f: impl FnOnce(&LightChunkReadCache<'_>) -> R) -> R {
        let mut chunks = LightChunkSlotArray::new();

        for chunk_slot in 0..self.chunks.slot_count() {
            let Some(workset_chunk) = self.chunks.get_slot(chunk_slot) else {
                continue;
            };
            if !workset_chunk.section_readable {
                continue;
            }
            let Some(chunk) = workset_chunk.holder.try_chunk(ChunkStatus::Empty) else {
                continue;
            };
            chunks.insert_slot(chunk_slot, chunk);
        }

        let mut light_chunks = LightChunkSlotArray::new();
        for chunk_slot in 0..self.chunks.slot_count() {
            let Some(workset_chunk) = self.chunks.get_slot(chunk_slot) else {
                continue;
            };
            if !workset_chunk.light_writable {
                continue;
            }
            let Some(chunk) = workset_chunk.holder.try_chunk(ChunkStatus::Empty) else {
                continue;
            };
            light_chunks.insert_slot(chunk_slot, chunk);
        }

        let cache = LightChunkReadCache {
            layout: self.layout,
            chunks,
            light_chunks,
        };
        f(&cache)
    }

    fn try_get_holder(
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
    light_chunks: LightChunkSlotArray<RwLockReadGuard<'a, ChunkAccess>>,
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

        for chunk_slot in 0..self.light_chunks.slot_count() {
            let Some(chunk_guard) = self.light_chunks.get_slot(chunk_slot) else {
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

    /// Returns whether a cached section exists and is non-empty.
    #[must_use]
    pub fn has_non_empty_section(&self, section_pos: SectionPos) -> bool {
        let Some(cached_section) = self.layout.cached_section(section_pos) else {
            return false;
        };
        self.sections
            .get_slot(cached_section.section_slot)
            .is_some_and(|section| !section.is_empty())
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

    /// Returns true when a cached section has a writable nibble entry.
    #[must_use]
    pub fn has_cached_section(&self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        self.nibble(section_slot).is_some()
    }

    /// Returns true when the cached chunk column has a known emptiness map for this layer.
    #[must_use]
    pub fn has_emptiness_map(&self, chunk_pos: ChunkPos) -> bool {
        let Some(cached_chunk) = self.layout.cached_chunk(chunk_pos) else {
            return false;
        };
        let Some(light_data) = self.chunks.get_slot(cached_chunk.chunk_slot) else {
            return false;
        };

        Self::layer_storage(light_data, self.layer)
            .emptiness_map()
            .is_some()
    }

    /// Returns known real-section emptiness for a cached chunk column.
    #[must_use]
    pub fn section_empty(&self, section_pos: SectionPos) -> Option<bool> {
        let chunk_pos = ChunkPos::new(section_pos.x(), section_pos.z());
        let cached_chunk = self.layout.cached_chunk(chunk_pos)?;
        let light_data = self.chunks.get_slot(cached_chunk.chunk_slot)?;

        Self::layer_storage(light_data, self.layer).section_empty(section_pos.y())
    }

    /// Marks a cached light section non-null without allocating light bytes.
    ///
    /// Returns false when the section has no writable cached nibble.
    pub fn set_section_non_null(&mut self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        self.set_section_slot_non_null(section_slot)
    }

    /// Marks a cached light section null and drops its updating bytes.
    ///
    /// Returns false when the section has no writable cached nibble.
    pub fn set_section_null(&mut self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        let Some(nibble) = self.nibble_mut(section_slot) else {
            return false;
        };
        let was_non_null = !nibble.is_null_updating();
        nibble.set_null();
        was_non_null
    }

    /// Hides a cached block-light section from external packet/save conversion.
    ///
    /// Returns false when the section has no writable cached nibble.
    pub fn set_section_hidden(&mut self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        let Some(nibble) = self.nibble_mut(section_slot) else {
            return false;
        };
        let was_visible = !nibble.is_null_updating();
        nibble.set_hidden();
        was_visible
    }

    /// Replaces one cached chunk column's layer nibbles with fresh null nibbles.
    ///
    /// Initial chunk lighting in ScalableLux lights into a new null nibble array
    /// for the center chunk, then stores it back after propagation. Steel keeps
    /// the array inside the chunk, so this gives the center chunk the same fresh
    /// starting point before the propagation context runs.
    pub fn reset_chunk_nibbles_to_null(&mut self, chunk_pos: ChunkPos) -> bool {
        let Some(cached_chunk) = self.layout.cached_chunk(chunk_pos) else {
            return false;
        };
        let layer = self.layer;
        let Some(light_data) = self.chunks.get_mut_slot(cached_chunk.chunk_slot) else {
            return false;
        };

        Self::layer_storage_mut(light_data, layer).reset_nibbles_to_null();
        true
    }

    /// Marks a cached section slot non-null without allocating light bytes.
    ///
    /// Returns false when the section slot has no writable cached nibble.
    pub fn set_section_slot_non_null(&mut self, section_slot: usize) -> bool {
        let Some(nibble) = self.nibble_mut(section_slot) else {
            return false;
        };
        let was_null = nibble.is_null_updating();
        nibble.set_non_null();
        was_null
    }

    /// Returns true when a cached section has a writable non-null updating nibble.
    #[must_use]
    pub fn has_non_null_section(&self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        self.nibble(section_slot)
            .is_some_and(|nibble| !nibble.is_null_updating())
    }

    /// Returns true when a cached section has initialized updating light bytes.
    #[must_use]
    pub fn is_section_initialized_updating(&self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        self.nibble(section_slot)
            .is_some_and(LightNibbleArray::is_initialized_updating)
    }

    /// Returns true when a cached section has a null updating nibble.
    #[must_use]
    pub fn is_section_null_updating(&self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        self.nibble(section_slot)
            .is_some_and(LightNibbleArray::is_null_updating)
    }

    /// Fills a cached section with one updating light value.
    ///
    /// Returns false when the section has no writable cached nibble.
    pub fn fill_section(&mut self, section_pos: SectionPos, value: u8) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };
        let Some(nibble) = self.nibble_mut(section_slot) else {
            return false;
        };
        nibble.fill(value);
        true
    }

    /// Extrudes the lower row from the first non-null cached section above.
    ///
    /// Returns false when the target section or source section is unavailable.
    pub fn extrude_lower_from_first_section_above(&mut self, section_pos: SectionPos) -> bool {
        let Some(target_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };

        let mut source_slot = None;
        for source_y in (section_pos.y() + 1)..self.layout.range().max_section_y_exclusive() {
            let source_pos = SectionPos::new(section_pos.x(), source_y, section_pos.z());
            let Some(candidate_slot) = self.layout.section_slot(source_pos) else {
                continue;
            };
            let Some(nibble) = self.nibble(candidate_slot) else {
                continue;
            };
            if !nibble.is_null_updating() {
                source_slot = Some(candidate_slot);
                break;
            }
        }

        let Some(source_slot) = source_slot else {
            return false;
        };
        let Some(source_row) = self
            .nibble(source_slot)
            .and_then(|source| source.lower_row_for_extrusion().ok())
        else {
            return false;
        };
        let Some(target) = self.nibble_mut(target_slot) else {
            return false;
        };
        target.extrude_lower_row(source_row);
        true
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

    /// Publishes dirty or explicitly notified cached nibbles.
    ///
    /// Calls `on_update` once for every section whose visible data changed or
    /// was explicitly marked for update, then returns the number of callbacks.
    pub fn update_visible(
        &mut self,
        notifications: Option<&LightUpdateNotificationCache>,
        mut on_update: impl FnMut(SectionPos),
    ) -> usize {
        debug_assert!(notifications.is_none_or(|cache| cache.layout() == self.layout));
        let mut updated = 0;

        for section_slot in 0..self.nibbles.slot_count() {
            let marked =
                notifications.is_some_and(|cache| cache.is_marked_section_slot(section_slot));
            let dirty = self
                .nibble_mut(section_slot)
                .is_some_and(LightNibbleArray::update_visible);
            if (dirty || marked)
                && let Some(section_pos) = self.layout.section_pos_for_slot(section_slot)
            {
                on_update(section_pos);
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
    use steel_utils::{BlockPos, SectionPos};

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
    fn section_read_cache_reports_non_empty_sections() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 2, 3, vanilla_blocks::STONE.default_state());
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

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                assert!(section_cache.has_non_empty_section(SectionPos::new(0, 0, 0)));
                assert!(!section_cache.has_non_empty_section(SectionPos::new(0, 1, 0)));
                assert!(!section_cache.has_non_empty_section(SectionPos::new(1, 0, 0)));
            });
        });
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

                let mut updated_sections = Vec::new();
                assert_eq!(
                    light_cache.update_visible(None, |section_pos| {
                        updated_sections.push(section_pos);
                    }),
                    1
                );
                assert_eq!(updated_sections, vec![SectionPos::new(0, 0, 0)]);
                assert_eq!(light_cache.update_visible(None, |_| {}), 0);
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
    fn light_write_cache_publishes_explicit_notifications() {
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
        let mut notifications = LightUpdateNotificationCache::new(layout);
        assert!(notifications.mark_section(SectionPos::new(0, 0, 0)));

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                let mut updated_sections = Vec::new();
                assert_eq!(
                    light_cache.update_visible(Some(&notifications), |section_pos| {
                        updated_sections.push(section_pos);
                    }),
                    1
                );
                assert_eq!(updated_sections, vec![SectionPos::new(0, 0, 0)]);
            });
        });
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
    fn light_write_cache_can_mark_sections_non_null_without_allocating() {
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
        let section_pos = SectionPos::new(0, 0, 0);

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                assert!(light_cache.set_section_non_null(section_pos));
                assert!(!light_cache.set_section_non_null(section_pos));
                assert!(light_cache.has_cached_section(section_pos));
                assert!(light_cache.has_non_null_section(section_pos));
                assert!(!light_cache.is_section_initialized_updating(section_pos));

                let mut updated_sections = Vec::new();
                assert_eq!(
                    light_cache.update_visible(None, |updated| {
                        updated_sections.push(updated);
                    }),
                    1
                );
                assert_eq!(updated_sections, vec![section_pos]);
            });
        });

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should still be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.block.nibble(0) else {
            panic!("block nibble should be present");
        };
        assert!(!nibble.is_null_visible());
        assert!(!nibble.is_initialized_visible());
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

    #[test]
    fn workset_can_read_sections_without_writable_light_scope() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let east = ChunkPos::new(1, 0);
        let center_holder = holder_with_section(center, ChunkSection::new_empty());
        let mut east_section = ChunkSection::new_empty();
        east_section.set_block_state(0, 0, 0, vanilla_blocks::STONE.default_state());
        let east_holder = holder_with_section(east, east_section);
        set_nibble_non_null(&east_holder, LightLayer::Block, 0);
        let layout = LightCacheLayout::new(center, range());

        let Ok(workset) = LightWorkset::setup_with_scopes(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&center_holder))
                } else if pos == east {
                    Some(Arc::clone(&east_holder))
                } else {
                    None
                }
            },
            |cached_chunk, _, _| (true, cached_chunk.chunk_pos == center),
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        assert_eq!(Arc::strong_count(&center_holder), 2);
        assert_eq!(Arc::strong_count(&east_holder), 2);

        let Some(east_block) = layout.cached_block(BlockPos::new(16, 0, 0)) else {
            panic!("east block should be inside light cache");
        };
        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                assert!(section_cache.has_non_empty_section(SectionPos::new(1, 0, 0)));
            });
            chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                assert_eq!(light_cache.get_updating(east_block), 0);
                assert!(!light_cache.set(east_block, 9));
            });
        });
    }
}

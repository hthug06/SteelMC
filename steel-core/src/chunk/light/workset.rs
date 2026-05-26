use std::sync::Arc;

use parking_lot::RwLockReadGuard;
use steel_registry::{REGISTRY, vanilla_blocks};
use steel_utils::{BlockStateId, ChunkPos};

use crate::chunk::{
    chunk_access::{ChunkAccess, ChunkStatus},
    chunk_holder::ChunkHolder,
    section::ChunkSection,
};

use super::{
    CachedLightBlock, CachedLightChunk, LightCacheChunkScope, LightCacheLayout,
    LightCacheSetupRadius, LightChunkSlotArray, LightSectionSlotArray,
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

        let cache = LightSectionReadCache { sections };
        f(&cache)
    }
}

/// Flat cached chunk-section reads for ScalableLux-style block-state access.
pub struct LightSectionReadCache<'a> {
    sections: LightSectionSlotArray<RwLockReadGuard<'a, ChunkSection>>,
}

impl LightSectionReadCache<'_> {
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
}

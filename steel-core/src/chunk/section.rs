//! This module contains the `Sections` and `ChunkSection` structs.
use std::{fmt::Debug, io::Cursor, sync::LazyLock};

use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_registry::vanilla_biomes;
use steel_registry::{REGISTRY, RegistryEntry};
use steel_utils::{BlockPos, BlockStateId, ChunkPos, locks::SyncRwLock, serial::WriteTo};

use crate::behavior::{BLOCK_BEHAVIORS, BlockBehaviorRegistry};
use crate::chunk::paletted_container::{BiomePalette, BlockPalette};

/// A wrapper around a chunk section.
#[derive(Debug)]
pub struct SectionHolder {
    /// The chunk section data (requires lock to access).
    pub section: SyncRwLock<ChunkSection>,
}

impl SectionHolder {
    /// Creates a new section holder.
    #[must_use]
    pub const fn new(section: ChunkSection) -> Self {
        Self {
            section: SyncRwLock::new(section),
        }
    }

    /// Returns true if this section contains any randomly-ticking blocks.
    ///
    /// Performs an unsynchronized read of the ticking block count to avoid
    /// lock overhead on every section during random ticks. A stale read is
    /// acceptable: worst case we acquire an unnecessary lock.
    #[inline]
    #[must_use]
    pub fn is_randomly_ticking(&self) -> bool {
        // SAFETY: `ticking_block_count` is a `u16` — reads are atomic on all
        // supported platforms. A torn/stale value only causes a harmless
        // false-positive (we take the lock when we didn't need to).
        unsafe { (*self.section.data_ptr()).ticking_block_count > 0 }
    }

    /// Acquires a read lock on the section.
    #[inline]
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, ChunkSection> {
        self.section.read()
    }

    /// Acquires a write lock on the section.
    #[inline]
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, ChunkSection> {
        self.section.write()
    }
}

/// A collection of chunk sections.
#[derive(Debug)]
pub struct Sections {
    /// The sections in the collection.
    pub sections: Box<[SectionHolder]>,
}

/// Cached section counter traits for one block state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockStateSectionCounts {
    is_air: bool,
    has_fluid: bool,
    randomly_ticking: bool,
}

static BLOCK_STATE_SECTION_COUNTS: LazyLock<Box<[BlockStateSectionCounts]>> = LazyLock::new(|| {
    let mut counts = Vec::with_capacity(REGISTRY.blocks.state_to_block_lookup.len());
    for state_index in 0..REGISTRY.blocks.state_to_block_lookup.len() {
        let Ok(raw_state_id) = u16::try_from(state_index) else {
            panic!("block state registry exceeded BlockStateId range");
        };
        counts.push(ChunkSection::block_state_section_counts_with(
            BlockStateId(raw_state_id),
            &BLOCK_BEHAVIORS,
        ));
    }
    counts.into_boxed_slice()
});

impl Sections {
    /// Creates a new `Sections` from a box of owned `ChunkSection`s.
    #[must_use]
    pub fn from_owned(sections: Box<[ChunkSection]>) -> Self {
        let holders: Box<[SectionHolder]> = sections
            .into_vec()
            .into_iter()
            .map(SectionHolder::new)
            .collect();
        Self { sections: holders }
    }

    /// Recalculates cached counters for every section.
    pub fn recalculate_counts(&self) {
        for section in &self.sections {
            section.write().recalculate_counts();
        }
    }

    /// Returns ScalableLux-style section emptiness for every real chunk section.
    #[must_use]
    pub fn section_emptiness_map(&self) -> Box<[bool]> {
        self.sections
            .iter()
            .map(|section| section.read().is_empty())
            .collect()
    }

    /// Returns block-light source positions in ScalableLux section/local-index order.
    ///
    /// ScalableLux first checks the section palette for any emitting states,
    /// then scans local indices as `x | (z << 4) | (y << 8)`.
    #[must_use]
    pub fn block_light_sources(&self, chunk_pos: ChunkPos, min_y: i32) -> Vec<BlockPos> {
        let mut sources = Vec::new();
        let chunk_min_x = chunk_pos.0.x * BlockPalette::SIZE as i32;
        let chunk_min_z = chunk_pos.0.y * BlockPalette::SIZE as i32;

        for (section_index, section) in self.sections.iter().enumerate() {
            let section_min_y = min_y + (section_index * BlockPalette::SIZE) as i32;
            section.read().append_block_light_sources(
                chunk_min_x,
                section_min_y,
                chunk_min_z,
                &mut sources,
            );
        }

        sources
    }

    /// Gets a block at a relative position in the chunk.
    #[must_use]
    pub fn get_relative_block(
        &self,
        relative_x: usize,
        relative_y: usize,
        relative_z: usize,
    ) -> Option<BlockStateId> {
        debug_assert!(relative_x < BlockPalette::SIZE);
        debug_assert!(relative_z < BlockPalette::SIZE);

        let section_index = relative_y / BlockPalette::SIZE;
        let relative_y = relative_y % BlockPalette::SIZE;
        self.sections.get(section_index).map(|section| {
            section
                .read()
                .states
                .get(relative_x, relative_y, relative_z)
        })
    }

    /// Reads an entire column at `(x, z)` across all sections into a caller-owned buffer.
    ///
    /// Holds each section's read lock once for 16 Y reads instead of acquiring
    /// a lock per block. Indexed by `relative_y` (0 = chunk min-y).
    /// The buffer is resized if needed and reused across calls to avoid allocation.
    pub fn read_column_into(&self, x: usize, z: usize, buf: &mut Vec<BlockStateId>) {
        debug_assert!(x < BlockPalette::SIZE);
        debug_assert!(z < BlockPalette::SIZE);

        let total = self.sections.len() * 16;
        buf.clear();
        buf.resize(total, BlockStateId(0));
        for (i, holder) in self.sections.iter().enumerate() {
            let guard = holder.read();
            let base = i * 16;
            for ly in 0..16 {
                buf[base + ly] = guard.states.get(x, ly, z);
            }
        }
    }

    /// Reads all biome palette values into a flat array.
    ///
    /// Indexed as `[section_idx * 64 + qy * 16 + qz * 4 + qx]`.
    /// Holds each section's read lock once for all 64 biome reads.
    #[must_use]
    pub fn read_all_biomes(&self) -> Box<[u16]> {
        let total = self.sections.len() * 64;
        let mut biomes = vec![0u16; total];
        for (i, holder) in self.sections.iter().enumerate() {
            let guard = holder.read();
            let base = i * 64;
            for qy in 0..4 {
                for qz in 0..4 {
                    for qx in 0..4 {
                        biomes[base + qy * 16 + qz * 4 + qx] = guard.biomes.get(qx, qy, qz);
                    }
                }
            }
        }
        biomes.into_boxed_slice()
    }

    /// Visits every biome palette value in section order while holding each
    /// section's read lock once.
    pub fn for_each_biome_id(&self, mut visitor: impl FnMut(u16)) {
        for holder in &self.sections {
            let guard = holder.read();
            for qy in 0..4 {
                for qz in 0..4 {
                    for qx in 0..4 {
                        visitor(guard.biomes.get(qx, qy, qz));
                    }
                }
            }
        }
    }

    /// Writes multiple blocks in one column, holding each section's write guard
    /// across all writes to that section. Most efficient when blocks are grouped
    /// by section (e.g. descending `relative_y` from a top-to-bottom scan).
    pub fn write_column_blocks(&self, x: usize, z: usize, blocks: &[(usize, BlockStateId)]) {
        debug_assert!(x < BlockPalette::SIZE);
        debug_assert!(z < BlockPalette::SIZE);

        let mut i = 0;
        while i < blocks.len() {
            let section_idx = blocks[i].0 / BlockPalette::SIZE;
            let mut guard = self.sections[section_idx].write();
            while i < blocks.len() && blocks[i].0 / BlockPalette::SIZE == section_idx {
                let (rel_y, value) = blocks[i];
                let new_counts = ChunkSection::block_state_section_counts(value);
                guard.set_block_state_with_known_new_counts(
                    x,
                    rel_y % BlockPalette::SIZE,
                    z,
                    value,
                    new_counts,
                );
                i += 1;
            }
        }
    }

    /// Writes a batch of blocks at arbitrary positions, holding each section's
    /// write guard across consecutive entries in the same section. Blocks should
    /// be roughly grouped by section index for best performance.
    pub fn write_block_batch(&self, blocks: &[(usize, usize, usize, BlockStateId)]) {
        let mut i = 0;
        while i < blocks.len() {
            let section_idx = blocks[i].1 / BlockPalette::SIZE;
            let mut guard = self.sections[section_idx].write();
            while i < blocks.len() && blocks[i].1 / BlockPalette::SIZE == section_idx {
                let (x, rel_y, z, value) = blocks[i];
                let new_counts = ChunkSection::block_state_section_counts(value);
                guard.set_block_state_with_known_new_counts(
                    x,
                    rel_y % BlockPalette::SIZE,
                    z,
                    value,
                    new_counts,
                );
                i += 1;
            }
        }
    }

    /// Sets a block at a relative position in the chunk.
    pub fn set_relative_block(
        &self,
        relative_x: usize,
        relative_y: usize,
        relative_z: usize,
        value: BlockStateId,
    ) {
        debug_assert!(relative_x < BlockPalette::SIZE);
        debug_assert!(relative_z < BlockPalette::SIZE);

        let idx = relative_y / BlockPalette::SIZE;
        let relative_y = relative_y % BlockPalette::SIZE;
        let new_counts = ChunkSection::block_state_section_counts(value);
        self.sections[idx]
            .write()
            .set_block_state_with_known_new_counts(
                relative_x, relative_y, relative_z, value, new_counts,
            );
    }
}

/// A chunk section.
///
/// Contains a 16x16x16 cube of block states and biomes, along with cached
/// counts for optimization (similar to vanilla's `LevelChunkSection`).
#[derive(Debug)]
pub struct ChunkSection {
    /// The block states in the section.
    pub states: BlockPalette,
    /// The biomes in the section.
    pub biomes: BiomePalette,
    /// Number of non-air blocks in this section (0-4096).
    /// Used to quickly check if a section is empty.
    non_empty_block_count: u16,
    /// Number of fluid-containing blocks in this section (0-4096).
    /// Includes water, lava, and waterlogged blocks.
    fluid_count: u16,
    /// Number of randomly-ticking blocks in this section (0-4096).
    pub ticking_block_count: u16,
}

impl ChunkSection {
    /// Creates a new chunk section with the given block states and biomes.
    ///
    /// Note: You must call `recalculate_counts()` after creation to initialize
    /// the cached counters if the states palette contains non-air blocks.
    #[must_use]
    pub const fn new_with_biomes(states: BlockPalette, biomes: BiomePalette) -> Self {
        Self {
            states,
            biomes,
            non_empty_block_count: 0,
            fluid_count: 0,
            ticking_block_count: 0,
        }
    }

    /// Creates a new empty chunk section.
    #[must_use]
    pub fn new_empty() -> Self {
        let plains_id = vanilla_biomes::PLAINS.id() as u16;
        Self {
            states: BlockPalette::Homogeneous(BlockStateId(0)),
            biomes: BiomePalette::Homogeneous(plains_id),
            non_empty_block_count: 0,
            fluid_count: 0,
            ticking_block_count: 0,
        }
    }

    /// Returns true if this section contains no non-air blocks.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.non_empty_block_count == 0
    }

    /// Returns true if this section contains any randomly-ticking blocks.
    #[must_use]
    pub const fn is_randomly_ticking(&self) -> bool {
        self.ticking_block_count > 0
    }

    /// Returns true if this section's palette may contain block-light sources.
    #[must_use]
    pub fn maybe_has_block_light_sources(&self) -> bool {
        !self.is_empty()
            && self
                .states
                .maybe_has(|state| state.get_light_emission() > 0)
    }

    /// Appends block-light source positions in ScalableLux local-index order.
    pub fn append_block_light_sources(
        &self,
        chunk_min_x: i32,
        section_min_y: i32,
        chunk_min_z: i32,
        sources: &mut Vec<BlockPos>,
    ) {
        if !self.maybe_has_block_light_sources() {
            return;
        }

        for local_index in 0..BlockPalette::VOLUME {
            let state = self.states.get_at_index(local_index);
            if state.get_light_emission() == 0 {
                continue;
            }

            sources.push(BlockPos::new(
                chunk_min_x + (local_index & 15) as i32,
                section_min_y + (local_index >> 8) as i32,
                chunk_min_z + ((local_index >> 4) & 15) as i32,
            ));
        }
    }

    /// Returns the number of non-air blocks in this section.
    #[must_use]
    pub const fn non_empty_block_count(&self) -> u16 {
        self.non_empty_block_count
    }

    /// Returns the number of fluid-containing blocks in this section.
    #[must_use]
    pub const fn fluid_count(&self) -> u16 {
        self.fluid_count
    }

    /// Returns if the chunk has fluid.
    #[must_use]
    pub const fn has_fluid(&self) -> bool {
        self.fluid_count > 0
    }

    /// Returns the number of randomly-ticking blocks in this section.
    #[must_use]
    pub const fn ticking_block_count(&self) -> u16 {
        self.ticking_block_count
    }

    /// Recalculates both cached counters by iterating all blocks.
    ///
    /// This should be called after chunk loading or generation to initialize
    /// the counters. It requires the block behavior registry to be initialized.
    ///
    /// # Panics
    /// Panics if the block behavior registry has not been initialized.
    pub fn recalculate_counts(&mut self) {
        self.recalculate_counts_with(&BLOCK_BEHAVIORS);
    }

    /// Recalculates all cached counters using the provided behavior registry.
    pub fn recalculate_counts_with(&mut self, block_behaviors: &BlockBehaviorRegistry) {
        let mut non_empty: u16 = 0;
        let mut fluid: u16 = 0;
        let mut ticking: u16 = 0;

        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let state = self.states.get(x, y, z);
                    if !state.is_air() {
                        non_empty += 1;
                        let block = state.get_block();
                        let behavior = block_behaviors.get_behavior(block);
                        if behavior.is_randomly_ticking(state) {
                            ticking += 1;
                        }
                    }
                    let fluid_state = block_behaviors
                        .get_behavior(state.get_block())
                        .get_fluid_state(state);
                    if !fluid_state.is_empty() {
                        fluid += 1;
                    }
                }
            }
        }

        self.non_empty_block_count = non_empty;
        self.fluid_count = fluid;
        self.ticking_block_count = ticking;
    }

    /// Sets a block state and updates the cached counters.
    ///
    /// Returns the old block state.
    ///
    /// # Panics
    /// Panics if the block behavior registry has not been initialized.
    pub fn set_block_state(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        new_state: BlockStateId,
    ) -> BlockStateId {
        self.set_block_state_with(x, y, z, new_state, &BLOCK_BEHAVIORS)
    }

    /// Sets a block state and updates the cached counters using the provided behavior registry.
    ///
    /// Returns the old block state.
    pub fn set_block_state_with(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        new_state: BlockStateId,
        block_behaviors: &BlockBehaviorRegistry,
    ) -> BlockStateId {
        let old_state = self.states.set(x, y, z, new_state);

        if old_state != new_state {
            let old_counts = Self::block_state_section_counts_with(old_state, block_behaviors);
            let new_counts = Self::block_state_section_counts_with(new_state, block_behaviors);
            self.apply_count_change(old_counts, new_counts);
        }

        old_state
    }

    /// Sets a block state and updates counters when the caller already knows
    /// the replacement state's counter traits.
    pub(crate) fn set_block_state_with_known_new_counts(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        new_state: BlockStateId,
        new_counts: BlockStateSectionCounts,
    ) -> BlockStateId {
        let old_state = self.states.set(x, y, z, new_state);
        if old_state != new_state {
            let old_counts = Self::block_state_section_counts(old_state);
            self.apply_count_change(old_counts, new_counts);
        }

        old_state
    }

    /// Returns the cached-counter traits for a block state using the global
    /// behavior registry.
    pub(crate) fn block_state_section_counts(state: BlockStateId) -> BlockStateSectionCounts {
        let Some(&counts) = BLOCK_STATE_SECTION_COUNTS.get(state.0 as usize) else {
            panic!("invalid block state id {}", state.0);
        };
        counts
    }

    fn block_state_section_counts_with(
        state: BlockStateId,
        block_behaviors: &BlockBehaviorRegistry,
    ) -> BlockStateSectionCounts {
        let behavior = block_behaviors.get_behavior(state.get_block());
        BlockStateSectionCounts {
            is_air: state.is_air(),
            has_fluid: !behavior.get_fluid_state(state).is_empty(),
            randomly_ticking: behavior.is_randomly_ticking(state),
        }
    }

    const fn apply_count_change(
        &mut self,
        old_counts: BlockStateSectionCounts,
        new_counts: BlockStateSectionCounts,
    ) {
        if !old_counts.is_air && new_counts.is_air {
            self.non_empty_block_count -= 1;
        } else if old_counts.is_air && !new_counts.is_air {
            self.non_empty_block_count += 1;
        }

        if old_counts.has_fluid && !new_counts.has_fluid {
            self.fluid_count -= 1;
        } else if !old_counts.has_fluid && new_counts.has_fluid {
            self.fluid_count += 1;
        }

        if old_counts.randomly_ticking && !new_counts.randomly_ticking {
            self.ticking_block_count -= 1;
        } else if !old_counts.randomly_ticking && new_counts.randomly_ticking {
            self.ticking_block_count += 1;
        }
    }

    /// Writes the chunk section to a writer.
    ///
    /// # Panics
    /// - If the writer fails to write.
    pub fn write(&self, writer: &mut Cursor<Vec<u8>>) {
        self.non_empty_block_count
            .write(writer)
            .expect("Failed to write block count");
        self.fluid_count
            .write(writer)
            .expect("Failed to write fluid count");

        self.states
            .write(writer)
            .expect("Failed to write block states");
        self.biomes.write(writer).expect("Failed to write biomes");
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::{REGISTRY, test_support::init_test_registry, vanilla_blocks};

    use super::{ChunkSection, Sections};
    use crate::behavior::init_behaviors;

    fn init() {
        init_test_registry();
        init_behaviors();
    }

    #[test]
    fn relative_block_writes_update_section_counts() {
        init();
        let sections = Sections::from_owned(vec![ChunkSection::new_empty()].into_boxed_slice());
        let stone = vanilla_blocks::STONE.default_state();
        let air = vanilla_blocks::AIR.default_state();

        sections.set_relative_block(1, 2, 3, stone);
        {
            let section = sections.sections[0].read();
            assert_eq!(section.non_empty_block_count(), 1);
            assert!(!section.is_empty());
        }

        sections.set_relative_block(1, 2, 3, air);
        let section = sections.sections[0].read();
        assert_eq!(section.non_empty_block_count(), 0);
        assert!(section.is_empty());
    }

    #[test]
    fn batched_generation_writes_update_section_counts() {
        init();
        let sections = Sections::from_owned(
            vec![ChunkSection::new_empty(), ChunkSection::new_empty()].into_boxed_slice(),
        );
        let stone = vanilla_blocks::STONE.default_state();
        let water = REGISTRY.blocks.get_default_state_id(&vanilla_blocks::WATER);
        let air = vanilla_blocks::AIR.default_state();

        sections.write_block_batch(&[(1, 0, 1, stone), (2, 16, 2, water)]);
        {
            let lower = sections.sections[0].read();
            assert_eq!(lower.non_empty_block_count(), 1);
            assert_eq!(lower.fluid_count(), 0);
            assert!(!lower.is_empty());
        }
        {
            let upper = sections.sections[1].read();
            assert_eq!(upper.non_empty_block_count(), 1);
            assert_eq!(upper.fluid_count(), 1);
            assert!(!upper.is_empty());
        }

        sections.write_column_blocks(1, 1, &[(0, air), (17, stone)]);
        {
            let lower = sections.sections[0].read();
            assert_eq!(lower.non_empty_block_count(), 0);
            assert_eq!(lower.fluid_count(), 0);
            assert!(lower.is_empty());
        }
        let upper = sections.sections[1].read();
        assert_eq!(upper.non_empty_block_count(), 2);
        assert_eq!(upper.fluid_count(), 1);
        assert!(!upper.is_empty());
    }
}

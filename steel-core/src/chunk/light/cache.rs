use steel_utils::{BlockPos, ChunkPos, Direction, SectionPos};

use super::LightSectionRange;

/// Horizontal cache radius used by ScalableLux light propagation.
pub const LIGHT_CACHE_RADIUS: i32 = 2;
/// Horizontal cache width and depth used by ScalableLux light propagation.
pub const LIGHT_CACHE_DIAMETER: usize = LIGHT_CACHE_RADIUS as usize * 2 + 1;
/// Number of chunk columns in one light-engine cache window.
pub const LIGHT_CACHE_CHUNK_SLOTS: usize = LIGHT_CACHE_DIAMETER * LIGHT_CACHE_DIAMETER;

const LIGHT_CACHE_DIAMETER_I64: i64 = LIGHT_CACHE_DIAMETER as i64;
const LIGHT_CACHE_CHUNK_SLOTS_I64: i64 = LIGHT_CACHE_CHUNK_SLOTS as i64;
const LIGHT_LOCAL_BLOCK_MASK: usize = 15;
const LIGHT_LOCAL_BLOCK_Z_SHIFT: usize = 4;
const LIGHT_LOCAL_BLOCK_Y_SHIFT: usize = 8;
const LIGHT_ENCODED_HORIZONTAL_BITS: i64 = 6;
const LIGHT_ENCODED_VERTICAL_BITS: i64 = 16;
const LIGHT_ENCODED_HORIZONTAL_MASK: i64 = (1 << LIGHT_ENCODED_HORIZONTAL_BITS) - 1;
const LIGHT_ENCODED_VERTICAL_MASK: i64 = (1 << LIGHT_ENCODED_VERTICAL_BITS) - 1;
const LIGHT_ENCODED_POSITION_MASK: u32 =
    (1 << (LIGHT_ENCODED_HORIZONTAL_BITS * 2 + LIGHT_ENCODED_VERTICAL_BITS)) - 1;
const LIGHT_ENCODED_Z_SHIFT: u32 = LIGHT_ENCODED_HORIZONTAL_BITS as u32;
const LIGHT_ENCODED_Y_SHIFT: u32 = (LIGHT_ENCODED_HORIZONTAL_BITS * 2) as u32;

/// ScalableLux packed block position used in light propagation queue entries.
///
/// The lower 28 bits store `x | (z << 6) | (y << 12)`. X and Z are encoded in
/// a 64-block window around the active chunk; Y is encoded relative to the
/// section below the vanilla light-section range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedLightBlockPos(u32);

impl PackedLightBlockPos {
    /// Creates a packed light block position from raw queue bits.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw & LIGHT_ENCODED_POSITION_MASK)
    }

    /// Returns the raw lower 28 queue-position bits.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns the encoded 6-bit X coordinate.
    #[must_use]
    pub const fn encoded_x(self) -> u8 {
        (self.0 & LIGHT_ENCODED_HORIZONTAL_MASK as u32) as u8
    }

    /// Returns the encoded 6-bit Z coordinate.
    #[must_use]
    pub const fn encoded_z(self) -> u8 {
        ((self.0 >> LIGHT_ENCODED_Z_SHIFT) & LIGHT_ENCODED_HORIZONTAL_MASK as u32) as u8
    }

    /// Returns the encoded 16-bit Y coordinate.
    #[must_use]
    pub const fn encoded_y(self) -> u16 {
        ((self.0 >> LIGHT_ENCODED_Y_SHIFT) & LIGHT_ENCODED_VERTICAL_MASK as u32) as u16
    }
}

/// Cached section slot and local nibble index for one block.
///
/// ScalableLux propagation uses `sectionIndex` plus local index
/// `x | (z << 4) | (y << 8)` instead of repeatedly materializing section
/// positions and local coordinates inside the hot propagation loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedLightBlock {
    /// World block position for this cached block.
    pub block_pos: BlockPos,
    /// Slot into ScalableLux's section/nibble cache arrays.
    pub section_slot: usize,
    /// Local block index inside the 16x16x16 light section.
    pub local_index: usize,
}

/// Section-slot notification flags used while publishing visible light updates.
///
/// ScalableLux keeps `notifyUpdateCache` beside its nibble cache and marks the
/// cached sections touched by a block's one-block lighting neighborhood. The
/// light engine later scans the same section slots while publishing dirty
/// nibbles and notifying clients.
#[derive(Debug, Clone)]
pub struct LightUpdateNotificationCache {
    layout: LightCacheLayout,
    marked: Box<[bool]>,
}

impl LightUpdateNotificationCache {
    /// Creates an empty notification cache for a light-engine cache window.
    #[must_use]
    pub fn new(layout: LightCacheLayout) -> Self {
        Self {
            layout,
            marked: vec![false; layout.section_slot_count()].into_boxed_slice(),
        }
    }

    /// Returns the layout this notification cache is indexed by.
    #[must_use]
    pub const fn layout(&self) -> LightCacheLayout {
        self.layout
    }

    /// Returns true when no section slots are marked for notification.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.marked.iter().all(|marked| !marked)
    }

    /// Removes every pending section notification.
    pub fn clear(&mut self) {
        self.marked.fill(false);
    }

    /// Marks a cached section, returning true only when it was newly marked.
    pub fn mark_section(&mut self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };

        self.mark_section_slot(section_slot)
    }

    /// Marks every cached section touched by a block's lighting neighborhood.
    ///
    /// Returns `None` if the full one-block neighborhood is not inside this
    /// cache window, so callers do not accidentally publish a partial update.
    pub fn mark_block_neighborhood(&mut self, block_pos: BlockPos) -> Option<usize> {
        let mut contained = true;
        SectionPos::around_and_at_block_pos(block_pos, |section_pos| {
            contained &= self.layout.section_slot(section_pos).is_some();
        });
        if !contained {
            return None;
        }

        let mut newly_marked = 0;
        SectionPos::around_and_at_block_pos(block_pos, |section_pos| {
            if self.mark_section(section_pos) {
                newly_marked += 1;
            }
        });
        Some(newly_marked)
    }

    /// Returns whether a cached section is marked for notification.
    #[must_use]
    pub fn is_marked_section(&self, section_pos: SectionPos) -> bool {
        let Some(section_slot) = self.layout.section_slot(section_pos) else {
            return false;
        };

        self.is_marked_section_slot(section_slot)
    }

    /// Returns whether a section slot is marked for notification.
    #[must_use]
    pub fn is_marked_section_slot(&self, section_slot: usize) -> bool {
        self.marked.get(section_slot).copied().unwrap_or(false)
    }

    /// Iterates marked section positions in cache-slot order.
    pub fn marked_section_positions(&self) -> impl Iterator<Item = SectionPos> + '_ {
        self.marked
            .iter()
            .enumerate()
            .filter_map(move |(section_slot, marked)| {
                if *marked {
                    self.layout.section_pos_for_slot(section_slot)
                } else {
                    None
                }
            })
    }

    fn mark_section_slot(&mut self, section_slot: usize) -> bool {
        let Some(marked) = self.marked.get_mut(section_slot) else {
            return false;
        };

        let newly_marked = !*marked;
        *marked = true;
        newly_marked
    }
}

/// ScalableLux cache-window layout for chunk, section, and nibble arrays.
///
/// ScalableLux keeps a 5x5 chunk window around the active chunk and stores
/// light sections in flat arrays with one extra cached section below and above
/// the vanilla light-section range. This type owns that index math so the
/// light engine can share the same slots for chunk sections, nibbles, and
/// update notifications without repeating coordinate transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightCacheLayout {
    center_chunk: ChunkPos,
    range: LightSectionRange,
    cached_min_section_y: i32,
    cached_section_count: usize,
    chunk_index_offset: i64,
    chunk_section_index_offset: i64,
    encode_offset_x: i64,
    encode_offset_y: i64,
    encode_offset_z: i64,
    encoded_min_block_x: i64,
    encoded_min_block_z: i64,
}

impl LightCacheLayout {
    /// Creates a cache layout centered on one chunk.
    #[must_use]
    pub fn new(center_chunk: ChunkPos, range: LightSectionRange) -> Self {
        let cached_min_section_y = range.min_section_y() - 1;
        let chunk_offset_x = i64::from(LIGHT_CACHE_RADIUS) - i64::from(center_chunk.0.x);
        let chunk_offset_z = i64::from(LIGHT_CACHE_RADIUS) - i64::from(center_chunk.0.y);
        let chunk_index_offset = chunk_offset_x + LIGHT_CACHE_DIAMETER_I64 * chunk_offset_z;
        let chunk_offset_y = -i64::from(cached_min_section_y);
        let chunk_section_index_offset =
            chunk_index_offset + LIGHT_CACHE_CHUNK_SLOTS_I64 * chunk_offset_y;
        let center_block_x = i64::from(center_chunk.0.x) * 16 + 7;
        let center_block_z = i64::from(center_chunk.0.y) * 16 + 7;
        let encode_offset_x = 31 - center_block_x;
        let encode_offset_y = -(i64::from(cached_min_section_y) * 16);
        let encode_offset_z = 31 - center_block_z;

        Self {
            center_chunk,
            range,
            cached_min_section_y,
            cached_section_count: range.section_count() + 2,
            chunk_index_offset,
            chunk_section_index_offset,
            encode_offset_x,
            encode_offset_y,
            encode_offset_z,
            encoded_min_block_x: center_block_x - 31,
            encoded_min_block_z: center_block_z - 31,
        }
    }

    /// Returns the chunk at the center of this cache window.
    #[must_use]
    pub const fn center_chunk(self) -> ChunkPos {
        self.center_chunk
    }

    /// Returns the vanilla padded light-section range.
    #[must_use]
    pub const fn range(self) -> LightSectionRange {
        self.range
    }

    /// Returns the first cached section Y coordinate, including the lower buffer.
    #[must_use]
    pub const fn cached_min_section_y(self) -> i32 {
        self.cached_min_section_y
    }

    /// Returns the section Y coordinate one past the last cached section.
    #[must_use]
    pub fn cached_max_section_y_exclusive(self) -> i32 {
        self.cached_min_section_y + self.cached_section_count as i32
    }

    /// Returns the number of cached vertical sections, including both buffers.
    #[must_use]
    pub const fn cached_section_count(self) -> usize {
        self.cached_section_count
    }

    /// Returns the number of section/nibble slots in this cache window.
    #[must_use]
    pub const fn section_slot_count(self) -> usize {
        LIGHT_CACHE_CHUNK_SLOTS * self.cached_section_count
    }

    /// Returns the slot for a cached chunk column.
    #[must_use]
    pub fn chunk_slot(self, chunk_pos: ChunkPos) -> Option<usize> {
        self.chunk_slot_by_coords(chunk_pos.0.x, chunk_pos.0.y)
    }

    /// Returns the slot for a cached chunk column by chunk coordinates.
    #[must_use]
    pub fn chunk_slot_by_coords(self, chunk_x: i32, chunk_z: i32) -> Option<usize> {
        if !self.contains_chunk_coords(chunk_x, chunk_z) {
            return None;
        }

        let slot = i64::from(chunk_x)
            + LIGHT_CACHE_DIAMETER_I64 * i64::from(chunk_z)
            + self.chunk_index_offset;
        usize::try_from(slot).ok()
    }

    /// Returns the section/nibble slot for a section position.
    #[must_use]
    pub fn section_slot(self, section_pos: SectionPos) -> Option<usize> {
        self.section_slot_by_coords(section_pos.x(), section_pos.y(), section_pos.z())
    }

    /// Returns the section/nibble slot for the section containing a block.
    #[must_use]
    pub fn section_slot_for_block(self, block_pos: BlockPos) -> Option<usize> {
        self.section_slot(SectionPos::from_block_pos(block_pos))
    }

    /// Converts a section/nibble slot back to its cached section position.
    #[must_use]
    pub fn section_pos_for_slot(self, section_slot: usize) -> Option<SectionPos> {
        if section_slot >= self.section_slot_count() {
            return None;
        }

        let section_x = self.center_chunk.0.x - LIGHT_CACHE_RADIUS
            + (section_slot % LIGHT_CACHE_DIAMETER) as i32;
        let section_z = self.center_chunk.0.y - LIGHT_CACHE_RADIUS
            + ((section_slot / LIGHT_CACHE_DIAMETER) % LIGHT_CACHE_DIAMETER) as i32;
        let section_y = self.cached_min_section_y + (section_slot / LIGHT_CACHE_CHUNK_SLOTS) as i32;

        Some(SectionPos::new(section_x, section_y, section_z))
    }

    /// Returns cache slot data for a block position.
    #[must_use]
    pub fn cached_block(self, block_pos: BlockPos) -> Option<CachedLightBlock> {
        self.cached_block_by_coords(block_pos.x(), block_pos.y(), block_pos.z())
    }

    /// Returns cache slot data for block coordinates.
    #[must_use]
    pub fn cached_block_by_coords(
        self,
        block_x: i32,
        block_y: i32,
        block_z: i32,
    ) -> Option<CachedLightBlock> {
        let section_slot = self.section_slot_by_coords(
            SectionPos::block_to_section_coord(block_x),
            SectionPos::block_to_section_coord(block_y),
            SectionPos::block_to_section_coord(block_z),
        )?;

        Some(CachedLightBlock {
            block_pos: BlockPos::new(block_x, block_y, block_z),
            section_slot,
            local_index: Self::local_block_index_by_coords(block_x, block_y, block_z),
        })
    }

    /// Returns cache slot data for a cached block's neighboring block.
    #[must_use]
    pub fn cached_neighbor(
        self,
        cached_block: CachedLightBlock,
        direction: Direction,
    ) -> Option<CachedLightBlock> {
        let (dx, dy, dz) = direction.offset();
        self.cached_block_by_coords(
            cached_block.block_pos.x().checked_add(dx)?,
            cached_block.block_pos.y().checked_add(dy)?,
            cached_block.block_pos.z().checked_add(dz)?,
        )
    }

    /// Decodes a packed queue position and returns its cache slot data.
    #[must_use]
    pub fn cached_block_from_packed(self, packed: PackedLightBlockPos) -> Option<CachedLightBlock> {
        self.cached_block(self.decode_block_pos(packed)?)
    }

    /// Returns the local light-section index for a block position.
    #[must_use]
    pub const fn local_block_index(block_pos: BlockPos) -> usize {
        Self::local_block_index_by_coords(block_pos.x(), block_pos.y(), block_pos.z())
    }

    /// Returns the local light-section index for block coordinates.
    #[must_use]
    pub const fn local_block_index_by_coords(block_x: i32, block_y: i32, block_z: i32) -> usize {
        (block_x as usize & LIGHT_LOCAL_BLOCK_MASK)
            | ((block_z as usize & LIGHT_LOCAL_BLOCK_MASK) << LIGHT_LOCAL_BLOCK_Z_SHIFT)
            | ((block_y as usize & LIGHT_LOCAL_BLOCK_MASK) << LIGHT_LOCAL_BLOCK_Y_SHIFT)
    }

    /// Returns the first block X coordinate that can be packed into queue entries.
    #[must_use]
    pub fn encoded_min_block_x(self) -> i32 {
        self.encoded_min_block_x as i32
    }

    /// Returns the block X coordinate one past the packed queue window.
    #[must_use]
    pub fn encoded_max_block_x_exclusive(self) -> i32 {
        (self.encoded_min_block_x + LIGHT_ENCODED_HORIZONTAL_MASK + 1) as i32
    }

    /// Returns the first block Z coordinate that can be packed into queue entries.
    #[must_use]
    pub fn encoded_min_block_z(self) -> i32 {
        self.encoded_min_block_z as i32
    }

    /// Returns the block Z coordinate one past the packed queue window.
    #[must_use]
    pub fn encoded_max_block_z_exclusive(self) -> i32 {
        (self.encoded_min_block_z + LIGHT_ENCODED_HORIZONTAL_MASK + 1) as i32
    }

    /// Packs a block position for ScalableLux queue storage.
    #[must_use]
    pub fn encode_block_pos(self, block_pos: BlockPos) -> Option<PackedLightBlockPos> {
        if !self.contains_encoded_block_pos(block_pos) {
            return None;
        }

        let encoded_x =
            (i64::from(block_pos.x()) + self.encode_offset_x) & LIGHT_ENCODED_HORIZONTAL_MASK;
        let encoded_y =
            (i64::from(block_pos.y()) + self.encode_offset_y) & LIGHT_ENCODED_VERTICAL_MASK;
        let encoded_z =
            (i64::from(block_pos.z()) + self.encode_offset_z) & LIGHT_ENCODED_HORIZONTAL_MASK;

        Some(PackedLightBlockPos::from_raw(
            encoded_x as u32 | (encoded_z as u32) << 6 | (encoded_y as u32) << 12,
        ))
    }

    /// Decodes ScalableLux queue-position bits back to a world block position.
    #[must_use]
    pub fn decode_block_pos(self, packed: PackedLightBlockPos) -> Option<BlockPos> {
        let x = i64::from(packed.encoded_x()) - self.encode_offset_x;
        let y = i64::from(packed.encoded_y()) - self.encode_offset_y;
        let z = i64::from(packed.encoded_z()) - self.encode_offset_z;

        Some(BlockPos::new(
            i32::try_from(x).ok()?,
            i32::try_from(y).ok()?,
            i32::try_from(z).ok()?,
        ))
    }

    /// Returns true if a block position is inside the packed queue-coordinate window.
    #[must_use]
    pub fn contains_encoded_block_pos(self, block_pos: BlockPos) -> bool {
        let x = i64::from(block_pos.x());
        let z = i64::from(block_pos.z());
        x >= self.encoded_min_block_x
            && x <= self.encoded_min_block_x + LIGHT_ENCODED_HORIZONTAL_MASK
            && z >= self.encoded_min_block_z
            && z <= self.encoded_min_block_z + LIGHT_ENCODED_HORIZONTAL_MASK
            && self.contains_section_y(SectionPos::block_to_section_coord(block_pos.y()))
    }

    /// Returns the section/nibble slot for section coordinates.
    #[must_use]
    pub fn section_slot_by_coords(
        self,
        section_x: i32,
        section_y: i32,
        section_z: i32,
    ) -> Option<usize> {
        if !self.contains_chunk_coords(section_x, section_z) || !self.contains_section_y(section_y)
        {
            return None;
        }

        let slot = i64::from(section_x)
            + LIGHT_CACHE_DIAMETER_I64 * i64::from(section_z)
            + LIGHT_CACHE_CHUNK_SLOTS_I64 * i64::from(section_y)
            + self.chunk_section_index_offset;
        usize::try_from(slot).ok()
    }

    /// Returns the cached vertical index for a section Y coordinate.
    #[must_use]
    pub fn cached_section_index(self, section_y: i32) -> Option<usize> {
        if !self.contains_section_y(section_y) {
            return None;
        }

        usize::try_from(section_y - self.cached_min_section_y).ok()
    }

    /// Converts a cached vertical index back to section Y.
    #[must_use]
    pub fn cached_section_y(self, index: usize) -> Option<i32> {
        if index >= self.cached_section_count {
            return None;
        }

        Some(self.cached_min_section_y + index as i32)
    }

    /// Returns true if a chunk coordinate is inside the 5x5 cache window.
    #[must_use]
    pub fn contains_chunk_coords(self, chunk_x: i32, chunk_z: i32) -> bool {
        let dx = i64::from(chunk_x) - i64::from(self.center_chunk.0.x);
        let dz = i64::from(chunk_z) - i64::from(self.center_chunk.0.y);
        dx.abs().max(dz.abs()) <= i64::from(LIGHT_CACHE_RADIUS)
    }

    /// Returns true if a section Y coordinate is inside the cached vertical range.
    #[must_use]
    pub fn contains_section_y(self, section_y: i32) -> bool {
        section_y >= self.cached_min_section_y && section_y < self.cached_max_section_y_exclusive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(min_y: i32, height: i32) -> LightSectionRange {
        let Ok(range) = LightSectionRange::from_world_height(min_y, height) else {
            panic!("test world height should create a light range");
        };
        range
    }

    #[test]
    fn cache_layout_matches_scalable_lux_chunk_indexing() {
        let layout = LightCacheLayout::new(ChunkPos::new(10, -20), range(0, 16));

        assert_eq!(layout.center_chunk(), ChunkPos::new(10, -20));
        assert_eq!(layout.range(), range(0, 16));
        assert_eq!(layout.chunk_slot(ChunkPos::new(10, -20)), Some(12));
        assert_eq!(layout.chunk_slot(ChunkPos::new(8, -22)), Some(0));
        assert_eq!(layout.chunk_slot(ChunkPos::new(12, -18)), Some(24));
        assert_eq!(layout.chunk_slot(ChunkPos::new(9, -20)), Some(11));
        assert_eq!(layout.chunk_slot(ChunkPos::new(13, -20)), None);
        assert_eq!(layout.chunk_slot(ChunkPos::new(10, -23)), None);
    }

    #[test]
    fn cache_layout_adds_vertical_buffer_around_light_range() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));

        assert_eq!(layout.cached_min_section_y(), -2);
        assert_eq!(layout.cached_max_section_y_exclusive(), 3);
        assert_eq!(layout.cached_section_count(), 5);
        assert_eq!(layout.section_slot_count(), 125);

        assert_eq!(layout.cached_section_index(-2), Some(0));
        assert_eq!(layout.cached_section_index(-1), Some(1));
        assert_eq!(layout.cached_section_index(2), Some(4));
        assert_eq!(layout.cached_section_index(3), None);

        assert_eq!(layout.cached_section_y(0), Some(-2));
        assert_eq!(layout.cached_section_y(4), Some(2));
        assert_eq!(layout.cached_section_y(5), None);
    }

    #[test]
    fn cache_layout_matches_scalable_lux_section_indexing() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));

        assert_eq!(layout.section_slot_by_coords(0, -2, 0), Some(12));
        assert_eq!(layout.section_slot_by_coords(0, -1, 0), Some(37));
        assert_eq!(layout.section_slot_by_coords(0, 1, 0), Some(87));
        assert_eq!(layout.section_slot_by_coords(0, 2, 0), Some(112));
        assert_eq!(layout.section_slot_by_coords(0, 3, 0), None);
    }

    #[test]
    fn cache_layout_decodes_scalable_lux_section_slots() {
        let layout = LightCacheLayout::new(ChunkPos::new(10, -20), range(0, 16));

        assert_eq!(
            layout.section_pos_for_slot(0),
            Some(SectionPos::new(8, -2, -22))
        );
        assert_eq!(
            layout.section_pos_for_slot(12),
            Some(SectionPos::new(10, -2, -20))
        );
        assert_eq!(
            layout.section_pos_for_slot(37),
            Some(SectionPos::new(10, -1, -20))
        );
        assert_eq!(
            layout.section_pos_for_slot(112),
            Some(SectionPos::new(10, 2, -20))
        );
        assert_eq!(layout.section_pos_for_slot(125), None);
    }

    #[test]
    fn cache_layout_section_slot_round_trips_every_cached_section() {
        let layout = LightCacheLayout::new(ChunkPos::new(-4, 6), range(-64, 384));

        for section_slot in 0..layout.section_slot_count() {
            let Some(section_pos) = layout.section_pos_for_slot(section_slot) else {
                panic!("valid cache slot did not decode");
            };
            assert_eq!(layout.section_slot(section_pos), Some(section_slot));
        }
    }

    #[test]
    fn cache_layout_maps_block_positions_to_section_slots() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));

        assert_eq!(
            layout.section_slot_for_block(BlockPos::new(31, 0, -32)),
            Some(53)
        );
        assert_eq!(layout.section_slot_for_block(BlockPos::new(48, 0, 0)), None);
    }

    #[test]
    fn cache_layout_uses_scalable_lux_local_block_indices() {
        assert_eq!(
            LightCacheLayout::local_block_index(BlockPos::new(0, 0, 0)),
            0
        );
        assert_eq!(
            LightCacheLayout::local_block_index(BlockPos::new(15, 15, 15)),
            15 | (15 << 4) | (15 << 8)
        );
        assert_eq!(
            LightCacheLayout::local_block_index(BlockPos::new(-1, -1, -1)),
            15 | (15 << 4) | (15 << 8)
        );
        assert_eq!(
            LightCacheLayout::local_block_index(BlockPos::new(16, 16, 16)),
            0
        );
    }

    #[test]
    fn cache_layout_maps_block_positions_to_cached_blocks() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));

        assert_eq!(
            layout.cached_block(BlockPos::new(31, 0, -32)),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(31, 0, -32),
                section_slot: 53,
                local_index: 15,
            })
        );
        assert_eq!(
            layout.cached_block(BlockPos::new(-1, -1, -1)),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(-1, -1, -1),
                section_slot: 31,
                local_index: 4095,
            })
        );
        assert_eq!(layout.cached_block(BlockPos::new(48, 0, 0)), None);
    }

    #[test]
    fn cache_layout_maps_cached_neighbors_with_scalable_lux_local_indices() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let Some(block) = layout.cached_block(BlockPos::new(7, 7, 7)) else {
            panic!("test block should be cached");
        };

        assert_eq!(
            layout.cached_neighbor(block, Direction::East),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(8, 7, 7),
                section_slot: 62,
                local_index: 8 | (7 << 4) | (7 << 8),
            })
        );
        assert_eq!(
            layout.cached_neighbor(block, Direction::Down),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(7, 6, 7),
                section_slot: 62,
                local_index: 7 | (7 << 4) | (6 << 8),
            })
        );
    }

    #[test]
    fn cache_layout_maps_cached_neighbors_across_section_edges() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let Some(block) = layout.cached_block(BlockPos::new(15, 15, 15)) else {
            panic!("test block should be cached");
        };

        assert_eq!(
            layout.cached_neighbor(block, Direction::East),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(16, 15, 15),
                section_slot: 63,
                local_index: (15 << 4) | (15 << 8),
            })
        );
        assert_eq!(
            layout.cached_neighbor(block, Direction::Up),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(15, 16, 15),
                section_slot: 87,
                local_index: 15 | (15 << 4),
            })
        );
    }

    #[test]
    fn cache_layout_rejects_cached_neighbors_outside_cache_window() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let Some(east_edge) = layout.cached_block(BlockPos::new(47, 0, 0)) else {
            panic!("edge block should be inside section cache");
        };
        let Some(bottom_edge) = layout.cached_block(BlockPos::new(0, -32, 0)) else {
            panic!("bottom block should be inside section cache");
        };

        assert_eq!(layout.cached_neighbor(east_edge, Direction::East), None);
        assert_eq!(layout.cached_neighbor(bottom_edge, Direction::Down), None);
    }

    #[test]
    fn overworld_cache_layout_uses_light_sections_plus_two_buffers() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(-64, 384));

        assert_eq!(layout.range().min_section_y(), -5);
        assert_eq!(layout.range().max_section_y_exclusive(), 21);
        assert_eq!(layout.range().section_count(), 26);
        assert_eq!(layout.cached_min_section_y(), -6);
        assert_eq!(layout.cached_max_section_y_exclusive(), 22);
        assert_eq!(layout.cached_section_count(), 28);
        assert_eq!(layout.section_slot_count(), 700);
    }

    #[test]
    fn packed_light_block_pos_masks_to_scalable_lux_position_bits() {
        let packed = PackedLightBlockPos::from_raw(u32::MAX);

        assert_eq!(packed.raw(), (1 << 28) - 1);
        assert_eq!(packed.encoded_x(), 63);
        assert_eq!(packed.encoded_z(), 63);
        assert_eq!(packed.encoded_y(), u16::MAX);
    }

    #[test]
    fn cache_layout_encodes_scalable_lux_queue_position_window() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));

        assert_eq!(layout.encoded_min_block_x(), -24);
        assert_eq!(layout.encoded_max_block_x_exclusive(), 40);
        assert_eq!(layout.encoded_min_block_z(), -24);
        assert_eq!(layout.encoded_max_block_z_exclusive(), 40);

        let Some(center) = layout.encode_block_pos(BlockPos::new(7, 0, 7)) else {
            panic!("center chunk block should encode");
        };
        assert_eq!(center.encoded_x(), 31);
        assert_eq!(center.encoded_z(), 31);
        assert_eq!(center.encoded_y(), 32);
        assert_eq!(
            layout.decode_block_pos(center),
            Some(BlockPos::new(7, 0, 7))
        );

        let Some(min) = layout.encode_block_pos(BlockPos::new(-24, -32, -24)) else {
            panic!("minimum queue block should encode");
        };
        assert_eq!(min.raw(), 0);
        assert_eq!(
            layout.decode_block_pos(min),
            Some(BlockPos::new(-24, -32, -24))
        );

        let Some(max) = layout.encode_block_pos(BlockPos::new(39, 47, 39)) else {
            panic!("maximum queue block should encode");
        };
        assert_eq!(max.encoded_x(), 63);
        assert_eq!(max.encoded_z(), 63);
        assert_eq!(max.encoded_y(), 79);
        assert_eq!(
            layout.decode_block_pos(max),
            Some(BlockPos::new(39, 47, 39))
        );

        assert_eq!(layout.encode_block_pos(BlockPos::new(40, 0, 0)), None);
        assert_eq!(layout.encode_block_pos(BlockPos::new(0, 0, 40)), None);
        assert_eq!(layout.encode_block_pos(BlockPos::new(0, 48, 0)), None);
    }

    #[test]
    fn cache_layout_maps_packed_positions_to_cached_blocks() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let Some(packed) = layout.encode_block_pos(BlockPos::new(7, 0, 7)) else {
            panic!("center block should encode");
        };

        assert_eq!(
            layout.cached_block_from_packed(packed),
            Some(CachedLightBlock {
                block_pos: BlockPos::new(7, 0, 7),
                section_slot: 62,
                local_index: 7 | (7 << 4),
            })
        );
        assert_eq!(
            layout.cached_block_from_packed(PackedLightBlockPos::from_raw(u32::MAX)),
            None
        );
    }

    #[test]
    fn notification_cache_marks_cached_sections_once() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let mut notifications = LightUpdateNotificationCache::new(layout);
        let section = SectionPos::new(0, 0, 0);

        assert_eq!(notifications.layout(), layout);
        assert!(notifications.is_empty());
        assert!(notifications.mark_section(section));
        assert!(!notifications.is_empty());
        assert!(notifications.is_marked_section(section));
        assert!(!notifications.mark_section(section));
        assert_eq!(
            notifications.marked_section_positions().collect::<Vec<_>>(),
            vec![section]
        );

        notifications.clear();
        assert!(notifications.is_empty());
        assert!(!notifications.is_marked_section(section));
    }

    #[test]
    fn notification_cache_marks_one_block_light_neighborhood() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let mut notifications = LightUpdateNotificationCache::new(layout);

        assert_eq!(
            notifications.mark_block_neighborhood(BlockPos::new(8, 8, 8)),
            Some(1)
        );
        assert_eq!(
            notifications.marked_section_positions().collect::<Vec<_>>(),
            vec![SectionPos::new(0, 0, 0)]
        );

        notifications.clear();
        assert_eq!(
            notifications.mark_block_neighborhood(BlockPos::new(16, 16, 16)),
            Some(8)
        );

        let marked = notifications.marked_section_positions().collect::<Vec<_>>();
        assert_eq!(marked.len(), 8);
        assert!(marked.contains(&SectionPos::new(0, 0, 0)));
        assert!(marked.contains(&SectionPos::new(1, 0, 0)));
        assert!(marked.contains(&SectionPos::new(0, 1, 0)));
        assert!(marked.contains(&SectionPos::new(1, 1, 1)));
    }

    #[test]
    fn notification_cache_rejects_partial_block_neighborhoods() {
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range(0, 16));
        let mut notifications = LightUpdateNotificationCache::new(layout);

        assert_eq!(
            notifications.mark_block_neighborhood(BlockPos::new(48, 8, 8)),
            None
        );
        assert!(notifications.is_empty());
    }
}

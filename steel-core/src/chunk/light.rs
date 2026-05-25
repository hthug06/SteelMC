//! Light storage primitives used by chunk and world lighting.

use rustc_hash::{FxHashMap, FxHashSet};
use steel_protocol::packets::game::LightUpdatePacketData;
use steel_registry::{
    blocks::{block_state_ext::BlockStateExt, shapes::VoxelShape},
    vanilla_blocks,
};
use steel_utils::{
    BlockPos, BlockStateId, ChunkPos, Direction, PackedSectionPos, SectionPos, codec::BitSet,
};

use crate::{
    chunk::section::Sections,
    physics::shapes::{face_shape_occludes, merged_face_occludes},
};

/// Maximum light value stored by vanilla lighting.
pub const MAX_LIGHT_LEVEL: u8 = 15;
/// Minimum opacity used while propagating vanilla light.
pub const MIN_LIGHT_OPACITY: u8 = 1;
/// Opacity returned when a block face fully blocks light.
pub const LIGHT_BLOCKED: u8 = MAX_LIGHT_LEVEL + 1;
/// Vanilla stores one extra light section below and above the build height.
pub const LIGHT_SECTION_PADDING: i32 = 1;

/// Number of blocks along one edge of a light section.
pub const DATA_LAYER_EDGE: usize = 16;
/// Number of blocks in a light section.
pub const DATA_LAYER_BLOCK_COUNT: usize = DATA_LAYER_EDGE * DATA_LAYER_EDGE * DATA_LAYER_EDGE;
/// Number of packed bytes in a light section.
pub const DATA_LAYER_SIZE: usize = DATA_LAYER_BLOCK_COUNT / 2;
const DATA_LAYER_Y_STRIDE: usize = DATA_LAYER_EDGE * DATA_LAYER_EDGE / 2;
const CHUNK_EDGE: usize = 16;
const CHUNK_COLUMN_COUNT: usize = CHUNK_EDGE * CHUNK_EDGE;
const NEGATIVE_INFINITY: i32 = i32::MIN;
const POSITIVE_INFINITY: i32 = i32::MAX;
const SECTION_HAS_DATA_BIT: u8 = 0b0010_0000;
const SECTION_NEIGHBOR_COUNT_BITS: u8 = 0b0001_1111;
const MAX_SECTION_NEIGHBORS: i32 = 26;
const QUEUE_ENTRY_LEVEL_MASK: u64 = 0b1111;
const QUEUE_ENTRY_DIRECTIONS_MASK: u64 = 0b11_1111_0000;
const QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE: u64 = 1 << 10;
const QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION: u64 = 1 << 11;
const LIGHT_QUEUE_MIN_CAPACITY: usize = 512;

/// Vanilla light layer kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightLayer {
    /// Sky light propagated from dimensions with skylight.
    Sky,
    /// Block light emitted by blocks.
    Block,
}

/// Returns whether vanilla must re-check lighting after a block-state change.
#[must_use]
pub fn has_different_light_properties(old_state: BlockStateId, new_state: BlockStateId) -> bool {
    old_state != new_state
        && (old_state.get_light_dampening() != new_state.get_light_dampening()
            || old_state.get_light_emission() != new_state.get_light_emission()
            || old_state.use_shape_for_light_occlusion()
            || new_state.use_shape_for_light_occlusion())
}

/// Returns vanilla's simple opacity for light propagation.
///
/// Vanilla clamps block light dampening to at least one while propagating
/// through neighbors.
#[must_use]
pub fn get_light_opacity(state: BlockStateId) -> u8 {
    state.get_light_dampening().max(MIN_LIGHT_OPACITY)
}

/// Returns the occlusion shape vanilla lighting uses for a block state.
#[must_use]
pub fn light_occlusion_shape(state: BlockStateId) -> VoxelShape {
    if !state.get_block().config.can_occlude || !state.use_shape_for_light_occlusion() {
        return &[];
    }

    state.get_occlusion_shape()
}

/// Returns vanilla's `LightEngine.getLightBlockInto` result.
#[must_use]
pub fn get_light_block_into(
    from_state: BlockStateId,
    to_state: BlockStateId,
    direction: Direction,
    simple_opacity: u8,
) -> u8 {
    let from_shape = light_occlusion_shape(from_state);
    let to_shape = light_occlusion_shape(to_state);
    if from_shape.is_empty() && to_shape.is_empty() {
        return simple_opacity;
    }

    if merged_face_occludes(from_shape, to_shape, direction) {
        LIGHT_BLOCKED
    } else {
        simple_opacity
    }
}

/// Returns whether the selected state faces fully occlude light.
#[must_use]
pub fn light_face_occludes(
    from_state: BlockStateId,
    to_state: BlockStateId,
    direction: Direction,
) -> bool {
    let from_shape = light_occlusion_shape(from_state);
    let to_shape = light_occlusion_shape(to_state);
    face_shape_occludes(from_shape, direction, to_shape, direction.opposite())
}

/// Vanilla's packed light-propagation queue entry.
///
/// `LightEngine.QueueEntry` stores the source level in bits 0..3, one
/// propagation bit per vanilla `Direction.ordinal()` in bits 4..9, and two
/// increase flags in bits 10 and 11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightQueueEntry(u64);

impl LightQueueEntry {
    /// Creates a decrease entry that propagates to all directions except one.
    #[must_use]
    pub const fn decrease_skip_one_direction(
        old_from_level: u8,
        skip_direction: Direction,
    ) -> Self {
        Self::with_level(
            Self::without_direction(QUEUE_ENTRY_DIRECTIONS_MASK, skip_direction),
            old_from_level,
        )
    }

    /// Creates a decrease entry that propagates to all directions.
    #[must_use]
    pub const fn decrease_all_directions(old_from_level: u8) -> Self {
        Self::with_level(QUEUE_ENTRY_DIRECTIONS_MASK, old_from_level)
    }

    /// Creates an increase entry sourced from a block's light emission.
    #[must_use]
    pub const fn increase_light_from_emission(new_from_level: u8, from_empty_shape: bool) -> Self {
        let mut entry = QUEUE_ENTRY_DIRECTIONS_MASK | QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION;
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(entry, new_from_level)
    }

    /// Creates an increase entry that propagates to all directions except one.
    #[must_use]
    pub const fn increase_skip_one_direction(
        new_from_level: u8,
        from_empty_shape: bool,
        skip_direction: Direction,
    ) -> Self {
        let mut entry = Self::without_direction(QUEUE_ENTRY_DIRECTIONS_MASK, skip_direction);
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(entry, new_from_level)
    }

    /// Creates an increase entry that propagates to exactly one direction.
    #[must_use]
    pub const fn increase_only_one_direction(
        new_from_level: u8,
        from_empty_shape: bool,
        direction: Direction,
    ) -> Self {
        let mut entry = 0;
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(Self::with_direction(entry, direction), new_from_level)
    }

    /// Creates a sky-source increase entry for selected directions.
    #[must_use]
    pub const fn increase_sky_source_in_directions(
        down: bool,
        north: bool,
        south: bool,
        west: bool,
        east: bool,
    ) -> Self {
        let mut entry = MAX_LIGHT_LEVEL as u64;
        if down {
            entry = Self::with_direction(entry, Direction::Down);
        }
        if north {
            entry = Self::with_direction(entry, Direction::North);
        }
        if south {
            entry = Self::with_direction(entry, Direction::South);
        }
        if west {
            entry = Self::with_direction(entry, Direction::West);
        }
        if east {
            entry = Self::with_direction(entry, Direction::East);
        }

        Self(entry)
    }

    /// Creates a queue entry from vanilla's packed representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns vanilla's packed representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns the source light level stored in this entry.
    #[must_use]
    pub const fn from_level(self) -> u8 {
        (self.0 & QUEUE_ENTRY_LEVEL_MASK) as u8
    }

    /// Returns true if propagation starts from an empty occlusion shape.
    #[must_use]
    pub const fn is_from_empty_shape(self) -> bool {
        self.0 & QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE != 0
    }

    /// Returns true if this increase came from block light emission.
    #[must_use]
    pub const fn is_increase_from_emission(self) -> bool {
        self.0 & QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION != 0
    }

    /// Returns true if this entry propagates in `direction`.
    #[must_use]
    pub const fn should_propagate_in_direction(self, direction: Direction) -> bool {
        self.0 & Self::direction_bit(direction) != 0
    }

    const fn with_level(entry: u64, level: u8) -> Self {
        Self(entry & !QUEUE_ENTRY_LEVEL_MASK | (level as u64 & QUEUE_ENTRY_LEVEL_MASK))
    }

    const fn with_direction(entry: u64, direction: Direction) -> u64 {
        entry | Self::direction_bit(direction)
    }

    const fn without_direction(entry: u64, direction: Direction) -> u64 {
        entry & !Self::direction_bit(direction)
    }

    const fn direction_bit(direction: Direction) -> u64 {
        1 << (Self::vanilla_direction_index(direction) + 4)
    }

    const fn vanilla_direction_index(direction: Direction) -> u64 {
        match direction {
            Direction::Down => 0,
            Direction::Up => 1,
            Direction::North => 2,
            Direction::South => 3,
            Direction::West => 4,
            Direction::East => 5,
        }
    }
}

/// One typed light propagation queue item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuedLightUpdate {
    /// Block position whose light should propagate.
    pub block_pos: BlockPos,
    /// Packed vanilla propagation metadata.
    pub entry: LightQueueEntry,
}

/// Array-backed FIFO used for vanilla light propagation work.
///
/// Vanilla stores alternating packed block positions and `QueueEntry` longs in
/// `LongArrayFIFOQueue`. Steel keeps typed records instead, while preserving
/// the FIFO ordering and packed queue-entry semantics that propagation depends
/// on.
#[derive(Debug)]
pub struct LightPropagationQueue {
    entries: Vec<QueuedLightUpdate>,
    read_index: usize,
}

impl LightPropagationQueue {
    /// Creates an empty propagation queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(LIGHT_QUEUE_MIN_CAPACITY),
            read_index: 0,
        }
    }

    /// Returns true when no queued work remains.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_index >= self.entries.len()
    }

    /// Returns the number of queued items that have not been dequeued yet.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len() - self.read_index
    }

    /// Adds propagation work to the back of the queue.
    pub fn enqueue(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.entries.push(QueuedLightUpdate { block_pos, entry });
    }

    /// Removes propagation work from the front of the queue.
    pub fn dequeue(&mut self) -> Option<QueuedLightUpdate> {
        if self.is_empty() {
            self.clear();
            return None;
        }

        let update = self.entries[self.read_index];
        self.read_index += 1;
        if self.is_empty() {
            self.clear();
        }

        Some(update)
    }

    /// Removes all queued work while keeping allocated storage for reuse.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.read_index = 0;
    }
}

impl Default for LightPropagationQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Vanilla's separate increase and decrease propagation queues.
#[derive(Debug, Default)]
pub struct LightPropagationQueues {
    increase: LightPropagationQueue,
    decrease: LightPropagationQueue,
}

impl LightPropagationQueues {
    /// Creates empty increase and decrease queues.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when either propagation queue contains work.
    #[must_use]
    pub fn has_work(&self) -> bool {
        !self.increase.is_empty() || !self.decrease.is_empty()
    }

    /// Enqueues decrease propagation work.
    pub fn enqueue_decrease(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.decrease.enqueue(block_pos, entry);
    }

    /// Enqueues increase propagation work.
    pub fn enqueue_increase(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.increase.enqueue(block_pos, entry);
    }

    /// Dequeues decrease propagation work.
    pub fn dequeue_decrease(&mut self) -> Option<QueuedLightUpdate> {
        self.decrease.dequeue()
    }

    /// Dequeues increase propagation work.
    pub fn dequeue_increase(&mut self) -> Option<QueuedLightUpdate> {
        self.increase.dequeue()
    }

    /// Removes all increase and decrease work.
    pub fn clear(&mut self) {
        self.increase.clear();
        self.decrease.clear();
    }
}

/// Error returned when a light section state would hold an invalid neighbor count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightSectionStateError {
    /// Requested neighbor count.
    pub neighbor_count: i32,
}

/// Error returned when a block light value is written without section light data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingLightDataLayerError {
    /// Section that does not currently store a light data layer.
    pub section_pos: SectionPos,
}

/// Vanilla's packed light-section state byte.
///
/// Bits 0..4 store the count of neighboring data sections and bit 5 stores
/// whether this section itself has block data. A non-zero neighbor count keeps
/// an otherwise empty section alive as `LIGHT_ONLY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightSectionState(u8);

impl LightSectionState {
    /// Empty section with no data and no neighboring data sections.
    pub const EMPTY: Self = Self(0);

    /// Creates a state from vanilla's packed byte representation.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        Self(raw)
    }

    /// Returns vanilla's packed byte representation.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Returns true when this section contains chunk block data.
    #[must_use]
    pub const fn has_data(self) -> bool {
        self.0 & SECTION_HAS_DATA_BIT != 0
    }

    /// Returns the count of neighboring data sections.
    #[must_use]
    pub const fn neighbor_count(self) -> u8 {
        self.0 & SECTION_NEIGHBOR_COUNT_BITS
    }

    /// Returns this section's debug type.
    #[must_use]
    pub const fn section_type(self) -> LightSectionType {
        if self.0 == 0 {
            LightSectionType::Empty
        } else if self.has_data() {
            LightSectionType::LightAndData
        } else {
            LightSectionType::LightOnly
        }
    }

    /// Sets the data bit.
    #[must_use]
    pub const fn with_has_data(self, has_data: bool) -> Self {
        if has_data {
            Self(self.0 | SECTION_HAS_DATA_BIT)
        } else {
            Self(self.0 & !SECTION_HAS_DATA_BIT)
        }
    }

    /// Sets the neighboring data-section count.
    pub fn with_neighbor_count(self, neighbor_count: i32) -> Result<Self, LightSectionStateError> {
        if !(0..=MAX_SECTION_NEIGHBORS).contains(&neighbor_count) {
            return Err(LightSectionStateError { neighbor_count });
        }

        Ok(Self(
            self.0 & !SECTION_NEIGHBOR_COUNT_BITS
                | (neighbor_count as u8 & SECTION_NEIGHBOR_COUNT_BITS),
        ))
    }
}

/// Debug type for a light section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightSectionType {
    /// No light layer is stored for the section.
    Empty,
    /// Light-only section kept alive by neighboring block data.
    LightOnly,
    /// Section with block data and light storage.
    LightAndData,
}

/// Error returned when a world height cannot produce a valid light-section range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightSectionRangeError {
    /// Minimum build height used to create the range.
    pub min_y: i32,
    /// Build height used to create the range.
    pub height: i32,
}

/// Inclusive/exclusive vertical range of light sections for a level.
///
/// Vanilla's `LevelLightEngine` exposes this as `getMinLightSection()`,
/// `getMaxLightSection()`, and `getLightSectionCount()`. The range is padded by
/// one section below and one section above the real chunk sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightSectionRange {
    min_section_y: i32,
    section_count: i32,
}

impl LightSectionRange {
    /// Creates the vanilla padded light-section range for a world height.
    pub fn from_world_height(min_y: i32, height: i32) -> Result<Self, LightSectionRangeError> {
        if height <= 0 {
            return Err(LightSectionRangeError { min_y, height });
        }

        let Some(max_y) = min_y.checked_add(height - 1) else {
            return Err(LightSectionRangeError { min_y, height });
        };

        let min_chunk_section_y = SectionPos::block_to_section_coord(min_y);
        let max_chunk_section_y = SectionPos::block_to_section_coord(max_y);
        let section_count =
            max_chunk_section_y - min_chunk_section_y + 1 + LIGHT_SECTION_PADDING * 2;

        Ok(Self {
            min_section_y: min_chunk_section_y - LIGHT_SECTION_PADDING,
            section_count,
        })
    }

    /// Returns the first light section Y coordinate.
    #[must_use]
    pub const fn min_section_y(self) -> i32 {
        self.min_section_y
    }

    /// Returns the section Y coordinate one past the last light section.
    #[must_use]
    pub const fn max_section_y_exclusive(self) -> i32 {
        self.min_section_y + self.section_count
    }

    /// Returns the number of light sections in this range.
    #[must_use]
    pub const fn section_count(self) -> usize {
        self.section_count as usize
    }

    /// Converts a packet light-section index to a section Y coordinate.
    #[must_use]
    pub fn section_y(self, section_index: usize) -> Option<i32> {
        if section_index >= self.section_count() {
            return None;
        }

        Some(self.min_section_y + section_index as i32)
    }

    /// Converts a section Y coordinate to a packet light-section index.
    #[must_use]
    pub fn section_index(self, section_y: i32) -> Option<usize> {
        if section_y < self.min_section_y || section_y >= self.max_section_y_exclusive() {
            return None;
        }

        Some((section_y - self.min_section_y) as usize)
    }

    /// Creates a section position for this light range's chunk column.
    #[must_use]
    pub fn section_pos(self, chunk_pos: ChunkPos, section_y: i32) -> SectionPos {
        SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y)
    }

    #[must_use]
    fn empty_bit_set(self) -> BitSet {
        BitSet(vec![0; self.section_count().div_ceil(64)].into_boxed_slice())
    }
}

/// `DataLayer` storage keyed by vanilla packed section position.
///
/// This mirrors vanilla's `DataLayerStorageMap`: it owns the section light
/// layers and offers explicit `copy_*` methods for the future light engine's
/// visible/updating map split.
#[derive(Debug, Default)]
pub struct DataLayerStorageMap {
    layers: FxHashMap<PackedSectionPos, DataLayer>,
}

impl DataLayerStorageMap {
    /// Creates an empty section light storage map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: FxHashMap::default(),
        }
    }

    /// Returns true when no light layers are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the number of stored section light layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true when a section has a stored light layer.
    #[must_use]
    pub fn has_layer(&self, section_pos: SectionPos) -> bool {
        self.layers.contains_key(&Self::key(section_pos))
    }

    /// Returns the light layer for a section.
    #[must_use]
    pub fn get_layer(&self, section_pos: SectionPos) -> Option<&DataLayer> {
        self.layers.get(&Self::key(section_pos))
    }

    /// Returns a mutable light layer for a section.
    #[must_use]
    pub fn get_layer_mut(&mut self, section_pos: SectionPos) -> Option<&mut DataLayer> {
        self.layers.get_mut(&Self::key(section_pos))
    }

    /// Inserts a light layer for a section, returning the old layer if present.
    pub fn set_layer(&mut self, section_pos: SectionPos, layer: DataLayer) -> Option<DataLayer> {
        self.layers.insert(Self::key(section_pos), layer)
    }

    /// Removes a light layer for a section.
    pub fn remove_layer(&mut self, section_pos: SectionPos) -> Option<DataLayer> {
        self.layers.remove(&Self::key(section_pos))
    }

    /// Copies a section's light layer in place and returns the mutable copy.
    pub fn copy_data_layer(&mut self, section_pos: SectionPos) -> Option<&mut DataLayer> {
        let key = Self::key(section_pos);
        let copied = self.layers.get(&key).map(DataLayer::copy)?;
        self.layers.insert(key, copied);
        self.layers.get_mut(&key)
    }

    /// Returns a deep copy of this storage map.
    #[must_use]
    pub fn copy_map(&self) -> Self {
        let mut layers = FxHashMap::default();
        layers.reserve(self.layers.len());
        for (&section_pos, layer) in &self.layers {
            layers.insert(section_pos, layer.copy());
        }
        Self { layers }
    }

    fn key(section_pos: SectionPos) -> PackedSectionPos {
        PackedSectionPos::from(section_pos)
    }
}

/// Common light-section storage state shared by vanilla block and sky lighting.
///
/// This mirrors vanilla `LayerLightSectionStorage` for section states, queued
/// data, pending removals, and visible/updating data maps.
#[derive(Debug)]
pub struct LayerLightSectionStorage {
    layer: LightLayer,
    section_states: FxHashMap<PackedSectionPos, LightSectionState>,
    columns_with_sources: FxHashSet<PackedSectionPos>,
    visible_section_data: DataLayerStorageMap,
    updating_section_data: DataLayerStorageMap,
    changed_sections: FxHashSet<PackedSectionPos>,
    sections_affected_by_light_updates: FxHashSet<PackedSectionPos>,
    queued_sections: FxHashMap<PackedSectionPos, DataLayer>,
    columns_to_retain_queued_data_for: FxHashSet<PackedSectionPos>,
    to_remove: FxHashSet<PackedSectionPos>,
    has_inconsistencies: bool,
    sky_data: Option<SkyLightSectionData>,
}

impl LayerLightSectionStorage {
    /// Creates empty section storage for one light layer.
    #[must_use]
    pub fn new(layer: LightLayer) -> Self {
        let updating_section_data = DataLayerStorageMap::new();
        let visible_section_data = updating_section_data.copy_map();
        Self {
            layer,
            section_states: FxHashMap::default(),
            columns_with_sources: FxHashSet::default(),
            visible_section_data,
            updating_section_data,
            changed_sections: FxHashSet::default(),
            sections_affected_by_light_updates: FxHashSet::default(),
            queued_sections: FxHashMap::default(),
            columns_to_retain_queued_data_for: FxHashSet::default(),
            to_remove: FxHashSet::default(),
            has_inconsistencies: false,
            sky_data: SkyLightSectionData::for_layer(layer),
        }
    }

    /// Returns the layer kind stored here.
    #[must_use]
    pub const fn layer(&self) -> LightLayer {
        self.layer
    }

    /// Returns whether pending queued/removal work needs to be reconciled.
    #[must_use]
    pub const fn has_inconsistencies(&self) -> bool {
        self.has_inconsistencies
    }

    /// Returns whether the updating map stores light for this section.
    #[must_use]
    pub fn storing_light_for_section(&self, section_pos: SectionPos) -> bool {
        self.updating_section_data.has_layer(section_pos)
    }

    /// Returns visible light data or queued replacement data for packet/saving reads.
    #[must_use]
    pub fn get_data_layer_data(&self, section_pos: SectionPos) -> Option<&DataLayer> {
        let key = Self::key(section_pos);
        if let Some(layer) = self.queued_sections.get(&key) {
            return Some(layer);
        }

        self.visible_section_data.get_layer(section_pos)
    }

    /// Returns the updating data layer for this section.
    #[must_use]
    pub fn get_updating_data_layer(&self, section_pos: SectionPos) -> Option<&DataLayer> {
        self.updating_section_data.get_layer(section_pos)
    }

    /// Returns the visible light value for a block.
    ///
    /// Vanilla implements this in separate block/sky storage subclasses. Steel
    /// keeps shared section storage in one type, so the layer selects the
    /// vanilla read path here.
    #[must_use]
    pub fn get_light_value(&self, block_pos: BlockPos) -> u8 {
        match self.layer {
            LightLayer::Block => self.get_block_light_value(block_pos),
            LightLayer::Sky => self.get_sky_light_value(block_pos),
        }
    }

    /// Returns the updating light value stored for a block.
    ///
    /// Vanilla assumes the caller already checked `storingLightForSection`;
    /// Steel returns `None` for sections without light data.
    #[must_use]
    pub fn get_stored_level(&self, block_pos: BlockPos) -> Option<u8> {
        let section_pos = SectionPos::from_block_pos(block_pos);
        self.updating_section_data
            .get_layer(section_pos)
            .map(|layer| Self::get_data_layer_block_value(layer, block_pos))
    }

    /// Writes an updating light value for a block.
    ///
    /// Vanilla assumes the caller already checked `storingLightForSection`;
    /// Steel returns an error instead of panicking on missing section data.
    pub fn set_stored_level(
        &mut self,
        block_pos: BlockPos,
        level: u8,
    ) -> Result<(), MissingLightDataLayerError> {
        let section_pos = SectionPos::from_block_pos(block_pos);
        let Some(layer) = self.get_data_layer_to_write(section_pos) else {
            return Err(MissingLightDataLayerError { section_pos });
        };

        Self::set_data_layer_block_value(layer, block_pos, level);
        self.mark_sections_around_block_as_affected(block_pos);
        Ok(())
    }

    /// Returns a mutable copy-on-write updating layer for this section.
    pub fn get_data_layer_to_write(&mut self, section_pos: SectionPos) -> Option<&mut DataLayer> {
        if !self.updating_section_data.has_layer(section_pos) {
            return None;
        }

        let key = Self::key(section_pos);
        if self.changed_sections.insert(key) {
            return self.updating_section_data.copy_data_layer(section_pos);
        }

        self.updating_section_data.get_layer_mut(section_pos)
    }

    /// Queues externally loaded section data.
    pub fn queue_section_data(&mut self, section_pos: SectionPos, data: Option<DataLayer>) {
        let key = Self::key(section_pos);
        if let Some(data) = data {
            self.queued_sections.insert(key, data);
            self.has_inconsistencies = true;
        } else {
            self.queued_sections.remove(&key);
        }
    }

    /// Keeps queued data for a chunk column when currently stored data is removed.
    pub fn retain_data(&mut self, section_zero_pos: SectionPos, retain: bool) {
        let zero_key = Self::zero_key(section_zero_pos);
        if retain {
            self.columns_to_retain_queued_data_for.insert(zero_key);
        } else {
            self.columns_to_retain_queued_data_for.remove(&zero_key);
        }
    }

    /// Enables or disables light sources for a chunk column.
    pub fn set_light_enabled(&mut self, section_zero_pos: SectionPos, enable: bool) {
        let zero_key = Self::zero_key(section_zero_pos);
        if enable {
            self.columns_with_sources.insert(zero_key);
        } else {
            self.columns_with_sources.remove(&zero_key);
        }
    }

    /// Enables sky sources and fills existing empty fully-sourced sky layers.
    ///
    /// Vanilla does this in `SkyLightEngine.setLightEnabled`; Steel keeps the
    /// storage mutation here because it depends only on sky section metadata and
    /// `ChunkSkyLightSources`.
    pub fn enable_sky_light_sources(
        &mut self,
        section_zero_pos: SectionPos,
        sources: &ChunkSkyLightSources,
    ) -> Option<()> {
        self.sky_data.as_ref()?;

        self.set_light_enabled(section_zero_pos, true);

        let highest_non_source_y = sources.get_highest_lowest_source_y().wrapping_sub(1);
        let lowest_fully_source_section_y =
            SectionPos::block_to_section_coord(highest_non_source_y) + 1;
        let top_section_y = self.get_top_section_y(section_zero_pos)?;
        let bottom_section_y = self
            .get_bottom_section_y()?
            .max(lowest_fully_source_section_y);

        for section_y in (bottom_section_y..top_section_y).rev() {
            let section_pos =
                SectionPos::new(section_zero_pos.x(), section_y, section_zero_pos.z());
            if let Some(data_layer) = self.get_data_layer_to_write(section_pos) {
                if data_layer.is_empty() {
                    data_layer.fill(MAX_LIGHT_LEVEL);
                }
            }
        }

        Some(())
    }

    /// Returns true when this section's column has light sources enabled.
    #[must_use]
    pub fn light_on_in_section(&self, section_pos: SectionPos) -> bool {
        self.columns_with_sources
            .contains(&Self::zero_key(section_pos))
    }

    /// Returns true when this zero-section column has light sources enabled.
    #[must_use]
    pub fn light_on_in_column(&self, section_zero_pos: SectionPos) -> bool {
        self.columns_with_sources
            .contains(&Self::zero_key(section_zero_pos))
    }

    /// Updates whether a chunk section contains block data.
    pub fn update_section_status(
        &mut self,
        section_pos: SectionPos,
        section_empty: bool,
    ) -> Result<(), LightSectionStateError> {
        let state = self.section_state(section_pos);
        let new_state = state.with_has_data(!section_empty);
        if state == new_state {
            return Ok(());
        }

        self.put_section_state(section_pos, new_state);
        let neighbor_increment = if section_empty { -1 } else { 1 };

        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                for offset_y in -1..=1 {
                    if offset_x == 0 && offset_y == 0 && offset_z == 0 {
                        continue;
                    }

                    let neighbor_pos = Self::offset(section_pos, offset_x, offset_y, offset_z);
                    let neighbor_state = self.section_state(neighbor_pos);
                    let neighbor_count =
                        i32::from(neighbor_state.neighbor_count()) + neighbor_increment;
                    self.put_section_state(
                        neighbor_pos,
                        neighbor_state.with_neighbor_count(neighbor_count)?,
                    );
                }
            }
        }

        Ok(())
    }

    /// Reconciles queued section data and pending removals.
    pub fn mark_new_inconsistencies(&mut self) {
        if !self.has_inconsistencies {
            return;
        }

        self.has_inconsistencies = false;
        let to_remove = std::mem::take(&mut self.to_remove);
        for key in &to_remove {
            let section_pos = key.to_section_pos();
            let queued = self.queued_sections.remove(key);
            let stored = self.updating_section_data.remove_layer(section_pos);
            if self
                .columns_to_retain_queued_data_for
                .contains(&Self::zero_key(section_pos))
            {
                if let Some(layer) = queued {
                    self.queued_sections.insert(*key, layer);
                } else if let Some(layer) = stored {
                    self.queued_sections.insert(*key, layer);
                }
            }
        }

        for key in to_remove {
            self.on_node_removed(key.to_section_pos());
            self.changed_sections.insert(key);
        }

        let queued_keys: Vec<PackedSectionPos> = self.queued_sections.keys().copied().collect();
        for key in queued_keys {
            let section_pos = key.to_section_pos();
            if self.storing_light_for_section(section_pos) {
                if let Some(data) = self.queued_sections.remove(&key) {
                    self.updating_section_data.set_layer(section_pos, data);
                    self.changed_sections.insert(key);
                }
            }
        }
    }

    /// Copies changed updating data into the visible map and returns affected sections.
    pub fn swap_section_map(&mut self) -> Vec<SectionPos> {
        if !self.changed_sections.is_empty() {
            self.visible_section_data = self.updating_section_data.copy_map();
            if let Some(sky_data) = self.sky_data.as_mut() {
                sky_data.visible = sky_data.updating.copy_map();
            }
            self.changed_sections.clear();
        }

        let mut affected = Vec::with_capacity(self.sections_affected_by_light_updates.len());
        for key in std::mem::take(&mut self.sections_affected_by_light_updates) {
            affected.push(key.to_section_pos());
        }
        affected
    }

    /// Returns this section's debug type.
    #[must_use]
    pub fn get_debug_section_type(&self, section_pos: SectionPos) -> LightSectionType {
        self.section_state(section_pos).section_type()
    }

    /// Returns whether sky storage has light data at or below `section_y`.
    ///
    /// This is a sky-light-only query. Block-light storage returns `None`.
    #[must_use]
    pub fn has_light_data_at_or_below(&self, section_y: i32) -> Option<bool> {
        self.sky_data
            .as_ref()
            .map(|sky_data| section_y >= sky_data.updating.current_lowest_y)
    }

    /// Returns whether a sky section is above all stored sky data in its column.
    ///
    /// This is a sky-light-only query. Block-light storage returns `None`.
    #[must_use]
    pub fn is_above_data(&self, section_pos: SectionPos) -> Option<bool> {
        let sky_data = self.sky_data.as_ref()?;
        let zero_key = Self::zero_key(section_pos);
        let top_section = sky_data.updating.top_section(zero_key);
        Some(top_section == sky_data.updating.current_lowest_y || section_pos.y() >= top_section)
    }

    /// Returns the sky top-section Y for a zero-section column.
    ///
    /// This is a sky-light-only query. Block-light storage returns `None`.
    #[must_use]
    pub fn get_top_section_y(&self, section_zero_pos: SectionPos) -> Option<i32> {
        self.sky_data.as_ref().map(|sky_data| {
            sky_data
                .updating
                .top_section(Self::zero_key(section_zero_pos))
        })
    }

    /// Returns the lowest section Y ever observed by sky storage.
    ///
    /// This is a sky-light-only query. Block-light storage returns `None`.
    #[must_use]
    pub fn get_bottom_section_y(&self) -> Option<i32> {
        self.sky_data
            .as_ref()
            .map(|sky_data| sky_data.updating.current_lowest_y)
    }

    fn section_state(&self, section_pos: SectionPos) -> LightSectionState {
        match self.section_states.get(&Self::key(section_pos)) {
            Some(state) => *state,
            None => LightSectionState::EMPTY,
        }
    }

    fn put_section_state(&mut self, section_pos: SectionPos, state: LightSectionState) {
        let key = Self::key(section_pos);
        if state != LightSectionState::EMPTY {
            if self.section_states.insert(key, state).is_none() {
                self.initialize_section(section_pos);
            }
        } else if self.section_states.remove(&key).is_some() {
            self.remove_section(section_pos);
        }
    }

    fn initialize_section(&mut self, section_pos: SectionPos) {
        let key = Self::key(section_pos);
        if self.to_remove.remove(&key) {
            return;
        }

        let data_layer = self.create_data_layer(section_pos);
        self.updating_section_data
            .set_layer(section_pos, data_layer);
        self.changed_sections.insert(key);
        self.on_node_added(section_pos);
        self.mark_section_and_neighbors_as_affected(section_pos);
        self.has_inconsistencies = true;
    }

    fn remove_section(&mut self, section_pos: SectionPos) {
        self.to_remove.insert(Self::key(section_pos));
        self.has_inconsistencies = true;
    }

    fn mark_section_and_neighbors_as_affected(&mut self, section_pos: SectionPos) {
        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                for offset_y in -1..=1 {
                    self.sections_affected_by_light_updates
                        .insert(Self::key(Self::offset(
                            section_pos,
                            offset_x,
                            offset_y,
                            offset_z,
                        )));
                }
            }
        }
    }

    fn mark_sections_around_block_as_affected(&mut self, block_pos: BlockPos) {
        SectionPos::around_and_at_block_pos(block_pos, |section_pos| {
            self.sections_affected_by_light_updates
                .insert(Self::key(section_pos));
        });
    }

    fn get_block_light_value(&self, block_pos: BlockPos) -> u8 {
        let section_pos = SectionPos::from_block_pos(block_pos);
        match self.visible_section_data.get_layer(section_pos) {
            Some(layer) => Self::get_data_layer_block_value(layer, block_pos),
            None => 0,
        }
    }

    fn get_sky_light_value(&self, block_pos: BlockPos) -> u8 {
        let Some(sky_data) = self.sky_data.as_ref() else {
            return MAX_LIGHT_LEVEL;
        };

        let section_pos = SectionPos::from_block_pos(block_pos);
        let section_y = section_pos.y();
        let top_section = sky_data.visible.top_section(Self::zero_key(section_pos));
        if top_section == sky_data.visible.current_lowest_y || section_y >= top_section {
            return MAX_LIGHT_LEVEL;
        }

        if let Some(layer) = self.visible_section_data.get_layer(section_pos) {
            return Self::get_data_layer_block_value(layer, block_pos);
        }

        let mut current_section_pos = section_pos;
        let mut current_section_y = section_y;
        loop {
            current_section_y += 1;
            if current_section_y >= top_section {
                return MAX_LIGHT_LEVEL;
            }

            current_section_pos = Self::offset(current_section_pos, 0, 1, 0);
            if let Some(layer) = self.visible_section_data.get_layer(current_section_pos) {
                return Self::get_data_layer_column_bottom_value(layer, block_pos);
            }
        }
    }

    fn get_data_layer_block_value(layer: &DataLayer, block_pos: BlockPos) -> u8 {
        let local_pos = SectionPos::section_relative_pos(block_pos);
        layer.get(
            local_pos.x() as usize,
            local_pos.y() as usize,
            local_pos.z() as usize,
        )
    }

    fn set_data_layer_block_value(layer: &mut DataLayer, block_pos: BlockPos, level: u8) {
        let local_pos = SectionPos::section_relative_pos(block_pos);
        layer.set(
            local_pos.x() as usize,
            local_pos.y() as usize,
            local_pos.z() as usize,
            level,
        );
    }

    fn get_data_layer_column_bottom_value(layer: &DataLayer, block_pos: BlockPos) -> u8 {
        let local_pos = SectionPos::section_relative_pos(block_pos);
        layer.get(local_pos.x() as usize, 0, local_pos.z() as usize)
    }

    fn create_data_layer(&mut self, section_pos: SectionPos) -> DataLayer {
        let key = Self::key(section_pos);
        if let Some(layer) = self.queued_sections.remove(&key) {
            return layer;
        }

        let Some(sky_data) = self.sky_data.as_ref() else {
            return DataLayer::new();
        };

        let zero_key = Self::zero_key(section_pos);
        let top_section = sky_data.updating.top_section(zero_key);
        if top_section != sky_data.updating.current_lowest_y && section_pos.y() < top_section {
            let above_data = self.first_data_layer_above(section_pos, top_section);
            return Self::repeat_first_layer(above_data);
        }

        if self.light_on_in_section(section_pos) {
            DataLayer::filled(MAX_LIGHT_LEVEL)
        } else {
            DataLayer::new()
        }
    }

    fn first_data_layer_above(&self, section_pos: SectionPos, top_section: i32) -> &DataLayer {
        let mut above_section = Self::offset(section_pos, 0, 1, 0);
        while above_section.y() < top_section {
            if let Some(layer) = self.get_updating_data_layer(above_section) {
                return layer;
            }

            above_section = Self::offset(above_section, 0, 1, 0);
        }

        panic!(
            "skylight top section invariant broken for section {:?} below top {}",
            section_pos, top_section
        );
    }

    fn repeat_first_layer(data: &DataLayer) -> DataLayer {
        if data.is_homogeneous() {
            return data.copy();
        }

        let input = data.to_bytes();
        let mut output = Box::new([0; DATA_LAYER_SIZE]);
        for y in 0..DATA_LAYER_EDGE {
            let start = y * DATA_LAYER_Y_STRIDE;
            output[start..start + DATA_LAYER_Y_STRIDE]
                .copy_from_slice(&input[0..DATA_LAYER_Y_STRIDE]);
        }

        DataLayer::from_packed_data(output)
    }

    fn on_node_added(&mut self, section_pos: SectionPos) {
        let Some(sky_data) = self.sky_data.as_mut() else {
            return;
        };

        let section_y = section_pos.y();
        if sky_data.updating.current_lowest_y > section_y {
            sky_data.updating.current_lowest_y = section_y;
        }

        let zero_key = Self::zero_key(section_pos);
        if sky_data.updating.top_section(zero_key) < section_y + 1 {
            sky_data.updating.set_top_section(zero_key, section_y + 1);
        }
    }

    fn on_node_removed(&mut self, section_pos: SectionPos) {
        let Some(sky_data) = self.sky_data.as_ref() else {
            return;
        };

        let zero_key = Self::zero_key(section_pos);
        let current_lowest_y = sky_data.updating.current_lowest_y;
        let section_y = section_pos.y();
        if sky_data.updating.top_section(zero_key) != section_y + 1 {
            return;
        }

        let mut new_top_section = section_pos;
        let mut new_top_y = section_y;
        while !self.storing_light_for_section(new_top_section) && new_top_y >= current_lowest_y {
            new_top_y -= 1;
            new_top_section = Self::offset(new_top_section, 0, -1, 0);
        }

        let stores_light = self.storing_light_for_section(new_top_section);
        let Some(sky_data) = self.sky_data.as_mut() else {
            return;
        };

        if stores_light {
            sky_data.updating.set_top_section(zero_key, new_top_y + 1);
        } else {
            sky_data.updating.remove_top_section(zero_key);
        }
    }

    fn key(section_pos: SectionPos) -> PackedSectionPos {
        PackedSectionPos::from(section_pos)
    }

    fn zero_key(section_pos: SectionPos) -> PackedSectionPos {
        PackedSectionPos::from(SectionPos::new(section_pos.x(), 0, section_pos.z()))
    }

    fn offset(section_pos: SectionPos, x: i32, y: i32, z: i32) -> SectionPos {
        SectionPos::new(
            section_pos.x() + x,
            section_pos.y() + y,
            section_pos.z() + z,
        )
    }
}

/// Sky-light metadata that vanilla stores in `SkyDataLayerStorageMap`.
///
/// Steel keeps the light data itself in `DataLayerStorageMap` and stores the
/// sky-only column metadata alongside the common storage. The visible/updating
/// split still matches vanilla's map-copy behavior.
#[derive(Debug)]
struct SkyLightSectionData {
    visible: SkyDataLayerStorageMap,
    updating: SkyDataLayerStorageMap,
}

impl SkyLightSectionData {
    fn for_layer(layer: LightLayer) -> Option<Self> {
        if layer != LightLayer::Sky {
            return None;
        }

        let updating = SkyDataLayerStorageMap::new();
        let visible = updating.copy_map();
        Some(Self { visible, updating })
    }
}

#[derive(Debug)]
struct SkyDataLayerStorageMap {
    current_lowest_y: i32,
    top_sections: FxHashMap<PackedSectionPos, i32>,
}

impl SkyDataLayerStorageMap {
    fn new() -> Self {
        Self {
            current_lowest_y: POSITIVE_INFINITY,
            top_sections: FxHashMap::default(),
        }
    }

    fn copy_map(&self) -> Self {
        let mut top_sections = FxHashMap::default();
        top_sections.reserve(self.top_sections.len());
        for (&section_pos, &top_section_y) in &self.top_sections {
            top_sections.insert(section_pos, top_section_y);
        }

        Self {
            current_lowest_y: self.current_lowest_y,
            top_sections,
        }
    }

    fn top_section(&self, zero_key: PackedSectionPos) -> i32 {
        match self.top_sections.get(&zero_key) {
            Some(top_section_y) => *top_section_y,
            None => self.current_lowest_y,
        }
    }

    fn set_top_section(&mut self, zero_key: PackedSectionPos, top_section_y: i32) {
        self.top_sections.insert(zero_key, top_section_y);
    }

    fn remove_top_section(&mut self, zero_key: PackedSectionPos) {
        self.top_sections.remove(&zero_key);
    }
}

/// Builds protocol light-update data for one chunk column.
///
/// This follows vanilla `ClientboundLightUpdatePacketData.prepareSectionData`:
/// missing layers are omitted, empty layers use the empty mask, and non-empty
/// layers are copied into the update payload.
#[must_use]
pub fn build_light_update_packet(
    chunk_pos: ChunkPos,
    range: LightSectionRange,
    sky_layers: Option<&DataLayerStorageMap>,
    block_layers: Option<&DataLayerStorageMap>,
) -> LightUpdatePacketData {
    let mut sky_y_mask = range.empty_bit_set();
    let mut block_y_mask = range.empty_bit_set();
    let mut empty_sky_y_mask = range.empty_bit_set();
    let mut empty_block_y_mask = range.empty_bit_set();
    let mut sky_updates = Vec::new();
    let mut block_updates = Vec::new();

    for section_index in 0..range.section_count() {
        let section_y = range.min_section_y + section_index as i32;
        let section_pos = range.section_pos(chunk_pos, section_y);

        if let Some(layers) = sky_layers {
            prepare_section_data(
                layers,
                section_pos,
                section_index,
                &mut sky_y_mask,
                &mut empty_sky_y_mask,
                &mut sky_updates,
            );
        }

        if let Some(layers) = block_layers {
            prepare_section_data(
                layers,
                section_pos,
                section_index,
                &mut block_y_mask,
                &mut empty_block_y_mask,
                &mut block_updates,
            );
        }
    }

    LightUpdatePacketData {
        sky_y_mask,
        block_y_mask,
        empty_sky_y_mask,
        empty_block_y_mask,
        sky_updates,
        block_updates,
    }
}

fn prepare_section_data(
    layers: &DataLayerStorageMap,
    section_pos: SectionPos,
    section_index: usize,
    mask: &mut BitSet,
    empty_mask: &mut BitSet,
    updates: &mut Vec<Vec<u8>>,
) {
    let Some(layer) = layers.get_layer(section_pos) else {
        return;
    };

    if layer.is_empty() {
        empty_mask.set(section_index, true);
    } else {
        mask.set(section_index, true);
        let bytes = layer.to_bytes();
        updates.push(bytes.as_ref().to_vec());
    }
}

/// Per-chunk cache of the lowest skylight source edge in each X/Z column.
///
/// Vanilla stores this in a 256-entry `SimpleBitStorage`. Steel keeps absolute
/// `i32` Y values instead; the cached semantics are the same, and this avoids a
/// new bit-storage abstraction before another system needs one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSkyLightSources {
    min_y: i32,
    heightmap: [i32; CHUNK_COLUMN_COUNT],
}

impl ChunkSkyLightSources {
    /// Creates an empty skylight-source cache for a level height.
    pub fn new(min_y: i32, height: i32) -> Result<Self, LightSectionRangeError> {
        if height <= 0 || min_y.checked_sub(1).is_none() || min_y.checked_add(height).is_none() {
            return Err(LightSectionRangeError { min_y, height });
        }

        let min_y = min_y - 1;
        Ok(Self {
            min_y,
            heightmap: [min_y; CHUNK_COLUMN_COUNT],
        })
    }

    /// Creates a cache for world heights already accepted by chunk construction.
    ///
    /// Invalid world heights are fatal because chunks and light sections cannot
    /// be indexed coherently without a valid vertical range.
    #[must_use]
    pub fn for_valid_world_height(min_y: i32, height: i32) -> Self {
        match Self::new(min_y, height) {
            Ok(sources) => sources,
            Err(error) => panic!("invalid world height for skylight sources: {error:?}"),
        }
    }

    /// Fills this cache from a chunk's sections.
    pub fn fill_from_sections(&mut self, sections: &Sections) {
        let Some(top_section_index) = sections
            .sections
            .iter()
            .rposition(|section| !section.read().is_empty())
        else {
            self.fill(self.min_y);
            return;
        };

        for z in 0..CHUNK_EDGE {
            for x in 0..CHUNK_EDGE {
                let initial_edge_y = self.find_lowest_source_y(sections, top_section_index, x, z);
                self.set(Self::index(x, z), initial_edge_y.max(self.min_y));
            }
        }
    }

    /// Updates one column after a block change.
    ///
    /// `state_at` is called with section-local X/Z and world Y coordinates.
    /// Returns true when the cached source edge changed.
    pub fn update(
        &mut self,
        x: usize,
        y: i32,
        z: usize,
        mut state_at: impl FnMut(usize, i32, usize) -> BlockStateId,
    ) -> bool {
        debug_assert!(x < CHUNK_EDGE);
        debug_assert!(z < CHUNK_EDGE);

        let Some(upper_edge_y) = y.checked_add(1) else {
            return false;
        };
        let index = Self::index(x, z);
        let current_lowest_source_y = self.get(index);
        if upper_edge_y < current_lowest_source_y {
            return false;
        }

        let top_state = state_at(x, upper_edge_y, z);
        let middle_state = state_at(x, y, z);
        if self.update_edge(
            index,
            current_lowest_source_y,
            x,
            z,
            upper_edge_y,
            top_state,
            y,
            middle_state,
            &mut state_at,
        ) {
            return true;
        }

        let Some(bottom_y) = y.checked_sub(1) else {
            return false;
        };
        let bottom_state = state_at(x, bottom_y, z);
        self.update_edge(
            index,
            current_lowest_source_y,
            x,
            z,
            y,
            middle_state,
            bottom_y,
            bottom_state,
            &mut state_at,
        )
    }

    /// Returns the lowest skylight source Y for a local X/Z column.
    #[must_use]
    pub fn get_lowest_source_y(&self, x: usize, z: usize) -> i32 {
        self.extend_sources_below_world(self.get(Self::index(x, z)))
    }

    /// Returns the highest cached lowest-source Y across all columns.
    #[must_use]
    pub fn get_highest_lowest_source_y(&self) -> i32 {
        let mut max_value = NEGATIVE_INFINITY;
        for value in self.heightmap {
            if value > max_value {
                max_value = value;
            }
        }
        self.extend_sources_below_world(max_value)
    }

    fn find_lowest_source_y(
        &self,
        sections: &Sections,
        top_section_index: usize,
        x: usize,
        z: usize,
    ) -> i32 {
        let mut top_y =
            Self::section_to_block_coord(self.section_y_from_index(top_section_index) + 1);
        let mut bottom_y = top_y - 1;
        let mut top_state = Self::air_state();

        for section_index in (0..=top_section_index).rev() {
            let section = sections.sections[section_index].read();
            if section.is_empty() {
                top_state = Self::air_state();
                top_y = Self::section_to_block_coord(self.section_y_from_index(section_index));
                bottom_y = top_y - 1;
                continue;
            }

            for y in (0..CHUNK_EDGE).rev() {
                let bottom_state = section.states.get(x, y, z);
                if Self::is_edge_occluded(top_state, bottom_state) {
                    return top_y;
                }

                top_state = bottom_state;
                top_y = bottom_y;
                bottom_y -= 1;
            }
        }

        self.min_y
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors vanilla's updateEdge inputs without bundling temporary positions"
    )]
    fn update_edge(
        &mut self,
        index: usize,
        old_top_edge_y: i32,
        x: usize,
        z: usize,
        checked_edge_y: i32,
        top_state: BlockStateId,
        bottom_y: i32,
        bottom_state: BlockStateId,
        state_at: &mut impl FnMut(usize, i32, usize) -> BlockStateId,
    ) -> bool {
        if Self::is_edge_occluded(top_state, bottom_state) {
            if checked_edge_y > old_top_edge_y {
                self.set(index, checked_edge_y);
                return true;
            }
        } else if checked_edge_y == old_top_edge_y {
            let new_source_y =
                self.find_lowest_source_below(x, z, bottom_y, bottom_state, state_at);
            self.set(index, new_source_y);
            return true;
        }

        false
    }

    fn find_lowest_source_below(
        &self,
        x: usize,
        z: usize,
        start_y: i32,
        start_state: BlockStateId,
        state_at: &mut impl FnMut(usize, i32, usize) -> BlockStateId,
    ) -> i32 {
        let mut top_y = start_y;
        let mut top_state = start_state;
        let Some(mut bottom_y) = start_y.checked_sub(1) else {
            return self.min_y;
        };

        while bottom_y >= self.min_y {
            let bottom_state = state_at(x, bottom_y, z);
            if Self::is_edge_occluded(top_state, bottom_state) {
                return top_y;
            }

            top_state = bottom_state;
            top_y = bottom_y;
            let Some(next_bottom_y) = bottom_y.checked_sub(1) else {
                break;
            };
            bottom_y = next_bottom_y;
        }

        self.min_y
    }

    fn is_edge_occluded(top_state: BlockStateId, bottom_state: BlockStateId) -> bool {
        if bottom_state.get_light_dampening() != 0 {
            return true;
        }

        light_face_occludes(top_state, bottom_state, Direction::Down)
    }

    fn fill(&mut self, lowest_source_y: i32) {
        self.heightmap.fill(lowest_source_y);
    }

    fn set(&mut self, index: usize, value: i32) {
        self.heightmap[index] = value;
    }

    fn get(&self, index: usize) -> i32 {
        self.heightmap[index]
    }

    fn extend_sources_below_world(&self, value: i32) -> i32 {
        if value == self.min_y {
            NEGATIVE_INFINITY
        } else {
            value
        }
    }

    fn section_y_from_index(&self, section_index: usize) -> i32 {
        SectionPos::block_to_section_coord(self.min_y + 1) + section_index as i32
    }

    const fn section_to_block_coord(section_y: i32) -> i32 {
        section_y << 4
    }

    const fn index(x: usize, z: usize) -> usize {
        x + z * CHUNK_EDGE
    }

    fn air_state() -> BlockStateId {
        vanilla_blocks::AIR.default_state()
    }
}

/// Error returned when packed light data has the wrong length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataLayerLengthError {
    /// Actual number of bytes provided.
    pub actual: usize,
}

/// Packed 4-bit light values for one 16x16x16 light section.
///
/// This mirrors vanilla's `DataLayer`: values are indexed as
/// `y << 8 | z << 4 | x`, with two light nibbles packed into each byte. A
/// homogeneous layer stores only a default value until bytes are needed.
#[derive(Debug, PartialEq, Eq)]
pub struct DataLayer {
    data: Option<Box<[u8; DATA_LAYER_SIZE]>>,
    default_value: u8,
}

impl DataLayer {
    /// Creates an empty all-zero layer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: None,
            default_value: 0,
        }
    }

    /// Creates a homogeneous layer filled with `value`.
    #[must_use]
    pub const fn filled(value: u8) -> Self {
        Self {
            data: None,
            default_value: value & MAX_LIGHT_LEVEL,
        }
    }

    /// Creates a layer from packed bytes.
    ///
    /// The byte order is vanilla's low-nibble-first order.
    pub fn from_bytes(bytes: Box<[u8]>) -> Result<Self, DataLayerLengthError> {
        let actual = bytes.len();
        let Ok(data) = bytes.try_into() else {
            return Err(DataLayerLengthError { actual });
        };

        Ok(Self::from_packed_data(data))
    }

    /// Returns the light value at local section coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        self.get_at_index(Self::index(x, y, z))
    }

    /// Sets the light value at local section coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: u8) {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        self.set_at_index(Self::index(x, y, z), value);
    }

    /// Fills the layer with one homogeneous value.
    pub fn fill(&mut self, value: u8) {
        self.default_value = value & MAX_LIGHT_LEVEL;
        self.data = None;
    }

    /// Returns true when the layer is represented by one homogeneous value.
    #[must_use]
    pub const fn is_homogeneous(&self) -> bool {
        self.data.is_none()
    }

    /// Returns true when the layer is known to be filled with `value`.
    #[must_use]
    pub const fn is_filled_with(&self, value: u8) -> bool {
        self.data.is_none() && self.default_value == (value & MAX_LIGHT_LEVEL)
    }

    /// Returns true when this layer is an all-zero homogeneous layer.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_none() && self.default_value == 0
    }

    /// Returns a deep copy of this layer.
    #[must_use]
    pub fn copy(&self) -> Self {
        Self {
            data: self.data.as_ref().map(|data| Box::new(**data)),
            default_value: self.default_value,
        }
    }

    /// Returns packed bytes without changing the layer representation.
    #[must_use]
    pub fn to_bytes(&self) -> Box<[u8; DATA_LAYER_SIZE]> {
        if let Some(data) = &self.data {
            Box::new(**data)
        } else {
            Box::new([Self::pack_filled(self.default_value); DATA_LAYER_SIZE])
        }
    }

    fn from_packed_data(data: Box<[u8; DATA_LAYER_SIZE]>) -> Self {
        Self {
            data: Some(data),
            default_value: 0,
        }
    }

    fn get_at_index(&self, index: usize) -> u8 {
        if let Some(data) = &self.data {
            let packed = data[Self::byte_index(index)];
            packed >> (4 * Self::nibble_index(index)) & MAX_LIGHT_LEVEL
        } else {
            self.default_value
        }
    }

    fn set_at_index(&mut self, index: usize, value: u8) {
        let data = self.data.get_or_insert_with(|| {
            Box::new([Self::pack_filled(self.default_value); DATA_LAYER_SIZE])
        });
        let byte_index = Self::byte_index(index);
        let shift = 4 * Self::nibble_index(index);
        let mask = !(MAX_LIGHT_LEVEL << shift);
        let value_to_set = (value & MAX_LIGHT_LEVEL) << shift;
        data[byte_index] = data[byte_index] & mask | value_to_set;
    }

    const fn index(x: usize, y: usize, z: usize) -> usize {
        y << 8 | z << 4 | x
    }

    const fn byte_index(index: usize) -> usize {
        index >> 1
    }

    const fn nibble_index(index: usize) -> usize {
        index & 1
    }

    const fn pack_filled(value: u8) -> u8 {
        let value = value & MAX_LIGHT_LEVEL;
        value | value << 4
    }
}

impl Default for DataLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::blocks::block_state_ext::BlockStateExt;
    use steel_registry::blocks::properties::{BlockStateProperties, SlabType};
    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::BlockStateId;
    use steel_utils::Direction::{Down, East, North, South, Up, West};
    use steel_utils::{BlockPos, ChunkPos, PackedSectionPos, SectionPos};

    use crate::{
        behavior::init_behaviors,
        chunk::section::{ChunkSection, Sections},
    };

    use super::{
        ChunkSkyLightSources, DATA_LAYER_SIZE, DataLayer, DataLayerStorageMap,
        LayerLightSectionStorage, LightLayer, LightPropagationQueue, LightPropagationQueues,
        LightQueueEntry, LightSectionRange, LightSectionState, LightSectionStateError,
        LightSectionType, MissingLightDataLayerError, QueuedLightUpdate, build_light_update_packet,
        get_light_block_into, get_light_opacity, has_different_light_properties,
        light_face_occludes,
    };

    fn init_light_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn empty_sections(section_count: usize) -> Sections {
        let sections: Vec<ChunkSection> = (0..section_count)
            .map(|_| ChunkSection::new_empty())
            .collect();
        Sections::from_owned(sections.into_boxed_slice())
    }

    fn single_section_with_block(local_y: usize, state: BlockStateId) -> Sections {
        let mut section = ChunkSection::new_empty();
        section.set_block_state(0, local_y, 0, state);
        Sections::from_owned(vec![section].into_boxed_slice())
    }

    fn new_test_sky_sources() -> ChunkSkyLightSources {
        let Ok(sources) = ChunkSkyLightSources::new(0, 16) else {
            panic!("valid single-section height rejected");
        };
        sources
    }

    fn sky_sources_with_highest_lowest_source_y(source_y: i32) -> ChunkSkyLightSources {
        let mut sources = new_test_sky_sources();
        sources.heightmap.fill(source_y);
        sources
    }

    #[test]
    fn light_opacity_uses_vanilla_minimum_opacity() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let stone = vanilla_blocks::STONE.default_state();

        assert_eq!(get_light_opacity(air), 1);
        assert_eq!(get_light_opacity(stone), 15);
    }

    #[test]
    fn light_block_into_uses_simple_opacity_for_empty_light_shapes() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let stone = vanilla_blocks::STONE.default_state();

        assert_eq!(get_light_block_into(air, stone, Down, 1), 1);
        assert_eq!(get_light_block_into(stone, air, Up, 7), 7);
    }

    #[test]
    fn light_block_into_uses_merged_shape_occlusion() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let bottom_slab = vanilla_blocks::STONE_SLAB
            .default_state()
            .set_value(&BlockStateProperties::SLAB_TYPE, SlabType::Bottom);

        assert_eq!(get_light_block_into(bottom_slab, air, Down, 1), 16);
        assert_eq!(get_light_block_into(bottom_slab, air, Up, 1), 1);
    }

    #[test]
    fn light_face_occludes_uses_face_occlusion_shapes() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let bottom_slab = vanilla_blocks::STONE_SLAB
            .default_state()
            .set_value(&BlockStateProperties::SLAB_TYPE, SlabType::Bottom);
        let top_slab = vanilla_blocks::STONE_SLAB
            .default_state()
            .set_value(&BlockStateProperties::SLAB_TYPE, SlabType::Top);

        assert!(light_face_occludes(bottom_slab, air, Down));
        assert!(!light_face_occludes(bottom_slab, air, Up));
        assert!(light_face_occludes(top_slab, air, Up));
        assert!(!light_face_occludes(top_slab, air, Down));
    }

    #[test]
    fn light_queue_entry_decrease_entries_match_vanilla_bits() {
        let all = LightQueueEntry::decrease_all_directions(7);

        assert_eq!(all.raw(), 1008 | 7);
        assert_eq!(all.from_level(), 7);
        for direction in [Down, Up, North, South, West, East] {
            assert!(all.should_propagate_in_direction(direction));
        }
        assert!(!all.is_from_empty_shape());
        assert!(!all.is_increase_from_emission());

        let skip_north = LightQueueEntry::decrease_skip_one_direction(7, North);
        assert_eq!(skip_north.raw(), 951);
        assert!(!skip_north.should_propagate_in_direction(North));
        assert!(skip_north.should_propagate_in_direction(South));
    }

    #[test]
    fn light_queue_entry_increase_entries_match_vanilla_bits() {
        let emission = LightQueueEntry::increase_light_from_emission(15, true);
        assert_eq!(emission.raw(), 4095);
        assert_eq!(emission.from_level(), 15);
        assert!(emission.is_from_empty_shape());
        assert!(emission.is_increase_from_emission());

        let skip_up = LightQueueEntry::increase_skip_one_direction(10, false, Up);
        assert_eq!(skip_up.raw(), 986);
        assert!(!skip_up.is_from_empty_shape());
        assert!(!skip_up.is_increase_from_emission());
        assert!(!skip_up.should_propagate_in_direction(Up));
        assert!(skip_up.should_propagate_in_direction(Down));

        let east_only = LightQueueEntry::increase_only_one_direction(4, true, East);
        assert_eq!(east_only.raw(), 1540);
        assert!(east_only.is_from_empty_shape());
        assert!(east_only.should_propagate_in_direction(East));
        assert!(!east_only.should_propagate_in_direction(West));
    }

    #[test]
    fn light_queue_entry_sky_source_entry_selects_horizontal_and_down_directions() {
        let entry =
            LightQueueEntry::increase_sky_source_in_directions(true, true, false, true, false);

        assert_eq!(entry.raw(), 351);
        assert_eq!(entry.from_level(), 15);
        assert!(entry.should_propagate_in_direction(Down));
        assert!(!entry.should_propagate_in_direction(Up));
        assert!(entry.should_propagate_in_direction(North));
        assert!(!entry.should_propagate_in_direction(South));
        assert!(entry.should_propagate_in_direction(West));
        assert!(!entry.should_propagate_in_direction(East));
    }

    #[test]
    fn light_queue_entry_masks_levels_like_vanilla() {
        let entry = LightQueueEntry::increase_light_from_emission(31, false);

        assert_eq!(entry.from_level(), 15);
        assert_eq!(entry.raw(), 1008 | 2048 | 15);
    }

    #[test]
    fn light_propagation_queue_preserves_fifo_order() {
        let first_pos = BlockPos::new(1, 2, 3);
        let second_pos = BlockPos::new(4, 5, 6);
        let first_entry = LightQueueEntry::decrease_all_directions(3);
        let second_entry = LightQueueEntry::increase_skip_one_direction(12, false, North);
        let mut queue = LightPropagationQueue::new();

        queue.enqueue(first_pos, first_entry);
        queue.enqueue(second_pos, second_entry);

        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue.dequeue(),
            Some(QueuedLightUpdate {
                block_pos: first_pos,
                entry: first_entry,
            })
        );
        assert_eq!(
            queue.dequeue(),
            Some(QueuedLightUpdate {
                block_pos: second_pos,
                entry: second_entry,
            })
        );
        assert_eq!(queue.dequeue(), None);
        assert!(queue.is_empty());
    }

    #[test]
    fn light_propagation_queue_processes_entries_enqueued_while_draining() {
        let first_pos = BlockPos::new(1, 2, 3);
        let second_pos = BlockPos::new(4, 5, 6);
        let first_entry = LightQueueEntry::decrease_all_directions(3);
        let second_entry = LightQueueEntry::decrease_skip_one_direction(2, South);
        let mut queue = LightPropagationQueue::new();

        queue.enqueue(first_pos, first_entry);

        assert_eq!(
            queue.dequeue(),
            Some(QueuedLightUpdate {
                block_pos: first_pos,
                entry: first_entry,
            })
        );

        queue.enqueue(second_pos, second_entry);

        assert_eq!(
            queue.dequeue(),
            Some(QueuedLightUpdate {
                block_pos: second_pos,
                entry: second_entry,
            })
        );
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn light_propagation_queues_keep_increase_and_decrease_work_separate() {
        let decrease_pos = BlockPos::new(1, 2, 3);
        let increase_pos = BlockPos::new(4, 5, 6);
        let decrease_entry = LightQueueEntry::decrease_all_directions(4);
        let increase_entry = LightQueueEntry::increase_only_one_direction(9, true, East);
        let mut queues = LightPropagationQueues::new();

        assert!(!queues.has_work());

        queues.enqueue_decrease(decrease_pos, decrease_entry);
        queues.enqueue_increase(increase_pos, increase_entry);

        assert!(queues.has_work());
        assert_eq!(
            queues.dequeue_increase(),
            Some(QueuedLightUpdate {
                block_pos: increase_pos,
                entry: increase_entry,
            })
        );
        assert!(queues.has_work());
        assert_eq!(
            queues.dequeue_decrease(),
            Some(QueuedLightUpdate {
                block_pos: decrease_pos,
                entry: decrease_entry,
            })
        );
        assert!(!queues.has_work());
    }

    #[test]
    fn new_layer_is_homogeneous_zero() {
        let layer = DataLayer::new();

        assert!(layer.is_empty());
        assert!(layer.is_homogeneous());
        assert_eq!(layer.get(0, 0, 0), 0);
        assert_eq!(layer.get(15, 15, 15), 0);
    }

    #[test]
    fn filled_layer_reads_same_value_everywhere() {
        let layer = DataLayer::filled(15);

        assert!(layer.is_filled_with(15));
        assert_eq!(layer.get(0, 0, 0), 15);
        assert_eq!(layer.get(3, 12, 7), 15);
        assert_eq!(layer.get(15, 15, 15), 15);
    }

    #[test]
    fn set_uses_vanilla_section_index_order() {
        let mut layer = DataLayer::new();

        layer.set(0, 0, 0, 1);
        layer.set(1, 0, 0, 2);
        layer.set(0, 0, 1, 3);
        layer.set(0, 1, 0, 4);

        assert_eq!(layer.to_bytes()[0], 0x21);
        assert_eq!(layer.to_bytes()[8], 0x03);
        assert_eq!(layer.to_bytes()[128], 0x04);
    }

    #[test]
    fn set_masks_to_nibble() {
        let mut layer = DataLayer::new();

        layer.set(0, 0, 0, 0x2f);

        assert_eq!(layer.get(0, 0, 0), 15);
        assert_eq!(layer.to_bytes()[0], 0x0f);
    }

    #[test]
    fn fill_returns_to_homogeneous_storage() {
        let mut layer = DataLayer::new();
        layer.set(4, 5, 6, 9);

        assert!(!layer.is_homogeneous());

        layer.fill(7);

        assert!(layer.is_homogeneous());
        assert!(layer.is_filled_with(7));
        assert_eq!(layer.to_bytes()[0], 0x77);
        assert_eq!(layer.to_bytes()[DATA_LAYER_SIZE - 1], 0x77);
    }

    #[test]
    fn copy_is_independent() {
        let mut original = DataLayer::new();
        original.set(2, 3, 4, 8);
        let mut copied = original.copy();

        copied.set(2, 3, 4, 1);

        assert_eq!(original.get(2, 3, 4), 8);
        assert_eq!(copied.get(2, 3, 4), 1);
    }

    #[test]
    fn from_bytes_rejects_wrong_length() {
        let err = DataLayer::from_bytes(vec![0; DATA_LAYER_SIZE - 1].into_boxed_slice());

        assert_eq!(
            err,
            Err(super::DataLayerLengthError {
                actual: DATA_LAYER_SIZE - 1,
            }),
        );
    }

    #[test]
    fn from_bytes_uses_existing_packed_data() {
        let mut bytes = vec![0; DATA_LAYER_SIZE];
        bytes[0] = 0xba;
        bytes[DATA_LAYER_SIZE - 1] = 0x65;

        let result = DataLayer::from_bytes(bytes.into_boxed_slice());
        let Ok(layer) = result else {
            panic!("valid data layer length was rejected");
        };

        assert_eq!(layer.get(0, 0, 0), 10);
        assert_eq!(layer.get(1, 0, 0), 11);
        assert_eq!(layer.get(14, 15, 15), 5);
        assert_eq!(layer.get(15, 15, 15), 6);
    }

    #[test]
    fn light_section_range_matches_vanilla_padding() {
        let Ok(range) = LightSectionRange::from_world_height(-64, 384) else {
            panic!("valid overworld height rejected");
        };

        assert_eq!(range.min_section_y(), -5);
        assert_eq!(range.max_section_y_exclusive(), 21);
        assert_eq!(range.section_count(), 26);
        assert_eq!(range.section_y(0), Some(-5));
        assert_eq!(range.section_y(25), Some(20));
        assert_eq!(range.section_y(26), None);
        assert_eq!(range.section_index(-5), Some(0));
        assert_eq!(range.section_index(20), Some(25));
        assert_eq!(range.section_index(-6), None);
        assert_eq!(range.section_index(21), None);
    }

    #[test]
    fn data_layer_storage_map_copies_layers_independently() {
        let section = SectionPos::new(4, -1, 7);
        let mut storage = DataLayerStorageMap::new();
        let mut layer = DataLayer::new();
        layer.set(2, 3, 4, 6);
        storage.set_layer(section, layer);

        let copied = storage.copy_map();
        let Some(original_layer) = storage.get_layer_mut(section) else {
            panic!("stored layer missing");
        };
        original_layer.set(2, 3, 4, 1);

        let Some(copied_layer) = copied.get_layer(section) else {
            panic!("copied layer missing");
        };
        assert_eq!(original_layer.get(2, 3, 4), 1);
        assert_eq!(copied_layer.get(2, 3, 4), 6);
    }

    #[test]
    fn light_section_state_matches_vanilla_bit_layout() {
        let data = LightSectionState::EMPTY.with_has_data(true);
        assert_eq!(data.raw(), 32);
        assert!(data.has_data());
        assert_eq!(data.neighbor_count(), 0);
        assert_eq!(data.section_type(), LightSectionType::LightAndData);

        let result = LightSectionState::EMPTY.with_neighbor_count(26);
        let Ok(light_only) = result else {
            panic!("valid neighbor count rejected");
        };
        assert_eq!(light_only.raw(), 26);
        assert!(!light_only.has_data());
        assert_eq!(light_only.neighbor_count(), 26);
        assert_eq!(light_only.section_type(), LightSectionType::LightOnly);
    }

    #[test]
    fn light_section_state_rejects_invalid_neighbor_count() {
        assert_eq!(
            LightSectionState::EMPTY.with_neighbor_count(27),
            Err(LightSectionStateError { neighbor_count: 27 })
        );
        assert_eq!(
            LightSectionState::EMPTY.with_neighbor_count(-1),
            Err(LightSectionStateError { neighbor_count: -1 })
        );
    }

    #[test]
    fn layer_storage_creates_data_and_light_only_neighbors() {
        let center = SectionPos::new(4, 5, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);

        assert_eq!(storage.update_section_status(center, false), Ok(()));

        assert_eq!(
            storage.get_debug_section_type(center),
            LightSectionType::LightAndData
        );
        assert!(storage.storing_light_for_section(center));
        assert_eq!(storage.updating_section_data.len(), 27);

        let neighbor = SectionPos::new(5, 5, 6);
        assert_eq!(
            storage.get_debug_section_type(neighbor),
            LightSectionType::LightOnly
        );
        assert!(storage.storing_light_for_section(neighbor));
    }

    #[test]
    fn layer_storage_removes_data_after_section_becomes_empty() {
        let center = SectionPos::new(4, 5, 6);
        let neighbor = SectionPos::new(5, 5, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        assert_eq!(storage.update_section_status(center, false), Ok(()));

        assert_eq!(storage.update_section_status(center, true), Ok(()));

        assert_eq!(
            storage.get_debug_section_type(center),
            LightSectionType::Empty
        );
        assert_eq!(
            storage.get_debug_section_type(neighbor),
            LightSectionType::Empty
        );
        assert!(storage.storing_light_for_section(center));
        assert!(storage.has_inconsistencies());

        storage.mark_new_inconsistencies();

        assert!(!storage.storing_light_for_section(center));
        assert!(!storage.storing_light_for_section(neighbor));
        assert_eq!(storage.updating_section_data.len(), 0);
    }

    #[test]
    fn layer_storage_retains_removed_column_data_when_requested() {
        let center = SectionPos::new(4, 5, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        assert_eq!(storage.update_section_status(center, false), Ok(()));

        let Some(layer) = storage.get_data_layer_to_write(center) else {
            panic!("center layer missing");
        };
        layer.set(1, 2, 3, 9);
        storage.retain_data(SectionPos::new(4, 0, 6), true);
        assert_eq!(storage.update_section_status(center, true), Ok(()));

        storage.mark_new_inconsistencies();

        let Some(retained) = storage.queued_sections.get(&PackedSectionPos::from(center)) else {
            panic!("removed section data was not retained");
        };
        assert_eq!(retained.get(1, 2, 3), 9);
    }

    #[test]
    fn layer_storage_swap_updates_visible_map_and_returns_affected_sections() {
        let center = SectionPos::new(4, 5, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        assert_eq!(storage.update_section_status(center, false), Ok(()));
        assert!(storage.get_data_layer_data(center).is_none());

        let affected = storage.swap_section_map();

        assert!(affected.contains(&center));
        assert!(storage.get_data_layer_data(center).is_some());
        assert!(storage.changed_sections.is_empty());
        assert!(storage.sections_affected_by_light_updates.is_empty());
    }

    #[test]
    fn layer_storage_reads_and_writes_stored_levels() {
        let center = SectionPos::new(4, 5, 6);
        let block = BlockPos::new((4 << 4) + 2, (5 << 4) + 3, (6 << 4) + 4);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        assert_eq!(storage.update_section_status(center, false), Ok(()));
        drop(storage.swap_section_map());

        assert_eq!(storage.get_stored_level(block), Some(0));
        assert_eq!(storage.set_stored_level(block, 12), Ok(()));
        assert_eq!(storage.get_stored_level(block), Some(12));

        let Some(visible_before_swap) = storage.get_data_layer_data(center) else {
            panic!("visible layer missing before stored level swap");
        };
        assert_eq!(visible_before_swap.get(2, 3, 4), 0);

        let affected = storage.swap_section_map();
        assert_eq!(affected.len(), 1);
        assert!(affected.contains(&center));

        let Some(visible_after_swap) = storage.get_data_layer_data(center) else {
            panic!("visible layer missing after stored level swap");
        };
        assert_eq!(visible_after_swap.get(2, 3, 4), 12);
    }

    #[test]
    fn layer_storage_rejects_stored_level_write_without_section_data() {
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        let block = BlockPos::new(1, 2, 3);

        assert_eq!(storage.get_stored_level(block), None);
        assert_eq!(
            storage.set_stored_level(block, 4),
            Err(MissingLightDataLayerError {
                section_pos: SectionPos::new(0, 0, 0)
            })
        );
    }

    #[test]
    fn layer_storage_stored_level_write_marks_vanilla_adjacent_block_sections() {
        let center = SectionPos::new(1, 2, -1);
        let block = BlockPos::new(16, 32, -1);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        assert_eq!(storage.update_section_status(center, false), Ok(()));
        drop(storage.swap_section_map());

        assert_eq!(storage.set_stored_level(block, 6), Ok(()));

        let affected = storage.swap_section_map();
        assert_eq!(affected.len(), 8);
        assert!(affected.contains(&SectionPos::new(0, 1, -1)));
        assert!(affected.contains(&SectionPos::new(0, 1, 0)));
        assert!(affected.contains(&SectionPos::new(0, 2, -1)));
        assert!(affected.contains(&SectionPos::new(0, 2, 0)));
        assert!(affected.contains(&SectionPos::new(1, 1, -1)));
        assert!(affected.contains(&SectionPos::new(1, 1, 0)));
        assert!(affected.contains(&SectionPos::new(1, 2, -1)));
        assert!(affected.contains(&SectionPos::new(1, 2, 0)));
    }

    #[test]
    fn block_storage_light_value_reads_visible_data() {
        let section = SectionPos::new(1, 2, 3);
        let block = BlockPos::new((1 << 4) + 2, (2 << 4) + 3, (3 << 4) + 4);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);

        assert_eq!(storage.get_light_value(block), 0);
        assert_eq!(storage.update_section_status(section, false), Ok(()));
        assert_eq!(storage.set_stored_level(block, 11), Ok(()));
        assert_eq!(storage.get_light_value(block), 0);

        drop(storage.swap_section_map());

        assert_eq!(storage.get_light_value(block), 11);
    }

    #[test]
    fn sky_storage_light_value_defaults_to_full_bright_without_data() {
        let storage = LayerLightSectionStorage::new(LightLayer::Sky);

        assert_eq!(storage.get_light_value(BlockPos::new(1, -64, 1)), 15);
    }

    #[test]
    fn sky_storage_light_value_reads_visible_data() {
        let section = SectionPos::new(0, 4, 0);
        let block = BlockPos::new(2, (4 << 4) + 3, 5);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);

        assert_eq!(storage.update_section_status(section, false), Ok(()));
        assert_eq!(storage.set_stored_level(block, 8), Ok(()));
        assert_eq!(storage.get_light_value(block), 15);

        drop(storage.swap_section_map());

        assert_eq!(storage.get_light_value(block), 8);
    }

    #[test]
    fn sky_storage_light_value_scans_to_first_visible_layer_above() {
        let source_section = SectionPos::new(0, 5, 0);
        let first_above_bottom_block = BlockPos::new(2, 4 << 4, 5);
        let first_above_same_column_block = BlockPos::new(2, (4 << 4) + 11, 5);
        let missing_section_block = BlockPos::new(2, (3 << 4) + 11, 5);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);

        assert_eq!(storage.update_section_status(source_section, false), Ok(()));
        assert_eq!(
            storage.set_stored_level(first_above_bottom_block, 9),
            Ok(())
        );
        assert_eq!(
            storage.set_stored_level(first_above_same_column_block, 3),
            Ok(())
        );

        drop(storage.swap_section_map());

        assert_eq!(storage.get_light_value(missing_section_block), 9);
    }

    #[test]
    fn sky_storage_tracks_top_and_bottom_sections() {
        let center = SectionPos::new(4, 5, 6);
        let zero = SectionPos::new(4, 0, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);

        assert_eq!(storage.update_section_status(center, false), Ok(()));

        assert_eq!(storage.get_bottom_section_y(), Some(4));
        assert_eq!(storage.get_top_section_y(zero), Some(7));
        assert_eq!(storage.has_light_data_at_or_below(3), Some(false));
        assert_eq!(storage.has_light_data_at_or_below(4), Some(true));
        assert_eq!(storage.is_above_data(SectionPos::new(4, 6, 6)), Some(false));
        assert_eq!(storage.is_above_data(SectionPos::new(4, 7, 6)), Some(true));
    }

    #[test]
    fn sky_storage_creates_full_bright_layers_when_sources_enabled() {
        let section = SectionPos::new(0, 0, 0);
        let below = SectionPos::new(0, -1, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        storage.set_light_enabled(SectionPos::new(0, 0, 0), true);

        assert_eq!(storage.update_section_status(section, false), Ok(()));

        let Some(section_layer) = storage.get_updating_data_layer(section) else {
            panic!("section layer missing");
        };
        assert!(section_layer.is_filled_with(super::MAX_LIGHT_LEVEL));

        let Some(below_layer) = storage.get_updating_data_layer(below) else {
            panic!("below layer missing");
        };
        assert!(below_layer.is_filled_with(super::MAX_LIGHT_LEVEL));
    }

    #[test]
    fn sky_storage_enable_sky_sources_fills_existing_fully_sourced_layers() {
        let zero = SectionPos::new(0, 0, 0);
        let lower = SectionPos::new(0, 3, 0);
        let upper = SectionPos::new(0, 6, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        let sources = sky_sources_with_highest_lowest_source_y(72);

        assert_eq!(storage.update_section_status(lower, false), Ok(()));
        assert_eq!(storage.update_section_status(upper, false), Ok(()));

        assert_eq!(storage.enable_sky_light_sources(zero, &sources), Some(()));
        assert!(storage.light_on_in_column(zero));

        for section_y in 5..=7 {
            let section = SectionPos::new(0, section_y, 0);
            let Some(layer) = storage.get_updating_data_layer(section) else {
                panic!("fully sourced section layer missing");
            };
            assert!(layer.is_filled_with(super::MAX_LIGHT_LEVEL));
        }

        let Some(lower_layer) = storage.get_updating_data_layer(SectionPos::new(0, 4, 0)) else {
            panic!("lower section layer missing");
        };
        assert!(lower_layer.is_empty());
    }

    #[test]
    fn block_storage_rejects_sky_source_enable() {
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        let sources = sky_sources_with_highest_lowest_source_y(72);

        assert_eq!(
            storage.enable_sky_light_sources(SectionPos::new(0, 0, 0), &sources),
            None
        );
        assert!(!storage.light_on_in_column(SectionPos::new(0, 0, 0)));
    }

    #[test]
    fn sky_storage_repeats_first_layer_below_top_data() {
        let upper = SectionPos::new(0, 5, 0);
        let copied = SectionPos::new(0, 3, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        storage.set_light_enabled(SectionPos::new(0, 0, 0), true);
        assert_eq!(storage.update_section_status(upper, false), Ok(()));

        let Some(source_layer) = storage.get_data_layer_to_write(SectionPos::new(0, 4, 0)) else {
            panic!("source layer missing");
        };
        source_layer.set(0, 0, 0, 3);
        source_layer.set(0, 1, 0, 12);

        assert_eq!(
            storage.update_section_status(SectionPos::new(0, 2, 0), false),
            Ok(())
        );

        let Some(copied_layer) = storage.get_updating_data_layer(copied) else {
            panic!("copied layer missing");
        };
        assert_eq!(copied_layer.get(0, 0, 0), 3);
        assert_eq!(copied_layer.get(0, 1, 0), 3);
    }

    #[test]
    fn sky_storage_moves_top_down_after_highest_section_removal() {
        let upper = SectionPos::new(0, 5, 0);
        let lower = SectionPos::new(0, 2, 0);
        let zero = SectionPos::new(0, 0, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        assert_eq!(storage.update_section_status(upper, false), Ok(()));
        assert_eq!(storage.update_section_status(lower, false), Ok(()));
        assert_eq!(storage.get_top_section_y(zero), Some(7));

        assert_eq!(storage.update_section_status(upper, true), Ok(()));
        storage.mark_new_inconsistencies();

        assert_eq!(storage.get_top_section_y(zero), Some(4));
        assert!(storage.storing_light_for_section(SectionPos::new(0, 3, 0)));
        assert!(!storage.storing_light_for_section(SectionPos::new(0, 6, 0)));
    }

    #[test]
    fn sky_storage_uses_queued_data_when_section_is_created() {
        let section = SectionPos::new(0, 0, 0);
        let mut queued = DataLayer::new();
        queued.set(1, 2, 3, 6);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        storage.queue_section_data(section, Some(queued));

        assert_eq!(storage.update_section_status(section, false), Ok(()));

        let Some(layer) = storage.get_updating_data_layer(section) else {
            panic!("queued layer missing");
        };
        assert_eq!(layer.get(1, 2, 3), 6);
    }

    #[test]
    fn light_update_packet_masks_match_vanilla_section_preparation() {
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("valid single-section height rejected");
        };
        let chunk_pos = ChunkPos::new(2, -3);

        let mut sky_layers = DataLayerStorageMap::new();
        sky_layers.set_layer(
            range.section_pos(chunk_pos, -1),
            DataLayer::filled(super::MAX_LIGHT_LEVEL),
        );
        sky_layers.set_layer(range.section_pos(chunk_pos, 0), DataLayer::new());

        let mut block_layers = DataLayerStorageMap::new();
        let mut block_layer = DataLayer::new();
        block_layer.set(0, 0, 0, 7);
        block_layers.set_layer(range.section_pos(chunk_pos, 1), block_layer);

        let packet =
            build_light_update_packet(chunk_pos, range, Some(&sky_layers), Some(&block_layers));

        assert_eq!(packet.sky_y_mask.0[0], 0b001);
        assert_eq!(packet.empty_sky_y_mask.0[0], 0b010);
        assert_eq!(packet.block_y_mask.0[0], 0b100);
        assert_eq!(packet.empty_block_y_mask.0[0], 0);
        assert_eq!(packet.sky_updates.len(), 1);
        assert!(packet.sky_updates[0].iter().all(|byte| *byte == 0xff));
        assert_eq!(
            packet.block_updates,
            vec![{
                let mut bytes = vec![0; DATA_LAYER_SIZE];
                bytes[0] = 0x07;
                bytes
            }]
        );
    }

    #[test]
    fn light_update_packet_omits_disabled_layers() {
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("valid single-section height rejected");
        };
        let packet = build_light_update_packet(ChunkPos::new(0, 0), range, None, None);

        assert_eq!(packet.sky_y_mask.0[0], 0);
        assert_eq!(packet.block_y_mask.0[0], 0);
        assert_eq!(packet.empty_sky_y_mask.0[0], 0);
        assert_eq!(packet.empty_block_y_mask.0[0], 0);
        assert!(packet.sky_updates.is_empty());
        assert!(packet.block_updates.is_empty());
    }

    #[test]
    fn different_light_properties_match_vanilla_conditions() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let stone = vanilla_blocks::STONE.default_state();

        assert!(!has_different_light_properties(air, air));
        assert!(has_different_light_properties(air, stone));

        let light = vanilla_blocks::LIGHT.default_state();
        let dim_light = light.set_value(
            &steel_registry::blocks::properties::BlockStateProperties::LEVEL,
            7,
        );
        assert!(has_different_light_properties(light, dim_light));
    }

    #[test]
    fn sky_light_sources_empty_chunk_extends_below_world() {
        init_light_tests();
        let sections = empty_sections(1);
        let mut sources = new_test_sky_sources();

        sources.fill_from_sections(&sections);

        assert_eq!(sources.get_lowest_source_y(0, 0), i32::MIN);
        assert_eq!(sources.get_lowest_source_y(15, 15), i32::MIN);
        assert_eq!(sources.get_highest_lowest_source_y(), i32::MIN);
    }

    #[test]
    fn sky_light_sources_find_lowest_occluding_edge() {
        init_light_tests();
        let stone = vanilla_blocks::STONE.default_state();
        let sections = single_section_with_block(4, stone);
        let mut sources = new_test_sky_sources();

        sources.fill_from_sections(&sections);

        assert_eq!(sources.get_lowest_source_y(0, 0), 5);
        assert_eq!(sources.get_lowest_source_y(1, 0), i32::MIN);
        assert_eq!(sources.get_highest_lowest_source_y(), 5);
    }

    #[test]
    fn sky_light_sources_update_adds_and_removes_occluding_edge() {
        init_light_tests();
        let air = vanilla_blocks::AIR.default_state();
        let stone = vanilla_blocks::STONE.default_state();
        let sections = empty_sections(1);
        let mut sources = new_test_sky_sources();
        sources.fill_from_sections(&sections);

        let added = sources.update(0, 4, 0, |_x, y, _z| if y == 4 { stone } else { air });

        assert!(added);
        assert_eq!(sources.get_lowest_source_y(0, 0), 5);

        let removed = sources.update(0, 4, 0, |_x, _y, _z| air);

        assert!(removed);
        assert_eq!(sources.get_lowest_source_y(0, 0), i32::MIN);
    }

    #[test]
    fn sky_light_sources_update_ignores_changes_below_current_source_edge() {
        init_light_tests();
        let stone = vanilla_blocks::STONE.default_state();
        let sections = single_section_with_block(10, stone);
        let mut sources = new_test_sky_sources();
        sources.fill_from_sections(&sections);

        let changed = sources.update(0, 4, 0, |_x, _y, _z| stone);

        assert!(!changed);
        assert_eq!(sources.get_lowest_source_y(0, 0), 11);
    }
}

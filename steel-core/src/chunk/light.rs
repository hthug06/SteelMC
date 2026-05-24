//! Light storage primitives used by chunk and world lighting.

use rustc_hash::FxHashMap;
use steel_protocol::packets::game::LightUpdatePacketData;
use steel_registry::{blocks::block_state_ext::BlockStateExt, vanilla_blocks};
use steel_utils::{BlockStateId, ChunkPos, Direction, PackedSectionPos, SectionPos, codec::BitSet};

use crate::{chunk::section::Sections, physics::shapes::merged_face_occludes};

/// Maximum light value stored by vanilla lighting.
pub const MAX_LIGHT_LEVEL: u8 = 15;
/// Vanilla stores one extra light section below and above the build height.
pub const LIGHT_SECTION_PADDING: i32 = 1;

/// Number of blocks along one edge of a light section.
pub const DATA_LAYER_EDGE: usize = 16;
/// Number of blocks in a light section.
pub const DATA_LAYER_BLOCK_COUNT: usize = DATA_LAYER_EDGE * DATA_LAYER_EDGE * DATA_LAYER_EDGE;
/// Number of packed bytes in a light section.
pub const DATA_LAYER_SIZE: usize = DATA_LAYER_BLOCK_COUNT / 2;
const CHUNK_EDGE: usize = 16;
const CHUNK_COLUMN_COUNT: usize = CHUNK_EDGE * CHUNK_EDGE;
const NEGATIVE_INFINITY: i32 = i32::MIN;

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

        let top_shape = Self::light_occlusion_shape(top_state);
        let bottom_shape = Self::light_occlusion_shape(bottom_state);
        merged_face_occludes(top_shape, bottom_shape, Direction::Down)
    }

    fn light_occlusion_shape(
        state: BlockStateId,
    ) -> &'static [steel_registry::blocks::shapes::AABB] {
        if !state.get_block().config.can_occlude || !state.use_shape_for_light_occlusion() {
            return &[];
        }

        state.get_occlusion_shape()
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

        Ok(Self {
            data: Some(data),
            default_value: 0,
        })
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
    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::BlockStateId;
    use steel_utils::{ChunkPos, SectionPos};

    use crate::{
        behavior::init_behaviors,
        chunk::section::{ChunkSection, Sections},
    };

    use super::{
        ChunkSkyLightSources, DATA_LAYER_SIZE, DataLayer, DataLayerStorageMap, LightSectionRange,
        build_light_update_packet, has_different_light_properties,
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

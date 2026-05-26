use std::sync::Arc;

use steel_utils::{BlockPos, SectionPos};

use crate::chunk::section::Sections;

use super::{
    DATA_LAYER_BLOCK_COUNT, DATA_LAYER_EDGE, DATA_LAYER_SIZE, DATA_LAYER_Y_STRIDE, DataLayer,
    DataLayerLengthError, LightLayer, LightSectionRange, LightSectionRangeError, MAX_LIGHT_LEVEL,
};

/// ScalableLux-style light nibble state for one 16x16x16 light section.
///
/// Vanilla models missing light sections by omitting `DataLayer`s from a
/// section map. ScalableLux keeps section-indexed chunk arrays and marks each
/// nibble as null, uninitialized, initialized, or hidden. Steel follows that
/// ownership model for the light engine, while preserving vanilla packet/save
/// conversion through `DataLayer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightNibbleState {
    /// No light data exists for this section; reads behave as zero.
    Null,
    /// Section exists but stores only zeroes without backing bytes.
    Uninitialized,
    /// Section stores concrete nibble bytes.
    Initialized,
    /// Section stores concrete bytes but should be treated as missing externally.
    Hidden,
}

/// Single-writer, multi-reader light nibbles with visible/updating state.
///
/// This mirrors ScalableLux `SWMRNibbleArray`: writes mutate the updating view,
/// `update_visible` publishes it, and reads choose either visible or updating
/// values explicitly. Steel uses `Arc` copy-on-write for the shared byte array
/// instead of Java reference identity.
#[derive(Debug)]
pub struct LightNibbleArray {
    updating_state: LightNibbleState,
    visible_state: LightNibbleState,
    updating_data: Option<Arc<[u8; DATA_LAYER_SIZE]>>,
    visible_data: Option<Arc<[u8; DATA_LAYER_SIZE]>>,
    updating_dirty: bool,
}

/// Vanilla-save representation of one visible light nibble section.
#[derive(Debug, PartialEq, Eq)]
pub struct LightNibbleSaveState {
    /// State to persist for this light section.
    pub state: LightNibbleState,
    /// Packed nibble bytes when the section has non-zero initialized data.
    pub data: Option<Box<[u8; DATA_LAYER_SIZE]>>,
}

impl LightNibbleArray {
    /// Creates a null nibble section.
    #[must_use]
    pub const fn null() -> Self {
        Self {
            updating_state: LightNibbleState::Null,
            visible_state: LightNibbleState::Null,
            updating_data: None,
            visible_data: None,
            updating_dirty: false,
        }
    }

    /// Creates an uninitialized, all-zero nibble section.
    #[must_use]
    pub const fn uninitialized() -> Self {
        Self {
            updating_state: LightNibbleState::Uninitialized,
            visible_state: LightNibbleState::Uninitialized,
            updating_data: None,
            visible_data: None,
            updating_dirty: false,
        }
    }

    /// Creates an initialized nibble section from packed vanilla bytes.
    pub fn initialized_from_bytes(bytes: Box<[u8]>) -> Result<Self, DataLayerLengthError> {
        let actual = bytes.len();
        let Ok(data) = bytes.try_into() else {
            return Err(DataLayerLengthError { actual });
        };

        Ok(Self::from_packed_data(data))
    }

    /// Creates a hidden initialized nibble section from packed vanilla bytes.
    pub fn hidden_from_bytes(bytes: Box<[u8]>) -> Result<Self, DataLayerLengthError> {
        let mut nibble = Self::initialized_from_bytes(bytes)?;
        nibble.updating_state = LightNibbleState::Hidden;
        nibble.visible_state = LightNibbleState::Hidden;
        Ok(nibble)
    }

    /// Creates a nibble section from optional vanilla data.
    #[must_use]
    pub fn from_data_layer(layer: Option<&DataLayer>) -> Self {
        let Some(layer) = layer else {
            return Self::null();
        };

        if layer.is_empty() {
            Self::uninitialized()
        } else {
            Self::from_packed_data(layer.to_bytes())
        }
    }

    /// Creates an initialized homogeneous nibble section.
    #[must_use]
    pub fn filled(value: u8) -> Self {
        let packed = Self::pack_filled(value);
        Self::from_packed_data(Box::new([packed; DATA_LAYER_SIZE]))
    }

    /// Returns the updating state.
    #[must_use]
    pub const fn updating_state(&self) -> LightNibbleState {
        self.updating_state
    }

    /// Returns the visible state.
    #[must_use]
    pub const fn visible_state(&self) -> LightNibbleState {
        self.visible_state
    }

    /// Returns whether updating and visible views differ.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.updating_dirty || self.updating_state != self.visible_state
    }

    /// Returns true when the updating view is null.
    #[must_use]
    pub const fn is_null_updating(&self) -> bool {
        matches!(self.updating_state, LightNibbleState::Null)
    }

    /// Returns true when the visible view is null.
    #[must_use]
    pub const fn is_null_visible(&self) -> bool {
        matches!(self.visible_state, LightNibbleState::Null)
    }

    /// Returns true when the updating view is initialized.
    #[must_use]
    pub const fn is_initialized_updating(&self) -> bool {
        matches!(self.updating_state, LightNibbleState::Initialized)
    }

    /// Returns true when the visible view is initialized.
    #[must_use]
    pub const fn is_initialized_visible(&self) -> bool {
        matches!(self.visible_state, LightNibbleState::Initialized)
    }

    /// Marks the updating view as non-null without allocating storage.
    pub fn set_non_null(&mut self) {
        match self.updating_state {
            LightNibbleState::Hidden => self.updating_state = LightNibbleState::Initialized,
            LightNibbleState::Null => self.updating_state = LightNibbleState::Uninitialized,
            LightNibbleState::Uninitialized | LightNibbleState::Initialized => {}
        }
    }

    /// Marks the updating view as null and drops updating bytes.
    pub fn set_null(&mut self) {
        self.updating_state = LightNibbleState::Null;
        self.updating_data = None;
        self.updating_dirty = false;
    }

    /// Marks the updating view as uninitialized and drops updating bytes.
    pub fn set_uninitialized(&mut self) {
        self.updating_state = LightNibbleState::Uninitialized;
        self.updating_data = None;
        self.updating_dirty = false;
    }

    /// Hides initialized updating bytes from external conversion.
    pub fn set_hidden(&mut self) {
        match self.updating_state {
            LightNibbleState::Hidden => {}
            LightNibbleState::Initialized => self.updating_state = LightNibbleState::Hidden,
            LightNibbleState::Null | LightNibbleState::Uninitialized => self.set_null(),
        }
    }

    /// Fills the updating view with one light value.
    pub fn fill(&mut self, value: u8) {
        let packed = Self::pack_filled(value);
        let data = self.ensure_updating_data();
        data.fill(packed);
    }

    /// Extrudes the first updating row of `above` into every row of this updating view.
    pub fn extrude_lower(&mut self, above: &Self) -> Result<(), LightNibbleExtrudeNullSourceError> {
        let row = above.lower_row_for_extrusion()?;
        self.extrude_lower_row(row);
        Ok(())
    }

    /// Returns the bottom row used when another nibble extrudes from this one.
    pub fn lower_row_for_extrusion(
        &self,
    ) -> Result<Option<[u8; DATA_LAYER_Y_STRIDE]>, LightNibbleExtrudeNullSourceError> {
        if self.updating_state == LightNibbleState::Null {
            return Err(LightNibbleExtrudeNullSourceError);
        }

        let Some(source) = self.updating_data.as_ref() else {
            return Ok(None);
        };

        let mut row = [0; DATA_LAYER_Y_STRIDE];
        row.copy_from_slice(&source[..DATA_LAYER_Y_STRIDE]);
        Ok(Some(row))
    }

    /// Extrudes one bottom row into every row of this updating view.
    pub fn extrude_lower_row(&mut self, row: Option<[u8; DATA_LAYER_Y_STRIDE]>) {
        let Some(row) = row else {
            self.set_uninitialized();
            return;
        };

        let data = self.ensure_updating_data();
        for y in 0..DATA_LAYER_EDGE {
            let start = y * DATA_LAYER_Y_STRIDE;
            data[start..start + DATA_LAYER_Y_STRIDE].copy_from_slice(&row);
        }
    }

    /// Returns an updating light value at local section coordinates.
    #[must_use]
    pub fn get_updating(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        Self::get_from_data(&self.updating_data, Self::index(x, y, z))
    }

    /// Returns an updating light value at a ScalableLux local section index.
    #[must_use]
    pub fn get_updating_at_index(&self, index: usize) -> u8 {
        debug_assert!(index < DATA_LAYER_BLOCK_COUNT);

        Self::get_from_data(&self.updating_data, index)
    }

    /// Returns a visible light value at local section coordinates.
    #[must_use]
    pub fn get_visible(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        Self::get_from_data(&self.visible_data, Self::index(x, y, z))
    }

    /// Returns a visible light value at a ScalableLux local section index.
    #[must_use]
    pub fn get_visible_at_index(&self, index: usize) -> u8 {
        debug_assert!(index < DATA_LAYER_BLOCK_COUNT);

        Self::get_from_data(&self.visible_data, index)
    }

    /// Sets an updating light value at local section coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: u8) {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        let index = Self::index(x, y, z);
        let data = self.ensure_updating_data();
        Self::set_in_data(data, index, value);
    }

    /// Sets an updating light value at a ScalableLux local section index.
    pub fn set_at_index(&mut self, index: usize, value: u8) {
        debug_assert!(index < DATA_LAYER_BLOCK_COUNT);

        let data = self.ensure_updating_data();
        Self::set_in_data(data, index, value);
    }

    /// Publishes the updating view and returns whether anything changed.
    pub fn update_visible(&mut self) -> bool {
        if !self.is_dirty() {
            return false;
        }

        self.visible_state = self.updating_state;
        self.visible_data = match self.updating_state {
            LightNibbleState::Null | LightNibbleState::Uninitialized => None,
            LightNibbleState::Initialized | LightNibbleState::Hidden => {
                self.updating_data.as_ref().map(Arc::clone)
            }
        };
        self.updating_dirty = false;
        true
    }

    /// Converts the visible view into vanilla `DataLayer` packet/save data.
    #[must_use]
    pub fn to_data_layer(&self) -> Option<DataLayer> {
        match self.visible_state {
            LightNibbleState::Null | LightNibbleState::Hidden => None,
            LightNibbleState::Uninitialized => Some(DataLayer::new()),
            LightNibbleState::Initialized => {
                let Some(data) = self.visible_data.as_ref() else {
                    panic!("initialized visible light nibble is missing data");
                };
                Some(DataLayer::from_packed_data(Box::new(**data)))
            }
        }
    }

    /// Converts the visible view into ScalableLux-style save state.
    ///
    /// All-zero initialized sections are persisted as uninitialized, and all-zero
    /// hidden sections are omitted, matching ScalableLux's canonical save state.
    #[must_use]
    pub fn to_save_state(&self) -> Option<LightNibbleSaveState> {
        match self.visible_state {
            LightNibbleState::Null => None,
            LightNibbleState::Uninitialized => Some(LightNibbleSaveState {
                state: LightNibbleState::Uninitialized,
                data: None,
            }),
            LightNibbleState::Initialized | LightNibbleState::Hidden => {
                let Some(data) = self.visible_data.as_ref() else {
                    panic!("initialized visible light nibble is missing data");
                };

                if Self::is_all_zero(data) {
                    if self.visible_state == LightNibbleState::Hidden {
                        return None;
                    }

                    return Some(LightNibbleSaveState {
                        state: LightNibbleState::Uninitialized,
                        data: None,
                    });
                }

                Some(LightNibbleSaveState {
                    state: self.visible_state,
                    data: Some(Box::new(**data)),
                })
            }
        }
    }

    fn from_packed_data(data: Box<[u8; DATA_LAYER_SIZE]>) -> Self {
        let data = Arc::new(*data);
        Self {
            updating_state: LightNibbleState::Initialized,
            visible_state: LightNibbleState::Initialized,
            updating_data: Some(Arc::clone(&data)),
            visible_data: Some(data),
            updating_dirty: false,
        }
    }

    fn ensure_updating_data(&mut self) -> &mut [u8; DATA_LAYER_SIZE] {
        if self.updating_state != LightNibbleState::Hidden {
            self.updating_state = LightNibbleState::Initialized;
        }

        let data = self
            .updating_data
            .get_or_insert_with(|| Arc::new([0; DATA_LAYER_SIZE]));
        self.updating_dirty = true;
        Arc::make_mut(data)
    }

    fn get_from_data(data: &Option<Arc<[u8; DATA_LAYER_SIZE]>>, index: usize) -> u8 {
        let Some(data) = data.as_ref() else {
            return 0;
        };
        let packed = data[index >> 1];
        packed >> ((index & 1) << 2) & MAX_LIGHT_LEVEL
    }

    fn set_in_data(data: &mut [u8; DATA_LAYER_SIZE], index: usize, value: u8) {
        let byte_index = index >> 1;
        let shift = (index & 1) << 2;
        let mask = !(MAX_LIGHT_LEVEL << shift);
        let value = (value & MAX_LIGHT_LEVEL) << shift;
        data[byte_index] = data[byte_index] & mask | value;
    }

    const fn index(x: usize, y: usize, z: usize) -> usize {
        x | (z << 4) | (y << 8)
    }

    const fn pack_filled(value: u8) -> u8 {
        let value = value & MAX_LIGHT_LEVEL;
        value | value << 4
    }

    fn is_all_zero(data: &[u8; DATA_LAYER_SIZE]) -> bool {
        data.iter().all(|value| *value == 0)
    }
}

impl Default for LightNibbleArray {
    fn default() -> Self {
        Self::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_nibble_index_access_matches_scalable_lux_local_index() {
        let mut nibble = LightNibbleArray::uninitialized();
        let index = 1 | (3 << 4) | (2 << 8);

        nibble.set_at_index(index, 12);

        assert_eq!(nibble.get_updating_at_index(index), 12);
        assert_eq!(nibble.get_updating(1, 2, 3), 12);
        assert_eq!(nibble.get_updating(2, 2, 3), 0);
        assert_eq!(nibble.get_visible_at_index(index), 0);

        assert!(nibble.update_visible());
        assert_eq!(nibble.get_visible_at_index(index), 12);
        assert_eq!(nibble.get_visible(1, 2, 3), 12);
    }

    #[test]
    fn light_nibble_index_access_masks_values_like_coordinate_access() {
        let mut nibble = LightNibbleArray::null();

        nibble.set_at_index(0, 31);

        assert_eq!(nibble.get_updating_at_index(0), MAX_LIGHT_LEVEL);
        assert_eq!(nibble.get_updating(0, 0, 0), MAX_LIGHT_LEVEL);
    }

    #[test]
    fn chunk_light_data_reads_visible_block_light() {
        let mut light = ChunkLightData::for_valid_world_height(0, 16);
        let block_pos = BlockPos::new(1, 2, 3);

        assert_eq!(light.get_light_value(LightLayer::Block, block_pos), 0);

        let Some(nibble) = light.block.nibble_mut(0) else {
            panic!("test section should be inside light range");
        };
        nibble.set_non_null();
        nibble.set(1, 2, 3, 12);
        assert!(nibble.update_visible());

        assert_eq!(light.get_light_value(LightLayer::Block, block_pos), 12);
    }

    #[test]
    fn chunk_light_data_reads_visible_sky_light_with_upward_extrusion() {
        let mut light = ChunkLightData::for_valid_world_height(0, 16);
        let block_pos = BlockPos::new(1, 15, 3);

        assert_eq!(
            light.get_light_value(LightLayer::Sky, block_pos),
            MAX_LIGHT_LEVEL
        );

        let Some(upper_nibble) = light.sky.nibble_mut(1) else {
            panic!("upper test section should be inside light range");
        };
        upper_nibble.set_non_null();
        upper_nibble.set(1, 0, 3, 9);
        assert!(upper_nibble.update_visible());

        assert_eq!(light.get_light_value(LightLayer::Sky, block_pos), 9);

        let Some(current_nibble) = light.sky.nibble_mut(0) else {
            panic!("current test section should be inside light range");
        };
        current_nibble.set_non_null();
        current_nibble.set(1, 15, 3, 7);
        assert!(current_nibble.update_visible());

        assert_eq!(light.get_light_value(LightLayer::Sky, block_pos), 7);
    }
}

/// Error returned when trying to extrude from a null light section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightNibbleExtrudeNullSourceError;

/// Error returned when a chunk light emptiness map has the wrong length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkLightEmptinessMapLengthError {
    /// Expected section count.
    pub expected: usize,
    /// Actual section count.
    pub actual: usize,
}

/// Per-layer chunk-owned light storage used by the ScalableLux-style engine.
#[derive(Debug)]
pub struct ChunkLightLayerStorage {
    layer: LightLayer,
    range: LightSectionRange,
    chunk_section_count: usize,
    nibbles: Box<[LightNibbleArray]>,
    emptiness_map: Option<Box<[bool]>>,
}

impl ChunkLightLayerStorage {
    /// Creates null light nibbles for every light section in a chunk.
    #[must_use]
    pub fn new(layer: LightLayer, range: LightSectionRange, chunk_section_count: usize) -> Self {
        let nibbles = (0..range.section_count())
            .map(|_| LightNibbleArray::null())
            .collect();
        Self {
            layer,
            range,
            chunk_section_count,
            nibbles,
            emptiness_map: None,
        }
    }

    /// Returns this storage's light layer.
    #[must_use]
    pub const fn layer(&self) -> LightLayer {
        self.layer
    }

    /// Returns the vertical light-section range.
    #[must_use]
    pub const fn range(&self) -> LightSectionRange {
        self.range
    }

    /// Returns all chunk light nibbles.
    #[must_use]
    pub fn nibbles(&self) -> &[LightNibbleArray] {
        &self.nibbles
    }

    /// Returns all chunk light nibbles mutably.
    #[must_use]
    pub fn nibbles_mut(&mut self) -> &mut [LightNibbleArray] {
        &mut self.nibbles
    }

    /// Replaces every light-section nibble with a fresh null nibble.
    ///
    /// ScalableLux does this for the center chunk before first lighting it via
    /// `StarLightEngine.getFilledEmptyLight`, so neighbor-initialized data from
    /// earlier chunk passes cannot become the center chunk's canonical light.
    pub fn reset_nibbles_to_null(&mut self) {
        for nibble in &mut self.nibbles {
            *nibble = LightNibbleArray::null();
        }
    }

    /// Returns the visible light value for one block position.
    #[must_use]
    pub fn get_light_value(&self, block_pos: BlockPos) -> u8 {
        match self.layer {
            LightLayer::Sky => self.get_sky_light_value(block_pos),
            LightLayer::Block => self.get_block_light_value(block_pos),
        }
    }

    /// Returns the number of real chunk sections tracked by the emptiness map.
    #[must_use]
    pub const fn chunk_section_count(&self) -> usize {
        self.chunk_section_count
    }

    /// Returns a light nibble for a section Y.
    #[must_use]
    pub fn nibble(&self, section_y: i32) -> Option<&LightNibbleArray> {
        self.range
            .section_index(section_y)
            .and_then(|index| self.nibbles.get(index))
    }

    /// Returns a mutable light nibble for a section Y.
    pub fn nibble_mut(&mut self, section_y: i32) -> Option<&mut LightNibbleArray> {
        let index = self.range.section_index(section_y)?;
        self.nibbles.get_mut(index)
    }

    /// Returns the current section emptiness map, if known.
    #[must_use]
    pub fn emptiness_map(&self) -> Option<&[bool]> {
        self.emptiness_map.as_deref()
    }

    /// Returns the known emptiness for one real chunk section Y.
    #[must_use]
    pub fn section_empty(&self, section_y: i32) -> Option<bool> {
        let index = self.chunk_section_index(section_y)?;
        self.emptiness_map
            .as_deref()
            .and_then(|emptiness_map| emptiness_map.get(index).copied())
    }

    /// Replaces the section emptiness map.
    pub fn set_emptiness_map(
        &mut self,
        emptiness_map: Box<[bool]>,
    ) -> Result<(), ChunkLightEmptinessMapLengthError> {
        let actual = emptiness_map.len();
        if actual != self.chunk_section_count {
            return Err(ChunkLightEmptinessMapLengthError {
                expected: self.chunk_section_count,
                actual,
            });
        }

        self.emptiness_map = Some(emptiness_map);
        Ok(())
    }

    /// Replaces the section emptiness map from current section counters.
    pub fn refresh_emptiness_map_from_sections(
        &mut self,
        sections: &Sections,
    ) -> Result<(), ChunkLightEmptinessMapLengthError> {
        self.set_emptiness_map(sections.section_emptiness_map())
    }

    /// Updates one known section emptiness entry, returning the previous value.
    pub fn set_section_empty(&mut self, section_y: i32, empty: bool) -> Option<bool> {
        let index = self.chunk_section_index(section_y)?;
        let emptiness_map = self.emptiness_map.as_deref_mut()?;
        let previous = *emptiness_map.get(index)?;
        emptiness_map[index] = empty;
        Some(previous)
    }

    fn get_block_light_value(&self, block_pos: BlockPos) -> u8 {
        self.visible_nibble_value(block_pos).unwrap_or(0)
    }

    fn get_sky_light_value(&self, block_pos: BlockPos) -> u8 {
        if let Some(value) = self.visible_nibble_value(block_pos) {
            return value;
        }

        let local_x = section_relative_coord(block_pos.x());
        let local_z = section_relative_coord(block_pos.z());
        let mut section_y = SectionPos::block_to_section_coord(block_pos.y()) + 1;
        while section_y < self.range.max_section_y_exclusive() {
            if let Some(nibble) = self.nibble(section_y)
                && !nibble.is_null_visible()
            {
                return nibble.get_visible(local_x, 0, local_z);
            }
            section_y += 1;
        }

        MAX_LIGHT_LEVEL
    }

    fn visible_nibble_value(&self, block_pos: BlockPos) -> Option<u8> {
        let section_y = SectionPos::block_to_section_coord(block_pos.y());
        let nibble = self.nibble(section_y)?;
        if nibble.is_null_visible() {
            return None;
        }

        Some(nibble.get_visible(
            section_relative_coord(block_pos.x()),
            section_relative_coord(block_pos.y()),
            section_relative_coord(block_pos.z()),
        ))
    }

    fn chunk_section_index(&self, section_y: i32) -> Option<usize> {
        let index = self.range.chunk_section_index(section_y)?;
        (index < self.chunk_section_count).then_some(index)
    }
}

fn section_relative_coord(block_coord: i32) -> usize {
    (block_coord & 15) as usize
}

/// Chunk-owned block and sky light storage.
#[derive(Debug)]
pub struct ChunkLightData {
    /// Block light nibbles and section emptiness metadata.
    pub block: ChunkLightLayerStorage,
    /// Sky light nibbles and section emptiness metadata.
    pub sky: ChunkLightLayerStorage,
}

impl ChunkLightData {
    /// Creates empty ScalableLux-style light storage for one chunk.
    pub fn new(min_y: i32, height: i32) -> Result<Self, LightSectionRangeError> {
        let range = LightSectionRange::from_world_height(min_y, height)?;
        Ok(Self {
            block: ChunkLightLayerStorage::new(
                LightLayer::Block,
                range,
                range.chunk_section_count(),
            ),
            sky: ChunkLightLayerStorage::new(LightLayer::Sky, range, range.chunk_section_count()),
        })
    }

    /// Creates storage for world heights already accepted by chunk construction.
    ///
    /// Invalid world heights are fatal because chunk-owned light arrays cannot
    /// be indexed coherently without the vanilla padded light-section range.
    #[must_use]
    pub fn for_valid_world_height(min_y: i32, height: i32) -> Self {
        match Self::new(min_y, height) {
            Ok(data) => data,
            Err(error) => panic!("invalid world height for chunk light data: {error:?}"),
        }
    }

    /// Refreshes both layer emptiness maps from current chunk section counters.
    pub fn refresh_emptiness_maps_from_sections(
        &mut self,
        sections: &Sections,
    ) -> Result<(), ChunkLightEmptinessMapLengthError> {
        self.block.refresh_emptiness_map_from_sections(sections)?;
        self.sky.refresh_emptiness_map_from_sections(sections)
    }

    /// Updates one real chunk section's known emptiness in both light layers.
    pub fn set_section_empty(&mut self, section_y: i32, empty: bool) -> bool {
        let block_changed = self
            .block
            .set_section_empty(section_y, empty)
            .is_some_and(|previous| previous != empty);
        let sky_changed = self
            .sky
            .set_section_empty(section_y, empty)
            .is_some_and(|previous| previous != empty);
        block_changed || sky_changed
    }

    /// Returns the visible light value for one layer at a block position.
    #[must_use]
    pub fn get_light_value(&self, layer: LightLayer, block_pos: BlockPos) -> u8 {
        match layer {
            LightLayer::Sky => self.sky.get_light_value(block_pos),
            LightLayer::Block => self.block.get_light_value(block_pos),
        }
    }
}

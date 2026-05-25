use std::sync::Arc;

use super::{
    DATA_LAYER_EDGE, DATA_LAYER_SIZE, DATA_LAYER_Y_STRIDE, DataLayer, DataLayerLengthError,
    LightLayer, LightSectionRange, LightSectionRangeError, MAX_LIGHT_LEVEL,
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
        if above.updating_state == LightNibbleState::Null {
            return Err(LightNibbleExtrudeNullSourceError);
        }

        let Some(source) = above.updating_data.as_ref() else {
            self.set_uninitialized();
            return Ok(());
        };

        let mut row = [0; DATA_LAYER_Y_STRIDE];
        row.copy_from_slice(&source[..DATA_LAYER_Y_STRIDE]);
        let data = self.ensure_updating_data();
        for y in 0..DATA_LAYER_EDGE {
            let start = y * DATA_LAYER_Y_STRIDE;
            data[start..start + DATA_LAYER_Y_STRIDE].copy_from_slice(&row);
        }

        Ok(())
    }

    /// Returns an updating light value at local section coordinates.
    #[must_use]
    pub fn get_updating(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        Self::get_from_data(&self.updating_data, Self::index(x, y, z))
    }

    /// Returns a visible light value at local section coordinates.
    #[must_use]
    pub fn get_visible(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        Self::get_from_data(&self.visible_data, Self::index(x, y, z))
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
        let chunk_section_count = (height / DATA_LAYER_EDGE as i32) as usize;
        Ok(Self {
            block: ChunkLightLayerStorage::new(LightLayer::Block, range, chunk_section_count),
            sky: ChunkLightLayerStorage::new(LightLayer::Sky, range, chunk_section_count),
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
}

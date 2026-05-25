use rustc_hash::FxHashMap;
use steel_utils::{PackedSectionPos, SectionPos};

use super::{DATA_LAYER_EDGE, DATA_LAYER_SIZE, MAX_LIGHT_LEVEL};

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

    pub(super) fn from_packed_data(data: Box<[u8; DATA_LAYER_SIZE]>) -> Self {
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

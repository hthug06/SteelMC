//! A paletted container is a container that can be either homogeneous or heterogeneous.
use std::{
    fmt::Debug,
    hash::Hash,
    io::{Result, Write},
};

use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_utils::{BlockStateId, codec::VarInt, serial::WriteTo};

/// A trait for converting a value to a global ID.
pub trait ToGlobalId {
    /// Converts the value to a global ID.
    fn to_global_id(&self) -> u32;
}

impl ToGlobalId for BlockStateId {
    fn to_global_id(&self) -> u32 {
        u32::from(self.0)
    }
}

impl ToGlobalId for u16 {
    fn to_global_id(&self) -> u32 {
        u32::from(*self)
    }
}

/// 3d array indexed by y,z,x
type Cube<T, const DIM: usize> = [[[T; DIM]; DIM]; DIM];

/// A heterogeneous palette container.
#[derive(Debug, Clone)]
pub struct HeterogeneousPalette<V: Hash + Eq + Copy, const DIM: usize> {
    pub(crate) cube: Box<Cube<V, DIM>>,
    // Keeps track of how many different times each value appears in the cube. (value, count)
    pub(crate) palette: Vec<(V, u16)>,
}

impl<V: Hash + Eq + Copy, const DIM: usize> HeterogeneousPalette<V, DIM> {
    fn get(&self, x: usize, y: usize, z: usize) -> V {
        debug_assert!(x < DIM);
        debug_assert!(y < DIM);
        debug_assert!(z < DIM);

        self.cube[y][z][x]
    }

    fn get_at_index(&self, index: usize) -> V {
        debug_assert!(index < DIM * DIM * DIM);

        let y = index / (DIM * DIM);
        let z = (index / DIM) % DIM;
        let x = index % DIM;
        self.cube[y][z][x]
    }

    /// Returns an iterator over all values in the cube in y, z, x order.
    pub fn iter_values(&self) -> impl Iterator<Item = &V> {
        self.cube.iter().flatten().flatten()
    }

    fn set(&mut self, x: usize, y: usize, z: usize, value: V) -> V {
        debug_assert!(x < DIM);
        debug_assert!(y < DIM);
        debug_assert!(z < DIM);

        let old_value = self.cube[y][z][x];

        if let Some((_, count)) = self.palette.iter_mut().find(|(v, _)| *v == value) {
            *count += 1;
        } else {
            self.palette.push((value, 1));
        }

        if let Some((index, (_, count))) = self
            .palette
            .iter_mut()
            .enumerate()
            .find(|(_, (v, _))| *v == old_value)
        {
            *count -= 1;
            if *count == 0 {
                self.palette.swap_remove(index);
            }
        }

        self.cube[y][z][x] = value;

        old_value
    }
}

/// A paletted container.
#[derive(Debug, Clone)]
pub enum PalettedContainer<V: Hash + Eq + Copy + Default, const DIM: usize> {
    /// A homogeneous container, where all values are the same.
    Homogeneous(V),
    /// A heterogeneous container, where values can be different.
    Heterogeneous(HeterogeneousPalette<V, DIM>),
}

enum PaletteMode {
    Linear,
    Hash,
    Global,
}

impl<V: Hash + Eq + Copy + Default + Debug, const DIM: usize> PalettedContainer<V, DIM> {
    /// The size of the container in one dimension.
    pub const SIZE: usize = DIM;
    /// The volume of the container.
    pub const VOLUME: usize = DIM * DIM * DIM;

    /// Creates a `PalettedContainer` from a pre-built cube.
    ///
    /// Will automatically determine if the result should be homogeneous or heterogeneous.
    #[must_use]
    pub fn from_cube(cube: Box<Cube<V, DIM>>) -> Self {
        let mut palette: Vec<(V, u16)> = Vec::new();
        cube.iter().flatten().flatten().for_each(|v| {
            if let Some((_, count)) = palette.iter_mut().find(|(value, _)| value == v) {
                *count += 1;
            } else {
                palette.push((*v, 1));
            }
        });

        if palette.len() == 1 {
            Self::Homogeneous(palette[0].0)
        } else {
            Self::Heterogeneous(HeterogeneousPalette { cube, palette })
        }
    }

    /// Gets the value at the given coordinates.
    pub fn get(&self, x: usize, y: usize, z: usize) -> V {
        match self {
            Self::Homogeneous(value) => *value,
            Self::Heterogeneous(data) => data.get(x, y, z),
        }
    }

    /// Gets the value at a y,z,x linear index.
    ///
    /// The index layout is `x + z * DIM + y * DIM * DIM`, matching the flat
    /// order used when serializing palette data and the local block index used
    /// by ScalableLux light propagation.
    pub fn get_at_index(&self, index: usize) -> V {
        debug_assert!(index < Self::VOLUME);

        match self {
            Self::Homogeneous(value) => *value,
            Self::Heterogeneous(data) => data.get_at_index(index),
        }
    }

    /// Returns whether this container's palette may contain a matching value.
    ///
    /// This checks palette entries instead of every cell, matching vanilla and
    /// ScalableLux's fast pre-scan before doing a full section pass.
    #[must_use]
    pub fn maybe_has(&self, mut predicate: impl FnMut(V) -> bool) -> bool {
        match self {
            Self::Homogeneous(value) => predicate(*value),
            Self::Heterogeneous(data) => data.palette.iter().any(|(value, _)| predicate(*value)),
        }
    }

    /// Collects all values in the container in y, z, x order.
    #[must_use]
    pub fn collect_values(&self) -> Vec<V> {
        match self {
            Self::Homogeneous(value) => vec![*value; Self::VOLUME],
            Self::Heterogeneous(data) => data.iter_values().copied().collect(),
        }
    }

    /// Sets the value at the given coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: V) -> V {
        debug_assert!(x < Self::SIZE);
        debug_assert!(y < Self::SIZE);
        debug_assert!(z < Self::SIZE);

        match self {
            Self::Homogeneous(original) => {
                let original = *original;
                if value != original {
                    let mut cube = Box::new([[[original; DIM]; DIM]; DIM]);
                    cube[y][z][x] = value;
                    *self = Self::from_cube(cube);
                }
                original
            }
            Self::Heterogeneous(data) => {
                let original = data.set(x, y, z, value);
                if data.palette.len() == 1 {
                    *self = Self::Homogeneous(data.palette[0].0);
                }
                original
            }
        }
    }

    /// Writes the container to the given writer.
    ///
    /// # Errors
    /// - If the writer fails to write.
    #[expect(
        clippy::missing_panics_doc,
        clippy::unwrap_used,
        reason = "position() is guaranteed to exist: palette was built from the cube's own values"
    )]
    pub fn write(&self, writer: &mut impl Write) -> Result<()>
    where
        V: ToGlobalId,
    {
        match self {
            Self::Homogeneous(value) => {
                // bits per entry = 0 (ZeroBitStorage)
                0u8.write(writer)?;
                // Single-value palette
                VarInt(value.to_global_id() as i32).write(writer)?;
                // writeFixedSizeLongArray(new long[0]) writes nothing
            }
            Self::Heterogeneous(data) => {
                let (bits, mode) = Self::calculate_strategy(data.palette.len());

                // Write bits per entry
                bits.write(writer)?;

                // Write Palette
                match mode {
                    PaletteMode::Linear | PaletteMode::Hash => {
                        VarInt(data.palette.len() as i32).write(writer)?;
                        for (val, _) in &data.palette {
                            VarInt(val.to_global_id() as i32).write(writer)?;
                        }
                    }
                    PaletteMode::Global => {}
                }

                // Pack data
                let indices: Vec<u32> = data
                    .cube
                    .iter()
                    .flatten()
                    .flatten()
                    .map(|val| {
                        if matches!(mode, PaletteMode::Global) {
                            val.to_global_id()
                        } else {
                            data.palette.iter().position(|(v, _)| v == val).unwrap() as u32
                        }
                    })
                    .collect();

                let packed = pack_bits(&indices, bits as usize);

                // writeFixedSizeLongArray: raw longs, no VarInt length prefix
                for long in packed {
                    long.write(writer)?;
                }
            }
        }
        Ok(())
    }

    fn calculate_strategy(count: usize) -> (u8, PaletteMode) {
        if DIM == 16 {
            // Block states
            match count {
                0..=1 => unreachable!("Homogeneous handled separately"),
                2..=16 => (4, PaletteMode::Linear),
                17..=32 => (5, PaletteMode::Hash),
                33..=64 => (6, PaletteMode::Hash),
                65..=128 => (7, PaletteMode::Hash),
                129..=256 => (8, PaletteMode::Hash),
                _ => (15, PaletteMode::Global), // ceil(log2(max_block_state_id)) approx 15
            }
        } else {
            // Biomes
            match count {
                0..=1 => unreachable!("Homogeneous handled separately"),
                2 => (1, PaletteMode::Linear),
                3..=4 => (2, PaletteMode::Linear),
                5..=8 => (3, PaletteMode::Hash),
                _ => (6, PaletteMode::Global), // ceil(log2(max_biome_id)) approx 6
            }
        }
    }
}

fn pack_bits(indices: &[u32], bits: usize) -> Vec<u64> {
    let values_per_long = 64 / bits;
    let len = indices.len().div_ceil(values_per_long);
    let mut data = vec![0u64; len];

    for (i, &index) in indices.iter().enumerate() {
        let array_index = i / values_per_long;
        let offset = (i % values_per_long) * bits;
        data[array_index] |= u64::from(index) << offset;
    }

    data
}

/// A palette container for blocks.
pub type BlockPalette = PalettedContainer<BlockStateId, 16>;
/// A palette container for biomes.
pub type BiomePalette = PalettedContainer<u16, 4>;

impl BlockPalette {
    /// Gets the number of non-empty blocks in the container.
    #[must_use]
    pub fn non_empty_block_count(&self) -> u16 {
        match self {
            Self::Homogeneous(v) => {
                if v.0 == 0 {
                    0
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "VOLUME = 16^3 = 4096, fits in u16"
                    )]
                    {
                        Self::VOLUME as u16
                    }
                }
            }
            Self::Heterogeneous(data) => {
                let mut count = 0;
                for (v, c) in &data.palette {
                    if v.0 != 0 {
                        count += c;
                    }
                }
                count
            }
        }
    }

    /// Returns `true` if this palette contains only air blocks.
    #[must_use]
    pub fn has_only_air(&self) -> bool {
        match self {
            Self::Homogeneous(v) => v.is_air(),
            //TODO: Use a nonEmpty counter?
            Self::Heterogeneous(_data) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use steel_utils::BlockStateId;

    use super::{BiomePalette, BlockPalette};

    #[test]
    fn get_at_index_reads_homogeneous_palette_values() {
        let blocks = BlockPalette::Homogeneous(BlockStateId(7));
        let biomes = BiomePalette::Homogeneous(12);

        assert_eq!(
            blocks.get_at_index(15 | (15 << 4) | (15 << 8)),
            BlockStateId(7)
        );
        assert_eq!(biomes.get_at_index(3 | (3 << 2) | (3 << 4)), 12);
    }

    #[test]
    fn get_at_index_uses_y_z_x_linear_order_for_blocks() {
        let mut blocks = BlockPalette::Homogeneous(BlockStateId(0));
        blocks.set(3, 4, 5, BlockStateId(42));
        blocks.set(15, 15, 15, BlockStateId(99));

        assert_eq!(
            blocks.get_at_index(3 | (5 << 4) | (4 << 8)),
            BlockStateId(42)
        );
        assert_eq!(
            blocks.get_at_index(15 | (15 << 4) | (15 << 8)),
            BlockStateId(99)
        );
        assert_eq!(blocks.get_at_index(0), BlockStateId(0));
    }

    #[test]
    fn get_at_index_uses_y_z_x_linear_order_for_biomes() {
        let mut biomes = BiomePalette::Homogeneous(0);
        biomes.set(2, 3, 1, 11);

        assert_eq!(biomes.get_at_index(2 | (1 << 2) | (3 << 4)), 11);
        assert_eq!(biomes.get_at_index(0), 0);
    }

    #[test]
    fn maybe_has_checks_palette_values_without_scanning_cells() {
        let homogeneous = BlockPalette::Homogeneous(BlockStateId(7));
        assert!(homogeneous.maybe_has(|state| state == BlockStateId(7)));
        assert!(!homogeneous.maybe_has(|state| state == BlockStateId(8)));

        let mut heterogeneous = BlockPalette::Homogeneous(BlockStateId(0));
        heterogeneous.set(3, 4, 5, BlockStateId(42));
        heterogeneous.set(15, 15, 15, BlockStateId(99));

        assert!(heterogeneous.maybe_has(|state| state == BlockStateId(42)));
        assert!(heterogeneous.maybe_has(|state| state == BlockStateId(99)));
        assert!(!heterogeneous.maybe_has(|state| state == BlockStateId(7)));
    }
}

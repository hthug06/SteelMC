use rustc_hash::{FxHashMap, FxHashSet};
use steel_utils::{BlockPos, ChunkPos, PackedSectionPos, SectionPos, codec::BitSet};

use super::{
    ADD_SKY_SOURCE_ENTRY, CHUNK_EDGE, ChunkSkyLightSources, DATA_LAYER_EDGE, DATA_LAYER_SIZE,
    DATA_LAYER_Y_STRIDE, DataLayer, DataLayerStorageMap, LIGHT_SECTION_PADDING, LightLayer,
    LightPropagationQueues, LightQueueEntry, MAX_LIGHT_LEVEL, MAX_SECTION_NEIGHBORS,
    POSITIVE_INFINITY, REMOVE_SKY_SOURCE_ENTRY, REMOVE_TOP_SKY_SOURCE_ENTRY, SECTION_HAS_DATA_BIT,
    SECTION_NEIGHBOR_COUNT_BITS, SkyLightSourceNeighborhood,
};

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
    pub(super) fn empty_bit_set(self) -> BitSet {
        BitSet(vec![0; self.section_count().div_ceil(64)].into_boxed_slice())
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
    pub(super) updating_section_data: DataLayerStorageMap,
    pub(super) changed_sections: FxHashSet<PackedSectionPos>,
    pub(super) sections_affected_by_light_updates: FxHashSet<PackedSectionPos>,
    pub(super) queued_sections: FxHashMap<PackedSectionPos, DataLayer>,
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

    /// Fills skylight source cells and queues their propagation edges.
    ///
    /// This mirrors vanilla `SkyLightEngine.propagateLightSources`; callers
    /// supply the center chunk's `ChunkSkyLightSources` and its four horizontal
    /// neighbors.
    pub fn propagate_sky_light_sources(
        &mut self,
        section_zero_pos: SectionPos,
        source_neighborhood: SkyLightSourceNeighborhood<'_>,
        queues: &mut LightPropagationQueues,
    ) -> Option<()> {
        self.sky_data.as_ref()?;

        self.set_light_enabled(section_zero_pos, true);

        let top_section_y = self.get_top_section_y(section_zero_pos)?;
        let bottom_section_y = self.get_bottom_section_y()?;
        let section_min_x = section_zero_pos.x() << 4;
        let section_min_z = section_zero_pos.z() << 4;

        for section_y in (bottom_section_y..top_section_y).rev() {
            let section_pos =
                SectionPos::new(section_zero_pos.x(), section_y, section_zero_pos.z());
            let Some(data_layer) = self.get_data_layer_to_write(section_pos) else {
                continue;
            };

            let section_min_y = section_y << 4;
            let section_max_y = section_min_y + 15;
            let mut sources_below = false;

            for z in 0..CHUNK_EDGE {
                for x in 0..CHUNK_EDGE {
                    let lowest_source_y = source_neighborhood.center.get_lowest_source_y(x, z);
                    if lowest_source_y > section_max_y {
                        continue;
                    }

                    let north_lowest_source_y = if z == 0 {
                        source_neighborhood
                            .north
                            .get_lowest_source_y(x, CHUNK_EDGE - 1)
                    } else {
                        source_neighborhood.center.get_lowest_source_y(x, z - 1)
                    };
                    let south_lowest_source_y = if z == CHUNK_EDGE - 1 {
                        source_neighborhood.south.get_lowest_source_y(x, 0)
                    } else {
                        source_neighborhood.center.get_lowest_source_y(x, z + 1)
                    };
                    let west_lowest_source_y = if x == 0 {
                        source_neighborhood
                            .west
                            .get_lowest_source_y(CHUNK_EDGE - 1, z)
                    } else {
                        source_neighborhood.center.get_lowest_source_y(x - 1, z)
                    };
                    let east_lowest_source_y = if x == CHUNK_EDGE - 1 {
                        source_neighborhood.east.get_lowest_source_y(0, z)
                    } else {
                        source_neighborhood.center.get_lowest_source_y(x + 1, z)
                    };
                    let neighbor_lowest_source_y = north_lowest_source_y
                        .max(south_lowest_source_y)
                        .max(west_lowest_source_y)
                        .max(east_lowest_source_y);
                    let min_source_y = section_min_y.max(lowest_source_y);

                    for y in (min_source_y..=section_max_y).rev() {
                        data_layer.set(x, Self::section_relative_coord(y), z, MAX_LIGHT_LEVEL);

                        if y == lowest_source_y || y < neighbor_lowest_source_y {
                            queues.enqueue_increase(
                                BlockPos::new(
                                    section_min_x + x as i32,
                                    y,
                                    section_min_z + z as i32,
                                ),
                                LightQueueEntry::increase_sky_source_in_directions(
                                    y == lowest_source_y,
                                    y < north_lowest_source_y,
                                    y < south_lowest_source_y,
                                    y < west_lowest_source_y,
                                    y < east_lowest_source_y,
                                ),
                            );
                        }
                    }

                    if lowest_source_y < section_min_y {
                        sources_below = true;
                    }
                }
            }

            if !sources_below {
                break;
            }
        }

        Some(())
    }

    /// Updates one sky-source column after its source edge changes.
    ///
    /// This mirrors vanilla `SkyLightEngine.updateSourcesInColumn`, split from
    /// `checkNode` so the future engine can supply chunk-neighbor source data
    /// without this storage type owning world chunk lookup.
    pub fn update_sky_sources_in_column(
        &mut self,
        x: i32,
        z: i32,
        lowest_source_y: i32,
        neighbor_lowest_source_y: i32,
        queues: &mut LightPropagationQueues,
    ) -> Result<Option<()>, MissingLightDataLayerError> {
        if self.sky_data.is_none() {
            return Ok(None);
        }

        let Some(bottom_section_y) = self.get_bottom_section_y() else {
            return Ok(Some(()));
        };
        if bottom_section_y == POSITIVE_INFINITY {
            return Ok(Some(()));
        }

        let world_bottom_y = Self::section_to_block_coord(bottom_section_y);
        self.remove_sky_sources_below(x, z, lowest_source_y, world_bottom_y, queues)?;
        self.add_sky_sources_above(
            x,
            z,
            lowest_source_y,
            neighbor_lowest_source_y,
            world_bottom_y,
            queues,
        )?;

        Ok(Some(()))
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

    fn section_relative_coord(block_coord: i32) -> usize {
        (block_coord & 15) as usize
    }

    fn section_to_block_coord(section_y: i32) -> i32 {
        section_y << 4
    }

    fn remove_sky_sources_below(
        &mut self,
        x: i32,
        z: i32,
        lowest_source_y: i32,
        world_bottom_y: i32,
        queues: &mut LightPropagationQueues,
    ) -> Result<(), MissingLightDataLayerError> {
        if lowest_source_y <= world_bottom_y {
            return Ok(());
        }

        let section_x = SectionPos::block_to_section_coord(x);
        let section_z = SectionPos::block_to_section_coord(z);
        let start_y = lowest_source_y - 1;
        let mut section_y = SectionPos::block_to_section_coord(start_y);

        loop {
            let Some(has_light_data) = self.has_light_data_at_or_below(section_y) else {
                return Ok(());
            };
            if !has_light_data {
                return Ok(());
            }

            let section_pos = SectionPos::new(section_x, section_y, section_z);
            if self.storing_light_for_section(section_pos) {
                let section_bottom_y = Self::section_to_block_coord(section_y);
                let section_top_y = section_bottom_y + 15;

                for y in (section_bottom_y..=section_top_y.min(start_y)).rev() {
                    let block_pos = BlockPos::new(x, y, z);
                    if self.get_stored_level(block_pos) != Some(MAX_LIGHT_LEVEL) {
                        return Ok(());
                    }

                    self.set_stored_level(block_pos, 0)?;
                    let entry = if y == start_y {
                        REMOVE_TOP_SKY_SOURCE_ENTRY
                    } else {
                        REMOVE_SKY_SOURCE_ENTRY
                    };
                    queues.enqueue_decrease(block_pos, entry);
                }
            }

            section_y -= 1;
        }
    }

    fn add_sky_sources_above(
        &mut self,
        x: i32,
        z: i32,
        lowest_source_y: i32,
        neighbor_lowest_source_y: i32,
        world_bottom_y: i32,
        queues: &mut LightPropagationQueues,
    ) -> Result<(), MissingLightDataLayerError> {
        let section_x = SectionPos::block_to_section_coord(x);
        let section_z = SectionPos::block_to_section_coord(z);
        let start_y = lowest_source_y.max(world_bottom_y);
        let mut section_y = SectionPos::block_to_section_coord(start_y);

        loop {
            let section_pos = SectionPos::new(section_x, section_y, section_z);
            let Some(is_above_data) = self.is_above_data(section_pos) else {
                return Ok(());
            };
            if is_above_data {
                return Ok(());
            }

            if self.storing_light_for_section(section_pos) {
                let section_bottom_y = Self::section_to_block_coord(section_y);
                let section_top_y = section_bottom_y + 15;

                for y in section_bottom_y.max(start_y)..=section_top_y {
                    let block_pos = BlockPos::new(x, y, z);
                    if self.get_stored_level(block_pos) == Some(MAX_LIGHT_LEVEL) {
                        return Ok(());
                    }

                    self.set_stored_level(block_pos, MAX_LIGHT_LEVEL)?;
                    if y < neighbor_lowest_source_y || y == lowest_source_y {
                        queues.enqueue_increase(block_pos, ADD_SKY_SOURCE_ENTRY);
                    }
                }
            }

            section_y += 1;
        }
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

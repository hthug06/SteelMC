use steel_registry::{blocks::block_state_ext::BlockStateExt, vanilla_blocks};
use steel_utils::{BlockPos, ChunkPos, Direction, SectionPos};

use super::{
    CachedLightBlock, LIGHT_BLOCKED, LightAxisDirection, LightCacheLayout, LightDirectionSet,
    LightLayer, LightLayerWriteCache, LightQueueFlags, LightSectionEmptinessChange,
    LightSectionReadCache, LightWorkset, MAX_LIGHT_LEVEL, PackedLightPropagationQueues,
    PackedLightQueueEntry, get_light_block_into, get_light_opacity, light_occlusion_shape,
};

/// Error returned when a sky-light propagation context is built from mismatched caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyLightPropagationContextError {
    /// Sky-light propagation requires a sky light write cache.
    WrongLayer {
        /// Layer supplied by the write cache.
        layer: LightLayer,
    },
    /// Section and light caches were built from different cache layouts.
    LayoutMismatch {
        /// Layout used by the section cache.
        section_layout: LightCacheLayout,
        /// Layout used by the light cache.
        light_layout: LightCacheLayout,
    },
    /// The workset does not contain its center chunk.
    MissingCenterChunk {
        /// Missing center chunk position.
        chunk_pos: ChunkPos,
    },
}

/// Sections whose visible sky-light data changed during a scoped update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkyLightUpdateResult {
    /// Light sections that should be reported to the world/chunk update layer.
    pub updated_sections: Vec<SectionPos>,
}

/// Whether chunk sky-light generation must validate edge consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyLightChunkEdgeChecks {
    /// Seed skylight and validate this chunk's horizontal edges against neighbors.
    Required,
    /// Trust existing neighboring light and pull initialized edge levels inward.
    Skipped,
}

/// Seeds and propagates sky light for the center chunk without edge checks.
///
/// This matches ScalableLux `SkyStarLightEngine.lightChunk` for the
/// no-edge-check path used during ordered relighting: sky nibbles around
/// non-empty sections are initialized, full skylight is propagated down through
/// transparent columns, initialized neighbor levels are pulled inward, and
/// dirty visible nibbles are published.
pub fn propagate_sky_light_chunk_without_edge_checks(
    workset: &LightWorkset,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    propagate_sky_light_chunk(workset, SkyLightChunkEdgeChecks::Skipped)
}

/// Seeds and propagates sky light for the center chunk of a scoped workset.
///
/// This matches ScalableLux `SkyStarLightEngine.lightChunk`: sky nibbles around
/// non-empty sections are initialized, full skylight is propagated downward,
/// then the caller chooses between validating edge consistency or pulling
/// already-initialized neighbor levels inward.
pub fn propagate_sky_light_chunk(
    workset: &LightWorkset,
    edge_checks: SkyLightChunkEdgeChecks,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = chunk_cache.layout();
        let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        };
        if chunk_cache.chunk(center_slot).is_none() {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        }

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    context.reset_center_chunk_nibbles();
                    context.handle_unlit_empty_section_changes(layout.center_chunk());
                    context.light_chunk(layout.center_chunk(), edge_checks);
                    if edge_checks == SkyLightChunkEdgeChecks::Required {
                        context.deinit_and_lazy_init_empty_sections(layout.center_chunk(), true);
                    }
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(SkyLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Force-synchronizes sky-light nibbles for an already-lit loaded chunk.
///
/// This matches the sky layer of ScalableLux `forceLoadInChunk`: existing
/// light data is kept, empty-section nibbles are synchronized, and dirty
/// visible nibbles are published before the later edge-check pass.
pub fn force_load_sky_light_chunk(
    workset: &LightWorkset,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = ensure_center_chunk(chunk_cache)?;

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    context.handle_loaded_empty_section_changes(layout.center_chunk());
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(SkyLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Validates already-loaded sky-light chunk edges without resetting nibbles.
///
/// This matches ScalableLux `checkSkyEdges`: the force-load pass has already
/// synchronized empty-section nibbles, so this pass only checks horizontal
/// consistency against loaded neighbors and publishes its own dirty nibbles.
pub fn check_sky_light_chunk_edges(
    workset: &LightWorkset,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = ensure_center_chunk(chunk_cache)?;

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    context.light.rewrite_null_nibbles_for_skylight();
                    for section_y in (layout.range().min_section_y()
                        ..layout.range().max_section_y_exclusive())
                        .rev()
                    {
                        context.check_null_section(layout.center_chunk(), section_y, true);
                    }
                    context.check_chunk_edges(
                        layout.center_chunk(),
                        layout.range().min_section_y(),
                        layout.range().max_section_y_exclusive() - 1,
                    );
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(SkyLightUpdateResult { updated_sections })
            })
        })
    })
}

/// Loads already-persisted sky light and validates chunk edges without resetting nibbles.
///
/// This is the complete sky-layer `lit == true` path: force-load
/// empty-section state first, then run the edge-check pass.
pub fn load_sky_light_chunk(
    workset: &LightWorkset,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    let mut updated_sections = force_load_sky_light_chunk(workset)?.updated_sections;
    updated_sections.extend(check_sky_light_chunk_edges(workset)?.updated_sections);
    Ok(SkyLightUpdateResult { updated_sections })
}

fn ensure_center_chunk(
    chunk_cache: &super::LightChunkReadCache<'_>,
) -> Result<LightCacheLayout, SkyLightPropagationContextError> {
    let layout = chunk_cache.layout();
    let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
        return Err(SkyLightPropagationContextError::MissingCenterChunk {
            chunk_pos: layout.center_chunk(),
        });
    };
    if chunk_cache.chunk(center_slot).is_none() {
        return Err(SkyLightPropagationContextError::MissingCenterChunk {
            chunk_pos: layout.center_chunk(),
        });
    }

    Ok(layout)
}

/// Runs ScalableLux-style sky-light propagation for changed blocks in a scoped workset.
pub fn propagate_sky_light_changes(
    workset: &LightWorkset,
    positions: impl IntoIterator<Item = BlockPos>,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    propagate_sky_light_changes_with_empty_sections(workset, positions, [])
}

/// Runs sky-light propagation after applying real section emptiness transitions.
pub fn propagate_sky_light_changes_with_empty_sections(
    workset: &LightWorkset,
    positions: impl IntoIterator<Item = BlockPos>,
    empty_sections: impl IntoIterator<Item = LightSectionEmptinessChange>,
) -> Result<SkyLightUpdateResult, SkyLightPropagationContextError> {
    let positions = positions.into_iter().collect::<Vec<_>>();
    let empty_sections = empty_sections.into_iter().collect::<Vec<_>>();

    workset.with_chunk_read_cache(|chunk_cache| {
        let layout = chunk_cache.layout();
        let Some(center_slot) = layout.cached_chunk(layout.center_chunk()) else {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        };
        if chunk_cache.chunk(center_slot).is_none() {
            return Err(SkyLightPropagationContextError::MissingCenterChunk {
                chunk_pos: layout.center_chunk(),
            });
        }

        chunk_cache.with_section_read_cache(|section_cache| {
            chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                let mut queues = PackedLightPropagationQueues::new();

                {
                    let mut context =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues)?;
                    let mut changed_chunks = Vec::new();
                    for change in &empty_sections {
                        let chunk_pos =
                            ChunkPos::new(change.section_pos.x(), change.section_pos.z());
                        context
                            .light
                            .set_section_empty(change.section_pos, change.empty);
                        if !changed_chunks.contains(&chunk_pos) {
                            changed_chunks.push(chunk_pos);
                        }
                    }
                    for chunk_pos in changed_chunks {
                        context.deinit_and_lazy_init_empty_sections(chunk_pos, false);
                    }
                    context.propagate_block_changes(&positions);
                }

                let mut updated_sections = Vec::new();
                light_cache.update_visible(None, |section_pos| {
                    updated_sections.push(section_pos);
                });
                Ok(SkyLightUpdateResult { updated_sections })
            })
        })
    })
}

/// ScalableLux-style sky-light propagation over scoped Steel light caches.
pub struct SkyLightPropagationContext<'a, 'sections, 'light> {
    layout: LightCacheLayout,
    sections: &'a LightSectionReadCache<'sections>,
    light: &'a mut LightLayerWriteCache<'light>,
    queues: &'a mut PackedLightPropagationQueues,
    null_section_checked: Vec<bool>,
}

impl<'a, 'sections, 'light> SkyLightPropagationContext<'a, 'sections, 'light> {
    /// Creates a sky-light propagation context from matching scoped caches.
    pub fn new(
        sections: &'a LightSectionReadCache<'sections>,
        light: &'a mut LightLayerWriteCache<'light>,
        queues: &'a mut PackedLightPropagationQueues,
    ) -> Result<Self, SkyLightPropagationContextError> {
        if light.layer() != LightLayer::Sky {
            return Err(SkyLightPropagationContextError::WrongLayer {
                layer: light.layer(),
            });
        }

        if sections.layout() != light.layout() {
            return Err(SkyLightPropagationContextError::LayoutMismatch {
                section_layout: sections.layout(),
                light_layout: light.layout(),
            });
        }

        let layout = light.layout();
        let section_count = layout.range().section_count();

        Ok(Self {
            layout,
            sections,
            light,
            queues,
            null_section_checked: vec![false; section_count],
        })
    }

    /// Initializes the sky nibbles required around non-empty center sections.
    pub fn handle_unlit_empty_section_changes(&mut self, chunk_pos: ChunkPos) {
        self.initialize_chunk_nibbles(chunk_pos, true);
        self.deinit_and_lazy_init_empty_sections(chunk_pos, true);
    }

    /// Synchronizes sky nibbles for an already-lit loaded chunk without resetting light data.
    pub fn handle_loaded_empty_section_changes(&mut self, chunk_pos: ChunkPos) {
        self.initialize_chunk_nibbles(chunk_pos, false);
        self.deinit_and_lazy_init_empty_sections(chunk_pos, false);
    }

    fn initialize_chunk_nibbles(&mut self, chunk_pos: ChunkPos, unlit: bool) {
        for section_y in (self.layout.range().min_chunk_section_y()
            ..self.layout.range().max_chunk_section_y_exclusive())
            .rev()
        {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.section_is_non_empty(section_pos) {
                continue;
            }

            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    let extrude = (offset_x | offset_z) != 0 || !unlit;
                    for offset_y in (-1..=1).rev() {
                        self.init_nibble(
                            SectionPos::new(
                                chunk_pos.0.x + offset_x,
                                section_y + offset_y,
                                chunk_pos.0.y + offset_z,
                            ),
                            extrude,
                            false,
                        );
                    }
                }
            }
        }
    }

    fn deinit_and_lazy_init_empty_sections(&mut self, chunk_pos: ChunkPos, unlit: bool) {
        for offset_z in -1..=1 {
            for offset_x in -1..=1 {
                let target_chunk =
                    ChunkPos::new(chunk_pos.0.x + offset_x, chunk_pos.0.y + offset_z);

                for section_y in (self.layout.range().min_section_y()
                    ..self.layout.range().max_section_y_exclusive())
                    .rev()
                {
                    let section_pos =
                        SectionPos::new(target_chunk.0.x, section_y, target_chunk.0.y);
                    match self.section_neighborhood_all_empty_if_known(target_chunk, section_y) {
                        Some(true) => {
                            self.light.set_section_null(section_pos);
                        }
                        Some(false) => {
                            self.init_nibble(
                                section_pos,
                                (offset_x | offset_z) != 0 || !unlit,
                                false,
                            );
                        }
                        None => {
                            if !self.section_neighborhood_all_empty(target_chunk, section_y) {
                                self.init_nibble(
                                    section_pos,
                                    (offset_x | offset_z) != 0 || !unlit,
                                    false,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn section_neighborhood_all_empty(&self, chunk_pos: ChunkPos, section_y: i32) -> bool {
        for offset_y in -1..=1 {
            let neighbor_y = section_y + offset_y;
            if neighbor_y < self.layout.range().min_chunk_section_y()
                || neighbor_y >= self.layout.range().max_chunk_section_y_exclusive()
            {
                continue;
            }

            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    let section_pos = SectionPos::new(
                        chunk_pos.0.x + offset_x,
                        neighbor_y,
                        chunk_pos.0.y + offset_z,
                    );
                    if let Some(empty) = self.sections.section_empty(section_pos) {
                        if !empty {
                            return false;
                        }
                    } else if let Some(empty) = self.light.section_empty(section_pos) {
                        if !empty {
                            return false;
                        }
                    } else if self.sections.has_non_empty_section(section_pos) {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn section_neighborhood_all_empty_if_known(
        &self,
        chunk_pos: ChunkPos,
        section_y: i32,
    ) -> Option<bool> {
        for offset_y in -1..=1 {
            let neighbor_y = section_y + offset_y;
            if neighbor_y < self.layout.range().min_chunk_section_y()
                || neighbor_y >= self.layout.range().max_chunk_section_y_exclusive()
            {
                continue;
            }

            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    let section_pos = SectionPos::new(
                        chunk_pos.0.x + offset_x,
                        neighbor_y,
                        chunk_pos.0.y + offset_z,
                    );
                    let empty = self.sections.section_empty(section_pos)?;
                    if !empty {
                        return Some(false);
                    }
                }
            }
        }

        Some(true)
    }

    /// Resets the center chunk to ScalableLux's fresh all-null lighting state.
    pub fn reset_center_chunk_nibbles(&mut self) {
        self.light
            .reset_chunk_nibbles_to_null(self.layout.center_chunk());
    }

    /// Runs sky chunk lighting with the selected ScalableLux edge-check mode.
    pub fn light_chunk(&mut self, chunk_pos: ChunkPos, edge_checks: SkyLightChunkEdgeChecks) {
        self.light.rewrite_null_nibbles_for_skylight();
        self.null_section_checked.fill(false);

        let min_section = self.layout.range().min_chunk_section_y();
        let mut highest_non_empty_section = self.layout.range().max_chunk_section_y_exclusive() - 1;

        loop {
            let section_pos =
                SectionPos::new(chunk_pos.0.x, highest_non_empty_section, chunk_pos.0.y);
            if highest_non_empty_section != min_section - 1
                && self.sections.has_non_empty_section(section_pos)
            {
                break;
            }

            self.check_null_section(chunk_pos, highest_non_empty_section, false);
            self.propagate_full_empty_section_edges(chunk_pos, highest_non_empty_section);

            if highest_non_empty_section == min_section - 1 {
                highest_non_empty_section -= 1;
                break;
            }
            highest_non_empty_section -= 1;
        }

        if highest_non_empty_section >= min_section {
            self.propagate_sky_sources_from_top(chunk_pos, highest_non_empty_section);
        }

        match edge_checks {
            SkyLightChunkEdgeChecks::Required => {
                self.perform_light_increase();
                for section_y in
                    (self.layout.range().min_section_y()..=highest_non_empty_section).rev()
                {
                    self.check_null_section(chunk_pos, section_y, false);
                }
                self.check_chunk_edges(
                    chunk_pos,
                    self.layout.range().min_section_y(),
                    highest_non_empty_section,
                );
            }
            SkyLightChunkEdgeChecks::Skipped => {
                for section_y in
                    (self.layout.range().min_section_y()..=highest_non_empty_section).rev()
                {
                    self.check_null_section(chunk_pos, section_y, false);
                }
                self.propagate_neighbor_levels(
                    chunk_pos,
                    self.layout.range().min_section_y(),
                    highest_non_empty_section,
                );
                self.perform_light_increase();
            }
        }
    }

    /// Handles one sky-light opacity change, matching ScalableLux `checkBlock`.
    ///
    /// Returns false when the changed block is outside this cache window.
    pub fn check_block(&mut self, block_pos: BlockPos) -> bool {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return false;
        };

        let current_level = self.light.get_updating(cached_block);
        if current_level == MAX_LIGHT_LEVEL {
            self.enqueue_increase(
                block_pos,
                current_level,
                LightDirectionSet::all(),
                LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS),
            );
        } else {
            self.light.set(cached_block, 0);
        }

        self.enqueue_decrease(
            block_pos,
            current_level,
            LightDirectionSet::all(),
            LightQueueFlags::EMPTY,
        );
        true
    }

    /// Handles sky-light source and opacity changes for blocks in the center chunk.
    pub fn propagate_block_changes(&mut self, positions: &[BlockPos]) {
        self.light.rewrite_null_nibbles_for_skylight();
        self.null_section_checked.fill(false);

        let chunk_pos = self.layout.center_chunk();
        self.initialize_changed_section_nibbles(chunk_pos, positions);

        let mut changed_column_max_y = [i32::MIN; 16 * 16];
        for position in positions {
            if SectionPos::block_to_section_coord(position.x()) != chunk_pos.0.x
                || SectionPos::block_to_section_coord(position.z()) != chunk_pos.0.y
            {
                continue;
            }

            let index = ((position.x() & 15) | ((position.z() & 15) << 4)) as usize;
            changed_column_max_y[index] = changed_column_max_y[index].max(position.y());
        }

        let mut delayed_increases = Vec::new();
        let mut delayed_decreases = Vec::new();
        for (index, max_y) in changed_column_max_y.into_iter().enumerate() {
            if max_y == i32::MIN {
                continue;
            }

            let x = (chunk_pos.0.x << 4) | (index as i32 & 15);
            let z = (chunk_pos.0.y << 4) | ((index as i32 >> 4) & 15);
            let max_propagation_y =
                self.try_propagate_skylight_delayed(x, max_y, z, true, &mut delayed_increases);
            self.remove_sky_sources_below(x, max_propagation_y, z, &mut delayed_decreases);
        }

        self.process_delayed_increases(&delayed_increases);
        self.process_delayed_decreases(&delayed_decreases);

        for position in positions {
            self.check_block(*position);
        }

        self.perform_light_decrease();
    }

    /// Calculates the sky-light value that should exist at `block_pos`.
    ///
    /// Returns `None` when the position is outside this cache window.
    #[must_use]
    pub fn calculate_light_value(&self, block_pos: BlockPos, expect: u8) -> Option<u8> {
        if expect == MAX_LIGHT_LEVEL {
            return Some(expect);
        }

        let cached_block = self.layout.cached_block(block_pos)?;
        let center_state = self.sections.get_block_state(cached_block);
        let opacity = get_light_opacity(center_state);
        let mut level = 0;

        for axis_direction in LightAxisDirection::ALL {
            let neighbor_pos = Self::offset(block_pos, axis_direction);
            let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                continue;
            };
            let neighbor_level = self.light.get_updating(neighbor_block);
            if neighbor_level.saturating_sub(1) <= level {
                continue;
            }

            let neighbor_state = self.sections.get_block_state(neighbor_block);
            if get_light_block_into(
                neighbor_state,
                center_state,
                axis_direction.opposite().direction(),
                opacity,
            ) == LIGHT_BLOCKED
            {
                continue;
            }

            level = level.max(neighbor_level.saturating_sub(opacity));
            if level > expect {
                return Some(level);
            }
        }

        Some(level)
    }

    fn init_nibble(&mut self, section_pos: SectionPos, extrude: bool, init_removed: bool) {
        if self.layout.section_slot(section_pos).is_none()
            || (!self.light.has_cached_section(section_pos)
                && (!init_removed || !self.light.materialize_removed_null_section(section_pos)))
        {
            return;
        }
        if !self.light.is_section_null_updating(section_pos) {
            return;
        }

        let mut highest_non_empty_section = self.layout.range().min_section_y() - 1;
        for section_y in (self.layout.range().min_chunk_section_y()
            ..self.layout.range().max_chunk_section_y_exclusive())
            .rev()
        {
            let candidate = SectionPos::new(section_pos.x(), section_y, section_pos.z());
            if self.section_is_non_empty(candidate) {
                highest_non_empty_section = section_y;
                break;
            }
        }

        if section_pos.y() > highest_non_empty_section {
            self.light.set_section_non_null(section_pos);
            self.light.fill_section(section_pos, MAX_LIGHT_LEVEL);
        } else if extrude {
            self.light
                .extrude_lower_from_first_section_above(section_pos);
        } else {
            self.light.set_section_non_null(section_pos);
        }
    }

    fn section_is_non_empty(&self, section_pos: SectionPos) -> bool {
        if let Some(empty) = self.sections.section_empty(section_pos) {
            return !empty;
        }

        if let Some(empty) = self.light.section_empty(section_pos) {
            return !empty;
        }

        self.sections.has_non_empty_section(section_pos)
    }

    /// Initializes null sections from the horizontal 1-radius neighbors that ScalableLux materializes.
    fn check_null_section(
        &mut self,
        chunk_pos: ChunkPos,
        section_y: i32,
        extrude_initialized: bool,
    ) -> bool {
        let Some(section_index) = self.layout.range().section_index(section_y) else {
            return false;
        };
        if self.null_section_checked[section_index] {
            return false;
        }
        self.null_section_checked[section_index] = true;

        let center_section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
        let mut need_init_neighbors = self.light.has_non_null_section(center_section_pos);
        if !need_init_neighbors {
            'neighbor_search: for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    let section_pos = SectionPos::new(
                        chunk_pos.0.x + offset_x,
                        section_y,
                        chunk_pos.0.y + offset_z,
                    );
                    if self.light.has_non_null_section(section_pos) {
                        need_init_neighbors = true;
                        break 'neighbor_search;
                    }
                }
            }
        }

        if need_init_neighbors {
            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    self.init_nibble(
                        SectionPos::new(
                            chunk_pos.0.x + offset_x,
                            section_y,
                            chunk_pos.0.y + offset_z,
                        ),
                        if (offset_x | offset_z) == 0 {
                            extrude_initialized
                        } else {
                            true
                        },
                        true,
                    );
                }
            }
        }

        need_init_neighbors
    }

    fn propagate_full_empty_section_edges(&mut self, chunk_pos: ChunkPos, section_y: i32) {
        for direction in LightAxisDirection::HORIZONTAL {
            let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
            let neighbor_section_pos = SectionPos::new(
                chunk_pos.0.x + neighbor_offset_x,
                section_y,
                chunk_pos.0.y + neighbor_offset_z,
            );
            if !self.light.has_non_null_section(neighbor_section_pos) {
                continue;
            }

            let (increment_x, increment_z, start_x, start_z) =
                Self::current_edge_scan(chunk_pos, direction);
            let directions = LightDirectionSet::only(direction);
            let min_y = section_y << 4;
            let max_y = min_y | 15;
            for y in min_y..=max_y {
                let mut x = start_x;
                let mut z = start_z;
                for _ in 0..16 {
                    self.enqueue_increase(
                        BlockPos::new(x, y, z),
                        MAX_LIGHT_LEVEL,
                        directions,
                        LightQueueFlags::EMPTY,
                    );
                    x += increment_x;
                    z += increment_z;
                }
            }
        }
    }

    fn propagate_sky_sources_from_top(&mut self, chunk_pos: ChunkPos, highest_section: i32) {
        let section_min_x = chunk_pos.0.x << 4;
        let section_min_z = chunk_pos.0.y << 4;
        let start_y = (highest_section << 4) | 15;

        for z in 0..super::CHUNK_EDGE {
            for x in 0..super::CHUNK_EDGE {
                self.try_propagate_skylight_inner(
                    section_min_x + x as i32,
                    start_y + 1,
                    section_min_z + z as i32,
                    false,
                    None,
                );
            }
        }
    }

    fn try_propagate_skylight_delayed(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        extrude_initialized: bool,
        delayed_increases: &mut Vec<PackedLightQueueEntry>,
    ) -> i32 {
        self.try_propagate_skylight_inner(x, y, z, extrude_initialized, Some(delayed_increases))
    }

    fn try_propagate_skylight_inner(
        &mut self,
        x: i32,
        mut y: i32,
        z: i32,
        extrude_initialized: bool,
        mut delayed_increases: Option<&mut Vec<PackedLightQueueEntry>>,
    ) -> i32 {
        if self.get_light_level_extruded(BlockPos::new(x, y + 1, z)) != MAX_LIGHT_LEVEL {
            return y;
        }

        self.check_null_section(
            ChunkPos::new(
                SectionPos::block_to_section_coord(x),
                SectionPos::block_to_section_coord(z),
            ),
            SectionPos::block_to_section_coord(y),
            extrude_initialized,
        );

        let mut above_state = self.block_state(BlockPos::new(x, y + 1, z));
        while y >= (self.layout.range().min_section_y() << 4) {
            if (y & 15) == 15 {
                self.check_null_section(
                    ChunkPos::new(
                        SectionPos::block_to_section_coord(x),
                        SectionPos::block_to_section_coord(z),
                    ),
                    SectionPos::block_to_section_coord(y),
                    extrude_initialized,
                );
            }

            let current_pos = BlockPos::new(x, y, z);
            let current_state = self.block_state(current_pos);
            let opacity = current_state.get_light_dampening();
            if get_light_block_into(above_state, current_state, Direction::Down, opacity)
                == LIGHT_BLOCKED
                || opacity > 0
            {
                break;
            }

            let section_pos = SectionPos::from_block_pos(current_pos);
            if !self.light.has_non_null_section(section_pos) {
                y &= !15;
                above_state = Self::air();
            } else {
                let Some(cached_block) = self.layout.cached_block(current_pos) else {
                    break;
                };
                let increase_entry = self.enqueue_increase(
                    current_pos,
                    MAX_LIGHT_LEVEL,
                    LightDirectionSet::all_except(LightAxisDirection::PositiveY),
                    Self::shape_flags(current_state),
                );
                above_state = current_state;

                if let Some(delayed_increases) = delayed_increases.as_deref_mut() {
                    if let Some(entry) = increase_entry {
                        delayed_increases.push(entry);
                    }
                } else {
                    self.light.set(cached_block, MAX_LIGHT_LEVEL);
                }
            }

            y -= 1;
        }

        y
    }

    fn initialize_changed_section_nibbles(&mut self, chunk_pos: ChunkPos, positions: &[BlockPos]) {
        let mut section_ys = Vec::new();
        for position in positions {
            if SectionPos::block_to_section_coord(position.x()) != chunk_pos.0.x
                || SectionPos::block_to_section_coord(position.z()) != chunk_pos.0.y
            {
                continue;
            }

            let section_y = SectionPos::block_to_section_coord(position.y());
            if !section_ys.contains(&section_y) {
                section_ys.push(section_y);
            }
        }

        for section_y in section_ys {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.sections.has_non_empty_section(section_pos) {
                continue;
            }

            for offset_z in -1..=1 {
                for offset_x in -1..=1 {
                    for offset_y in (-1..=1).rev() {
                        self.init_nibble(
                            SectionPos::new(
                                chunk_pos.0.x + offset_x,
                                section_y + offset_y,
                                chunk_pos.0.y + offset_z,
                            ),
                            true,
                            false,
                        );
                    }
                }
            }
        }
    }

    fn remove_sky_sources_below(
        &mut self,
        x: i32,
        mut y: i32,
        z: i32,
        delayed_decreases: &mut Vec<PackedLightQueueEntry>,
    ) {
        if self.get_light_level_extruded(BlockPos::new(x, y, z)) != MAX_LIGHT_LEVEL {
            return;
        }

        let min_y = self.layout.range().min_section_y() << 4;
        while y >= min_y {
            if (y & 15) == 15 {
                self.check_null_section(
                    ChunkPos::new(
                        SectionPos::block_to_section_coord(x),
                        SectionPos::block_to_section_coord(z),
                    ),
                    SectionPos::block_to_section_coord(y),
                    true,
                );
            }

            let current_pos = BlockPos::new(x, y, z);
            let section_pos = SectionPos::from_block_pos(current_pos);
            if !self.light.has_non_null_section(section_pos) {
                y &= !15;
                y -= 1;
                continue;
            }

            let Some(cached_block) = self.layout.cached_block(current_pos) else {
                break;
            };
            if self.light.get_updating(cached_block) != MAX_LIGHT_LEVEL {
                break;
            }

            if let Some(entry) = self.enqueue_decrease(
                current_pos,
                MAX_LIGHT_LEVEL,
                LightDirectionSet::all_except(LightAxisDirection::PositiveY),
                LightQueueFlags::EMPTY,
            ) {
                delayed_decreases.push(entry);
            }
            y -= 1;
        }
    }

    fn process_delayed_increases(&mut self, entries: &[PackedLightQueueEntry]) {
        for entry in entries {
            let Some(source_block) = self.cached_block_from_entry(*entry) else {
                continue;
            };
            self.light.set(source_block, entry.level());
        }
    }

    fn process_delayed_decreases(&mut self, entries: &[PackedLightQueueEntry]) {
        for entry in entries {
            let Some(source_block) = self.cached_block_from_entry(*entry) else {
                continue;
            };
            self.light.set(source_block, 0);
        }
    }

    fn get_light_level_extruded(&self, block_pos: BlockPos) -> u8 {
        let mut section_y = SectionPos::block_to_section_coord(block_pos.y());
        let section_x = SectionPos::block_to_section_coord(block_pos.x());
        let section_z = SectionPos::block_to_section_coord(block_pos.z());

        if let Some(cached_block) = self.layout.cached_block(block_pos)
            && self
                .light
                .has_non_null_section(SectionPos::new(section_x, section_y, section_z))
        {
            return self.light.get_updating(cached_block);
        }

        loop {
            section_y += 1;
            if section_y >= self.layout.range().max_section_y_exclusive() {
                return MAX_LIGHT_LEVEL;
            }

            let section_pos = SectionPos::new(section_x, section_y, section_z);
            if !self.light.has_non_null_section(section_pos) {
                continue;
            }
            let block_above = BlockPos::new(block_pos.x(), section_y << 4, block_pos.z());
            let Some(cached_block) = self.layout.cached_block(block_above) else {
                continue;
            };
            return self.light.get_updating(cached_block);
        }
    }

    fn propagate_neighbor_levels(
        &mut self,
        chunk_pos: ChunkPos,
        from_section: i32,
        to_section: i32,
    ) {
        for section_y in (from_section..=to_section).rev() {
            let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
            if !self.light.has_non_null_section(section_pos) {
                continue;
            }

            for direction in LightAxisDirection::HORIZONTAL {
                self.propagate_neighbor_level_section(chunk_pos, section_y, direction);
            }
        }
    }

    fn propagate_neighbor_level_section(
        &mut self,
        chunk_pos: ChunkPos,
        section_y: i32,
        direction: LightAxisDirection,
    ) {
        let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
        let neighbor_section_pos = SectionPos::new(
            chunk_pos.0.x + neighbor_offset_x,
            section_y,
            chunk_pos.0.y + neighbor_offset_z,
        );
        if !self
            .light
            .is_section_initialized_updating(neighbor_section_pos)
        {
            return;
        }

        let (increment_x, increment_z, start_x, start_z) =
            Self::neighbor_edge_scan(chunk_pos, direction);
        let directions = LightDirectionSet::only(direction.opposite());
        let flags = LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS);

        let min_y = section_y << 4;
        let max_y = min_y | 15;
        for y in min_y..=max_y {
            let mut x = start_x;
            let mut z = start_z;
            for _ in 0..16 {
                let source_pos = BlockPos::new(x, y, z);
                let Some(source_block) = self.layout.cached_block(source_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };
                let level = self.light.get_updating(source_block);
                if level > 1 {
                    self.enqueue_increase(source_pos, level, directions, flags);
                }
                x += increment_x;
                z += increment_z;
            }
        }
    }

    fn check_chunk_edges(&mut self, chunk_pos: ChunkPos, from_section: i32, to_section: i32) {
        for section_y in (from_section..=to_section).rev() {
            self.check_chunk_edge(chunk_pos, section_y);
        }

        self.perform_light_decrease();
    }

    fn check_chunk_edge(&mut self, chunk_pos: ChunkPos, section_y: i32) {
        let current_section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);
        if !self.light.has_non_null_section(current_section_pos) {
            return;
        }

        for direction in LightAxisDirection::HORIZONTAL {
            let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
            let neighbor_chunk_pos = ChunkPos::new(
                chunk_pos.0.x + neighbor_offset_x,
                chunk_pos.0.y + neighbor_offset_z,
            );
            let neighbor_section_pos =
                SectionPos::new(neighbor_chunk_pos.0.x, section_y, neighbor_chunk_pos.0.y);
            if !self.light.has_non_null_section(neighbor_section_pos) {
                continue;
            }
            if !self
                .light
                .is_section_initialized_updating(current_section_pos)
                && !self
                    .light
                    .is_section_initialized_updating(neighbor_section_pos)
            {
                continue;
            }

            self.check_chunk_edge_direction(chunk_pos, neighbor_chunk_pos, section_y, direction);
        }
    }

    fn check_chunk_edge_direction(
        &mut self,
        chunk_pos: ChunkPos,
        neighbor_chunk_pos: ChunkPos,
        section_y: i32,
        direction: LightAxisDirection,
    ) {
        let (neighbor_offset_x, _, neighbor_offset_z) = direction.offset();
        let (increment_x, increment_z, start_x, start_z) =
            Self::current_edge_scan(chunk_pos, direction);
        let mut center_delayed_checks = [0usize; 16 * 16];
        let mut neighbor_delayed_checks = [0usize; 16 * 16];
        let mut center_delayed_check_count = 0;
        let mut neighbor_delayed_check_count = 0;

        let min_y = section_y << 4;
        let max_y = min_y | 15;
        for y in min_y..=max_y {
            let mut x = start_x;
            let mut z = start_z;
            for _ in 0..16 {
                let current_pos = BlockPos::new(x, y, z);
                let neighbor_pos = BlockPos::new(x + neighbor_offset_x, y, z + neighbor_offset_z);
                let Some(current_block) = self.layout.cached_block(current_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    x += increment_x;
                    z += increment_z;
                    continue;
                };

                let current_level = self.light.get_updating(current_block);
                if self
                    .calculate_light_value(current_pos, current_level)
                    .is_some_and(|calculated| calculated != current_level)
                {
                    center_delayed_checks[center_delayed_check_count] = current_block.local_index;
                    center_delayed_check_count += 1;
                }

                let neighbor_level = self.light.get_updating(neighbor_block);
                if self
                    .calculate_light_value(neighbor_pos, neighbor_level)
                    .is_some_and(|calculated| calculated != neighbor_level)
                {
                    neighbor_delayed_checks[neighbor_delayed_check_count] =
                        neighbor_block.local_index;
                    neighbor_delayed_check_count += 1;
                }

                x += increment_x;
                z += increment_z;
            }
        }

        let current_chunk_offset_x = chunk_pos.0.x << 4;
        let current_chunk_offset_z = chunk_pos.0.y << 4;
        let neighbor_chunk_offset_x = neighbor_chunk_pos.0.x << 4;
        let neighbor_chunk_offset_z = neighbor_chunk_pos.0.y << 4;
        let chunk_offset_y = section_y << 4;
        let delayed_check_count = center_delayed_check_count.max(neighbor_delayed_check_count);
        for delayed_check_index in 0..delayed_check_count {
            if delayed_check_index < center_delayed_check_count {
                let local_index = center_delayed_checks[delayed_check_index];
                self.check_block(Self::block_pos_from_local_index(
                    current_chunk_offset_x,
                    chunk_offset_y,
                    current_chunk_offset_z,
                    local_index,
                ));
            }
            if delayed_check_index < neighbor_delayed_check_count {
                let local_index = neighbor_delayed_checks[delayed_check_index];
                self.check_block(Self::block_pos_from_local_index(
                    neighbor_chunk_offset_x,
                    chunk_offset_y,
                    neighbor_chunk_offset_z,
                    local_index,
                ));
            }
        }
    }

    fn perform_light_increase(&mut self) {
        while let Some(entry) = self.queues.dequeue_increase() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            if entry.should_recheck_level() {
                if self.light.get_updating(source_block) != entry.level() {
                    continue;
                }
            } else if entry.should_write_level() {
                self.light.set(source_block, entry.level());
            }

            let source_state = if entry.has_sided_transparent_blocks() {
                Some(self.sections.get_block_state(source_block))
            } else {
                None
            };

            for axis_direction in entry.directions().directions() {
                let neighbor_pos = Self::offset(source_block.block_pos, axis_direction);
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    continue;
                };
                if !self.light.has_non_null_updating(neighbor_block) {
                    continue;
                }
                let current_level = self.light.get_updating(neighbor_block);
                if current_level >= entry.level().saturating_sub(1) {
                    continue;
                }

                let neighbor_state = self.sections.get_block_state(neighbor_block);
                let Some((target_level, flags)) = Self::target_level(
                    entry.level(),
                    source_state,
                    neighbor_state,
                    axis_direction.direction(),
                ) else {
                    continue;
                };
                if target_level <= current_level {
                    continue;
                }

                self.light.set(neighbor_block, target_level);
                if target_level > 1 {
                    self.enqueue_increase(
                        neighbor_pos,
                        target_level,
                        LightDirectionSet::all_except_opposite(axis_direction),
                        flags,
                    );
                }
            }
        }
    }

    fn perform_light_decrease(&mut self) {
        while let Some(entry) = self.queues.dequeue_decrease() {
            let Some(source_block) = self.cached_block_from_entry(entry) else {
                continue;
            };
            let source_state = if entry.has_sided_transparent_blocks() {
                Some(self.sections.get_block_state(source_block))
            } else {
                None
            };

            for axis_direction in entry.directions().directions() {
                let neighbor_pos = Self::offset(source_block.block_pos, axis_direction);
                let Some(neighbor_block) = self.layout.cached_block(neighbor_pos) else {
                    continue;
                };
                if !self.light.has_non_null_updating(neighbor_block) {
                    continue;
                }
                let current_level = self.light.get_updating(neighbor_block);
                if current_level == 0 {
                    continue;
                }

                let neighbor_state = self.sections.get_block_state(neighbor_block);
                let Some((target_level, flags)) = Self::target_level_saturating(
                    entry.level(),
                    source_state,
                    neighbor_state,
                    axis_direction.direction(),
                ) else {
                    continue;
                };

                if current_level > target_level {
                    self.enqueue_increase(
                        neighbor_pos,
                        current_level,
                        LightDirectionSet::all(),
                        flags.with(LightQueueFlags::RECHECK_LEVEL),
                    );
                    continue;
                }

                self.light.set(neighbor_block, 0);
                if target_level > 0 {
                    self.enqueue_decrease(
                        neighbor_pos,
                        target_level,
                        LightDirectionSet::all_except_opposite(axis_direction),
                        flags,
                    );
                }
            }
        }

        self.perform_light_increase();
    }

    fn target_level(
        propagated_level: u8,
        source_state: Option<steel_utils::BlockStateId>,
        target_state: steel_utils::BlockStateId,
        direction: Direction,
    ) -> Option<(u8, LightQueueFlags)> {
        let source_state = source_state.unwrap_or_else(Self::air);
        let opacity = get_light_block_into(
            source_state,
            target_state,
            direction,
            get_light_opacity(target_state),
        );
        if opacity == LIGHT_BLOCKED || opacity >= propagated_level {
            return None;
        }

        Some((propagated_level - opacity, Self::shape_flags(target_state)))
    }

    fn target_level_saturating(
        propagated_level: u8,
        source_state: Option<steel_utils::BlockStateId>,
        target_state: steel_utils::BlockStateId,
        direction: Direction,
    ) -> Option<(u8, LightQueueFlags)> {
        let source_state = source_state.unwrap_or_else(Self::air);
        let opacity = get_light_block_into(
            source_state,
            target_state,
            direction,
            get_light_opacity(target_state),
        );
        if opacity == LIGHT_BLOCKED {
            return None;
        }

        Some((
            propagated_level.saturating_sub(opacity),
            Self::shape_flags(target_state),
        ))
    }

    fn cached_block_from_entry(&self, entry: PackedLightQueueEntry) -> Option<CachedLightBlock> {
        self.layout.cached_block_from_packed(entry.block_pos())
    }

    fn enqueue_decrease(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) -> Option<PackedLightQueueEntry> {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return None;
        };
        let entry = PackedLightQueueEntry::from_parts(packed_pos, level, directions, flags);
        self.queues.enqueue_decrease(entry);
        Some(entry)
    }

    fn enqueue_increase(
        &mut self,
        block_pos: BlockPos,
        level: u8,
        directions: LightDirectionSet,
        flags: LightQueueFlags,
    ) -> Option<PackedLightQueueEntry> {
        let Some(packed_pos) = self.layout.encode_block_pos(block_pos) else {
            return None;
        };
        let entry = PackedLightQueueEntry::from_parts(packed_pos, level, directions, flags);
        self.queues.enqueue_increase(entry);
        Some(entry)
    }

    fn block_state(&self, block_pos: BlockPos) -> steel_utils::BlockStateId {
        let Some(cached_block) = self.layout.cached_block(block_pos) else {
            return Self::air();
        };
        self.sections.get_block_state(cached_block)
    }

    fn current_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                chunk_pos.0.x << 4
            } else {
                (chunk_pos.0.x << 4) | 15
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            chunk_pos.0.y << 4
        } else {
            (chunk_pos.0.y << 4) | 15
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
    }

    fn neighbor_edge_scan(
        chunk_pos: ChunkPos,
        direction: LightAxisDirection,
    ) -> (i32, i32, i32, i32) {
        let (offset_x, _, offset_z) = direction.offset();
        if offset_x != 0 {
            let start_x = if offset_x < 0 {
                (chunk_pos.0.x << 4) - 1
            } else {
                (chunk_pos.0.x << 4) + 16
            };
            return (0, 1, start_x, chunk_pos.0.y << 4);
        }

        let start_z = if offset_z < 0 {
            (chunk_pos.0.y << 4) - 1
        } else {
            (chunk_pos.0.y << 4) + 16
        };
        (1, 0, chunk_pos.0.x << 4, start_z)
    }

    fn block_pos_from_local_index(
        chunk_offset_x: i32,
        chunk_offset_y: i32,
        chunk_offset_z: i32,
        local_index: usize,
    ) -> BlockPos {
        BlockPos::new(
            chunk_offset_x | (local_index & 15) as i32,
            chunk_offset_y | (local_index >> 8) as i32,
            chunk_offset_z | ((local_index >> 4) & 15) as i32,
        )
    }

    fn shape_flags(block_state: steel_utils::BlockStateId) -> LightQueueFlags {
        if light_occlusion_shape(block_state).is_empty() {
            LightQueueFlags::EMPTY
        } else {
            LightQueueFlags::EMPTY.with(LightQueueFlags::HAS_SIDED_TRANSPARENT_BLOCKS)
        }
    }

    fn offset(block_pos: BlockPos, direction: LightAxisDirection) -> BlockPos {
        let (dx, dy, dz) = direction.offset();
        block_pos.offset(dx, dy, dz)
    }

    fn air() -> steel_utils::BlockStateId {
        vanilla_blocks::AIR.default_state()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::types::UpdateFlags;

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::chunk::{
        chunk_access::{ChunkAccess, ChunkStatus},
        chunk_holder::ChunkHolder,
        light::{LightCacheSetupRadius, LightNibbleState, LightSectionRange, LightWorkset},
        proto_chunk::ProtoChunk,
        section::{ChunkSection, Sections},
    };

    fn init_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn range() -> LightSectionRange {
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        range
    }

    fn holder_with_section(pos: ChunkPos, section: ChunkSection) -> Arc<ChunkHolder> {
        let sections = Sections::from_owned(vec![section].into_boxed_slice());
        let proto = ProtoChunk::new(sections, pos, 0, 16, Weak::new());
        proto.initialize_light_sources();
        let holder = Arc::new(ChunkHolder::new(pos, 0, 0, 16));
        holder.insert_chunk(ChunkAccess::Proto(proto), ChunkStatus::Light);
        holder
    }

    fn initialize_holder_light(holder: &ChunkHolder) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        chunk.initialize_light_sources();
    }

    fn holder_with_sections(pos: ChunkPos, sections: Vec<ChunkSection>) -> Arc<ChunkHolder> {
        let height = (sections.len() * 16) as i32;
        let proto = ProtoChunk::new(
            Sections::from_owned(sections.into_boxed_slice()),
            pos,
            0,
            height,
            Weak::new(),
        );
        proto.initialize_light_sources();
        let holder = Arc::new(ChunkHolder::new(pos, 0, 0, height));
        holder.insert_chunk(ChunkAccess::Proto(proto), ChunkStatus::Light);
        holder
    }

    fn empty_holder_with_section_count(pos: ChunkPos, section_count: usize) -> Arc<ChunkHolder> {
        holder_with_sections(
            pos,
            (0..section_count)
                .map(|_| ChunkSection::new_empty())
                .collect(),
        )
    }

    fn horizontal_empty_neighbors(
        center: ChunkPos,
        section_count: usize,
    ) -> Vec<(ChunkPos, Arc<ChunkHolder>)> {
        [
            ChunkPos::new(center.0.x, center.0.y - 1),
            ChunkPos::new(center.0.x, center.0.y + 1),
            ChunkPos::new(center.0.x - 1, center.0.y),
            ChunkPos::new(center.0.x + 1, center.0.y),
        ]
        .into_iter()
        .map(|pos| (pos, empty_holder_with_section_count(pos, section_count)))
        .collect()
    }

    fn roofed_holder(
        pos: ChunkPos,
        section_count: usize,
        roof_section_index: usize,
        roof_local_y: usize,
    ) -> Arc<ChunkHolder> {
        let mut sections = (0..section_count)
            .map(|_| ChunkSection::new_empty())
            .collect::<Vec<_>>();
        for z in 0..16 {
            for x in 0..16 {
                sections[roof_section_index].set_block_state(
                    x,
                    roof_local_y,
                    z,
                    vanilla_blocks::STONE.default_state(),
                );
            }
        }
        holder_with_sections(pos, sections)
    }

    fn roofed_holder_square(
        center: ChunkPos,
        radius: i32,
        section_count: usize,
        roof_section_index: usize,
        roof_local_y: usize,
    ) -> Vec<(ChunkPos, Arc<ChunkHolder>)> {
        let mut holders = Vec::new();
        for z in -radius..=radius {
            for x in -radius..=radius {
                let pos = ChunkPos::new(center.0.x + x, center.0.y + z);
                holders.push((
                    pos,
                    roofed_holder(pos, section_count, roof_section_index, roof_local_y),
                ));
            }
        }
        holders
    }

    fn find_holder(
        holders: &[(ChunkPos, Arc<ChunkHolder>)],
        pos: ChunkPos,
    ) -> Option<Arc<ChunkHolder>> {
        holders
            .iter()
            .find(|(holder_pos, _)| *holder_pos == pos)
            .map(|(_, holder)| Arc::clone(holder))
    }

    fn set_visible_sky_light(
        holder: &ChunkHolder,
        section_y: i32,
        x: usize,
        y: usize,
        z: usize,
        level: u8,
    ) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let mut light = chunk.light_mut();
        let Some(nibble) = light.sky.nibble_mut(section_y) else {
            panic!("test nibble should be inside light range");
        };
        nibble.set_non_null();
        nibble.set(x, y, z, level);
        assert!(nibble.update_visible());
    }

    #[test]
    fn context_requires_sky_layer() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
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

        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_write_cache(LightLayer::Block, |light_cache| {
                    let mut queues = PackedLightPropagationQueues::new();
                    let result =
                        SkyLightPropagationContext::new(section_cache, light_cache, &mut queues);

                    assert_eq!(
                        result.err(),
                        Some(SkyLightPropagationContextError::WrongLayer {
                            layer: LightLayer::Block,
                        })
                    );
                });
            });
        });
    }

    #[test]
    fn sky_light_chunk_without_edge_checks_propagates_down_air_column() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        section.set_block_state(1, 0, 1, vanilla_blocks::STONE.default_state());
        let holder = holder_with_section(center, section);
        let neighbors = horizontal_empty_neighbors(center, 1);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_chunk_without_edge_checks(&workset) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(top_air) = layout.cached_block(BlockPos::new(1, 15, 1)) else {
            panic!("top air block should be cached");
        };
        let Some(lower_air) = layout.cached_block(BlockPos::new(1, 1, 1)) else {
            panic!("lower air block should be cached");
        };
        let Some(stone) = layout.cached_block(BlockPos::new(1, 0, 1)) else {
            panic!("stone block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(top_air.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(lower_air.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(stone.local_index), 0);
    }

    #[test]
    fn sky_light_chunk_without_edge_checks_propagates_from_virtual_top_source() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        for z in 0..16 {
            for x in 0..16 {
                section.set_block_state(x, 15, z, vanilla_blocks::ACACIA_LEAVES.default_state());
            }
        }
        let holder = holder_with_section(center, section);
        let neighbors = horizontal_empty_neighbors(center, 1);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_chunk_without_edge_checks(&workset) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(top_leaf) = layout.cached_block(BlockPos::new(8, 15, 8)) else {
            panic!("top leaf block should be cached");
        };
        let Some(air_below_leaf) = layout.cached_block(BlockPos::new(8, 14, 8)) else {
            panic!("air below leaf block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(top_leaf.local_index), 14);
        assert_eq!(nibble.get_visible_at_index(air_below_leaf.local_index), 13);
    }

    #[test]
    fn sky_light_chunk_initializes_diagonal_only_null_section_sources() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let diagonal = ChunkPos::new(1, 1);
        let Ok(range) = LightSectionRange::from_world_height(0, 48) else {
            panic!("test height should create a valid light range");
        };

        let mut center_lower = ChunkSection::new_empty();
        center_lower.set_block_state(1, 0, 1, vanilla_blocks::STONE.default_state());
        let center_holder = holder_with_sections(
            center,
            vec![
                center_lower,
                ChunkSection::new_empty(),
                ChunkSection::new_empty(),
            ],
        );
        let diagonal_holder = empty_holder_with_section_count(diagonal, 3);
        set_visible_sky_light(&diagonal_holder, 2, 0, 0, 0, 15);

        let mut holders = Vec::new();
        for z in -1..=1 {
            for x in -1..=1 {
                let pos = ChunkPos::new(center.0.x + x, center.0.y + z);
                if pos == center {
                    holders.push((pos, Arc::clone(&center_holder)));
                } else if pos == diagonal {
                    holders.push((pos, Arc::clone(&diagonal_holder)));
                } else {
                    holders.push((pos, empty_holder_with_section_count(pos, 3)));
                }
            }
        }

        let layout = LightCacheLayout::new(center, range);
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| find_holder(&holders, pos),
            |_| true,
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(result) = propagate_sky_light_chunk_without_edge_checks(&workset) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 1, 0)));

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(top_source_nibble) = light.sky.nibble(1) else {
            panic!("top source sky nibble should be inside light range");
        };
        let Some(above_source_nibble) = light.sky.nibble(2) else {
            panic!("above-source sky nibble should be inside light range");
        };

        assert_eq!(
            top_source_nibble.visible_state(),
            LightNibbleState::Initialized
        );
        assert_eq!(above_source_nibble.visible_state(), LightNibbleState::Null);
    }

    #[test]
    fn sky_light_chunk_without_edge_checks_keeps_sealed_roof_dark() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        for z in 0..16 {
            for x in 0..16 {
                section.set_block_state(x, 15, z, vanilla_blocks::STONE.default_state());
            }
        }
        let holder = holder_with_section(center, section);
        let neighbors = roofed_holder_square(center, 2, 1, 0, 15);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup_with_scopes(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |cached_chunk, _, _| (true, cached_chunk.chunk_pos == center),
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_chunk_without_edge_checks(&workset) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(under_roof) = layout.cached_block(BlockPos::new(8, 14, 8)) else {
            panic!("under-roof block should be cached");
        };
        let Some(roof) = layout.cached_block(BlockPos::new(8, 15, 8)) else {
            panic!("roof block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(under_roof.local_index), 0);
        assert_eq!(nibble.get_visible_at_index(roof.local_index), 0);
    }

    #[test]
    fn sky_light_chunk_resets_stale_center_nibbles_before_lighting() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let mut section = ChunkSection::new_empty();
        for z in 0..16 {
            for x in 0..16 {
                section.set_block_state(x, 15, z, vanilla_blocks::STONE.default_state());
            }
        }
        let holder = holder_with_section(center, section);
        let neighbors = roofed_holder_square(center, 2, 1, 0, 15);
        let layout = LightCacheLayout::new(center, range());

        {
            let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
                panic!("test chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(0) else {
                panic!("test sky nibble should exist");
            };
            nibble.fill(MAX_LIGHT_LEVEL);
            assert!(nibble.update_visible());
        }

        let Ok(workset) = LightWorkset::setup_with_scopes(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |cached_chunk, _, _| (true, cached_chunk.chunk_pos == center),
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required)
        else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test sky nibble should exist");
        };
        let Some(under_roof) = layout.cached_block(BlockPos::new(8, 14, 8)) else {
            panic!("under-roof block should be cached");
        };
        let Some(roof) = layout.cached_block(BlockPos::new(8, 15, 8)) else {
            panic!("roof block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(under_roof.local_index), 0);
        assert_eq!(nibble.get_visible_at_index(roof.local_index), 0);
    }

    #[test]
    fn sky_light_chunk_deinitializes_loaded_empty_neighbor_sections() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let stale_neighbor = ChunkPos::new(1, 0);
        let mut holders = Vec::new();
        let mut stale_holder = None;
        for z in -2..=2 {
            for x in -2..=2 {
                let pos = ChunkPos::new(x, z);
                let holder = holder_with_section(pos, ChunkSection::new_empty());
                initialize_holder_light(&holder);
                if pos == stale_neighbor {
                    stale_holder = Some(Arc::clone(&holder));
                }
                holders.push((pos, holder));
            }
        }
        let Some(stale_holder) = stale_holder else {
            panic!("stale neighbor holder should be created");
        };

        {
            let Some(chunk) = stale_holder.try_chunk(ChunkStatus::Empty) else {
                panic!("stale neighbor chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(0) else {
                panic!("stale neighbor sky nibble should exist");
            };
            nibble.fill(MAX_LIGHT_LEVEL);
            assert!(nibble.update_visible());
        }

        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                holders
                    .iter()
                    .find(|(holder_pos, _)| *holder_pos == pos)
                    .map(|(_, holder)| Arc::clone(holder))
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(_) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        let Some(chunk) = stale_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("stale neighbor chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("stale neighbor sky nibble should exist");
        };

        assert_eq!(nibble.visible_state(), LightNibbleState::Null);
    }

    #[test]
    fn sky_light_chunk_does_not_reinitialize_all_empty_section_above_neighbor_sources() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let stale_neighbor = ChunkPos::new(1, 0);
        let section_count = 6;
        let mut holders = Vec::new();
        let mut center_holder = None;
        let mut stale_holder = None;

        for z in -2..=2 {
            for x in -2..=2 {
                let pos = ChunkPos::new(x, z);
                let holder = if pos == center {
                    roofed_holder(pos, section_count, 2, 0)
                } else if pos == stale_neighbor {
                    roofed_holder(pos, section_count, 3, 0)
                } else {
                    empty_holder_with_section_count(pos, section_count)
                };
                if pos == center {
                    center_holder = Some(Arc::clone(&holder));
                }
                if pos == stale_neighbor {
                    stale_holder = Some(Arc::clone(&holder));
                }
                holders.push((pos, holder));
            }
        }

        let Some(center_holder) = center_holder else {
            panic!("center holder should be created");
        };
        let Some(stale_holder) = stale_holder else {
            panic!("stale neighbor holder should be created");
        };

        {
            let Some(chunk) = stale_holder.try_chunk(ChunkStatus::Empty) else {
                panic!("stale neighbor chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(5) else {
                panic!("stale neighbor sky nibble should exist");
            };
            nibble.fill(MAX_LIGHT_LEVEL);
            assert!(nibble.update_visible());
        }

        let Ok(range) = LightSectionRange::from_world_height(0, (section_count * 16) as i32) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(center, range);
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| find_holder(&holders, pos),
            |_| true,
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(_) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center chunk should be available");
        };
        let light = chunk.light();
        let Some(kept_nibble) = light.sky.nibble(4) else {
            panic!("source-padding sky nibble should exist");
        };
        let Some(omitted_nibble) = light.sky.nibble(5) else {
            panic!("all-empty sky nibble should exist");
        };

        assert_eq!(kept_nibble.visible_state(), LightNibbleState::Initialized);
        assert_eq!(omitted_nibble.visible_state(), LightNibbleState::Null);
    }

    #[test]
    fn sky_light_chunk_deinitializes_cached_empty_section_without_neighbor_light_maps() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let stale_neighbor = ChunkPos::new(1, 0);
        let section_count = 6;
        let mut holders = Vec::new();
        let mut stale_holder = None;

        for z in -2..=2 {
            for x in -2..=2 {
                let pos = ChunkPos::new(x, z);
                let holder = if pos == center {
                    roofed_holder(pos, section_count, 2, 0)
                } else if pos == stale_neighbor {
                    roofed_holder(pos, section_count, 3, 0)
                } else {
                    empty_holder_with_section_count(pos, section_count)
                };
                if pos == stale_neighbor {
                    stale_holder = Some(Arc::clone(&holder));
                }
                holders.push((pos, holder));
            }
        }

        let Some(stale_holder) = stale_holder else {
            panic!("stale neighbor holder should be created");
        };

        {
            let Some(chunk) = stale_holder.try_chunk(ChunkStatus::Empty) else {
                panic!("stale neighbor chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(5) else {
                panic!("stale neighbor sky nibble should exist");
            };
            nibble.fill(MAX_LIGHT_LEVEL);
            assert!(nibble.update_visible());
        }

        let Ok(range) = LightSectionRange::from_world_height(0, (section_count * 16) as i32) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(center, range);
        let Ok(workset) = LightWorkset::setup_with_scopes(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| find_holder(&holders, pos),
            |cached_chunk, _, _| {
                let writable =
                    cached_chunk.chunk_pos == center || cached_chunk.chunk_pos == stale_neighbor;
                (true, writable)
            },
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(_) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        let Some(chunk) = stale_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("stale neighbor chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(5) else {
            panic!("stale neighbor sky nibble should exist");
        };

        assert_eq!(nibble.visible_state(), LightNibbleState::Null);
    }

    #[test]
    fn sky_light_chunk_edge_checks_do_not_extrude_center_null_sections() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let east = ChunkPos::new(1, 0);
        let Ok(range) = LightSectionRange::from_world_height(0, 48) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(center, range);

        let lower = ChunkSection::new_empty();
        let middle = ChunkSection::new_empty();
        let mut upper = ChunkSection::new_empty();
        for z in 0..16 {
            for x in 0..16 {
                upper.set_block_state(x, 0, z, vanilla_blocks::STONE.default_state());
            }
        }
        let center_holder = holder_with_sections(center, vec![lower, middle, upper]);
        let east_holder = roofed_holder(east, 3, 2, 0);
        let neighbors = roofed_holder_square(center, 2, 3, 2, 0);

        {
            let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
                panic!("center test chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(1) else {
                panic!("middle sky nibble should exist");
            };
            nibble.set_non_null();
            for z in 0..16 {
                for x in 0..16 {
                    nibble.set(x, 0, z, MAX_LIGHT_LEVEL);
                }
            }
            assert!(nibble.update_visible());
        }
        {
            let Some(chunk) = east_holder.try_chunk(ChunkStatus::Empty) else {
                panic!("east test chunk should be available");
            };
            let mut light = chunk.light_mut();
            let Some(nibble) = light.sky.nibble_mut(0) else {
                panic!("east lower sky nibble should exist");
            };
            nibble.set_non_null();
            assert!(nibble.update_visible());
        }

        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&center_holder))
                } else if pos == east {
                    Some(Arc::clone(&east_holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing optional chunks");
        };

        let Ok(_) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("lower sky nibble should exist");
        };
        let Some(lower_air) = layout.cached_block(BlockPos::new(8, 8, 8)) else {
            panic!("lower air block should be cached");
        };

        assert_eq!(nibble.visible_state(), LightNibbleState::Null);
        assert_eq!(nibble.get_visible_at_index(lower_air.local_index), 0);
    }

    #[test]
    fn sky_neighbor_level_pull_skips_null_current_sections() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let east = ChunkPos::new(1, 0);
        let center_holder = holder_with_section(center, ChunkSection::new_empty());
        let east_holder = holder_with_section(east, ChunkSection::new_empty());
        set_visible_sky_light(east_holder.as_ref(), 0, 0, 0, 0, 7);

        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&center_holder))
                } else if pos == east {
                    Some(Arc::clone(&east_holder))
                } else {
                    None
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing optional chunks");
        };

        let mut queues = PackedLightPropagationQueues::new();
        workset.with_chunk_read_cache(|chunk_cache| {
            chunk_cache.with_section_read_cache(|section_cache| {
                chunk_cache.with_light_write_cache(LightLayer::Sky, |light_cache| {
                    {
                        let Ok(mut context) = SkyLightPropagationContext::new(
                            section_cache,
                            light_cache,
                            &mut queues,
                        ) else {
                            panic!("sky propagation context should initialize");
                        };
                        context.propagate_neighbor_levels(center, 0, 0);
                    }

                    assert!(!queues.has_work());
                });
            });
        });
    }

    #[test]
    fn sky_light_changes_add_and_remove_air_column_shadow() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let holder = holder_with_section(center, ChunkSection::new_empty());
        let changed_pos = BlockPos::new(1, 14, 1);
        let layout = LightCacheLayout::new(center, range());

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        assert!(
            chunk
                .set_block_state(
                    changed_pos,
                    vanilla_blocks::STONE.default_state(),
                    UpdateFlags::UPDATE_CLIENTS,
                )
                .is_some()
        );
        drop(chunk);

        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_changes_with_empty_sections(
            &workset,
            [changed_pos],
            [LightSectionEmptinessChange {
                section_pos: SectionPos::new(0, 0, 0),
                empty: false,
            }],
        ) else {
            panic!("matching sky caches should run sky block changes");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(above) = layout.cached_block(BlockPos::new(1, 15, 1)) else {
            panic!("above block should be cached");
        };
        let Some(stone) = layout.cached_block(changed_pos) else {
            panic!("changed block should be cached");
        };
        let Some(below) = layout.cached_block(BlockPos::new(1, 13, 1)) else {
            panic!("below block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(above.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(stone.local_index), 0);
        assert_eq!(nibble.get_visible_at_index(below.local_index), 14);
        drop(light);
        drop(chunk);

        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        assert!(
            chunk
                .set_block_state(
                    changed_pos,
                    vanilla_blocks::AIR.default_state(),
                    UpdateFlags::UPDATE_CLIENTS,
                )
                .is_some()
        );
        drop(chunk);

        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| (pos == center).then(|| Arc::clone(&holder)),
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };
        let Ok(_) = propagate_sky_light_changes_with_empty_sections(
            &workset,
            [changed_pos],
            [LightSectionEmptinessChange {
                section_pos: SectionPos::new(0, 0, 0),
                empty: true,
            }],
        ) else {
            panic!("matching sky caches should run sky block changes");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));
        let Some(chunk) = holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };

        assert_eq!(nibble.get_visible_at_index(above.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(stone.local_index), 15);
        assert_eq!(nibble.get_visible_at_index(below.local_index), 15);
        assert_eq!(light.sky.get_light_value(above.block_pos), 15);
        assert_eq!(light.sky.get_light_value(stone.block_pos), 15);
        assert_eq!(light.sky.get_light_value(below.block_pos), 15);
    }

    #[test]
    fn sky_light_changes_apply_empty_section_transitions() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let removed_pos = BlockPos::new(1, 1, 1);
        let mut holders = Vec::new();
        let mut center_holder = None;
        for z in -2..=2 {
            for x in -2..=2 {
                let pos = ChunkPos::new(x, z);
                let mut section = ChunkSection::new_empty();
                if pos == center {
                    section.set_block_state(1, 1, 1, vanilla_blocks::STONE.default_state());
                }
                let holder = holder_with_section(pos, section);
                if pos == center {
                    center_holder = Some(Arc::clone(&holder));
                }
                holders.push((pos, holder));
            }
        }
        let Some(center_holder) = center_holder else {
            panic!("center holder should be created");
        };
        set_visible_sky_light(&center_holder, 0, 1, 1, 1, 15);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center chunk should be available");
        };
        assert_eq!(
            chunk.set_block_state(
                removed_pos,
                vanilla_blocks::AIR.default_state(),
                UpdateFlags::UPDATE_NONE,
            ),
            Some(vanilla_blocks::STONE.default_state())
        );
        drop(chunk);

        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |pos| find_holder(&holders, pos),
            |_| true,
        ) else {
            panic!("relaxed setup should accept cached test chunks");
        };

        let Ok(_) = propagate_sky_light_changes_with_empty_sections(
            &workset,
            [removed_pos],
            [LightSectionEmptinessChange {
                section_pos: SectionPos::new(0, 0, 0),
                empty: true,
            }],
        ) else {
            panic!("matching sky caches should run sky light updates");
        };

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("center chunk should be available");
        };
        let light = chunk.light();
        assert_eq!(light.sky.section_empty(0), Some(true));
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("center sky nibble should exist");
        };
        assert_eq!(nibble.visible_state(), LightNibbleState::Null);
    }

    #[test]
    fn sky_light_chunk_edge_checks_pull_neighbor_under_ceiling() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let east_chunk = ChunkPos::new(1, 0);
        let mut center_section = ChunkSection::new_empty();
        for z in 0..16 {
            for x in 0..16 {
                center_section.set_block_state(x, 15, z, vanilla_blocks::STONE.default_state());
            }
        }
        let center_holder = holder_with_section(center, center_section);
        let east_holder = holder_with_section(east_chunk, ChunkSection::new_empty());
        let neighbors = horizontal_empty_neighbors(center, 1);
        set_visible_sky_light(&east_holder, 0, 0, 14, 1, 15);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |pos| {
                if pos == center {
                    Some(Arc::clone(&center_holder))
                } else if pos == east_chunk {
                    Some(Arc::clone(&east_holder))
                } else {
                    find_holder(&neighbors, pos)
                }
            },
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing neighbors");
        };

        let Ok(result) = propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required)
        else {
            panic!("matching sky caches should run sky chunk lighting");
        };

        assert!(result.updated_sections.contains(&SectionPos::new(0, 0, 0)));

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Empty) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Some(nibble) = light.sky.nibble(0) else {
            panic!("test nibble should be inside light range");
        };
        let Some(edge) = layout.cached_block(BlockPos::new(15, 14, 1)) else {
            panic!("center edge should be cached");
        };
        let Some(inner) = layout.cached_block(BlockPos::new(14, 14, 1)) else {
            panic!("center inner block should be cached");
        };
        let Some(blocked) = layout.cached_block(BlockPos::new(15, 15, 1)) else {
            panic!("ceiling block should be cached");
        };

        assert_eq!(nibble.get_visible_at_index(edge.local_index), 14);
        assert_eq!(nibble.get_visible_at_index(inner.local_index), 13);
        assert_eq!(nibble.get_visible_at_index(blocked.local_index), 0);
    }

    #[test]
    fn sky_light_chunk_requires_center_chunk() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let layout = LightCacheLayout::new(center, range());
        let Ok(workset) = LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Inner,
            true,
            |_| None,
            |_| true,
        ) else {
            panic!("relaxed setup should accept missing chunks");
        };

        assert_eq!(
            propagate_sky_light_chunk_without_edge_checks(&workset).err(),
            Some(SkyLightPropagationContextError::MissingCenterChunk { chunk_pos: center })
        );
    }
}

//! Light storage primitives used by chunk and world lighting.

use steel_registry::blocks::{block_state_ext::BlockStateExt, shapes::VoxelShape};
use steel_utils::{BlockStateId, Direction};

use crate::physics::shapes::{face_shape_occludes, merged_face_occludes};

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

mod cache;
mod data_layer;
mod nibble;
mod packet;
mod queue;
mod section_storage;
mod sky_sources;

pub use cache::{
    LIGHT_CACHE_CHUNK_SLOTS, LIGHT_CACHE_DIAMETER, LIGHT_CACHE_RADIUS, LightCacheLayout,
    PackedLightBlockPos,
};
pub use data_layer::{DataLayer, DataLayerLengthError, DataLayerStorageMap};
pub use nibble::{
    ChunkLightData, ChunkLightEmptinessMapLengthError, ChunkLightLayerStorage, LightNibbleArray,
    LightNibbleExtrudeNullSourceError, LightNibbleSaveState, LightNibbleState,
};
pub use packet::{build_chunk_light_update_packet, build_light_update_packet};
pub(crate) use queue::{
    ADD_SKY_SOURCE_ENTRY, REMOVE_SKY_SOURCE_ENTRY, REMOVE_TOP_SKY_SOURCE_ENTRY,
};
pub use queue::{
    LightAxisDirection, LightDirectionSet, LightPropagationQueue, LightPropagationQueues,
    LightQueueEntry, LightQueueFlags, PackedLightQueueEntry, QueuedLightUpdate,
};
pub use section_storage::{
    LayerLightSectionStorage, LightSectionRange, LightSectionRangeError, LightSectionState,
    LightSectionStateError, LightSectionType, MissingLightDataLayerError,
};
pub use sky_sources::{ChunkSkyLightSources, SkyLightSourceNeighborhood};

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
        ChunkLightData, ChunkSkyLightSources, DATA_LAYER_SIZE, DataLayer, DataLayerStorageMap,
        LayerLightSectionStorage, LightLayer, LightNibbleArray, LightNibbleSaveState,
        LightNibbleState, LightPropagationQueue, LightPropagationQueues, LightQueueEntry,
        LightSectionRange, LightSectionState, LightSectionStateError, LightSectionType,
        MissingLightDataLayerError, QueuedLightUpdate, SkyLightSourceNeighborhood,
        build_chunk_light_update_packet, build_light_update_packet, get_light_block_into,
        get_light_opacity, has_different_light_properties, light_face_occludes,
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

    fn sky_sources_with_column(
        default_source_y: i32,
        x: usize,
        z: usize,
        source_y: i32,
    ) -> ChunkSkyLightSources {
        let mut sources = sky_sources_with_highest_lowest_source_y(default_source_y);
        sources.heightmap[x + z * super::CHUNK_EDGE] = source_y;
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
    fn light_nibble_array_publishes_visible_state_after_update() {
        let mut nibble = LightNibbleArray::null();

        nibble.set(1, 2, 3, 12);

        assert_eq!(nibble.updating_state(), LightNibbleState::Initialized);
        assert_eq!(nibble.visible_state(), LightNibbleState::Null);
        assert_eq!(nibble.get_updating(1, 2, 3), 12);
        assert_eq!(nibble.get_visible(1, 2, 3), 0);
        assert!(nibble.update_visible());
        assert_eq!(nibble.visible_state(), LightNibbleState::Initialized);
        assert_eq!(nibble.get_visible(1, 2, 3), 12);
        assert!(!nibble.update_visible());
    }

    #[test]
    fn light_nibble_array_converts_external_states_to_data_layer() {
        let mut hidden = LightNibbleArray::filled(7);
        hidden.set_hidden();
        assert!(hidden.update_visible());

        let null = LightNibbleArray::null();
        let uninitialized = LightNibbleArray::uninitialized();
        let initialized = LightNibbleArray::filled(5);

        assert!(null.to_data_layer().is_none());
        assert!(hidden.to_data_layer().is_none());

        let Some(empty_layer) = uninitialized.to_data_layer() else {
            panic!("uninitialized nibble should convert to an empty data layer");
        };
        assert!(empty_layer.is_empty());

        let Some(filled_layer) = initialized.to_data_layer() else {
            panic!("initialized nibble should convert to a data layer");
        };
        assert_eq!(filled_layer.get(3, 9, 11), 5);
    }

    #[test]
    fn light_nibble_save_state_matches_scalable_lux_canonicalization() {
        let null = LightNibbleArray::null();
        let uninitialized = LightNibbleArray::uninitialized();
        let zero_initialized = LightNibbleArray::filled(0);

        assert_eq!(null.to_save_state(), None);
        assert_eq!(
            uninitialized.to_save_state(),
            Some(LightNibbleSaveState {
                state: LightNibbleState::Uninitialized,
                data: None,
            })
        );
        assert_eq!(
            zero_initialized.to_save_state(),
            Some(LightNibbleSaveState {
                state: LightNibbleState::Uninitialized,
                data: None,
            })
        );

        let mut zero_hidden = LightNibbleArray::filled(0);
        zero_hidden.set_hidden();
        assert!(zero_hidden.update_visible());
        assert_eq!(zero_hidden.to_save_state(), None);

        let mut hidden = LightNibbleArray::filled(3);
        hidden.set_hidden();
        assert!(hidden.update_visible());

        let Some(save_state) = hidden.to_save_state() else {
            panic!("non-zero hidden nibble should produce save state");
        };
        assert_eq!(save_state.state, LightNibbleState::Hidden);
        let Some(data) = save_state.data else {
            panic!("non-zero hidden nibble should save packed bytes");
        };
        assert!(data.iter().all(|byte| *byte == 0x33));
    }

    #[test]
    fn light_nibble_array_extrudes_lower_row_from_updating_source() {
        let mut source = LightNibbleArray::null();
        for z in 0..super::DATA_LAYER_EDGE {
            for x in 0..super::DATA_LAYER_EDGE {
                source.set(x, 0, z, ((x + z) & 15) as u8);
            }
        }

        let mut target = LightNibbleArray::null();
        if target.extrude_lower(&source).is_err() {
            panic!("initialized source should extrude");
        }

        for y in 0..super::DATA_LAYER_EDGE {
            assert_eq!(target.get_updating(3, y, 5), 8);
            assert_eq!(target.get_updating(11, y, 14), 9);
        }
    }

    #[test]
    fn chunk_light_data_uses_vanilla_padded_light_section_count() {
        let Ok(light) = ChunkLightData::new(-64, 384) else {
            panic!("valid overworld height rejected");
        };

        assert_eq!(light.block.range().min_section_y(), -5);
        assert_eq!(light.block.nibbles().len(), 26);
        assert_eq!(light.sky.nibbles().len(), 26);
        assert_eq!(
            light.sky.nibble(-5).map(LightNibbleArray::visible_state),
            Some(LightNibbleState::Null)
        );
        assert!(light.sky.nibble(21).is_none());
    }

    #[test]
    fn chunk_light_emptiness_maps_follow_section_counters() {
        init_light_tests();

        let mut non_empty_section = ChunkSection::new_empty();
        non_empty_section.set_block_state(0, 0, 0, vanilla_blocks::STONE.default_state());
        let sections = Sections::from_owned(
            vec![ChunkSection::new_empty(), non_empty_section].into_boxed_slice(),
        );

        let Ok(mut light) = ChunkLightData::new(0, 32) else {
            panic!("valid two-section height rejected");
        };
        assert_eq!(light.block.chunk_section_count(), 2);
        assert!(light.block.emptiness_map().is_none());

        if let Err(error) = light.refresh_emptiness_maps_from_sections(&sections) {
            panic!("valid section count rejected: {error:?}");
        }

        assert_eq!(light.block.emptiness_map(), Some(&[true, false][..]));
        assert_eq!(light.sky.emptiness_map(), Some(&[true, false][..]));
        assert_eq!(light.block.section_empty(0), Some(true));
        assert_eq!(light.block.section_empty(1), Some(false));
        assert_eq!(light.block.section_empty(-1), None);
        assert_eq!(light.block.section_empty(2), None);

        assert!(light.set_section_empty(0, false));
        assert!(!light.set_section_empty(0, false));
        assert_eq!(light.block.emptiness_map(), Some(&[false, false][..]));
        assert_eq!(light.sky.emptiness_map(), Some(&[false, false][..]));
    }

    #[test]
    fn chunk_light_update_packet_converts_chunk_owned_nibbles() {
        let Ok(mut light) = ChunkLightData::new(0, 16) else {
            panic!("valid single-section height rejected");
        };

        let Some(sky_nibble) = light.sky.nibble_mut(0) else {
            panic!("single-section light range should contain section 0");
        };
        sky_nibble.set_non_null();
        assert!(sky_nibble.update_visible());

        let Some(block_nibble) = light.block.nibble_mut(0) else {
            panic!("single-section light range should contain section 0");
        };
        block_nibble.set(0, 0, 0, 7);
        assert!(block_nibble.update_visible());

        let Some(hidden_nibble) = light.sky.nibble_mut(1) else {
            panic!("single-section light range should contain padded section 1");
        };
        hidden_nibble.set(0, 0, 0, 12);
        hidden_nibble.set_hidden();
        assert!(hidden_nibble.update_visible());

        let packet = build_chunk_light_update_packet(&light);

        assert_eq!(packet.sky_y_mask.0[0], 0);
        assert_eq!(packet.empty_sky_y_mask.0[0], 0b010);
        assert_eq!(packet.block_y_mask.0[0], 0b010);
        assert_eq!(packet.empty_block_y_mask.0[0], 0);
        assert!(packet.sky_updates.is_empty());
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
    fn sky_storage_propagates_sky_sources_and_queues_edge_updates() {
        let zero = SectionPos::new(0, 0, 0);
        let section = SectionPos::new(0, 4, 0);
        let source_section = SectionPos::new(0, 5, 0);
        let sources = sky_sources_with_column(1000, 0, 0, 95);
        let north_sources = sky_sources_with_highest_lowest_source_y(90);
        let south_sources = sky_sources_with_highest_lowest_source_y(1000);
        let west_sources = sky_sources_with_highest_lowest_source_y(1000);
        let east_sources = sky_sources_with_highest_lowest_source_y(1000);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        let mut queues = LightPropagationQueues::new();

        assert_eq!(storage.update_section_status(section, false), Ok(()));
        assert_eq!(
            storage.propagate_sky_light_sources(
                zero,
                SkyLightSourceNeighborhood::new(
                    &sources,
                    &north_sources,
                    &south_sources,
                    &west_sources,
                    &east_sources,
                ),
                &mut queues,
            ),
            Some(())
        );

        assert!(storage.light_on_in_column(zero));
        let Some(layer) = storage.get_updating_data_layer(source_section) else {
            panic!("source section light layer missing");
        };
        assert_eq!(layer.get(0, 15, 0), super::MAX_LIGHT_LEVEL);
        assert_eq!(layer.get(0, 14, 0), 0);

        let Some(update) = queues.dequeue_increase() else {
            panic!("skylight source edge was not queued");
        };
        assert_eq!(update.block_pos, BlockPos::new(0, 95, 0));
        assert_eq!(update.entry.from_level(), super::MAX_LIGHT_LEVEL);
        assert!(update.entry.should_propagate_in_direction(Down));
        assert!(!update.entry.should_propagate_in_direction(Up));
        assert!(!update.entry.should_propagate_in_direction(North));
        assert!(update.entry.should_propagate_in_direction(South));
        assert!(update.entry.should_propagate_in_direction(West));
        assert!(update.entry.should_propagate_in_direction(East));
        assert_eq!(queues.dequeue_increase(), None);
    }

    #[test]
    fn block_storage_rejects_sky_source_propagation() {
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        let sources = sky_sources_with_highest_lowest_source_y(95);
        let mut queues = LightPropagationQueues::new();

        assert_eq!(
            storage.propagate_sky_light_sources(
                SectionPos::new(0, 0, 0),
                SkyLightSourceNeighborhood::new(&sources, &sources, &sources, &sources, &sources),
                &mut queues,
            ),
            None
        );
        assert!(!queues.has_work());
        assert!(!storage.light_on_in_column(SectionPos::new(0, 0, 0)));
    }

    #[test]
    fn sky_storage_update_sources_column_adds_sources_and_queues_edges() {
        let section = SectionPos::new(0, 4, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        let mut queues = LightPropagationQueues::new();

        assert_eq!(storage.update_section_status(section, false), Ok(()));
        assert_eq!(
            storage.update_sky_sources_in_column(0, 0, 78, 80, &mut queues),
            Ok(Some(()))
        );

        assert_eq!(storage.get_stored_level(BlockPos::new(0, 78, 0)), Some(15));
        assert_eq!(storage.get_stored_level(BlockPos::new(0, 79, 0)), Some(15));
        assert_eq!(storage.get_stored_level(BlockPos::new(0, 80, 0)), Some(15));

        let Some(first) = queues.dequeue_increase() else {
            panic!("first skylight source edge was not queued");
        };
        assert_eq!(first.block_pos, BlockPos::new(0, 78, 0));
        assert_eq!(first.entry, super::ADD_SKY_SOURCE_ENTRY);
        assert!(first.entry.should_propagate_in_direction(Down));
        assert!(!first.entry.should_propagate_in_direction(Up));

        let Some(second) = queues.dequeue_increase() else {
            panic!("second skylight source edge was not queued");
        };
        assert_eq!(second.block_pos, BlockPos::new(0, 79, 0));
        assert_eq!(second.entry, super::ADD_SKY_SOURCE_ENTRY);
        assert_eq!(queues.dequeue_increase(), None);
    }

    #[test]
    fn sky_storage_update_sources_column_removes_old_sources_below_new_edge() {
        let section = SectionPos::new(0, 4, 0);
        let mut storage = LayerLightSectionStorage::new(LightLayer::Sky);
        let mut queues = LightPropagationQueues::new();

        assert_eq!(storage.update_section_status(section, false), Ok(()));
        assert_eq!(
            storage.set_stored_level(BlockPos::new(0, 78, 0), 15),
            Ok(())
        );
        assert_eq!(
            storage.set_stored_level(BlockPos::new(0, 79, 0), 15),
            Ok(())
        );
        assert_eq!(
            storage.set_stored_level(BlockPos::new(0, 80, 0), 15),
            Ok(())
        );

        assert_eq!(
            storage.update_sky_sources_in_column(0, 0, 80, 1000, &mut queues),
            Ok(Some(()))
        );

        assert_eq!(storage.get_stored_level(BlockPos::new(0, 78, 0)), Some(0));
        assert_eq!(storage.get_stored_level(BlockPos::new(0, 79, 0)), Some(0));
        assert_eq!(storage.get_stored_level(BlockPos::new(0, 80, 0)), Some(15));

        let Some(first) = queues.dequeue_decrease() else {
            panic!("top skylight source removal was not queued");
        };
        assert_eq!(first.block_pos, BlockPos::new(0, 79, 0));
        assert_eq!(first.entry, super::REMOVE_TOP_SKY_SOURCE_ENTRY);

        let Some(second) = queues.dequeue_decrease() else {
            panic!("lower skylight source removal was not queued");
        };
        assert_eq!(second.block_pos, BlockPos::new(0, 78, 0));
        assert_eq!(second.entry, super::REMOVE_SKY_SOURCE_ENTRY);
        assert_eq!(queues.dequeue_decrease(), None);
        assert_eq!(queues.dequeue_increase(), None);
    }

    #[test]
    fn block_storage_rejects_sky_source_column_update() {
        let mut storage = LayerLightSectionStorage::new(LightLayer::Block);
        let mut queues = LightPropagationQueues::new();

        assert_eq!(
            storage.update_sky_sources_in_column(0, 0, 80, 1000, &mut queues),
            Ok(None)
        );
        assert!(!queues.has_work());
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

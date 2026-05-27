use steel_protocol::packets::game::LightUpdatePacketData;
use steel_utils::{ChunkPos, SectionPos, codec::BitSet};

use super::{ChunkLightData, ChunkLightLayerStorage, DataLayerStorageMap, LightSectionRange};
use super::{LightLayer, LightNibbleState, MAX_LIGHT_LEVEL};

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
        let Some(section_y) = range.section_y(section_index) else {
            continue;
        };
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

/// Builds protocol light-update data from chunk-owned ScalableLux-style nibbles.
#[must_use]
pub fn build_chunk_light_update_packet(light: &ChunkLightData) -> LightUpdatePacketData {
    let range = light.sky.range();
    let mut sky_y_mask = range.empty_bit_set();
    let mut block_y_mask = range.empty_bit_set();
    let mut empty_sky_y_mask = range.empty_bit_set();
    let mut empty_block_y_mask = range.empty_bit_set();
    let mut sky_updates = Vec::new();
    let mut block_updates = Vec::new();

    for section_index in 0..range.section_count() {
        prepare_chunk_sky_nibble_section_data(
            light,
            section_index,
            &mut sky_y_mask,
            &mut empty_sky_y_mask,
            &mut sky_updates,
        );
        prepare_nibble_section_data(
            &light.block,
            section_index,
            &mut block_y_mask,
            &mut empty_block_y_mask,
            &mut block_updates,
        );
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

/// Builds protocol light-update data for the changed sections of one chunk column.
///
/// Vanilla writes update payloads in ascending light-section-index order. Keep
/// that order here even though callers pass sets/vectors of changed sections.
#[must_use]
pub fn build_chunk_light_update_packet_for_sections(
    chunk_pos: ChunkPos,
    light: &ChunkLightData,
    sky_sections: &[SectionPos],
    block_sections: &[SectionPos],
) -> LightUpdatePacketData {
    let range = light.sky.range();
    let mut sky_y_mask = range.empty_bit_set();
    let mut block_y_mask = range.empty_bit_set();
    let mut empty_sky_y_mask = range.empty_bit_set();
    let mut empty_block_y_mask = range.empty_bit_set();
    let mut sky_updates = Vec::new();
    let mut block_updates = Vec::new();

    for section_index in 0..range.section_count() {
        let Some(section_y) = range.section_y(section_index) else {
            continue;
        };
        let section_pos = range.section_pos(chunk_pos, section_y);

        if sky_sections.contains(&section_pos) {
            prepare_nibble_section_data(
                &light.sky,
                section_index,
                &mut sky_y_mask,
                &mut empty_sky_y_mask,
                &mut sky_updates,
            );
        }

        if block_sections.contains(&section_pos) {
            prepare_nibble_section_data(
                &light.block,
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

fn prepare_chunk_sky_nibble_section_data(
    light: &ChunkLightData,
    section_index: usize,
    mask: &mut BitSet,
    empty_mask: &mut BitSet,
    updates: &mut Vec<Vec<u8>>,
) {
    let Some(nibble) = light.sky.nibbles().get(section_index) else {
        return;
    };
    let Some(layer) = nibble.to_data_layer() else {
        if should_send_empty_nibble_section(&light.sky, section_index) {
            empty_mask.set(section_index, true);
        }
        return;
    };

    if layer.is_empty() {
        empty_mask.set(section_index, true);
        return;
    }

    let bytes = layer.to_bytes();
    if should_omit_full_sky_section_above_local_sources(light, section_index, bytes.as_ref()) {
        return;
    }

    mask.set(section_index, true);
    updates.push(bytes.as_ref().to_vec());
}

fn prepare_nibble_section_data(
    layers: &ChunkLightLayerStorage,
    section_index: usize,
    mask: &mut BitSet,
    empty_mask: &mut BitSet,
    updates: &mut Vec<Vec<u8>>,
) {
    let Some(nibble) = layers.nibbles().get(section_index) else {
        return;
    };
    let Some(layer) = nibble.to_data_layer() else {
        if should_send_empty_nibble_section(layers, section_index) {
            empty_mask.set(section_index, true);
        }
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

fn should_omit_full_sky_section_above_local_sources(
    light: &ChunkLightData,
    section_index: usize,
    bytes: &[u8; super::DATA_LAYER_SIZE],
) -> bool {
    let full_byte = MAX_LIGHT_LEVEL | (MAX_LIGHT_LEVEL << 4);
    if bytes.iter().any(|byte| *byte != full_byte) {
        return false;
    }

    let Some(section_y) = light.sky.range().section_y(section_index) else {
        return false;
    };
    let Some(highest) = highest_non_empty_section_y(&light.sky) else {
        return false;
    };
    if section_y <= highest + 1 {
        return false;
    }

    light
        .block
        .nibbles()
        .get(section_index)
        .is_none_or(|nibble| {
            matches!(
                nibble.visible_state(),
                LightNibbleState::Null | LightNibbleState::Hidden
            )
        })
}

fn should_send_empty_nibble_section(layers: &ChunkLightLayerStorage, section_index: usize) -> bool {
    let Some(section_y) = layers.range().section_y(section_index) else {
        return false;
    };

    if layers.layer() == LightLayer::Sky
        && highest_non_empty_section_y(layers).is_some_and(|highest| section_y > highest)
    {
        return false;
    }

    for neighbor_y in (section_y - 1)..=(section_y + 1) {
        if layers.section_empty(neighbor_y).is_some_and(|empty| !empty) {
            return true;
        }
    }

    false
}

fn highest_non_empty_section_y(layers: &ChunkLightLayerStorage) -> Option<i32> {
    let emptiness_map = layers.emptiness_map()?;
    for (index, empty) in emptiness_map.iter().copied().enumerate().rev() {
        if !empty {
            return layers.range().chunk_section_y(index);
        }
    }
    None
}

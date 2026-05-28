//! Chunk generation stage regression test.
//!
//! Verifies that Steel's chunk generation matches vanilla Minecraft at each stage
//! by comparing MD5 hashes of block data. When a mismatch is found and binary
//! reference data is available, shows exact block-level diffs.
//!
//! Tests all dimensions (overworld, nether, end) using the new JSON format
//! with a `dimensions` wrapper.

use std::env;
use std::fmt::Write;
use std::fs;
use std::io::{BufReader, Cursor, Read as IoRead};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Duration;

use flate2::read::GzDecoder;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use serde::Deserialize;
use steel_core::chunk::chunk_access::{ChunkAccess, ChunkStatus};
use steel_core::chunk::chunk_generation_task::StaticCache2D;
use steel_core::chunk::chunk_holder::ChunkHolder;
use steel_core::chunk::chunk_pyramid::GENERATION_PYRAMID;
use steel_core::chunk::chunk_request::{ChunkRequestState, ChunkTicketKind};
use steel_core::chunk::light::{
    ChunkLightData, DATA_LAYER_BLOCK_COUNT, DATA_LAYER_SIZE, LightLayer,
    build_chunk_light_update_packet,
};
use steel_core::chunk::proto_chunk::ProtoChunk;
use steel_core::chunk::section::{ChunkSection, Sections};
use steel_core::level_data::WorldGenerationSettings;
use steel_core::world::structure::StructureStart;
use steel_core::world::{World, WorldConfig, WorldStorageConfig};
use steel_core::worldgen::noise::beardifier::Beardifier;
use steel_core::worldgen::{ChunkGenerator, ChunkGeneratorType};
use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_registry::structure::TerrainAdjustment;
use steel_registry::{dimension_type::DimensionTypeRef, vanilla_dimension_types};
use steel_utils::types::{Difficulty, GameType};
use steel_utils::{BlockPos, ChunkPos, Identifier};
use tokio::runtime::Runtime;
use toml::map::Map;

#[derive(Clone, Deserialize, Debug, Eq, PartialEq)]
struct LightSectionDebug {
    state: String,
    #[serde(default)]
    non_zero_bytes: Option<usize>,
    #[serde(default)]
    byte_hash: Option<String>,
}

#[derive(Clone, Deserialize, Debug, Eq, PartialEq)]
struct LightChunkDebug {
    min_section_y: i32,
    section_count: usize,
    sky: Vec<LightSectionDebug>,
    block: Vec<LightSectionDebug>,
}

#[derive(Deserialize, Debug)]
struct ChunkStageEntry {
    x: i32,
    z: i32,
    stages: FxHashMap<String, String>,
    #[serde(default)]
    light_debug: Option<LightChunkDebug>,
}

#[derive(Deserialize, Debug)]
struct DimensionData {
    chunks: Vec<ChunkStageEntry>,
}

#[derive(Deserialize, Debug)]
struct ChunkStageHashesJson {
    seed: u64,
    chunk_generation_order: String,
    #[serde(default)]
    feature_hash_capture: Option<String>,
    #[serde(default)]
    hashset_iteration_order: Option<String>,
    #[serde(default)]
    light_hash_capture: Option<String>,
    #[serde(default)]
    light_dependency_radius: Option<i32>,
    #[serde(default)]
    light_hash_format: Option<String>,
    #[serde(default)]
    light_debug_format: Option<String>,
    #[serde(default)]
    light_binary_format: Option<String>,
    dimensions: FxHashMap<String, DimensionData>,
}

/// Stages to verify in vanilla generation order.
const STAGES: &[&str] = &[
    "minecraft:noise",
    "minecraft:surface",
    "minecraft:carvers",
    "minecraft:features",
];

/// Match the extractor run's structure setting.
///
/// Set this to `false` when the vanilla fixture was produced with
/// `-DMC_DEBUG_DISABLE_STRUCTURES=true`.
const GENERATE_STRUCTURES: bool = true;

/// Max block-level diffs to show per chunk before truncating.
const MAX_DIFFS_PER_CHUNK: usize = 30;
const MAX_LIGHT_DIFFS_PER_CHUNK: usize = 40;

/// Set specific chunk coordinates to test only those chunks.
/// When non-empty, only these chunks are generated and checked (ignores the JSON list).
/// Example: &[(24, 35)] to debug a single failing chunk.
const DEBUG_CHUNKS: &[(i32, i32)] = &[];
const DEBUG_CLUSTER_ENV: &str = "STEEL_HASH_DEBUG_CLUSTER";
const DEBUG_CHUNK_ENV: &str = "STEEL_HASH_DEBUG_CHUNK";
const DEBUG_DIMENSION_ENV: &str = "STEEL_HASH_DEBUG_DIMENSION";
const DEBUG_STAGE_ENV: &str = "STEEL_HASH_DEBUG_STAGE";
const DEBUG_LIGHT_SUMMARY_ENV: &str = "STEEL_HASH_DEBUG_LIGHT_SUMMARY";
const DEBUG_LIGHT_WINDOW_ENV: &str = "STEEL_HASH_DEBUG_LIGHT_WINDOW";
const DEBUG_EXPECTED_SOURCE_LIGHT_ENV: &str = "STEEL_HASH_DEBUG_EXPECTED_SOURCE_LIGHT";
const DEBUG_RAW_LIGHT_CHUNK_ENV: &str = "STEEL_HASH_DEBUG_RAW_LIGHT_CHUNK";
const DEBUG_STRUCTURE_REFS_ENV: &str = "STEEL_HASH_DEBUG_STRUCTURE_REFS";
const DEBUG_STOP_AFTER_FIRST_MISMATCH_ENV: &str = "STEEL_HASH_STOP_AFTER_FIRST_MISMATCH";
const DEBUG_FIXTURE_PATH_ENV: &str = "STEEL_HASH_FIXTURE_PATH";
const DEBUG_LIGHT_DATA_PATH_ENV: &str = "STEEL_HASH_LIGHT_DATA_PATH";

const CARVERS_STAGE: &str = "minecraft:carvers";
const FEATURE_STAGE: &str = "minecraft:features";
const LIGHT_STAGE: &str = "minecraft:light";
const CHUNK_GENERATION_ORDER_X_Z_ASCENDING: &str = "x_z_ascending";
const FEATURE_HASH_CAPTURE_AFTER_ALL_READY: &str = "after_all_tracked_features_ready";
const HASHSET_ITERATION_ORDER_INSERTION: &str = "insertion_order";
const LIGHT_HASH_CAPTURE_AFTER_ALL_TRACKED_LIGHT_READY_AND_PENDING_TASKS_DRAINED_AND_IDLE: &str =
    "after_all_tracked_light_ready_pending_tasks_drained_and_light_engine_idle";
const LIGHT_HASH_FORMAT_PACKET_DATA_LAYERS_V1: &str = "packet_data_layers_v1";
const LIGHT_DEBUG_FORMAT_SECTION_MARKERS_V1: &str = "section_markers_v1";
const LIGHT_BINARY_FORMAT_PACKET_DATA_LAYERS_V1: &str = "packet_data_layers_binary_v1";

fn load_expected_hashes() -> ChunkStageHashesJson {
    let json = if let Ok(path) = env::var(DEBUG_FIXTURE_PATH_ENV) {
        fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("Failed to read chunk_stage_hashes fixture from {path}: {err}")
        })
    } else {
        include_str!("../test_assets/chunk_stage_hashes.json").to_owned()
    };
    serde_json::from_str(&json).expect("Failed to parse chunk_stage_hashes.json")
}

fn sorted_positions(positions: &FxHashSet<(i32, i32)>) -> Vec<(i32, i32)> {
    let mut positions = positions.iter().copied().collect::<Vec<_>>();
    positions.sort_unstable();
    positions
}

fn debug_chunk_filter() -> Option<FxHashSet<(i32, i32)>> {
    let mut chunks = FxHashSet::default();
    chunks.extend(DEBUG_CHUNKS.iter().copied());

    if let Ok(chunk) = env::var(DEBUG_CHUNK_ENV) {
        let Some((x, z)) = chunk.split_once(',') else {
            panic!("{DEBUG_CHUNK_ENV} must be formatted as '<chunk_x>,<chunk_z>'");
        };
        let Ok(chunk_x) = x.parse::<i32>() else {
            panic!("{DEBUG_CHUNK_ENV} chunk_x is not an i32: {x}");
        };
        let Ok(chunk_z) = z.parse::<i32>() else {
            panic!("{DEBUG_CHUNK_ENV} chunk_z is not an i32: {z}");
        };
        chunks.insert((chunk_x, chunk_z));
    }

    if let Ok(cluster) = env::var(DEBUG_CLUSTER_ENV) {
        let Some((x, z)) = cluster.split_once(',') else {
            panic!("{DEBUG_CLUSTER_ENV} must be formatted as '<chunk_x>,<chunk_z>'");
        };
        let Ok(origin_x) = x.parse::<i32>() else {
            panic!("{DEBUG_CLUSTER_ENV} chunk_x is not an i32: {x}");
        };
        let Ok(origin_z) = z.parse::<i32>() else {
            panic!("{DEBUG_CLUSTER_ENV} chunk_z is not an i32: {z}");
        };

        for dx in 0..10 {
            for dz in 0..10 {
                chunks.insert((origin_x + dx, origin_z + dz));
            }
        }
    }

    (!chunks.is_empty()).then_some(chunks)
}

fn debug_dimension_filter() -> Option<String> {
    env::var(DEBUG_DIMENSION_ENV)
        .ok()
        .filter(|dimension| !dimension.is_empty())
}

fn debug_raw_light_chunk_filter() -> Option<(i32, i32)> {
    let value = env::var(DEBUG_RAW_LIGHT_CHUNK_ENV).ok()?;
    let Some((x, z)) = value.split_once(',') else {
        panic!("{DEBUG_RAW_LIGHT_CHUNK_ENV} must be formatted as '<chunk_x>,<chunk_z>'");
    };
    let Ok(chunk_x) = x.parse::<i32>() else {
        panic!("{DEBUG_RAW_LIGHT_CHUNK_ENV} chunk_x is not an i32: {x}");
    };
    let Ok(chunk_z) = z.parse::<i32>() else {
        panic!("{DEBUG_RAW_LIGHT_CHUNK_ENV} chunk_z is not an i32: {z}");
    };
    Some((chunk_x, chunk_z))
}

fn debug_structure_refs_filter() -> Option<(i32, i32)> {
    let value = env::var(DEBUG_STRUCTURE_REFS_ENV).ok()?;
    let Some((x, z)) = value.split_once(',') else {
        panic!("{DEBUG_STRUCTURE_REFS_ENV} must be formatted as '<chunk_x>,<chunk_z>'");
    };
    let Ok(chunk_x) = x.parse::<i32>() else {
        panic!("{DEBUG_STRUCTURE_REFS_ENV} chunk_x is not an i32: {x}");
    };
    let Ok(chunk_z) = z.parse::<i32>() else {
        panic!("{DEBUG_STRUCTURE_REFS_ENV} chunk_z is not an i32: {z}");
    };
    Some((chunk_x, chunk_z))
}

fn debug_stage_filter() -> Option<String> {
    env::var(DEBUG_STAGE_ENV)
        .ok()
        .filter(|stage| !stage.is_empty())
}

#[derive(Clone, Copy)]
struct DebugLightWindow {
    chunk_x: i32,
    chunk_z: i32,
    min_x: usize,
    min_y: i32,
    min_z: usize,
    size_x: usize,
    size_y: usize,
    size_z: usize,
}

fn debug_light_window_filter() -> Option<DebugLightWindow> {
    let value = env::var(DEBUG_LIGHT_WINDOW_ENV).ok()?;
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|err| panic!("{DEBUG_LIGHT_WINDOW_ENV} has invalid integer: {err}"));
    if parts.len() != 8 {
        panic!(
            "{DEBUG_LIGHT_WINDOW_ENV} must be '<chunk_x>,<chunk_z>,<local_x>,<world_y>,<local_z>,<size_x>,<size_y>,<size_z>'"
        );
    }

    let [
        chunk_x,
        chunk_z,
        min_x,
        min_y,
        min_z,
        size_x,
        size_y,
        size_z,
    ] = parts.as_slice()
    else {
        unreachable!();
    };

    Some(DebugLightWindow {
        chunk_x: *chunk_x,
        chunk_z: *chunk_z,
        min_x: usize::try_from(*min_x).expect("light window local_x must be non-negative"),
        min_y: *min_y,
        min_z: usize::try_from(*min_z).expect("light window local_z must be non-negative"),
        size_x: usize::try_from(*size_x).expect("light window size_x must be non-negative"),
        size_y: usize::try_from(*size_y).expect("light window size_y must be non-negative"),
        size_z: usize::try_from(*size_z).expect("light window size_z must be non-negative"),
    })
}

fn empty_proto_chunk(
    pos: (i32, i32),
    section_count: usize,
    min_y: i32,
    height: i32,
) -> ChunkAccess {
    let sections: Box<[ChunkSection]> = (0..section_count)
        .map(|_| ChunkSection::new_empty())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let proto = ProtoChunk::new(
        Sections::from_owned(sections),
        ChunkPos::new(pos.0, pos.1),
        min_y,
        height,
        Weak::new(),
    );
    ChunkAccess::Proto(proto)
}

fn chunk_or_panic(chunks: &FxHashMap<(i32, i32), ChunkAccess>, pos: (i32, i32)) -> &ChunkAccess {
    match chunks.get(&pos) {
        Some(chunk) => chunk,
        None => panic!("Missing test chunk ({}, {})", pos.0, pos.1),
    }
}

fn create_test_world(
    dim_key: &str,
    dim_type: DimensionTypeRef,
    seed: u64,
    generator: Arc<ChunkGeneratorType>,
) -> Arc<World> {
    let runtime = Arc::new(Runtime::new().expect("failed to create chunk-stage hash test runtime"));
    let generation_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|index| format!("chunk-stage-hashes-{index}"))
            .build()
            .expect("failed to create chunk-stage hash test rayon pool"),
    );
    let dim_short = dim_key.strip_prefix("minecraft:").unwrap_or(dim_key);
    let empty_config = toml::Value::Table(Map::new());
    let generation_settings = WorldGenerationSettings::from_generator_config(
        Identifier::new(Identifier::VANILLA_NAMESPACE, dim_short.to_owned()),
        &empty_config,
        dim_type.key.clone(),
        dim_type.min_y,
        dim_type.height,
    );
    let sea_level = match dim_key {
        "minecraft:the_nether" => 32,
        "minecraft:the_end" => 0,
        _ => 63,
    };

    runtime
        .block_on(World::new_with_config(
            runtime.clone(),
            Identifier::new(Identifier::VANILLA_NAMESPACE, dim_short.to_owned()),
            dim_type,
            seed as i64,
            WorldConfig {
                storage: WorldStorageConfig::RamOnly,
                level_data_path: None,
                generator,
                generation_settings,
                view_distance: 2,
                simulation_distance: 2,
                compression: None,
                is_flat: false,
                sea_level,
                default_gamemode: GameType::Survival,
                difficulty: Difficulty::Normal,
            },
            generation_pool,
        ))
        .expect("failed to create chunk-stage hash test world")
}

fn build_feature_holders(
    chunks: FxHashMap<(i32, i32), ChunkAccess>,
    carver_positions: &FxHashSet<(i32, i32)>,
    min_y: i32,
    height: i32,
) -> FxHashMap<(i32, i32), Arc<ChunkHolder>> {
    let mut holders = FxHashMap::with_capacity_and_hasher(chunks.len(), FxBuildHasher);
    for (pos, chunk) in chunks {
        let holder = Arc::new(ChunkHolder::new(
            ChunkPos::new(pos.0, pos.1),
            0,
            min_y,
            height,
        ));
        let status = if carver_positions.contains(&pos) {
            ChunkStatus::Carvers
        } else {
            ChunkStatus::StructureStarts
        };
        if let ChunkAccess::Proto(proto) = &chunk {
            proto.set_status(status);
        }
        holder.insert_chunk(chunk, status);
        holders.insert(pos, holder);
    }
    holders
}

fn compute_block_hash(sections: &Sections) -> String {
    let mut ctx = md5::Context::new();

    for section_holder in &sections.sections {
        let section = section_holder.read();
        // Match vanilla's `LevelChunkSection.hasOnlyAir()` — which returns
        // true when `nonEmptyBlockCount == 0`, i.e. every block is air /
        // cave_air / void_air. Steel's palette-level `has_only_air()` doesn't
        // treat a heterogeneous cave_air-only palette as "empty", so we scan
        // manually to match the extractor's shortcut.
        let mut all_air = true;
        'scan: for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    if !section.states.get(x, y, z).is_air() {
                        all_air = false;
                        break 'scan;
                    }
                }
            }
        }
        if all_air {
            ctx.consume([0u8]);
        } else {
            for y in 0..16 {
                for z in 0..16 {
                    for x in 0..16 {
                        let state = section.states.get(x, y, z);
                        let state_id = u32::from(state.0);
                        ctx.consume([(state_id >> 24) as u8]);
                        ctx.consume([(state_id >> 16) as u8]);
                        ctx.consume([(state_id >> 8) as u8]);
                        ctx.consume([state_id as u8]);
                    }
                }
            }
        }
    }

    format!("{:x}", ctx.finalize())
}

fn consume_i32(ctx: &mut md5::Context, value: i32) {
    ctx.consume([(value >> 24) as u8]);
    ctx.consume([(value >> 16) as u8]);
    ctx.consume([(value >> 8) as u8]);
    ctx.consume([value as u8]);
}

fn bitset_contains(bitset: &steel_utils::codec::BitSet, index: usize) -> bool {
    bitset
        .0
        .get(index / 64)
        .is_some_and(|bits| bits & (1 << (index % 64)) != 0)
}

fn md5_hex(bytes: &[u8]) -> String {
    let mut ctx = md5::Context::new();
    ctx.consume(bytes);
    format!("{:x}", ctx.finalize())
}

fn light_layer_debug(
    section_count: usize,
    data_mask: &steel_utils::codec::BitSet,
    empty_mask: &steel_utils::codec::BitSet,
    updates: &[Vec<u8>],
) -> Vec<LightSectionDebug> {
    let mut update_index = 0;
    let mut sections = Vec::with_capacity(section_count);
    for section_index in 0..section_count {
        if bitset_contains(empty_mask, section_index) {
            sections.push(LightSectionDebug {
                state: "empty".to_owned(),
                non_zero_bytes: None,
                byte_hash: None,
            });
        } else if bitset_contains(data_mask, section_index) {
            let update = &updates[update_index];
            let non_zero = update.iter().filter(|byte| **byte != 0).count();
            let all_full = update.iter().all(|byte| *byte == 0xff);
            sections.push(LightSectionDebug {
                state: if all_full { "full" } else { "data" }.to_owned(),
                non_zero_bytes: (non_zero != 0).then_some(non_zero),
                byte_hash: Some(md5_hex(update)),
            });
            update_index += 1;
        } else {
            sections.push(LightSectionDebug {
                state: "null".to_owned(),
                non_zero_bytes: None,
                byte_hash: None,
            });
        }
    }
    sections
}

fn actual_light_debug(light: &ChunkLightData, has_skylight: bool) -> LightChunkDebug {
    let packet = build_chunk_light_update_packet(light, has_skylight);
    let range = light.sky.range();
    let sky = light_layer_debug(
        range.section_count(),
        &packet.sky_y_mask,
        &packet.empty_sky_y_mask,
        &packet.sky_updates,
    );
    let block = light_layer_debug(
        range.section_count(),
        &packet.block_y_mask,
        &packet.empty_block_y_mask,
        &packet.block_updates,
    );

    LightChunkDebug {
        min_section_y: range.min_section_y(),
        section_count: range.section_count(),
        sky,
        block,
    }
}

fn light_layer_bytes(
    section_count: usize,
    data_mask: &steel_utils::codec::BitSet,
    empty_mask: &steel_utils::codec::BitSet,
    updates: &[Vec<u8>],
) -> Vec<LightSectionBytes> {
    let mut update_index = 0;
    let mut sections = Vec::with_capacity(section_count);
    for section_index in 0..section_count {
        if bitset_contains(empty_mask, section_index) {
            sections.push(LightSectionBytes {
                state: 1,
                bytes: None,
            });
        } else if bitset_contains(data_mask, section_index) {
            let update = &updates[update_index];
            sections.push(LightSectionBytes {
                state: 2,
                bytes: Some(update.clone()),
            });
            update_index += 1;
        } else {
            sections.push(LightSectionBytes {
                state: 0,
                bytes: None,
            });
        }
    }

    if update_index != updates.len() {
        panic!(
            "light packet carried {} unused updates after consuming {update_index}",
            updates.len()
        );
    }

    sections
}

fn actual_light_bytes(light: &ChunkLightData, has_skylight: bool) -> ChunkLightBytes {
    let packet = build_chunk_light_update_packet(light, has_skylight);
    let range = light.sky.range();
    let sky = light_layer_bytes(
        range.section_count(),
        &packet.sky_y_mask,
        &packet.empty_sky_y_mask,
        &packet.sky_updates,
    );
    let block = light_layer_bytes(
        range.section_count(),
        &packet.block_y_mask,
        &packet.empty_block_y_mask,
        &packet.block_updates,
    );

    ChunkLightBytes {
        min_section_y: range.min_section_y(),
        section_count: range.section_count(),
        sky,
        block,
    }
}

fn format_light_section_debug(section: &LightSectionDebug) -> String {
    let mut label = section.state.clone();
    if let Some(non_zero_bytes) = section.non_zero_bytes {
        let _ = write!(label, "({non_zero_bytes} nz-bytes");
        if let Some(byte_hash) = &section.byte_hash {
            let _ = write!(label, ", {byte_hash}");
        }
        label.push(')');
    } else if let Some(byte_hash) = &section.byte_hash {
        let _ = write!(label, "({byte_hash})");
    }
    label
}

fn format_light_debug(debug: &LightChunkDebug) -> String {
    let mut out = String::new();
    for section_index in 0..debug.section_count {
        let Ok(section_offset) = i32::try_from(section_index) else {
            continue;
        };
        let section_y = debug.min_section_y + section_offset;
        let sky = debug
            .sky
            .get(section_index)
            .map(format_light_section_debug)
            .unwrap_or_else(|| "missing".to_owned());
        let block = debug
            .block
            .get(section_index)
            .map(format_light_section_debug)
            .unwrap_or_else(|| "missing".to_owned());
        let _ = writeln!(out, "    y={section_y:4}: sky={:18} block={}", sky, block);
    }
    out
}

fn debug_light_differences(expected: &LightChunkDebug, actual: &LightChunkDebug) -> String {
    let mut out = String::new();
    if expected.min_section_y != actual.min_section_y
        || expected.section_count != actual.section_count
    {
        let _ = writeln!(
            out,
            "    range expected min={} count={}, actual min={} count={}",
            expected.min_section_y,
            expected.section_count,
            actual.min_section_y,
            actual.section_count
        );
    }

    let section_count = expected.section_count.min(actual.section_count);
    let mut shown = 0;
    for section_index in 0..section_count {
        let expected_sky = expected.sky.get(section_index);
        let actual_sky = actual.sky.get(section_index);
        let expected_block = expected.block.get(section_index);
        let actual_block = actual.block.get(section_index);
        if expected_sky == actual_sky && expected_block == actual_block {
            continue;
        }

        let Ok(section_offset) = i32::try_from(section_index) else {
            continue;
        };
        let section_y = expected.min_section_y + section_offset;
        let _ = writeln!(
            out,
            "    y={section_y:4}: expected sky={:18} block={}; actual sky={:18} block={}",
            expected_sky
                .map(format_light_section_debug)
                .unwrap_or_else(|| "missing".to_owned()),
            expected_block
                .map(format_light_section_debug)
                .unwrap_or_else(|| "missing".to_owned()),
            actual_sky
                .map(format_light_section_debug)
                .unwrap_or_else(|| "missing".to_owned()),
            actual_block
                .map(format_light_section_debug)
                .unwrap_or_else(|| "missing".to_owned()),
        );
        shown += 1;
        if shown == 12 {
            break;
        }
    }

    if shown == 0 && out.is_empty() {
        out.push_str("    no section-marker differences found\n");
    }
    out
}

fn light_section_state_name(state: u8) -> &'static str {
    match state {
        0 => "null",
        1 => "empty",
        2 => "data",
        _ => "invalid",
    }
}

fn light_value_from_section(section: &LightSectionBytes, index: usize) -> u8 {
    let Some(bytes) = section.bytes.as_ref() else {
        return 0;
    };
    let packed = bytes[index >> 1];
    packed >> ((index & 1) << 2) & 0x0f
}

fn debug_raw_light_differences(
    expected: &ChunkLightBytes,
    actual: &ChunkLightBytes,
    chunk: &ChunkAccess,
    chunk_x: i32,
    chunk_z: i32,
) -> String {
    let mut out = String::new();
    if expected.min_section_y != actual.min_section_y
        || expected.section_count != actual.section_count
    {
        let _ = writeln!(
            out,
            "    range expected min={} count={}, actual min={} count={}",
            expected.min_section_y,
            expected.section_count,
            actual.min_section_y,
            actual.section_count
        );
    }

    let mut shown = 0;
    for (layer_name, expected_layer, actual_layer) in [
        ("sky", expected.sky.as_slice(), actual.sky.as_slice()),
        ("block", expected.block.as_slice(), actual.block.as_slice()),
    ] {
        let section_count = expected_layer.len().min(actual_layer.len());
        for section_index in 0..section_count {
            let expected_section = &expected_layer[section_index];
            let actual_section = &actual_layer[section_index];
            let Ok(section_offset) = i32::try_from(section_index) else {
                continue;
            };
            let section_y = expected.min_section_y + section_offset;

            if expected_section.state != actual_section.state {
                let _ = writeln!(
                    out,
                    "    {layer_name:5} y={section_y:4}: state vanilla={} steel={}",
                    light_section_state_name(expected_section.state),
                    light_section_state_name(actual_section.state),
                );
                shown += 1;
                if shown == MAX_LIGHT_DIFFS_PER_CHUNK {
                    out.push_str("    ... more light differences omitted\n");
                    return out;
                }
            }

            for index in 0..DATA_LAYER_BLOCK_COUNT {
                let vanilla = light_value_from_section(expected_section, index);
                let steel = light_value_from_section(actual_section, index);
                if vanilla == steel {
                    continue;
                }

                let local_x = index & 15;
                let local_z = (index >> 4) & 15;
                let local_y = (index >> 8) & 15;
                let world_x = chunk_x * 16 + local_x as i32;
                let world_y = section_y * 16 + local_y as i32;
                let world_z = chunk_z * 16 + local_z as i32;
                let state_context = format_light_state_context(chunk, local_x, world_y, local_z);
                let _ = writeln!(
                    out,
                    "    {layer_name:5} y={section_y:4} local=({local_x:2},{local_y:2},{local_z:2}) world=({world_x},{world_y},{world_z}): vanilla={vanilla:2} steel={steel:2}{state_context}",
                );
                if shown == 0 {
                    out.push_str(&format_light_column_differences(
                        expected_section,
                        actual_section,
                        chunk,
                        local_x,
                        section_y,
                        local_z,
                    ));
                }
                shown += 1;
                if shown == MAX_LIGHT_DIFFS_PER_CHUNK {
                    out.push_str("    ... more light differences omitted\n");
                    return out;
                }
            }
        }
    }

    if shown == 0 {
        out.push_str("    no raw light value differences found\n");
    }
    out
}

fn format_debug_light_window(
    window: DebugLightWindow,
    expected: Option<&ChunkLightBytes>,
    actual: &ChunkLightData,
    chunk: &ChunkAccess,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "    window chunk=({}, {}) local=({}, {}, {}) size=({}, {}, {})",
        window.chunk_x,
        window.chunk_z,
        window.min_x,
        window.min_y,
        window.min_z,
        window.size_x,
        window.size_y,
        window.size_z
    );

    for dy in (0..window.size_y).rev() {
        let world_y = window.min_y + dy as i32;
        let _ = writeln!(out, "      y={world_y}");
        for dz in 0..window.size_z {
            let local_z = window.min_z + dz;
            let mut expected_row = String::with_capacity(window.size_x);
            let mut actual_row = String::with_capacity(window.size_x);
            let mut state_row = String::new();
            for dx in 0..window.size_x {
                let local_x = window.min_x + dx;
                let world_x = window.chunk_x * 16 + local_x as i32;
                let world_z = window.chunk_z * 16 + local_z as i32;
                let actual_level = actual
                    .get_light_value(LightLayer::Block, BlockPos::new(world_x, world_y, world_z));
                let expected_level = expected
                    .and_then(|light| light_bytes_block_value(light, world_y, local_x, local_z));
                push_light_digit(&mut actual_row, actual_level);
                match expected_level {
                    Some(level) => push_light_digit(&mut expected_row, level),
                    None => expected_row.push('?'),
                }

                if let Some(state) = chunk_state_at(chunk, local_x, world_y, local_z)
                    && (state.get_light_emission() > 0 || expected_level.unwrap_or(0) > 8)
                {
                    let _ = write!(
                        state_row,
                        " ({local_x},{local_z})={}:{}",
                        i32::from(state.0),
                        state.get_light_emission()
                    );
                }
            }
            let _ = writeln!(
                out,
                "        z={local_z:2} expected={expected_row} actual={actual_row}{state_row}"
            );
        }
    }

    out
}

fn format_debug_block_window(window: DebugLightWindow, chunk: &ChunkAccess) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "    block window chunk=({}, {}) local=({}, {}, {}) size=({}, {}, {})",
        window.chunk_x,
        window.chunk_z,
        window.min_x,
        window.min_y,
        window.min_z,
        window.size_x,
        window.size_y,
        window.size_z
    );

    for dy in (0..window.size_y).rev() {
        let world_y = window.min_y + dy as i32;
        let _ = writeln!(out, "      y={world_y}");
        for dz in 0..window.size_z {
            let local_z = window.min_z + dz;
            let mut state_row = String::new();
            for dx in 0..window.size_x {
                let local_x = window.min_x + dx;
                if let Some(state) = chunk_state_at(chunk, local_x, world_y, local_z) {
                    let _ = write!(
                        state_row,
                        " ({local_x},{local_z})={}",
                        describe_state(i32::from(state.0))
                    );
                }
            }
            let _ = writeln!(out, "        z={local_z:2}{state_row}");
        }
    }

    out
}

fn format_expected_source_light_audit(
    expected: &ChunkLightBytes,
    chunk: &ChunkAccess,
    chunk_x: i32,
    chunk_z: i32,
) -> Option<String> {
    let mut out = String::new();
    let mut source_count = 0usize;
    let mut underlit_count = 0usize;

    for (section_index, section_holder) in chunk.sections().sections.iter().enumerate() {
        let section_base_y = chunk.min_y() + section_index as i32 * 16;
        let section = section_holder.read();
        for local_y in 0..16usize {
            let world_y = section_base_y + local_y as i32;
            for local_z in 0..16usize {
                for local_x in 0..16usize {
                    let state = section.states.get(local_x, local_y, local_z);
                    let emission = state.get_light_emission();
                    if emission == 0 {
                        continue;
                    }

                    source_count += 1;
                    let expected_level =
                        light_bytes_block_value(expected, world_y, local_x, local_z).unwrap_or(0);
                    if expected_level >= emission {
                        continue;
                    }

                    underlit_count += 1;
                    if underlit_count <= MAX_LIGHT_DIFFS_PER_CHUNK {
                        let world_x = chunk_x * 16 + local_x as i32;
                        let world_z = chunk_z * 16 + local_z as i32;
                        let _ = writeln!(
                            out,
                            "    local=({local_x:2},{local_y:2},{local_z:2}) world=({world_x},{world_y},{world_z}): vanilla_block={expected_level:2} emission={emission:2} state={}",
                            describe_state(i32::from(state.0))
                        );
                    }
                }
            }
        }
    }

    if underlit_count == 0 {
        return None;
    }

    if underlit_count > MAX_LIGHT_DIFFS_PER_CHUNK {
        let _ = writeln!(
            out,
            "    ... and {} more underlit source blocks",
            underlit_count - MAX_LIGHT_DIFFS_PER_CHUNK
        );
    }

    let mut header = format!(
        "    {underlit_count}/{source_count} vanilla source blocks are below their own emission\n"
    );
    header.push_str(&out);
    Some(header)
}

fn light_bytes_block_value(
    light: &ChunkLightBytes,
    world_y: i32,
    local_x: usize,
    local_z: usize,
) -> Option<u8> {
    let section_y = world_y.div_euclid(16);
    let local_y = usize::try_from(world_y.rem_euclid(16)).ok()?;
    let section_index = usize::try_from(section_y - light.min_section_y).ok()?;
    let section = light.block.get(section_index)?;
    let index = local_x | (local_z << 4) | (local_y << 8);
    Some(light_value_from_section(section, index))
}

fn push_light_digit(out: &mut String, level: u8) {
    let digit = match level {
        0 => '.',
        1..=9 => char::from(b'0' + level),
        10..=15 => char::from(b'a' + (level - 10)),
        _ => '?',
    };
    out.push(digit);
}

fn format_light_column_differences(
    expected_section: &LightSectionBytes,
    actual_section: &LightSectionBytes,
    chunk: &ChunkAccess,
    local_x: usize,
    section_y: i32,
    local_z: usize,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "      column local=({local_x},*,{local_z}) values:");
    for local_y in (0..16).rev() {
        let index = local_x | (local_z << 4) | (local_y << 8);
        let vanilla = light_value_from_section(expected_section, index);
        let steel = light_value_from_section(actual_section, index);
        let world_y = (section_y << 4) | local_y as i32;
        let state = chunk_state_at(chunk, local_x, world_y, local_z)
            .map(format_light_state)
            .unwrap_or_else(|| "missing".to_owned());
        let _ = writeln!(
            out,
            "        y={world_y:4} local_y={local_y:2}: vanilla={vanilla:2} steel={steel:2} {state}",
        );
    }
    out
}

fn format_light_state_context(
    chunk: &ChunkAccess,
    local_x: usize,
    world_y: i32,
    local_z: usize,
) -> String {
    let current = chunk_state_at(chunk, local_x, world_y, local_z)
        .map(format_light_state)
        .unwrap_or_else(|| "missing".to_owned());
    let above = chunk_state_at(chunk, local_x, world_y + 1, local_z)
        .map(format_light_state)
        .unwrap_or_else(|| "missing".to_owned());
    let above_non_air = first_non_air_above(chunk, local_x, world_y, local_z)
        .map(|(y, state)| format!("{y}:{}", format_light_state(state)))
        .unwrap_or_else(|| "none".to_owned());
    let source_y = chunk
        .sky_light_sources()
        .get_lowest_source_y(local_x, local_z);
    let source_neighbors = format_light_source_neighbors(chunk, local_x, local_z);

    format!(
        " source_y={source_y} {source_neighbors} state={current} above={above} first_non_air_above={above_non_air}"
    )
}

fn format_light_source_neighbors(chunk: &ChunkAccess, local_x: usize, local_z: usize) -> String {
    let sources = chunk.sky_light_sources();
    let north = if local_z > 0 {
        sources
            .get_lowest_source_y(local_x, local_z - 1)
            .to_string()
    } else {
        "edge".to_owned()
    };
    let south = if local_z + 1 < 16 {
        sources
            .get_lowest_source_y(local_x, local_z + 1)
            .to_string()
    } else {
        "edge".to_owned()
    };
    let west = if local_x > 0 {
        sources
            .get_lowest_source_y(local_x - 1, local_z)
            .to_string()
    } else {
        "edge".to_owned()
    };
    let east = if local_x + 1 < 16 {
        sources
            .get_lowest_source_y(local_x + 1, local_z)
            .to_string()
    } else {
        "edge".to_owned()
    };

    format!("source_neighbors=n:{north} s:{south} w:{west} e:{east}")
}

fn format_light_state(state: steel_utils::BlockStateId) -> String {
    format!(
        "{} opacity={}",
        describe_state(i32::from(state.0)),
        state.get_light_dampening()
    )
}

fn chunk_state_at(
    chunk: &ChunkAccess,
    local_x: usize,
    world_y: i32,
    local_z: usize,
) -> Option<steel_utils::BlockStateId> {
    let offset_y = world_y.checked_sub(chunk.min_y())?;
    if offset_y < 0 {
        return None;
    }
    let section_index = usize::try_from(offset_y / 16).ok()?;
    let local_y = usize::try_from(offset_y & 15).ok()?;
    let section = chunk.sections().sections.get(section_index)?.read();
    Some(section.states.get(local_x, local_y, local_z))
}

fn first_non_air_above(
    chunk: &ChunkAccess,
    local_x: usize,
    world_y: i32,
    local_z: usize,
) -> Option<(i32, steel_utils::BlockStateId)> {
    let max_y = chunk.min_y() + chunk.sections().sections.len() as i32 * 16;
    for y in world_y + 1..max_y {
        let state = chunk_state_at(chunk, local_x, y, local_z)?;
        if !state.is_air() {
            return Some((y, state));
        }
    }

    None
}

fn debug_chunk_section_summary(chunk: &ChunkAccess) -> String {
    let mut out = String::new();
    for (section_index, section) in chunk.sections().sections.iter().enumerate() {
        let section_y = chunk.min_y() / 16 + section_index as i32;
        let section = section.read();
        let non_air = section.non_empty_block_count();
        if non_air == 0 {
            continue;
        }

        let opaque = (0..steel_core::chunk::paletted_container::BlockPalette::VOLUME)
            .filter(|local_index| {
                section
                    .states
                    .get_at_index(*local_index)
                    .get_light_dampening()
                    > 0
            })
            .count();
        let _ = writeln!(
            out,
            "    y={section_y:4}: non_air={non_air:4} dampens_light={opaque:4}"
        );
    }
    out
}

fn consume_light_layer_hash(
    ctx: &mut md5::Context,
    section_count: usize,
    data_mask: &steel_utils::codec::BitSet,
    empty_mask: &steel_utils::codec::BitSet,
    updates: &[Vec<u8>],
) {
    let mut update_index = 0;
    for section_index in 0..section_count {
        if bitset_contains(empty_mask, section_index) {
            ctx.consume([1]);
            continue;
        }

        if !bitset_contains(data_mask, section_index) {
            ctx.consume([0]);
            continue;
        }

        let Some(update) = updates.get(update_index) else {
            panic!("light packet data mask referenced missing update {update_index}");
        };
        ctx.consume([2]);
        ctx.consume(update);
        update_index += 1;
    }

    if update_index != updates.len() {
        panic!(
            "light packet carried {} unused updates after consuming {update_index}",
            updates.len()
        );
    }
}

fn compute_light_hash(light: &ChunkLightData, has_skylight: bool) -> String {
    let mut ctx = md5::Context::new();
    let range = light.sky.range();
    let Ok(section_count) = i32::try_from(range.section_count()) else {
        panic!("light section count does not fit in i32");
    };

    consume_i32(&mut ctx, range.min_section_y());
    consume_i32(&mut ctx, section_count);
    let packet = build_chunk_light_update_packet(light, has_skylight);
    for layer in [LightLayer::Sky, LightLayer::Block] {
        ctx.consume([match layer {
            LightLayer::Sky => 0,
            LightLayer::Block => 1,
        }]);
        match layer {
            LightLayer::Sky => consume_light_layer_hash(
                &mut ctx,
                range.section_count(),
                &packet.sky_y_mask,
                &packet.empty_sky_y_mask,
                &packet.sky_updates,
            ),
            LightLayer::Block => consume_light_layer_hash(
                &mut ctx,
                range.section_count(),
                &packet.block_y_mask,
                &packet.empty_block_y_mask,
                &packet.block_updates,
            ),
        }
    }

    format!("{:x}", ctx.finalize())
}

/// Per-chunk reference block data from the extractor binary.
struct ChunkBlockData {
    /// Sections, each None (all air) or Some(4096 state IDs in YZX order).
    sections: Vec<Option<Vec<i32>>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LightSectionBytes {
    state: u8,
    bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ChunkLightBytes {
    min_section_y: i32,
    section_count: usize,
    sky: Vec<LightSectionBytes>,
    block: Vec<LightSectionBytes>,
}

/// Loads binary reference block data for a given stage and dimension.
///
/// Binary format (gzip compressed, all integers big-endian):
///   `chunk_count`: i32
///   For each chunk:
///     `chunk_x`: i32
///     `chunk_z`: i32
///     `section_count`: i32
///     For each section:
///       `has_data`: u8
///       if `has_data` == 1: `state_ids`: [i32; 4096]
fn load_reference_blocks(
    stage: &str,
    dim_short: &str,
) -> Option<FxHashMap<(i32, i32), ChunkBlockData>> {
    let short_name = stage.strip_prefix("minecraft:").unwrap_or(stage);
    let path = format!(
        "{}/test_assets/chunk_stage_{dim_short}_{short_name}_blocks.bin.gz",
        env!("CARGO_MANIFEST_DIR"),
    );
    let compressed = fs::read(&path).ok()?;

    let decoder = GzDecoder::new(Cursor::new(compressed));
    let mut buf = Vec::new();
    BufReader::new(decoder).read_to_end(&mut buf).ok()?;

    let mut pos = 0;

    let read_i32 = |pos: &mut usize| -> i32 {
        let val = i32::from_be_bytes(
            buf[*pos..*pos + 4]
                .try_into()
                .expect("slice should be 4 bytes"),
        );
        *pos += 4;
        val
    };

    let chunk_count = read_i32(&mut pos) as usize;
    let mut map = FxHashMap::with_capacity_and_hasher(chunk_count, FxBuildHasher);

    for _ in 0..chunk_count {
        let cx = read_i32(&mut pos);
        let cz = read_i32(&mut pos);
        let section_count = read_i32(&mut pos) as usize;
        let mut sections = Vec::with_capacity(section_count);

        for _ in 0..section_count {
            let has_data = buf[pos];
            pos += 1;
            if has_data == 0 {
                sections.push(None);
            } else {
                let mut state_ids = Vec::with_capacity(4096);
                for _ in 0..4096 {
                    state_ids.push(read_i32(&mut pos));
                }
                sections.push(Some(state_ids));
            }
        }

        map.insert((cx, cz), ChunkBlockData { sections });
    }

    Some(map)
}

fn default_light_data_file_name(dim_short: &str) -> String {
    format!("chunk_stage_{dim_short}_light_layers.bin.gz")
}

fn default_light_data_path(dim_short: &str) -> PathBuf {
    let file_name = default_light_data_file_name(dim_short);
    if let Ok(path) = env::var(DEBUG_LIGHT_DATA_PATH_ENV) {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return path.join(file_name);
        }
        return path;
    }

    if let Ok(fixture_path) = env::var(DEBUG_FIXTURE_PATH_ENV) {
        let fixture_path = PathBuf::from(fixture_path);
        if let Some(parent) = fixture_path.parent() {
            return parent.join(file_name);
        }
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_assets")
        .join(file_name)
}

fn load_reference_lights(dim_short: &str) -> Option<FxHashMap<(i32, i32), ChunkLightBytes>> {
    let path = default_light_data_path(dim_short);
    let compressed = fs::read(&path).ok()?;

    let decoder = GzDecoder::new(Cursor::new(compressed));
    let mut buf = Vec::new();
    BufReader::new(decoder).read_to_end(&mut buf).ok()?;

    let mut pos = 0;
    let read_i32 = |pos: &mut usize| -> i32 {
        let val = i32::from_be_bytes(
            buf[*pos..*pos + 4]
                .try_into()
                .expect("slice should be 4 bytes"),
        );
        *pos += 4;
        val
    };

    let chunk_count = read_i32(&mut pos) as usize;
    let mut map = FxHashMap::with_capacity_and_hasher(chunk_count, FxBuildHasher);
    for _ in 0..chunk_count {
        let cx = read_i32(&mut pos);
        let cz = read_i32(&mut pos);
        let min_section_y = read_i32(&mut pos);
        let section_count = read_i32(&mut pos) as usize;
        let sky = read_light_layer_bytes(&buf, &mut pos, section_count);
        let block = read_light_layer_bytes(&buf, &mut pos, section_count);
        map.insert(
            (cx, cz),
            ChunkLightBytes {
                min_section_y,
                section_count,
                sky,
                block,
            },
        );
    }

    Some(map)
}

fn read_light_layer_bytes(
    buf: &[u8],
    pos: &mut usize,
    section_count: usize,
) -> Vec<LightSectionBytes> {
    let mut sections = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let state = buf[*pos];
        *pos += 1;
        let bytes = if state == 2 {
            let data = buf[*pos..*pos + DATA_LAYER_SIZE].to_vec();
            *pos += DATA_LAYER_SIZE;
            Some(data)
        } else {
            None
        };
        sections.push(LightSectionBytes { state, bytes });
    }
    sections
}

/// Format a state ID as "id (`block_name`[props])" for human-readable output.
fn describe_state(state_id: i32) -> String {
    use steel_registry::REGISTRY;
    use steel_utils::types::BlockStateId;

    let bsid = BlockStateId(state_id as u16);
    let Some(block) = REGISTRY.blocks.by_state_id(bsid) else {
        return format!("{state_id} (unknown)");
    };
    let props = REGISTRY.blocks.get_properties(bsid);
    if props.is_empty() {
        format!("{state_id} ({})", block.key)
    } else {
        let prop_str: Vec<_> = props.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{state_id} ({}[{}])", block.key, prop_str.join(","))
    }
}

fn format_structure_references(chunk: &ChunkAccess) -> String {
    let refs = chunk.structure_references();
    let mut entries = refs
        .iter()
        .map(|(structure_id, positions)| {
            let mut positions = positions
                .iter()
                .map(|pos| (pos.0.x, pos.0.y))
                .collect::<Vec<_>>();
            positions.sort_unstable();
            (structure_id.to_string(), positions)
        })
        .collect::<Vec<_>>();
    drop(refs);

    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    if entries.is_empty() {
        return "    no structure references\n".to_owned();
    }

    let mut out = String::new();
    for (structure_id, positions) in entries {
        let _ = writeln!(
            out,
            "    {structure_id}: {} source chunk(s)",
            positions.len()
        );
        for (source_x, source_z) in positions {
            let _ = writeln!(out, "      ({source_x}, {source_z})");
        }
    }
    out
}

fn referenced_structure_positions(chunk: &ChunkAccess, structure_key: &str) -> Vec<ChunkPos> {
    let refs = chunk.structure_references();
    let positions = refs
        .iter()
        .find(|(structure_id, _)| structure_id.to_string() == structure_key)
        .map(|(_, positions)| positions.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    drop(refs);
    positions
}

fn format_structure_start_summary(chunk: &ChunkAccess, structure_key: &str) -> String {
    let Ok(structure_id) = structure_key.parse::<Identifier>() else {
        return format!("      invalid structure key {structure_key}\n");
    };
    let starts = chunk.structure_starts();
    let Some(start) = starts.get(&structure_id) else {
        return format!("      {structure_key}: no start\n");
    };

    let mut out = String::new();
    let _ = writeln!(
        out,
        "      {structure_key}: pieces={} references={} bb={:?}",
        start.pieces.len(),
        start.references,
        start.bounding_box
    );
    for (index, piece) in start.pieces.iter().take(12).enumerate() {
        let _ = writeln!(
            out,
            "        piece[{index}] type={} depth={} bb={:?}",
            piece.piece_type, piece.gen_depth, piece.bounding_box
        );
    }
    if start.pieces.len() > 12 {
        let _ = writeln!(out, "        ... and {} more", start.pieces.len() - 12);
    }
    out
}

struct BlockDiff {
    x: usize,
    y: i32,
    z: usize,
    vanilla: i32,
    steel: i32,
}

/// Compare a chunk's sections against reference data, returning block-level diffs.
fn diff_chunk(sections: &Sections, reference: &ChunkBlockData, min_y: i32) -> Vec<BlockDiff> {
    let mut diffs = Vec::new();

    for (si, section_holder) in sections.sections.iter().enumerate() {
        let section = section_holder.read();
        let ref_section = reference.sections.get(si);
        let section_base_y = min_y + (si as i32) * 16;

        match ref_section {
            Some(Some(ref_ids)) => {
                if section.states.has_only_air() {
                    // Steel says all air, vanilla has data
                    for (idx, &vanilla_id) in ref_ids.iter().enumerate() {
                        if vanilla_id != 0 {
                            let y_local = idx / 256;
                            let z = (idx % 256) / 16;
                            let x = idx % 16;
                            diffs.push(BlockDiff {
                                x,
                                y: section_base_y + y_local as i32,
                                z,
                                vanilla: vanilla_id,
                                steel: 0,
                            });
                        }
                    }
                } else {
                    for y_local in 0..16usize {
                        for z in 0..16usize {
                            for x in 0..16usize {
                                let idx = y_local * 256 + z * 16 + x;
                                let vanilla_id = ref_ids[idx];
                                let steel_id =
                                    u32::from(section.states.get(x, y_local, z).0) as i32;
                                if vanilla_id != steel_id {
                                    diffs.push(BlockDiff {
                                        x,
                                        y: section_base_y + y_local as i32,
                                        z,
                                        vanilla: vanilla_id,
                                        steel: steel_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            Some(None) | None => {
                // Vanilla says all air (or section missing). Check if Steel also has air.
                if !section.states.has_only_air() {
                    for y_local in 0..16usize {
                        for z in 0..16usize {
                            for x in 0..16usize {
                                let steel_id =
                                    u32::from(section.states.get(x, y_local, z).0) as i32;
                                if steel_id != 0 {
                                    diffs.push(BlockDiff {
                                        x,
                                        y: section_base_y + y_local as i32,
                                        z,
                                        vanilla: 0,
                                        steel: steel_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    diffs
}

/// Format block diffs into a human-readable report for a single chunk.
fn format_chunk_diffs(diffs: &[BlockDiff], chunk_x: i32, chunk_z: i32, min_y: i32) -> String {
    let mut msg = format!(
        "  Chunk ({chunk_x:3},{chunk_z:3}): {} block differences\n",
        diffs.len()
    );

    // Group by section
    let mut by_section: FxHashMap<i32, Vec<&BlockDiff>> = FxHashMap::default();
    for d in diffs {
        let section_idx = (d.y - min_y) / 16;
        by_section.entry(section_idx).or_default().push(d);
    }

    let mut section_indices: Vec<_> = by_section.keys().copied().collect();
    section_indices.sort_unstable();

    let mut shown = 0;
    for si in section_indices {
        let section_diffs = &by_section[&si];
        let section_base = min_y + si * 16;
        let _ = writeln!(
            msg,
            "    Section {si} (y={section_base}..{}): {} differences",
            section_base + 15,
            section_diffs.len()
        );

        for d in section_diffs {
            if shown >= MAX_DIFFS_PER_CHUNK {
                let remaining = diffs.len() - shown;
                let _ = writeln!(msg, "      ... and {remaining} more");
                return msg;
            }
            let _ = writeln!(
                msg,
                "      ({:2},{:4},{:2}): vanilla={} steel={}",
                d.x,
                d.y,
                d.z,
                describe_state(d.vanilla),
                describe_state(d.steel),
            );
            shown += 1;
        }
    }

    msg
}

#[test]
#[ignore = "This test takes too long to run for normal testing; run with --release"]
fn chunk_stage_hashes() {
    use std::panic;
    use std::thread;

    // Run on a thread with a larger stack to avoid overflow in debug builds,
    // since pre-generating biome data for neighbor lookups increases stack usage.
    let result = thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(chunk_stage_hashes_inner)
        .expect("Failed to spawn test thread")
        .join();

    if let Err(payload) = result {
        panic::resume_unwind(payload);
    }
}

#[test]
#[ignore = "This test takes too long to run for normal testing; run with --release"]
fn chunk_light_hashes() {
    use std::panic;
    use std::thread;

    let result = thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(chunk_light_hashes_inner)
        .expect("Failed to spawn test thread")
        .join();

    if let Err(payload) = result {
        panic::resume_unwind(payload);
    }
}

/// Dimension order for deterministic test output (`HashMap` iteration is unordered).
const DIMENSION_ORDER: &[&str] = &[
    "minecraft:overworld",
    "minecraft:the_nether",
    "minecraft:the_end",
];

/// Build a beardifier for `chunk` using `chunks` as the chunk source. Mirrors the
/// production logic in `worldgen::stages::noise` but reads from a `HashMap` instead
/// of a chunk cache.
fn build_test_beardifier(
    chunk: &ChunkAccess,
    chunks: &FxHashMap<(i32, i32), ChunkAccess>,
) -> Option<Beardifier> {
    let pos = chunk.pos();
    let chunk_x = pos.0.x;
    let chunk_z = pos.0.y;

    let references = chunk.structure_references();

    let mut source_positions: FxHashSet<ChunkPos> = FxHashSet::default();
    for source_chunks in references.values() {
        source_positions.extend(source_chunks.iter().copied());
    }
    if source_positions.is_empty() {
        return None;
    }

    let source_chunk_refs: Vec<&ChunkAccess> = source_positions
        .iter()
        .filter_map(|p| chunks.get(&(p.0.x, p.0.y)))
        .collect();
    let mut source_indices: FxHashMap<ChunkPos, usize> = FxHashMap::default();
    let mut starts_guards = Vec::with_capacity(source_chunk_refs.len());
    for source_chunk in &source_chunk_refs {
        let source_pos = source_chunk.pos();
        source_indices.insert(source_pos, starts_guards.len());
        starts_guards.push(source_chunk.structure_starts());
    }

    let mut starts: Vec<&StructureStart> = Vec::new();
    for (structure_id, source_chunks_ref) in references.iter() {
        for &source_pos in source_chunks_ref {
            let Some(&guard_index) = source_indices.get(&source_pos) else {
                continue;
            };
            let guard = &starts_guards[guard_index];
            if let Some(start) = guard.get(structure_id)
                && start.chunk_pos == source_pos
                && start.terrain_adjustment != TerrainAdjustment::None
            {
                starts.push(start);
            }
        }
    }

    if starts.is_empty() {
        return None;
    }

    let beardifier = Beardifier::for_structures_in_chunk(starts.iter().copied(), chunk_x, chunk_z);
    (!beardifier.is_empty()).then_some(beardifier)
}

fn drive_chunk_request(
    world: &Arc<World>,
    request: &steel_core::chunk::chunk_request::ChunkRequestHandle,
    label: &str,
) {
    let runtime = Arc::clone(&world.chunk_map.chunk_runtime);
    runtime.block_on(async {
        for _ in 0..60_000 {
            world.chunk_map.tick_scheduling();
            match request.poll() {
                ChunkRequestState::Ready => return,
                ChunkRequestState::Cancelled => panic!("{label} chunk request was cancelled"),
                ChunkRequestState::Pending { .. } => {}
            }
            world.chunk_map.tick_scheduling();
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        panic!("{label} chunk request did not become ready");
    });
}

fn drive_chunk_generation_idle(world: &Arc<World>, label: &str) {
    let runtime = Arc::clone(&world.chunk_map.chunk_runtime);
    runtime.block_on(async {
        for _ in 0..60_000 {
            world.chunk_map.tick_scheduling();

            if world.chunk_map.pending_generation_tasks.lock().is_empty()
                && world.chunk_map.task_tracker.is_empty()
            {
                world.chunk_map.tick_scheduling();
                if world.chunk_map.pending_generation_tasks.lock().is_empty()
                    && world.chunk_map.task_tracker.is_empty()
                {
                    return;
                }
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        panic!("{label} chunk generation did not become idle");
    });
}

#[expect(
    clippy::too_many_lines,
    reason = "large ignored regression test mirrors chunk_stage_hashes setup"
)]
fn chunk_light_hashes_inner() {
    use steel_core::behavior::init_behaviors;
    use steel_core::block_entity::init_block_entities;
    use steel_core::entity::init_entities;
    use steel_core::worldgen::{
        BiomeSourceKind, EndGenerator, NetherGenerator, OverworldGenerator,
    };
    use steel_registry::{REGISTRY, Registry};

    let mut registry = Registry::new_vanilla();
    registry.freeze();
    let _ = REGISTRY.init(registry);
    init_behaviors();
    init_block_entities();
    init_entities();

    let expected = load_expected_hashes();
    assert_eq!(
        expected.chunk_generation_order, CHUNK_GENERATION_ORDER_X_Z_ASCENDING,
        "chunk light hash test only supports x/z ascending generation order"
    );

    let debug_dimension = debug_dimension_filter();
    let debug_filter = debug_chunk_filter();
    let stop_after_first_mismatch = env::var_os(DEBUG_STOP_AFTER_FIRST_MISMATCH_ENV).is_some();
    let emit_light_summary = env::var_os(DEBUG_LIGHT_SUMMARY_ENV).is_some();
    let audit_expected_source_light = env::var_os(DEBUG_EXPECTED_SOURCE_LIGHT_ENV).is_some();
    let debug_light_window = debug_light_window_filter();
    let debug_raw_light_chunk = debug_raw_light_chunk_filter();
    let debug_structure_refs = debug_structure_refs_filter();
    let mut saw_light_hashes = false;

    for &dim_key in DIMENSION_ORDER {
        if debug_dimension
            .as_deref()
            .is_some_and(|filter| filter != dim_key)
        {
            continue;
        }
        let Some(dim_data) = expected.dimensions.get(dim_key) else {
            continue;
        };

        let mut stage_entries = dim_data
            .chunks
            .iter()
            .filter(|entry| {
                debug_filter
                    .as_ref()
                    .is_none_or(|filter| filter.contains(&(entry.x, entry.z)))
            })
            .filter_map(|entry| {
                entry
                    .stages
                    .get(LIGHT_STAGE)
                    .map(|hash| (entry.x, entry.z, hash.as_str(), entry.light_debug.as_ref()))
            })
            .collect::<Vec<_>>();
        if stage_entries.is_empty() {
            continue;
        }
        saw_light_hashes = true;
        stage_entries.sort_unstable_by_key(|entry| (entry.0, entry.1));
        let dim_short = dim_key.strip_prefix("minecraft:").unwrap_or(dim_key);
        let light_dependency_radius = expected.light_dependency_radius.unwrap_or(0);
        assert!(
            light_dependency_radius >= 0,
            "light_dependency_radius must be non-negative"
        );
        let tracked_light_positions = stage_entries
            .iter()
            .map(|(chunk_x, chunk_z, _, _)| (*chunk_x, *chunk_z))
            .collect::<FxHashSet<_>>();
        let feature_write_radius = GENERATION_PYRAMID
            .get_step_to(ChunkStatus::Features)
            .block_state_write_radius;
        assert!(
            feature_write_radius >= 0,
            "features must declare a non-negative block write radius for light hash comparison"
        );
        let comparison_radius = (light_dependency_radius + feature_write_radius).max(1);
        let comparable_entries = stage_entries
            .iter()
            .copied()
            .filter(|(chunk_x, chunk_z, _, _)| {
                (-comparison_radius..=comparison_radius).all(|dx| {
                    (-comparison_radius..=comparison_radius)
                        .all(|dz| tracked_light_positions.contains(&(chunk_x + dx, chunk_z + dz)))
                })
            })
            .collect::<Vec<_>>();
        // Feature and light writes cross chunk borders; a chunk on the fixture
        // perimeter can depend on feature writes into its light dependencies that
        // this filtered fixture did not drive.
        if comparable_entries.len() != stage_entries.len() {
            eprintln!(
                "[{dim_short}/{LIGHT_STAGE}] comparing {} chunks with a fixture-local {comparison_radius}-radius feature/light halo ({} perimeter chunks skipped)",
                comparable_entries.len(),
                stage_entries.len() - comparable_entries.len()
            );
        }
        assert_eq!(
            expected.light_hash_capture.as_deref(),
            Some(
                LIGHT_HASH_CAPTURE_AFTER_ALL_TRACKED_LIGHT_READY_AND_PENDING_TASKS_DRAINED_AND_IDLE
            ),
            "light hashes must be extracted after all tracked LIGHT chunks are ready, pending light tasks are drained, and the light engine is idle; rerun the extractor"
        );
        assert_eq!(
            expected.light_hash_format.as_deref(),
            Some(LIGHT_HASH_FORMAT_PACKET_DATA_LAYERS_V1),
            "light hashes must use the packet data-layer format; rerun the extractor"
        );
        if stage_entries
            .iter()
            .any(|(_, _, _, light_debug)| light_debug.is_some())
        {
            assert_eq!(
                expected.light_debug_format.as_deref(),
                Some(LIGHT_DEBUG_FORMAT_SECTION_MARKERS_V1),
                "light debug data must use the section-marker format; rerun the extractor"
            );
        }

        let reference_lights = load_reference_lights(dim_short);
        if reference_lights.is_some() {
            assert_eq!(
                expected.light_binary_format.as_deref(),
                Some(LIGHT_BINARY_FORMAT_PACKET_DATA_LAYERS_V1),
                "light binary data must use the packet data-layer binary format; rerun the extractor"
            );
        }
        let dim_type = match dim_key {
            "minecraft:overworld" => &vanilla_dimension_types::OVERWORLD,
            "minecraft:the_nether" => &vanilla_dimension_types::THE_NETHER,
            "minecraft:the_end" => &vanilla_dimension_types::THE_END,
            _ => panic!("Unknown dimension: {dim_key}"),
        };
        let min_y = dim_type.min_y;
        let has_skylight = dim_type.has_skylight;
        let seed = expected.seed;
        let generator: Arc<ChunkGeneratorType> = Arc::new(match dim_key {
            "minecraft:overworld" => {
                let source = BiomeSourceKind::overworld(seed);
                ChunkGeneratorType::Overworld(OverworldGenerator::new(source, seed))
            }
            "minecraft:the_nether" => {
                let source = BiomeSourceKind::nether(seed);
                ChunkGeneratorType::Nether(NetherGenerator::new(source, seed))
            }
            "minecraft:the_end" => {
                let source = BiomeSourceKind::end(seed);
                ChunkGeneratorType::End(EndGenerator::new(source, seed))
            }
            _ => unreachable!(),
        });
        let world = create_test_world(dim_key, dim_type, seed, generator);
        let expected_feature_hashes = dim_data
            .chunks
            .iter()
            .filter_map(|entry| {
                entry
                    .stages
                    .get(FEATURE_STAGE)
                    .map(|hash| ((entry.x, entry.z), hash.as_str()))
            })
            .collect::<FxHashMap<_, _>>();
        let expected_carver_hashes = dim_data
            .chunks
            .iter()
            .filter_map(|entry| {
                entry
                    .stages
                    .get(CARVERS_STAGE)
                    .map(|hash| ((entry.x, entry.z), hash.as_str()))
            })
            .collect::<FxHashMap<_, _>>();
        eprintln!(
            "[{dim_short}/{LIGHT_STAGE}] preparing {} chunks to CARVERS in x/z order",
            stage_entries.len()
        );
        let mut carver_requests = Vec::with_capacity(stage_entries.len());
        for (chunk_x, chunk_z, _, _) in stage_entries.iter().copied() {
            let pos = ChunkPos::new(chunk_x, chunk_z);
            let request =
                world
                    .chunk_map
                    .request_chunk(pos, ChunkStatus::Carvers, ChunkTicketKind::Command);
            drive_chunk_request(&world, &request, dim_short);
            carver_requests.push(request);
        }

        let reference_carver_blocks = load_reference_blocks(CARVERS_STAGE, dim_short);
        let mut carver_mismatches = Vec::new();
        let total_carvers = stage_entries.len();
        eprintln!(
            "[{dim_short}/{CARVERS_STAGE}] verifying production carver state before {FEATURE_STAGE}"
        );
        for (i, ((chunk_x, chunk_z, _, _), request)) in stage_entries
            .iter()
            .copied()
            .zip(carver_requests.iter())
            .enumerate()
        {
            let Some(expected_carver_hash) =
                expected_carver_hashes.get(&(chunk_x, chunk_z)).copied()
            else {
                panic!(
                    "{dim_short}/{CARVERS_STAGE}: missing carver hash for light fixture chunk ({chunk_x}, {chunk_z})"
                );
            };
            let Some(ready_chunks) = request.ready_chunks() else {
                panic!(
                    "{dim_short}/{CARVERS_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                );
            };
            let Some(holder) = ready_chunks.holders.into_iter().next() else {
                panic!(
                    "{dim_short}/{CARVERS_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                );
            };
            let Some(chunk) = holder.try_chunk(ChunkStatus::Carvers) else {
                panic!("ready carver chunk missing at ({chunk_x}, {chunk_z})");
            };
            let actual_carver_hash = compute_block_hash(chunk.sections());
            let ok = actual_carver_hash == expected_carver_hash;
            if (i + 1) % 10 == 0 || i + 1 == total_carvers || !ok {
                let status = if ok { "OK" } else { "MISMATCH" };
                eprintln!(
                    "[{dim_short}/{CARVERS_STAGE}] ({chunk_x:3},{chunk_z:3}) {status} expected={expected_carver_hash} actual={actual_carver_hash}  [{}/{} before {FEATURE_STAGE}]",
                    i + 1,
                    total_carvers,
                );
            }

            if !ok {
                let block_diffs = reference_carver_blocks
                    .as_ref()
                    .and_then(|refs| refs.get(&(chunk_x, chunk_z)))
                    .map(|ref_data| diff_chunk(chunk.sections(), ref_data, min_y));
                carver_mismatches.push((
                    chunk_x,
                    chunk_z,
                    expected_carver_hash.to_owned(),
                    actual_carver_hash,
                    block_diffs,
                ));
                if stop_after_first_mismatch {
                    break;
                }
            }
        }

        if !carver_mismatches.is_empty() {
            let failed = carver_mismatches.len();
            let mut msg = format!(
                "{dim_short}/{CARVERS_STAGE}: {failed}/{total_carvers} chunks do not match vanilla before {FEATURE_STAGE}\n"
            );
            for (x, z, expected_hash, actual_hash, block_diffs) in &carver_mismatches {
                let _ = writeln!(
                    msg,
                    "  Chunk ({x:3},{z:3}): expected={expected_hash} actual={actual_hash}"
                );
                if let Some(diffs) = block_diffs {
                    msg.push_str(&format_chunk_diffs(diffs, *x, *z, min_y));
                }
            }
            panic!("{msg}");
        }

        if let Some(target) = debug_structure_refs {
            for ((chunk_x, chunk_z, _, _), request) in
                stage_entries.iter().copied().zip(carver_requests.iter())
            {
                if (chunk_x, chunk_z) != target {
                    continue;
                }
                let Some(ready_chunks) = request.ready_chunks() else {
                    panic!(
                        "{dim_short}/{CARVERS_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                    );
                };
                let Some(holder) = ready_chunks.holders.into_iter().next() else {
                    panic!(
                        "{dim_short}/{CARVERS_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                    );
                };
                let Some(chunk) = holder.try_chunk(ChunkStatus::Carvers) else {
                    panic!("ready carver chunk missing at ({chunk_x}, {chunk_z})");
                };
                eprintln!(
                    "[{dim_short}/{CARVERS_STAGE}] ({chunk_x},{chunk_z}) production structure references before {FEATURE_STAGE}:\n{}",
                    format_structure_references(&chunk)
                );
                let mut start_summary = String::new();
                for source_pos in referenced_structure_positions(&chunk, "minecraft:mineshaft") {
                    let _ = writeln!(
                        start_summary,
                        "    source ({}, {})",
                        source_pos.0.x, source_pos.0.y
                    );
                    if let Some(source_holder) = world
                        .chunk_map
                        .chunks
                        .read_sync(&source_pos, |_, holder| holder.clone())
                    {
                        if let Some(source_chunk) =
                            source_holder.try_chunk(ChunkStatus::StructureStarts)
                        {
                            start_summary.push_str(&format_structure_start_summary(
                                &source_chunk,
                                "minecraft:mineshaft",
                            ));
                        } else {
                            start_summary.push_str("      source chunk missing StructureStarts\n");
                        }
                    } else {
                        start_summary.push_str("      source holder missing\n");
                    }
                }
                if !start_summary.is_empty() {
                    eprintln!(
                        "[{dim_short}/{CARVERS_STAGE}] ({chunk_x},{chunk_z}) production referenced mineshaft starts before {FEATURE_STAGE}:\n{start_summary}"
                    );
                }
                break;
            }
        }

        eprintln!(
            "[{dim_short}/{LIGHT_STAGE}] preparing {} chunks to FEATURES in x/z order",
            stage_entries.len()
        );
        let mut feature_requests = Vec::with_capacity(stage_entries.len());
        for (chunk_x, chunk_z, _, _) in stage_entries.iter().copied() {
            let pos = ChunkPos::new(chunk_x, chunk_z);
            let request =
                world
                    .chunk_map
                    .request_chunk(pos, ChunkStatus::Features, ChunkTicketKind::Command);
            drive_chunk_request(&world, &request, dim_short);
            feature_requests.push(request);
        }
        let feature_request_by_pos = stage_entries
            .iter()
            .copied()
            .zip(feature_requests.iter())
            .map(|((chunk_x, chunk_z, _, _), request)| ((chunk_x, chunk_z), request))
            .collect::<FxHashMap<_, _>>();

        let reference_feature_blocks = load_reference_blocks(FEATURE_STAGE, dim_short);
        let mut feature_mismatches = Vec::new();
        let total_features = comparable_entries.len();
        eprintln!(
            "[{dim_short}/{FEATURE_STAGE}] verifying production feature state before {LIGHT_STAGE}"
        );
        for (i, (chunk_x, chunk_z, _, _)) in comparable_entries.iter().copied().enumerate() {
            let Some(request) = feature_request_by_pos.get(&(chunk_x, chunk_z)).copied() else {
                panic!(
                    "{dim_short}/{FEATURE_STAGE}: missing retained feature request for ({chunk_x}, {chunk_z})"
                );
            };
            let Some(expected_feature_hash) =
                expected_feature_hashes.get(&(chunk_x, chunk_z)).copied()
            else {
                panic!(
                    "{dim_short}/{FEATURE_STAGE}: missing feature hash for light fixture chunk ({chunk_x}, {chunk_z})"
                );
            };
            let Some(ready_chunks) = request.ready_chunks() else {
                panic!(
                    "{dim_short}/{FEATURE_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                );
            };
            let Some(holder) = ready_chunks.holders.into_iter().next() else {
                panic!(
                    "{dim_short}/{FEATURE_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                );
            };
            let Some(chunk) = holder.try_chunk(ChunkStatus::Features) else {
                panic!("ready feature chunk missing at ({chunk_x}, {chunk_z})");
            };
            let actual_feature_hash = compute_block_hash(chunk.sections());
            let ok = actual_feature_hash == expected_feature_hash;
            if (i + 1) % 10 == 0 || i + 1 == total_features || !ok {
                let status = if ok { "OK" } else { "MISMATCH" };
                eprintln!(
                    "[{dim_short}/{FEATURE_STAGE}] ({chunk_x:3},{chunk_z:3}) {status} expected={expected_feature_hash} actual={actual_feature_hash}  [{}/{} before {LIGHT_STAGE}]",
                    i + 1,
                    total_features,
                );
            }

            if !ok {
                let block_diffs = reference_feature_blocks
                    .as_ref()
                    .and_then(|refs| refs.get(&(chunk_x, chunk_z)))
                    .map(|ref_data| diff_chunk(chunk.sections(), ref_data, min_y));
                feature_mismatches.push((
                    chunk_x,
                    chunk_z,
                    expected_feature_hash.to_owned(),
                    actual_feature_hash,
                    block_diffs,
                ));
                if stop_after_first_mismatch {
                    break;
                }
            }
        }

        if !feature_mismatches.is_empty() {
            let failed = feature_mismatches.len();
            let mut msg = format!(
                "{dim_short}/{FEATURE_STAGE}: {failed}/{total_features} chunks do not match vanilla before {LIGHT_STAGE}\n"
            );
            for (x, z, expected_hash, actual_hash, block_diffs) in &feature_mismatches {
                let _ = writeln!(
                    msg,
                    "  Chunk ({x:3},{z:3}): expected={expected_hash} actual={actual_hash}"
                );
                if let Some(diffs) = block_diffs {
                    msg.push_str(&format_chunk_diffs(diffs, *x, *z, min_y));
                }
            }
            panic!("{msg}");
        }
        drop(carver_requests);

        if let Some(window) = debug_light_window {
            for ((chunk_x, chunk_z, _, _), request) in
                stage_entries.iter().copied().zip(feature_requests.iter())
            {
                if (chunk_x, chunk_z) != (window.chunk_x, window.chunk_z) {
                    continue;
                }
                let Some(ready_chunks) = request.ready_chunks() else {
                    panic!(
                        "{dim_short}/{FEATURE_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                    );
                };
                let Some(holder) = ready_chunks.holders.into_iter().next() else {
                    panic!(
                        "{dim_short}/{FEATURE_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                    );
                };
                let Some(chunk) = holder.try_chunk(ChunkStatus::Features) else {
                    panic!("ready feature chunk missing at ({chunk_x}, {chunk_z})");
                };
                eprintln!(
                    "[{dim_short}/{FEATURE_STAGE}] ({chunk_x},{chunk_z}) debug block window after tracked FEATURES:\n{}",
                    format_debug_block_window(window, &chunk)
                );
                break;
            }
        }
        let mut light_request_positions = FxHashSet::default();
        for (chunk_x, chunk_z, _, _) in stage_entries.iter().copied() {
            for dx in -light_dependency_radius..=light_dependency_radius {
                for dz in -light_dependency_radius..=light_dependency_radius {
                    light_request_positions.insert((chunk_x + dx, chunk_z + dz));
                }
            }
        }
        let light_request_positions = sorted_positions(&light_request_positions);

        eprintln!(
            "[{dim_short}/{LIGHT_STAGE}] requesting light for {} chunks in x/z order ({} compared, radius {})",
            light_request_positions.len(),
            comparable_entries.len(),
            light_dependency_radius
        );

        let total_requests = light_request_positions.len();
        let mut light_requests = FxHashMap::with_capacity_and_hasher(total_requests, FxBuildHasher);
        for (i, (chunk_x, chunk_z)) in light_request_positions.iter().copied().enumerate() {
            let pos = ChunkPos::new(chunk_x, chunk_z);
            let request =
                world
                    .chunk_map
                    .request_chunk(pos, ChunkStatus::Light, ChunkTicketKind::Command);
            drive_chunk_request(&world, &request, dim_short);
            let Some(ready_chunks) = request.ready_chunks() else {
                panic!(
                    "{dim_short}/{LIGHT_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                );
            };
            if ready_chunks.holders.is_empty() {
                panic!(
                    "{dim_short}/{LIGHT_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                );
            }
            light_requests.insert((chunk_x, chunk_z), request);

            if (i + 1) % 10 == 0 || i + 1 == total_requests {
                eprintln!(
                    "[{dim_short}/{LIGHT_STAGE}] ({chunk_x:3},{chunk_z:3}) ready [{}/{}]",
                    i + 1,
                    total_requests
                );
            }
        }

        drive_chunk_generation_idle(&world, dim_short);

        if let Some(window) = debug_light_window
            && let Some(holder) = world.chunk_map.chunks.read_sync(
                &ChunkPos::new(window.chunk_x, window.chunk_z),
                |_, holder| Arc::clone(holder),
            )
            && let Some(chunk) = holder.try_chunk(ChunkStatus::Light)
        {
            eprintln!(
                "[{dim_short}/{LIGHT_STAGE}] ({},{}) debug block window after tracked LIGHT:\n{}",
                window.chunk_x,
                window.chunk_z,
                format_debug_block_window(window, &chunk)
            );
        }

        eprintln!(
            "[{dim_short}/{LIGHT_STAGE}] comparing {} chunks after all tracked LIGHT requests are ready and generation is idle",
            comparable_entries.len()
        );

        let mut mismatches = Vec::new();
        let total = comparable_entries.len();
        for (i, (chunk_x, chunk_z, expected_hash, expected_light_debug)) in
            comparable_entries.iter().copied().enumerate()
        {
            let Some(request) = light_requests.get(&(chunk_x, chunk_z)) else {
                panic!(
                    "{dim_short}/{LIGHT_STAGE}: missing retained LIGHT request for ({chunk_x}, {chunk_z})"
                );
            };
            let Some(ready_chunks) = request.ready_chunks() else {
                panic!(
                    "{dim_short}/{LIGHT_STAGE}: request for ({chunk_x}, {chunk_z}) reported ready without ready chunk"
                );
            };
            let Some(holder) = ready_chunks.holders.into_iter().next() else {
                panic!(
                    "{dim_short}/{LIGHT_STAGE}: missing ready holder for ({chunk_x}, {chunk_z})"
                );
            };
            let Some(chunk) = holder.try_chunk(ChunkStatus::Light) else {
                panic!("ready light chunk missing at ({chunk_x}, {chunk_z})");
            };
            let light = chunk.light();
            let actual_debug = emit_light_summary.then(|| actual_light_debug(&light, has_skylight));
            let expected_light_bytes = reference_lights
                .as_ref()
                .and_then(|lights| lights.get(&(chunk_x, chunk_z)));
            let actual_light_bytes =
                expected_light_bytes.map(|_| actual_light_bytes(&light, has_skylight));
            if audit_expected_source_light
                && let Some(expected_light_bytes) = expected_light_bytes
                && let Some(audit) = format_expected_source_light_audit(
                    expected_light_bytes,
                    &chunk,
                    chunk_x,
                    chunk_z,
                )
            {
                eprintln!(
                    "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) expected source-light audit:\n{audit}"
                );
            }
            let summary = actual_debug.as_ref().map(|debug| {
                let mut summary = format_light_debug(debug);
                let section_summary = debug_chunk_section_summary(&chunk);
                if !section_summary.is_empty() {
                    let _ = writeln!(summary, "  generated non-empty sections:");
                    summary.push_str(&section_summary);
                }
                summary
            });
            let actual_hash = compute_light_hash(&light, has_skylight);
            let raw_light_diff = if actual_hash == expected_hash
                || debug_raw_light_chunk.is_some_and(|target| target != (chunk_x, chunk_z))
            {
                None
            } else if let (Some(expected_light_bytes), Some(actual_light_bytes)) =
                (expected_light_bytes, actual_light_bytes.as_ref())
            {
                Some(debug_raw_light_differences(
                    expected_light_bytes,
                    actual_light_bytes,
                    &chunk,
                    chunk_x,
                    chunk_z,
                ))
            } else {
                None
            };
            if let Some(window) = debug_light_window
                && window.chunk_x == chunk_x
                && window.chunk_z == chunk_z
            {
                eprintln!(
                    "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) debug light window:\n{}",
                    format_debug_light_window(window, expected_light_bytes, &light, &chunk)
                );
            }
            drop(light);
            drop(chunk);

            let ok = actual_hash == expected_hash;
            if (i + 1) % 10 == 0 || i + 1 == total || !ok {
                let status = if ok { "OK" } else { "MISMATCH" };
                eprintln!(
                    "[{dim_short}/{LIGHT_STAGE}] ({chunk_x:3},{chunk_z:3}) {status} expected={expected_hash} actual={actual_hash}  [{}/{total}]",
                    i + 1,
                );
            }

            if !ok {
                if let Some(summary) = summary {
                    eprintln!(
                        "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) actual packet light sections:\n{summary}"
                    );
                }
                if let (Some(expected_debug), Some(actual_debug)) =
                    (expected_light_debug, actual_debug.as_ref())
                {
                    eprintln!(
                        "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) expected vanilla packet light sections:\n{}",
                        format_light_debug(expected_debug)
                    );
                    eprintln!(
                        "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) differing light sections:\n{}",
                        debug_light_differences(expected_debug, &actual_debug)
                    );
                }
                if let Some(raw_light_diff) = raw_light_diff {
                    eprintln!(
                        "[{dim_short}/{LIGHT_STAGE}] ({chunk_x},{chunk_z}) raw light value differences:\n{}",
                        raw_light_diff
                    );
                }
                mismatches.push((chunk_x, chunk_z, expected_hash.to_owned(), actual_hash));
                if stop_after_first_mismatch {
                    break;
                }
            }
        }
        drop(feature_requests);

        if mismatches.is_empty() {
            continue;
        }

        let failed = mismatches.len();
        let mut msg =
            format!("{dim_short}/{LIGHT_STAGE}: {failed}/{total} chunks do not match vanilla\n");
        for (x, z, expected_hash, actual_hash) in &mismatches {
            let _ = writeln!(
                msg,
                "  ({x:3},{z:3}): expected {expected_hash}, got {actual_hash}"
            );
        }
        panic!("{msg}");
    }

    if !saw_light_hashes {
        eprintln!(
            "chunk_stage_hashes.json has no {LIGHT_STAGE} entries; skipping light hash regression"
        );
    }
}

#[expect(
    clippy::too_many_lines,
    clippy::similar_names,
    reason = "large test with many hash assertions"
)]
fn chunk_stage_hashes_inner() {
    use steel_core::behavior::init_behaviors;
    use steel_core::block_entity::init_block_entities;
    use steel_core::entity::init_entities;
    use steel_core::worldgen::{
        BiomeSourceKind, EndGenerator, NetherGenerator, OverworldGenerator,
    };
    use steel_registry::{REGISTRY, Registry};

    let mut registry = Registry::new_vanilla();
    registry.freeze();
    let _ = REGISTRY.init(registry);
    init_behaviors();
    init_block_entities();
    init_entities();

    let expected = load_expected_hashes();
    let seed = expected.seed;
    assert_eq!(seed, 13579, "Expected seed 13579");
    assert_eq!(
        expected.chunk_generation_order, CHUNK_GENERATION_ORDER_X_Z_ASCENDING,
        "chunk stage hash test only supports x/z ascending generation order"
    );
    let includes_features = STAGES.contains(&FEATURE_STAGE);
    assert!(
        !includes_features || STAGES.last().copied() == Some(FEATURE_STAGE),
        "features must remain the last checked stage because it consumes the local chunk map"
    );
    if includes_features {
        assert_eq!(
            expected.feature_hash_capture.as_deref(),
            Some(FEATURE_HASH_CAPTURE_AFTER_ALL_READY),
            "features stage hashes must be extracted after all tracked features are ready; rerun the extractor"
        );
        assert_eq!(
            expected.hashset_iteration_order.as_deref(),
            Some(HASHSET_ITERATION_ORDER_INSERTION),
            "features stage hashes must be extracted with deterministic insertion-order HashSet normalization; rerun the extractor"
        );
    }
    let feature_step = GENERATION_PYRAMID.get_step_to(ChunkStatus::Features);
    let feature_cache_radius = feature_step.direct_dependencies.get_radius() as i32;
    let feature_carver_radius = feature_step
        .direct_dependencies
        .get_radius_of(ChunkStatus::Carvers) as i32;
    let debug_dimension = debug_dimension_filter();
    let debug_stage = debug_stage_filter();
    let debug_structure_refs = debug_structure_refs_filter();
    let stop_after_first_mismatch = env::var_os(DEBUG_STOP_AFTER_FIRST_MISMATCH_ENV).is_some();

    for &dim_key in DIMENSION_ORDER {
        if debug_dimension
            .as_deref()
            .is_some_and(|filter| filter != dim_key)
        {
            continue;
        }
        let Some(dim_data) = expected.dimensions.get(dim_key) else {
            continue;
        };

        let dim_short = dim_key.strip_prefix("minecraft:").unwrap_or(dim_key);
        let dim_type = match dim_key {
            "minecraft:overworld" => &vanilla_dimension_types::OVERWORLD,
            "minecraft:the_nether" => &vanilla_dimension_types::THE_NETHER,
            "minecraft:the_end" => &vanilla_dimension_types::THE_END,
            _ => panic!("Unknown dimension: {dim_key}"),
        };

        let min_y = dim_type.min_y;
        let height = dim_type.height;
        let section_count = (height / 16) as usize;
        let min_qy = min_y >> 2;
        let total_quarts_y = (section_count * 4) as i32;

        let generator: Arc<ChunkGeneratorType> = Arc::new(match dim_key {
            "minecraft:overworld" => {
                let source = BiomeSourceKind::overworld(seed);
                ChunkGeneratorType::Overworld(OverworldGenerator::new(source, seed))
            }
            "minecraft:the_nether" => {
                let source = BiomeSourceKind::nether(seed);
                ChunkGeneratorType::Nether(NetherGenerator::new(source, seed))
            }
            "minecraft:the_end" => {
                let source = BiomeSourceKind::end(seed);
                ChunkGeneratorType::End(EndGenerator::new(source, seed))
            }
            _ => unreachable!(),
        });
        let feature_world = includes_features
            .then(|| create_test_world(dim_key, dim_type, seed, generator.clone()));
        let feature_context = feature_world
            .as_ref()
            .map(|world| world.chunk_map.world_gen_context.clone());

        eprintln!("=== {dim_key} ===");

        let debug_filter = debug_chunk_filter();
        let mut test_entries: Vec<&ChunkStageEntry> = if let Some(filter) = &debug_filter {
            dim_data
                .chunks
                .iter()
                .filter(|c| filter.contains(&(c.x, c.z)))
                .collect()
        } else {
            dim_data.chunks.iter().collect()
        };
        test_entries.sort_unstable_by_key(|entry| (entry.x, entry.z));
        let tracked_positions: FxHashSet<(i32, i32)> = test_entries
            .iter()
            .map(|entry| (entry.x, entry.z))
            .collect();

        // === Pre-pass: replicate vanilla's STRUCTURE_STARTS → STRUCTURE_REFERENCES →
        // BIOMES → NOISE pipeline before the per-stage hash loop. The beardifier in
        // production reads structure starts from referenced neighbor chunks, so the
        // test must populate those references the same way `generate_references` does
        // in `worldgen::stages::structures`. ===

        // 17×17 around each test chunk feeds STRUCTURE_REFERENCES. Surface and
        // feature dependency chunks add their required biome rings below.
        let mut starts_positions: FxHashSet<(i32, i32)> =
            FxHashSet::with_capacity_and_hasher(test_entries.len() * 289, FxBuildHasher);
        let mut biome_positions: FxHashSet<(i32, i32)> = FxHashSet::default();
        let mut feature_carver_positions: FxHashSet<(i32, i32)> = FxHashSet::default();
        for entry in &test_entries {
            if includes_features {
                for dx in -feature_cache_radius..=feature_cache_radius {
                    for dz in -feature_cache_radius..=feature_cache_radius {
                        starts_positions.insert((entry.x + dx, entry.z + dz));
                    }
                }
                for dx in -feature_carver_radius..=feature_carver_radius {
                    for dz in -feature_carver_radius..=feature_carver_radius {
                        feature_carver_positions.insert((entry.x + dx, entry.z + dz));
                    }
                }
            }
            for dx in -1i32..=1 {
                for dz in -1i32..=1 {
                    biome_positions.insert((entry.x + dx, entry.z + dz));
                }
            }
        }

        let reference_target_positions = if includes_features {
            sorted_positions(&feature_carver_positions)
        } else {
            test_entries
                .iter()
                .map(|entry| (entry.x, entry.z))
                .collect::<Vec<_>>()
        };
        if GENERATE_STRUCTURES {
            for &(target_x, target_z) in &reference_target_positions {
                for dx in -8i32..=8 {
                    for dz in -8i32..=8 {
                        starts_positions.insert((target_x + dx, target_z + dz));
                    }
                }
            }
        }
        if includes_features {
            for &(x, z) in &feature_carver_positions {
                for dx in -1i32..=1 {
                    for dz in -1i32..=1 {
                        biome_positions.insert((x + dx, z + dz));
                    }
                }
            }
        }
        if !GENERATE_STRUCTURES {
            starts_positions.extend(biome_positions.iter().copied());
        }

        let mut chunks: FxHashMap<(i32, i32), ChunkAccess> =
            FxHashMap::with_capacity_and_hasher(starts_positions.len(), FxBuildHasher);
        for &pos in &starts_positions {
            chunks.insert(pos, empty_proto_chunk(pos, section_count, min_y, height));
        }
        eprintln!(
            "[{dim_short}] Allocated {} proto chunks (structures: {GENERATE_STRUCTURES})",
            chunks.len()
        );

        // STRUCTURE_STARTS — per-chunk; uses biome_source directly (no chunk biomes
        // required). Most chunks early-exit at `placement.is_structure_chunk`.
        if GENERATE_STRUCTURES {
            for chunk in chunks.values() {
                generator.create_structures(chunk);
            }
        }

        // BIOMES — only for the 3×3 around each test chunk (surface stage's lookup).
        for &pos in &biome_positions {
            generator.create_biomes(chunk_or_panic(&chunks, pos));
        }

        // STRUCTURE_REFERENCES — mirror of `generate_references`: scan 17×17 for each
        // chunk that will be read at noise/carver stage, recording which neighbor chunks
        // hold a start whose inflated BB intersects it.
        if GENERATE_STRUCTURES {
            for &(target_x, target_z) in &reference_target_positions {
                let target_block_x = target_x * 16;
                let target_block_z = target_z * 16;

                for source_x in (target_x - 8)..=(target_x + 8) {
                    for source_z in (target_z - 8)..=(target_z + 8) {
                        let Some(source_chunk) = chunks.get(&(source_x, source_z)) else {
                            continue;
                        };
                        let starts = source_chunk.structure_starts();
                        for (structure_id, start) in starts.iter() {
                            // `start.bounding_box` is already inflated by `bb_inflate`,
                            // matching `worldgen::stages::structures::generate_references`.
                            let Some(bb) = start.bounding_box else {
                                continue;
                            };
                            if bb.intersects_xz(
                                target_block_x,
                                target_block_z,
                                target_block_x + 15,
                                target_block_z + 15,
                            ) {
                                chunk_or_panic(&chunks, (target_x, target_z))
                                    .structure_references_mut()
                                    .entry(structure_id.clone())
                                    .or_default()
                                    .insert(ChunkPos::new(source_x, source_z));
                            }
                        }
                    }
                }
            }
        }

        // NOISE — fill_from_noise with per-chunk beardifier built from references.
        let noise_positions = if includes_features {
            sorted_positions(&feature_carver_positions)
        } else {
            test_entries
                .iter()
                .map(|entry| (entry.x, entry.z))
                .collect()
        };
        for pos in noise_positions {
            let chunk = chunk_or_panic(&chunks, pos);
            let beardifier = if GENERATE_STRUCTURES {
                build_test_beardifier(chunk, &chunks)
            } else {
                None
            };
            generator.fill_from_noise(chunk, beardifier.as_ref());
        }

        for &stage in STAGES {
            if debug_stage.as_deref().is_some_and(|filter| filter != stage) {
                continue;
            }
            let reference_blocks = load_reference_blocks(stage, dim_short);
            let has_reference = reference_blocks.is_some();

            let stage_entries: Vec<_> = test_entries
                .iter()
                .filter_map(|e| e.stages.get(stage).map(|hash| (e.x, e.z, hash.as_str())))
                .collect();
            let total = stage_entries.len();
            let mut mismatches = Vec::new();
            let feature_holders = if stage == FEATURE_STAGE {
                // Vanilla requests all sampled chunks to CARVERS first, then requests
                // FEATURES in x/z order. Untracked radius-1 dependencies must reach
                // CARVERS, but their feature stage must not run.
                let dependency_positions = sorted_positions(&feature_carver_positions);
                let feature_stage_only = debug_stage.as_deref() == Some(FEATURE_STAGE);
                for &pos in &dependency_positions {
                    if !feature_stage_only && tracked_positions.contains(&pos) {
                        continue;
                    }
                    let chunk = chunk_or_panic(&chunks, pos);
                    let neighbor_biomes = |qx: i32, qy: i32, qz: i32| -> u16 {
                        let cx = qx >> 2;
                        let cz = qz >> 2;
                        let neighbor = chunk_or_panic(&chunks, (cx, cz));
                        let sections = neighbor.sections();
                        let local_qx = (qx - cx * 4) as usize;
                        let local_qz = (qz - cz * 4) as usize;
                        let qy_clamped = (qy - min_qy).clamp(0, total_quarts_y - 1) as usize;
                        let section_idx = qy_clamped / 4;
                        let local_qy = qy_clamped % 4;
                        sections.sections[section_idx]
                            .read()
                            .biomes
                            .get(local_qx, local_qy, local_qz)
                    };
                    generator.build_surface(chunk, &neighbor_biomes);
                }
                for &pos in &dependency_positions {
                    if !feature_stage_only && tracked_positions.contains(&pos) {
                        continue;
                    }
                    generator.apply_carvers(chunk_or_panic(&chunks, pos));
                }
                Some(Arc::new(build_feature_holders(
                    mem::take(&mut chunks),
                    &feature_carver_positions,
                    min_y,
                    height,
                )))
            } else {
                None
            };

            if stage == FEATURE_STAGE {
                let Some(holders) = &feature_holders else {
                    panic!("features stage missing chunk holders");
                };
                let Some(context) = &feature_context else {
                    panic!("features stage missing worldgen context");
                };

                for &(chunk_x, chunk_z, _) in &stage_entries {
                    let center = ChunkPos::new(chunk_x, chunk_z);
                    let Some(center_holder) = holders.get(&(chunk_x, chunk_z)) else {
                        panic!("Missing feature center chunk ({chunk_x}, {chunk_z})");
                    };
                    {
                        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Carvers) else {
                            panic!("Feature center chunk ({chunk_x}, {chunk_z}) missing");
                        };
                        chunk.prime_final_heightmaps();
                        if debug_structure_refs == Some((chunk_x, chunk_z)) {
                            eprintln!(
                                "[{dim_short}/{CARVERS_STAGE}] ({chunk_x},{chunk_z}) deterministic structure references before {FEATURE_STAGE}:\n{}",
                                format_structure_references(&chunk)
                            );
                            let mut start_summary = String::new();
                            for source_pos in
                                referenced_structure_positions(&chunk, "minecraft:mineshaft")
                            {
                                let _ = writeln!(
                                    start_summary,
                                    "    source ({}, {})",
                                    source_pos.0.x, source_pos.0.y
                                );
                                if let Some(source_holder) =
                                    holders.get(&(source_pos.0.x, source_pos.0.y))
                                {
                                    if let Some(source_chunk) =
                                        source_holder.try_chunk(ChunkStatus::StructureStarts)
                                    {
                                        start_summary.push_str(&format_structure_start_summary(
                                            &source_chunk,
                                            "minecraft:mineshaft",
                                        ));
                                    } else {
                                        start_summary.push_str(
                                            "      source chunk missing StructureStarts\n",
                                        );
                                    }
                                } else {
                                    start_summary.push_str("      source holder missing\n");
                                }
                            }
                            if !start_summary.is_empty() {
                                eprintln!(
                                    "[{dim_short}/{CARVERS_STAGE}] ({chunk_x},{chunk_z}) deterministic referenced mineshaft starts before {FEATURE_STAGE}:\n{start_summary}"
                                );
                            }
                        }
                    }
                    let cache_holders = holders.clone();
                    let cache = Arc::new(StaticCache2D::create(
                        chunk_x,
                        chunk_z,
                        feature_cache_radius,
                        move |x, z| match cache_holders.get(&(x, z)) {
                            Some(holder) => holder.clone(),
                            None => panic!("Missing feature dependency chunk ({x}, {z})"),
                        },
                    ));
                    let region_random =
                        generator.create_worldgen_region_random(seed as i64, center);
                    let mut region = steel_core::worldgen::WorldGenRegion::new(
                        context,
                        feature_step,
                        &cache,
                        center,
                        region_random,
                    );
                    generator.apply_biome_decorations(&mut region);
                }
            }

            for (i, &(chunk_x, chunk_z, expected_hash)) in stage_entries.iter().enumerate() {
                let actual_hash = if stage == FEATURE_STAGE {
                    let Some(holders) = &feature_holders else {
                        panic!("features stage missing chunk holders");
                    };
                    let Some(holder) = holders.get(&(chunk_x, chunk_z)) else {
                        panic!("Missing feature center chunk ({chunk_x}, {chunk_z})");
                    };
                    let Some(chunk) = holder.try_chunk(ChunkStatus::Carvers) else {
                        panic!("Feature center chunk ({chunk_x}, {chunk_z}) missing");
                    };
                    compute_block_hash(chunk.sections())
                } else {
                    let chunk = chunk_or_panic(&chunks, (chunk_x, chunk_z));

                    // Apply current stage (structure_starts, references, biomes, noise
                    // already done by pre-pass).
                    if stage != "minecraft:noise" {
                        let neighbor_biomes = |qx: i32, qy: i32, qz: i32| -> u16 {
                            let cx = qx >> 2;
                            let cz = qz >> 2;
                            let neighbor = chunk_or_panic(&chunks, (cx, cz));
                            let sections = neighbor.sections();
                            let local_qx = (qx - cx * 4) as usize;
                            let local_qz = (qz - cz * 4) as usize;
                            let qy_clamped = (qy - min_qy).clamp(0, total_quarts_y - 1) as usize;
                            let section_idx = qy_clamped / 4;
                            let local_qy = qy_clamped % 4;
                            sections.sections[section_idx]
                                .read()
                                .biomes
                                .get(local_qx, local_qy, local_qz)
                        };

                        match stage {
                            "minecraft:surface" => generator.build_surface(chunk, &neighbor_biomes),
                            "minecraft:carvers" => generator.apply_carvers(chunk),
                            _ => panic!("Stage {stage} not yet implemented in test harness"),
                        }
                    }

                    compute_block_hash(chunk.sections())
                };

                let ok = actual_hash == expected_hash;
                if (i + 1) % 10 == 0 || i + 1 == total || !ok {
                    let status = if ok { "OK" } else { "MISMATCH" };
                    eprintln!(
                        "[{dim_short}/{stage}] ({chunk_x:3},{chunk_z:3}) {status} expected={expected_hash} actual={actual_hash}  [{}/{total}]",
                        i + 1,
                    );
                }

                if actual_hash != expected_hash {
                    let block_diffs = reference_blocks
                        .as_ref()
                        .and_then(|refs| refs.get(&(chunk_x, chunk_z)))
                        .map(|ref_data| {
                            if stage == FEATURE_STAGE {
                                let Some(holders) = &feature_holders else {
                                    panic!("features stage missing chunk holders");
                                };
                                let Some(holder) = holders.get(&(chunk_x, chunk_z)) else {
                                    panic!("Missing feature center chunk ({chunk_x}, {chunk_z})");
                                };
                                let Some(chunk) = holder.try_chunk(ChunkStatus::Carvers) else {
                                    panic!("Feature center chunk ({chunk_x}, {chunk_z}) missing");
                                };
                                diff_chunk(chunk.sections(), ref_data, min_y)
                            } else {
                                let chunk = chunk_or_panic(&chunks, (chunk_x, chunk_z));
                                diff_chunk(chunk.sections(), ref_data, min_y)
                            }
                        });

                    mismatches.push((
                        chunk_x,
                        chunk_z,
                        expected_hash.to_owned(),
                        actual_hash,
                        block_diffs,
                    ));
                    if stop_after_first_mismatch {
                        break;
                    }
                }
            }

            if mismatches.is_empty() {
                continue;
            }

            let failed = mismatches.len();
            let mut msg =
                format!("{dim_short}/{stage}: {failed}/{total} chunks do not match vanilla");
            if !has_reference {
                msg.push_str(" (no binary reference data, showing hashes only)");
            }
            msg.push('\n');

            for (x, z, expected_hash, actual_hash, block_diffs) in &mismatches {
                match block_diffs {
                    Some(diffs) if !diffs.is_empty() => {
                        msg.push_str(&format_chunk_diffs(diffs, *x, *z, min_y));
                    }
                    _ => {
                        let _ = writeln!(
                            msg,
                            "  ({x:3},{z:3}): expected {expected_hash}, got {actual_hash}"
                        );
                    }
                }
            }

            panic!("{msg}");
        }
    }
}

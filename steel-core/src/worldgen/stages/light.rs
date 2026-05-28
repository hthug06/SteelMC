use std::sync::Arc;

use crate::chunk::{
    chunk_access::ChunkStatus,
    chunk_generation_task::StaticCache2D,
    chunk_holder::ChunkHolder,
    chunk_pyramid::ChunkStep,
    light::{
        BlockLightChunkEdgeChecks, LightCacheLayout, LightCacheSetupRadius, LightLayer,
        LightSectionRange, LightWorkset, SkyLightChunkEdgeChecks, check_block_light_chunk_edges,
        check_sky_light_chunk_edges, force_load_block_light_chunk, force_load_sky_light_chunk,
        propagate_block_light_chunk, propagate_sky_light_chunk,
    },
};
use crate::worldgen::context::WorldGenContext;
use steel_utils::SectionPos;

pub(crate) fn initialize(
    _context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    _cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let Some(chunk) = holder.try_chunk(ChunkStatus::Features) else {
        panic!("Chunk not found at status Features");
    };

    chunk.initialize_light_sources();
}

pub(crate) fn generate(
    context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let (sky_updates, block_updates) = run_light_stage(
        cache,
        holder.as_ref(),
        context.world().dimension_type.has_skylight,
    );
    publish_light_updates(&context, LightLayer::Sky, sky_updates);
    publish_light_updates(&context, LightLayer::Block, block_updates);
}

pub(crate) fn load(
    context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let (sky_updates, block_updates) = run_loaded_light_stage(
        cache,
        holder.as_ref(),
        context.world().dimension_type.has_skylight,
    );
    publish_light_updates(&context, LightLayer::Sky, sky_updates);
    publish_light_updates(&context, LightLayer::Block, block_updates);
}

fn run_light_stage(
    cache: &StaticCache2D<Arc<ChunkHolder>>,
    holder: &ChunkHolder,
    has_skylight: bool,
) -> (Vec<SectionPos>, Vec<SectionPos>) {
    let center = holder.get_pos();
    if holder.try_chunk(ChunkStatus::InitializeLight).is_none() {
        panic!("Chunk not found at status InitializeLight");
    }

    let Ok(range) = LightSectionRange::from_world_height(holder.min_y(), holder.height()) else {
        panic!("invalid world height for light stage");
    };

    let layout = LightCacheLayout::new(center, range);
    let Ok(workset) = LightWorkset::setup_with_scopes(
        layout,
        LightCacheSetupRadius::Full,
        true,
        |pos| {
            let holder = cache.try_get(pos.0.x, pos.0.y)?;
            holder
                .try_chunk(ChunkStatus::InitializeLight)
                .is_some()
                .then(|| Arc::clone(holder))
        },
        |cached_chunk, holder, _chunk| {
            let status = holder.persisted_status();
            let center_chunk = cached_chunk.chunk_pos == center;
            let initialized = status.is_some_and(|status| status >= ChunkStatus::InitializeLight);
            let lit = status.is_some_and(|status| status >= ChunkStatus::Light);
            // InitializeLight neighbors provide block/emptiness data. Only the center and already
            // lit neighbors expose writable light, matching ScalableLux's provisional neighbor
            // behavior without making unlit neighbors' temporary nibbles packet-visible.
            (center_chunk || initialized, center_chunk || lit)
        },
    ) else {
        panic!("required light-stage chunk is missing");
    };

    let sky_updates = if has_skylight {
        match propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) {
            Ok(result) => result.updated_sections,
            Err(error) => panic!("sky light chunk propagation failed: {error:?}"),
        }
    } else {
        Vec::new()
    };
    let block_result =
        match propagate_block_light_chunk(&workset, BlockLightChunkEdgeChecks::Required) {
            Ok(result) => result,
            Err(error) => panic!("block light chunk propagation failed: {error:?}"),
        };

    (sky_updates, block_result.updated_sections)
}

fn run_loaded_light_stage(
    cache: &StaticCache2D<Arc<ChunkHolder>>,
    holder: &ChunkHolder,
    has_skylight: bool,
) -> (Vec<SectionPos>, Vec<SectionPos>) {
    let center = holder.get_pos();
    if holder.try_chunk(ChunkStatus::Light).is_none() {
        panic!("Chunk not found at status Light");
    }

    let Ok(range) = LightSectionRange::from_world_height(holder.min_y(), holder.height()) else {
        panic!("invalid world height for loaded light stage");
    };

    let layout = LightCacheLayout::new(center, range);
    let Ok(workset) = LightWorkset::setup(
        layout,
        LightCacheSetupRadius::Full,
        true,
        |pos| {
            let holder = cache.try_get(pos.0.x, pos.0.y)?;
            holder
                .try_chunk(ChunkStatus::Light)
                .is_some()
                .then(|| Arc::clone(holder))
        },
        |_| true,
    ) else {
        panic!("required loaded light-stage chunk is missing");
    };

    let mut sky_updates = if has_skylight {
        match force_load_sky_light_chunk(&workset) {
            Ok(result) => result.updated_sections,
            Err(error) => panic!("loaded sky light force-load failed: {error:?}"),
        }
    } else {
        Vec::new()
    };
    let mut block_updates = match force_load_block_light_chunk(&workset) {
        Ok(result) => result.updated_sections,
        Err(error) => panic!("loaded block light force-load failed: {error:?}"),
    };

    if has_skylight {
        match check_sky_light_chunk_edges(&workset) {
            Ok(result) => sky_updates.extend(result.updated_sections),
            Err(error) => panic!("loaded sky light edge validation failed: {error:?}"),
        }
    }
    match check_block_light_chunk_edges(&workset) {
        Ok(result) => block_updates.extend(result.updated_sections),
        Err(error) => panic!("loaded block light edge validation failed: {error:?}"),
    }

    (sky_updates, block_updates)
}

fn publish_light_updates(
    context: &WorldGenContext,
    layer: LightLayer,
    updated_sections: Vec<SectionPos>,
) {
    if updated_sections.is_empty() {
        return;
    }

    let world = context.world();
    for section_pos in updated_sections {
        world.chunk_map.light_changed(layer, section_pos);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::{BlockPos, ChunkPos};

    use super::*;
    use crate::{
        behavior::init_behaviors,
        chunk::{
            chunk_access::ChunkAccess,
            proto_chunk::ProtoChunk,
            section::{ChunkSection, Sections},
        },
    };

    fn init_tests() {
        init_test_registry();
        init_behaviors();
    }

    fn holder_with_section_at_status(
        pos: ChunkPos,
        section: ChunkSection,
        status: ChunkStatus,
    ) -> Arc<ChunkHolder> {
        let sections = Sections::from_owned(vec![section].into_boxed_slice());
        let proto = ProtoChunk::new(sections, pos, 0, 16, Weak::new());
        proto.set_status(status);
        if status >= ChunkStatus::InitializeLight {
            proto.initialize_light_sources();
        }
        let holder = Arc::new(ChunkHolder::new(pos, 0, 0, 16));
        holder.insert_chunk(ChunkAccess::Proto(proto), status);
        holder
    }

    fn holder_with_section(pos: ChunkPos, section: ChunkSection) -> Arc<ChunkHolder> {
        holder_with_section_at_status(pos, section, ChunkStatus::InitializeLight)
    }

    fn set_visible_light(
        holder: &ChunkHolder,
        layer: LightLayer,
        section_y: i32,
        block_pos: BlockPos,
        level: u8,
    ) {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Light) else {
            panic!("test chunk should be available at Light");
        };
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(holder.get_pos(), range);
        let Some(cached_block) = layout.cached_block(block_pos) else {
            panic!("test block should be cached");
        };

        let mut light = chunk.light_mut();
        let storage = match layer {
            LightLayer::Sky => &mut light.sky,
            LightLayer::Block => &mut light.block,
        };
        let Some(nibble) = storage.nibble_mut(section_y) else {
            panic!("test nibble should be inside light range");
        };
        nibble.set_non_null();
        nibble.set_at_index(cached_block.local_index, level);
        assert!(nibble.update_visible());
    }

    #[test]
    fn light_stage_generates_center_sky_and_block_light() {
        init_tests();
        let cache = StaticCache2D::create(0, 0, 1, |x, z| {
            let pos = ChunkPos::new(x, z);
            let mut section = ChunkSection::new_empty();
            if pos == ChunkPos::new(0, 0) {
                section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
            }
            holder_with_section(pos, section)
        });
        let center_holder = Arc::clone(cache.get(0, 0));

        run_light_stage(&cache, center_holder.as_ref(), true);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::InitializeLight) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range);
        let Some(source) = layout.cached_block(BlockPos::new(1, 1, 1)) else {
            panic!("source block should be cached");
        };
        let Some(sky) = layout.cached_block(BlockPos::new(1, 15, 1)) else {
            panic!("sky block should be cached");
        };
        let Some(block_nibble) = light.block.nibble(0) else {
            panic!("block nibble should be inside light range");
        };
        let Some(sky_nibble) = light.sky.nibble(0) else {
            panic!("sky nibble should be inside light range");
        };

        assert_eq!(block_nibble.get_visible_at_index(source.local_index), 15);
        assert_eq!(sky_nibble.get_visible_at_index(sky.local_index), 15);
    }

    #[test]
    fn light_stage_skips_sky_light_when_dimension_has_no_skylight() {
        init_tests();
        let cache = StaticCache2D::create(0, 0, 1, |x, z| {
            let pos = ChunkPos::new(x, z);
            let mut section = ChunkSection::new_empty();
            if pos == ChunkPos::new(0, 0) {
                section.set_block_state(1, 1, 1, vanilla_blocks::LIGHT.default_state());
            }
            holder_with_section(pos, section)
        });
        let center_holder = Arc::clone(cache.get(0, 0));

        let (sky_updates, block_updates) = run_light_stage(&cache, center_holder.as_ref(), false);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::InitializeLight) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        let Ok(range) = LightSectionRange::from_world_height(0, 16) else {
            panic!("test height should create a valid light range");
        };
        let layout = LightCacheLayout::new(ChunkPos::new(0, 0), range);
        let Some(source) = layout.cached_block(BlockPos::new(1, 1, 1)) else {
            panic!("source block should be cached");
        };
        let Some(block_nibble) = light.block.nibble(0) else {
            panic!("block nibble should be inside light range");
        };
        let Some(sky_nibble) = light.sky.nibble(0) else {
            panic!("sky nibble should be inside light range");
        };

        assert!(sky_updates.is_empty());
        assert!(!block_updates.is_empty());
        assert_eq!(block_nibble.get_visible_at_index(source.local_index), 15);
        assert_eq!(
            sky_nibble.visible_state(),
            crate::chunk::light::LightNibbleState::Null
        );
    }

    #[test]
    fn loaded_light_stage_preserves_persisted_interior_block_light() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let cache = StaticCache2D::create(0, 0, 2, move |x, z| {
            let pos = ChunkPos::new(x, z);
            let mut section = ChunkSection::new_empty();
            if pos == center {
                section.set_block_state(0, 0, 0, vanilla_blocks::STONE.default_state());
            }
            holder_with_section_at_status(pos, section, ChunkStatus::Light)
        });
        let center_holder = Arc::clone(cache.get(0, 0));
        let block_pos = BlockPos::new(8, 1, 8);
        set_visible_light(&center_holder, LightLayer::Block, 0, block_pos, 7);

        run_loaded_light_stage(&cache, center_holder.as_ref(), false);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Light) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        assert_eq!(light.get_light_value(LightLayer::Block, block_pos), 7);
    }

    #[test]
    fn loaded_light_stage_validates_persisted_sky_light() {
        init_tests();
        let center = ChunkPos::new(0, 0);
        let cache = StaticCache2D::create(0, 0, 2, move |x, z| {
            let pos = ChunkPos::new(x, z);
            let mut section = ChunkSection::new_empty();
            if pos == center {
                section.set_block_state(0, 0, 0, vanilla_blocks::STONE.default_state());
            }
            holder_with_section_at_status(pos, section, ChunkStatus::Light)
        });
        let center_holder = Arc::clone(cache.get(0, 0));
        let sky_pos = BlockPos::new(8, 15, 8);
        set_visible_light(&center_holder, LightLayer::Sky, 0, sky_pos, 15);

        run_loaded_light_stage(&cache, center_holder.as_ref(), true);

        let Some(chunk) = center_holder.try_chunk(ChunkStatus::Light) else {
            panic!("test chunk should be available");
        };
        let light = chunk.light();
        assert_eq!(light.get_light_value(LightLayer::Sky, sky_pos), 15);
    }
}

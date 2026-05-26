use std::sync::Arc;

use crate::chunk::{
    chunk_access::ChunkStatus,
    chunk_generation_task::StaticCache2D,
    chunk_holder::ChunkHolder,
    chunk_pyramid::ChunkStep,
    light::{
        BlockLightChunkEdgeChecks, LightCacheLayout, LightCacheSetupRadius, LightLayer,
        LightSectionRange, LightWorkset, SkyLightChunkEdgeChecks, propagate_block_light_chunk,
        propagate_sky_light_chunk,
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
    let (sky_updates, block_updates) = run_light_stage(cache, holder.as_ref());
    publish_light_updates(&context, LightLayer::Sky, sky_updates);
    publish_light_updates(&context, LightLayer::Block, block_updates);
}

fn run_light_stage(
    cache: &StaticCache2D<Arc<ChunkHolder>>,
    holder: &ChunkHolder,
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
            (
                status.is_some_and(|status| status >= ChunkStatus::InitializeLight),
                cached_chunk.chunk_pos == center
                    || status.is_some_and(|status| status >= ChunkStatus::Light),
            )
        },
    ) else {
        panic!("required light-stage chunk is missing");
    };

    let sky_result = match propagate_sky_light_chunk(&workset, SkyLightChunkEdgeChecks::Required) {
        Ok(result) => result,
        Err(error) => panic!("sky light chunk propagation failed: {error:?}"),
    };
    let block_result =
        match propagate_block_light_chunk(&workset, BlockLightChunkEdgeChecks::Required) {
            Ok(result) => result,
            Err(error) => panic!("block light chunk propagation failed: {error:?}"),
        };

    (sky_result.updated_sections, block_result.updated_sections)
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

    fn holder_with_section(pos: ChunkPos, section: ChunkSection) -> Arc<ChunkHolder> {
        let sections = Sections::from_owned(vec![section].into_boxed_slice());
        let proto = ProtoChunk::new(sections, pos, 0, 16, Weak::new());
        let holder = Arc::new(ChunkHolder::new(pos, 0, 0, 16));
        holder.insert_chunk(ChunkAccess::Proto(proto), ChunkStatus::InitializeLight);
        holder
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

        run_light_stage(&cache, center_holder.as_ref());

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
}

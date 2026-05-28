use rayon::{
    ThreadPool,
    iter::{IntoParallelIterator, ParallelIterator},
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::{
    io, mem,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use steel_protocol::packet_traits::EncodedPacket;
use steel_protocol::packets::game::{
    BlockChange, CBlockUpdate, CLightUpdate, CSectionBlocksUpdate, CSetChunkCenter,
};
use steel_protocol::utils::ConnectionProtocol;
use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_registry::dimension_type::DimensionTypeRef;
use steel_utils::{BlockPos, ChunkPos, SectionPos, locks::SyncMutex};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::instrument;

use crate::behavior::BlockStateBehaviorExt;
use crate::behavior::{BLOCK_BEHAVIORS, FLUID_BEHAVIORS};
use crate::chunk::chunk_holder::ChunkHolder;
use crate::chunk::chunk_ticket_manager::{
    ChunkTicketManager, LevelChange, MAX_VIEW_DISTANCE, is_full,
};
use crate::chunk::light::{
    LightCacheLayout, LightCacheSetupRadius, LightLayer, LightSectionEmptinessChange,
    LightSectionRange, LightWorkset, build_chunk_light_update_packet_for_sections,
    propagate_block_light_changes_with_empty_sections,
    propagate_sky_light_changes_with_empty_sections,
};
use crate::chunk::player_chunk_view::PlayerChunkView;
use crate::chunk::{chunk_access::ChunkAccess, chunk_ticket_manager::is_ticked};
use crate::chunk::{chunk_access::ChunkStatus, chunk_generation_task::ChunkGenerationTask};
use crate::chunk_saver::ChunkStorage;
use crate::player::Player;
use crate::player::connection::NetworkConnection;
use crate::world::tick_scheduler::{BlockTick, FluidTick};
use crate::world::{ChunkUpdateRecipients, World};
use crate::worldgen::{ChunkGeneratorType, WorldGenContext};

/// Timing information for the game tick portion of chunk map operations.
#[derive(Debug, Default)]
pub struct ChunkMapGameTickTimings {
    /// Time spent broadcasting block changes.
    pub broadcast_changes: Duration,
    /// Time spent collecting tickable chunks.
    pub collect_tickable: Duration,
    /// Time spent ticking chunks (random ticks, etc.).
    pub tick_chunks: Duration,
    /// Number of chunks that were ticked.
    pub tickable_count: usize,
    /// Total number of loaded chunks.
    pub total_chunks: usize,
}

/// Timing information for the chunk scheduling tick operations.
#[derive(Debug, Default)]
pub struct ChunkMapSchedulingTimings {
    /// Time spent processing ticket updates.
    pub ticket_updates: Duration,
    /// Time spent creating/updating chunk holders.
    pub holder_creation: Duration,
    /// Time spent scheduling generation tasks.
    pub schedule_generation: Duration,
    /// Number of holders scheduled for generation.
    pub scheduled_count: usize,
    /// Time spent spawning generation tasks.
    pub run_generation: Duration,
    /// Time spent processing chunk unloads.
    pub process_unloads: Duration,
}

#[derive(Debug, Default)]
struct PendingLightUpdates {
    chunks: FxHashMap<ChunkPos, PendingChunkLightUpdates>,
    queued_chunks: Vec<ChunkPos>,
}

impl PendingLightUpdates {
    fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    fn queue_change(
        &mut self,
        chunk_pos: ChunkPos,
        pos: BlockPos,
        check_block: bool,
        empty_section_change: Option<LightSectionEmptinessChange>,
    ) {
        if !self.chunks.contains_key(&chunk_pos) {
            self.queued_chunks.push(chunk_pos);
        }

        let task = self.chunks.entry(chunk_pos).or_default();
        if check_block {
            task.changed_positions.insert(pos);
        }
        if let Some(change) = empty_section_change {
            task.changed_sections
                .insert(change.section_pos, change.empty);
        }
    }

    fn remove_chunk(&mut self, chunk_pos: ChunkPos) {
        self.chunks.remove(&chunk_pos);
    }

    fn drain(&mut self) -> Vec<(ChunkPos, PendingChunkLightUpdates)> {
        let mut chunks = mem::take(&mut self.chunks);
        let queued_chunks = mem::take(&mut self.queued_chunks);
        queued_chunks
            .into_iter()
            .filter_map(|chunk_pos| chunks.remove(&chunk_pos).map(|task| (chunk_pos, task)))
            .collect()
    }
}

#[derive(Debug, Default)]
struct PendingChunkLightUpdates {
    changed_positions: FxHashSet<BlockPos>,
    changed_sections: FxHashMap<SectionPos, bool>,
}

impl PendingChunkLightUpdates {
    fn is_empty(&self) -> bool {
        self.changed_positions.is_empty() && self.changed_sections.is_empty()
    }

    fn empty_section_changes(&self) -> Vec<LightSectionEmptinessChange> {
        let mut changes = self
            .changed_sections
            .iter()
            .map(|(&section_pos, &empty)| LightSectionEmptinessChange { section_pos, empty })
            .collect::<Vec<_>>();
        changes.sort_by(|left, right| {
            left.section_pos
                .x()
                .cmp(&right.section_pos.x())
                .then_with(|| left.section_pos.z().cmp(&right.section_pos.z()))
                .then_with(|| right.section_pos.y().cmp(&left.section_pos.y()))
        });
        changes
    }
}

/// A map of chunks managing their state, loading, and generation.
pub struct ChunkMap {
    /// Map of active chunks.
    pub chunks: scc::HashMap<ChunkPos, Arc<ChunkHolder>, FxBuildHasher>,
    /// Map of chunks currently being unloaded.
    pub unloading_chunks: scc::HashMap<ChunkPos, Arc<ChunkHolder>, FxBuildHasher>,
    /// Queue of pending generation tasks.
    pub pending_generation_tasks: SyncMutex<Vec<Arc<ChunkGenerationTask>>>,
    /// Tracker for background generation tasks.
    pub task_tracker: TaskTracker,
    /// Manager for chunk distances and tickets.
    pub chunk_tickets: SyncMutex<ChunkTicketManager>,
    /// The world generation context.
    pub world_gen_context: Arc<WorldGenContext>,
    /// The thread pool to use for chunk generation (throughput-oriented).
    pub generation_pool: Arc<ThreadPool>,
    /// The thread pool to use for chunk ticking (latency-oriented).
    //pub tick_pool: Arc<ThreadPool>,
    /// The runtime to use for chunk tasks.
    pub chunk_runtime: Arc<Runtime>,
    /// Storage backend for chunk saving and loading.
    pub storage: Arc<ChunkStorage>,
    /// Chunk holders with pending block changes to broadcast.
    pub chunks_to_broadcast: SyncMutex<Vec<Arc<ChunkHolder>>>,
    /// Coalesced block and section changes waiting for one ScalableLux-style light pass.
    pending_light_updates: SyncMutex<PendingLightUpdates>,
    /// Last length of `tickable_chunks` to pre-allocate with appropriate capacity.
    last_tickable_len: AtomicUsize,
    /// Parent cancellation token for all generation tasks.
    /// Child tokens are created per-task; cancelling this cancels everything.
    pub cancel_token: CancellationToken,
}

impl ChunkMap {
    /// Creates a new chunk map with a custom storage backend.
    ///
    /// This allows using different storage implementations (disk, RAM, etc.).
    #[must_use]
    pub fn new_with_storage(
        chunk_runtime: Arc<Runtime>,
        world: Weak<World>,
        _dimension_type: DimensionTypeRef,
        storage: Arc<ChunkStorage>,
        generator: Arc<ChunkGeneratorType>,
        generation_pool: Arc<ThreadPool>,
    ) -> Self {
        Self {
            chunks: scc::HashMap::default(),
            unloading_chunks: scc::HashMap::default(),
            pending_generation_tasks: SyncMutex::new(Vec::new()),
            task_tracker: TaskTracker::new(),
            chunk_tickets: SyncMutex::new(ChunkTicketManager::new()),
            world_gen_context: Arc::new(WorldGenContext::new(generator, world)),
            generation_pool,
            chunk_runtime,
            storage,
            chunks_to_broadcast: SyncMutex::new(Vec::new()),
            pending_light_updates: SyncMutex::new(PendingLightUpdates::default()),
            last_tickable_len: AtomicUsize::new(0),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Executes a function with access to a fully loaded chunk.
    /// Returns `None` if the chunk is not loaded or not at Full status.
    pub fn with_full_chunk<F, R>(&self, pos: ChunkPos, f: F) -> Option<R>
    where
        F: FnOnce(&ChunkAccess) -> R,
    {
        self.with_chunk_at_status(pos, ChunkStatus::Full, f)
    }

    /// Executes a function with access to a chunk at the requested generation status or later.
    /// Returns `None` if the chunk is not loaded or has not reached the requested status.
    pub(crate) fn with_chunk_at_status<F, R>(
        &self,
        pos: ChunkPos,
        status: ChunkStatus,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&ChunkAccess) -> R,
    {
        let chunk_holder = self.chunks.read_sync(&pos, |_, chunk| chunk.clone())?;
        let guard = chunk_holder.try_chunk(status)?;
        Some(f(&guard))
    }

    /// Loads full chunks in a square radius, runs `f`, then removes the temporary ticket.
    pub async fn with_full_chunks_in_radius<F, R>(
        self: &Arc<Self>,
        center: ChunkPos,
        radius: u8,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        let ticket_level = MAX_VIEW_DISTANCE.saturating_sub(radius);

        self.chunk_tickets.lock().add_ticket(center, ticket_level);
        self.tick_scheduling();

        let mut holders = Vec::new();
        let radius = i32::from(radius);
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let pos = ChunkPos::new(center.0.x + dx, center.0.y + dz);
                let Some(holder) = self.chunks.read_sync(&pos, |_, holder| holder.clone()) else {
                    self.chunk_tickets
                        .lock()
                        .remove_ticket(center, ticket_level);
                    self.tick_scheduling();
                    return None;
                };
                holders.push(holder);
            }
        }

        for holder in holders {
            if holder.await_chunk(ChunkStatus::Full).await.is_none() {
                self.chunk_tickets
                    .lock()
                    .remove_ticket(center, ticket_level);
                self.tick_scheduling();
                return None;
            }
        }

        let result = f();
        self.chunk_tickets
            .lock()
            .remove_ticket(center, ticket_level);
        self.tick_scheduling();

        Some(result)
    }

    /// Records a block change at the given position.
    /// This marks the chunk as having pending changes to broadcast.
    pub fn block_changed(&self, pos: BlockPos) {
        let chunk_pos = ChunkPos::new(
            SectionPos::block_to_section_coord(pos.0.x),
            SectionPos::block_to_section_coord(pos.0.z),
        );

        if let Some(holder) = self.chunks.read_sync(&chunk_pos, |_, h| Arc::clone(h))
            && holder.block_changed(pos)
        {
            // First change for this chunk - add to broadcast list
            self.chunks_to_broadcast.lock().push(holder);
        }
    }

    /// Records a light-section change at the given position.
    pub fn light_changed(&self, layer: LightLayer, section_pos: SectionPos) {
        let chunk_pos = ChunkPos::new(section_pos.x(), section_pos.z());

        if let Some(holder) = self.chunks.read_sync(&chunk_pos, |_, h| Arc::clone(h))
            && holder.light_changed(layer, section_pos)
        {
            self.chunks_to_broadcast.lock().push(holder);
        }
    }

    /// Queues a block or section light change for the next light propagation drain.
    ///
    /// ScalableLux batches changed block positions and section emptiness changes
    /// by chunk before running sky and block propagation. Steel keeps the same
    /// coalescing shape on the main tick thread.
    pub fn queue_light_change(
        &self,
        pos: BlockPos,
        check_block: bool,
        empty_section_change: Option<LightSectionEmptinessChange>,
    ) {
        if !check_block && empty_section_change.is_none() {
            return;
        }

        let chunk_pos = ChunkPos::new(
            SectionPos::block_to_section_coord(pos.0.x),
            SectionPos::block_to_section_coord(pos.0.z),
        );
        if !self.can_accept_queued_light_change(chunk_pos) {
            return;
        }

        let mut pending = self.pending_light_updates.lock();
        pending.queue_change(chunk_pos, pos, check_block, empty_section_change);
    }

    /// Drains all queued light updates and runs one scoped propagation per changed chunk.
    pub fn propagate_queued_light_changes(&self) {
        let tasks = {
            let mut pending = self.pending_light_updates.lock();
            if pending.is_empty() {
                return;
            }
            pending.drain()
        };

        for (center, task) in tasks {
            if task.is_empty() {
                continue;
            }
            self.propagate_queued_light_change(center, task);
        }
    }

    fn can_accept_queued_light_change(&self, center: ChunkPos) -> bool {
        self.chunks
            .read_sync(&center, |_, holder| {
                holder.try_chunk(ChunkStatus::Light).is_some()
            })
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(crate) fn has_pending_light_update_for_test(&self, chunk_pos: ChunkPos) -> bool {
        self.pending_light_updates
            .lock()
            .chunks
            .contains_key(&chunk_pos)
    }

    fn propagate_queued_light_change(&self, center: ChunkPos, task: PendingChunkLightUpdates) {
        let Some(workset) = self.light_workset_for_change(center) else {
            log::warn!("Failed to set up light workset for queued light update at {center:?}");
            return;
        };

        let empty_sections = task.empty_section_changes();
        let positions = task.changed_positions.into_iter().collect::<Vec<_>>();
        let world = self.world_gen_context.world();

        if world.dimension_type.has_skylight {
            let Ok(result) = propagate_sky_light_changes_with_empty_sections(
                &workset,
                positions.iter().copied(),
                empty_sections.iter().copied(),
            ) else {
                log::warn!("Failed to propagate queued sky-light change for {center:?}");
                return;
            };

            for section_pos in result.updated_sections {
                self.light_changed(LightLayer::Sky, section_pos);
            }
        }

        let Ok(result) =
            propagate_block_light_changes_with_empty_sections(&workset, positions, empty_sections)
        else {
            log::warn!("Failed to propagate queued block-light change for {center:?}");
            return;
        };

        for section_pos in result.updated_sections {
            self.light_changed(LightLayer::Block, section_pos);
        }
    }

    fn light_workset_for_change(&self, center: ChunkPos) -> Option<LightWorkset> {
        let Ok(range) = LightSectionRange::from_world_height(
            self.world_gen_context.min_y(),
            self.world_gen_context.height(),
        ) else {
            return None;
        };

        let layout = LightCacheLayout::new(center, range);
        LightWorkset::setup(
            layout,
            LightCacheSetupRadius::Full,
            true,
            |chunk_pos| {
                let holder = self
                    .chunks
                    .read_sync(&chunk_pos, |_, holder| Arc::clone(holder))?;
                if holder.try_chunk(ChunkStatus::Light).is_none() {
                    return None;
                }
                Some(holder)
            },
            |_| true,
        )
        .ok()
    }

    /// Broadcasts all pending block and light changes to nearby players.
    ///
    pub fn broadcast_changed_chunks(&self) {
        self.propagate_queued_light_changes();

        let holders = {
            let mut guard = self.chunks_to_broadcast.lock();
            if guard.is_empty() {
                return;
            }
            mem::take(&mut *guard)
        };

        let world = self.world_gen_context.world();
        let has_skylight = world.dimension_type.has_skylight;

        for holder in holders {
            let chunk_pos = holder.get_pos();
            let min_y = holder.min_y();

            holder.clear_broadcast_queued();

            let light_changes = holder.take_changed_light_sections();
            // Take all pending changes from this chunk holder
            let changes_by_section = holder.take_changed_blocks();
            let has_publishable_light_changes =
                !light_changes.block.is_empty() || (has_skylight && !light_changes.sky.is_empty());

            if !has_publishable_light_changes && changes_by_section.is_empty() {
                continue;
            }

            let light_players = if has_publishable_light_changes {
                world.player_area_map.get_chunk_update_players(
                    chunk_pos,
                    ChunkUpdateRecipients::TrackedBorder,
                    |entity_id, chunk| Self::is_chunk_pending_for_player(&world, entity_id, chunk),
                )
            } else {
                Vec::new()
            };
            let block_players = if changes_by_section.is_empty() {
                Vec::new()
            } else {
                world.player_area_map.get_chunk_update_players(
                    chunk_pos,
                    ChunkUpdateRecipients::Tracked,
                    |entity_id, chunk| Self::is_chunk_pending_for_player(&world, entity_id, chunk),
                )
            };

            if light_players.is_empty() && block_players.is_empty() {
                continue;
            }

            if has_publishable_light_changes
                && !light_players.is_empty()
                && let Some(chunk) = holder.try_chunk(ChunkStatus::Full)
            {
                let light_data = {
                    let light = chunk.light();
                    let sky_sections = if has_skylight {
                        light_changes.sky.as_slice()
                    } else {
                        &[]
                    };
                    build_chunk_light_update_packet_for_sections(
                        chunk_pos,
                        &light,
                        has_skylight,
                        sky_sections,
                        &light_changes.block,
                    )
                };
                let light_packet = CLightUpdate {
                    x: chunk_pos.0.x,
                    z: chunk_pos.0.y,
                    light_data,
                };

                let encoded = EncodedPacket::from_bare(
                    light_packet,
                    world.compression,
                    ConnectionProtocol::Play,
                );
                match encoded {
                    Ok(encoded) => {
                        for entity_id in &light_players {
                            if let Some(player) = world.players.get_by_entity_id(*entity_id) {
                                player.connection.send_encoded(encoded.clone());
                            }
                        }
                    }
                    Err(_) => log::warn!("Failed to encode light update packet"),
                }
            }

            // For each section with changes, send appropriate packet
            for (section_index, changed_positions) in changes_by_section {
                let section_y = min_y / 16 + section_index as i32;
                let section_pos = SectionPos::new(chunk_pos.0.x, section_y, chunk_pos.0.y);

                if changed_positions.len() == 1 {
                    // Single block change - use CBlockUpdate
                    let Some(&packed) = changed_positions.iter().next() else {
                        continue;
                    };
                    let block_pos = section_pos.relative_to_block_pos(packed);
                    let block_state = world.get_block_state(block_pos);

                    tracing::debug!(
                        ?block_pos,
                        ?block_state,
                        player_count = block_players.len(),
                        "Broadcasting single block update"
                    );

                    let update_packet = CBlockUpdate {
                        pos: block_pos,
                        block_state,
                    };

                    let Ok(encoded) = EncodedPacket::from_bare(
                        update_packet,
                        world.compression,
                        ConnectionProtocol::Play,
                    ) else {
                        log::warn!("Failed to encode block update packet");
                        continue;
                    };

                    for entity_id in &block_players {
                        if let Some(player) = world.players.get_by_entity_id(*entity_id) {
                            player.connection.send_encoded(encoded.clone());
                        }
                    }

                    Self::broadcast_block_entity_if_needed(
                        &world,
                        &block_players,
                        block_pos,
                        block_state,
                    );
                } else {
                    // Multiple block changes - use CSectionBlocksUpdate
                    let changes: Vec<BlockChange> = changed_positions
                        .iter()
                        .map(|&packed| {
                            let block_pos = section_pos.relative_to_block_pos(packed);
                            let block_state = world.get_block_state(block_pos);
                            BlockChange {
                                pos: packed,
                                block_state,
                            }
                        })
                        .collect();

                    tracing::debug!(
                        change_count = changes.len(),
                        ?section_pos,
                        player_count = block_players.len(),
                        "Broadcasting section block updates"
                    );

                    let block_entity_updates = changes
                        .iter()
                        .map(|change| {
                            (
                                section_pos.relative_to_block_pos(change.pos),
                                change.block_state,
                            )
                        })
                        .collect::<Vec<_>>();

                    let packet = CSectionBlocksUpdate {
                        section_pos,
                        changes,
                    };

                    let Ok(encoded) = EncodedPacket::from_bare(
                        packet,
                        world.compression,
                        ConnectionProtocol::Play,
                    ) else {
                        log::warn!("Failed to encode section block update packet");
                        continue;
                    };

                    for entity_id in &block_players {
                        if let Some(player) = world.players.get_by_entity_id(*entity_id) {
                            player.connection.send_encoded(encoded.clone());
                        }
                    }

                    for (block_pos, block_state) in block_entity_updates {
                        Self::broadcast_block_entity_if_needed(
                            &world,
                            &block_players,
                            block_pos,
                            block_state,
                        );
                    }
                }
            }
        }
    }

    fn broadcast_block_entity_if_needed(
        world: &World,
        players: &[i32],
        pos: BlockPos,
        state: steel_utils::BlockStateId,
    ) {
        let Some(packet) = world.block_entity_update_packet_for_state(pos, state) else {
            return;
        };

        Self::broadcast_encoded_to_players(world, players, packet, "block entity update");
    }

    fn broadcast_encoded_to_players<P: steel_protocol::packet_traits::ClientPacket>(
        world: &World,
        players: &[i32],
        packet: P,
        packet_name: &'static str,
    ) {
        let Ok(encoded) =
            EncodedPacket::from_bare(packet, world.compression, ConnectionProtocol::Play)
        else {
            log::warn!("Failed to encode {packet_name} packet");
            return;
        };

        for entity_id in players {
            if let Some(player) = world.players.get_by_entity_id(*entity_id) {
                player.connection.send_encoded(encoded.clone());
            }
        }
    }

    fn is_chunk_pending_for_player(world: &World, entity_id: i32, chunk: ChunkPos) -> bool {
        world
            .players
            .get_by_entity_id(entity_id)
            .is_some_and(|player| player.chunk_sender.lock().is_pending(chunk))
    }

    /// Schedules a new generation task.
    #[inline]
    #[instrument(level = "trace", skip(self), fields(chunk = ?pos, target = ?target_status))]
    pub(crate) fn schedule_generation_task_b(
        self: &Arc<Self>,
        target_status: ChunkStatus,
        pos: ChunkPos,
    ) -> Arc<ChunkGenerationTask> {
        let task = Arc::new(ChunkGenerationTask::new(
            pos,
            target_status,
            self.clone(),
            self.generation_pool.clone(),
            self.cancel_token.child_token(),
        ));
        self.pending_generation_tasks.lock().push(task.clone());
        task
    }

    /// Runs queued generation tasks.
    #[instrument(level = "trace", skip(self))]
    pub fn run_generation_tasks_b(&self) {
        let mut pending = self.pending_generation_tasks.lock();
        if pending.is_empty() {
            return;
        }
        let task_count = pending.len();
        tracing::trace!(task_count, "Running generation tasks");
        let tasks = pending.drain(..).collect::<Vec<_>>();
        drop(pending); // Release lock before spawning

        for task in tasks {
            self.task_tracker
                .spawn_on(async move { task.run().await }, self.chunk_runtime.handle());
        }
    }

    /// Updates scheduling for a chunk based on its new level.
    /// Returns the chunk holder if it is active.
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        clippy::unwrap_used,
        reason = "unwrap is on new_level which was already checked non-None via new_level?"
    )]
    pub fn update_chunk_level(
        self: &Arc<Self>,
        pos: ChunkPos,
        new_level: Option<u8>,
    ) -> Option<Arc<ChunkHolder>> {
        // Recover from unloading if possible, else create new holder.
        let chunk_holder =
            if let Some(holder) = self.chunks.read_sync(&pos, |_, holder| holder.clone()) {
                holder
            } else {
                new_level?;

                if let Some(entry) = self.unloading_chunks.remove_sync(&pos) {
                    let _ = self.chunks.insert_sync(pos, entry.1.clone());
                    entry.1
                } else {
                    let holder = Arc::new(ChunkHolder::new(
                        pos,
                        new_level.unwrap(),
                        self.world_gen_context.min_y(),
                        self.world_gen_context.height(),
                    ));
                    let _ = self.chunks.insert_sync(pos, holder.clone());
                    holder
                }
            };

        if let Some(level) = new_level {
            let old = chunk_holder.ticket_level.swap(level, Ordering::Relaxed);
            if old != level {
                chunk_holder.update_highest_allowed_status(level);
            }
            Some(chunk_holder)
        } else {
            //log::info!("Unloading chunk at {pos:?}");
            chunk_holder.cancel_generation_task();
            chunk_holder.ticket_level.store(u8::MAX, Ordering::Relaxed);
            chunk_holder.update_highest_allowed_status(u8::MAX);
            // Wake any await_chunk futures so generation tasks holding refs to
            // this chunk can detect the status is disallowed and exit.
            chunk_holder.wake_all_watchers();

            // Clean up POI data for this chunk column
            let world = self.world_gen_context.world();
            world.poi_storage.lock().remove_chunk(pos);
            self.pending_light_updates.lock().remove_chunk(pos);

            // Move to unloading_chunks for deferred unload
            if let Some((_, holder)) = self.chunks.remove_sync(&pos) {
                let _ = self.unloading_chunks.insert_sync(pos, holder);
            }
            None
        }
    }

    /// Processes chunk updates, ticks chunks, and executes ready scheduled ticks.
    ///
    /// # Arguments
    /// * `world` - The world reference (needed for executing scheduled tick callbacks)
    /// Game tick: broadcasts block changes, ticks chunks (random + scheduled ticks).
    ///
    /// Runs on the main game tick loop. Does NOT handle chunk generation or unloading.
    #[instrument(level = "trace", skip(self, world), name = "chunk_map_game_tick")]
    pub fn tick_game(
        self: &Arc<Self>,
        world: &Arc<World>,
        tick_count: u64,
        random_tick_speed: u32,
        runs_normally: bool,
    ) -> ChunkMapGameTickTimings {
        let mut timings = ChunkMapGameTickTimings::default();
        let mut ready_block_ticks = Vec::new();
        let mut ready_fluid_ticks = Vec::new();

        {
            let _span = tracing::trace_span!("broadcast_changes").entered();
            let start = Instant::now();
            self.broadcast_changed_chunks();
            timings.broadcast_changes = start.elapsed();
        }

        if tick_count.is_multiple_of(100) {
            tracing::debug!(
                chunks = self.chunks.len(),
                unloading = self.unloading_chunks.len(),
                "Chunk map status"
            );
        }

        if !runs_normally {
            return timings;
        }

        {
            let _span = tracing::trace_span!("collect_tickable").entered();
            let start = Instant::now();
            let mut total_chunks = 0;
            let last_len = self.last_tickable_len.load(Ordering::Relaxed);
            let mut tickable_chunks = Vec::with_capacity(last_len);
            self.chunks.iter_sync(|_, holder| {
                total_chunks += 1;
                let level = holder.ticket_level.load(Ordering::Relaxed);
                if is_ticked(level, world.view_distance, world.simulation_distance) {
                    tickable_chunks.push(holder.clone());
                }
                true
            });
            self.last_tickable_len
                .store(tickable_chunks.len(), Ordering::Relaxed);
            timings.collect_tickable = start.elapsed();
            timings.total_chunks = total_chunks;
            timings.tickable_count = tickable_chunks.len();

            if !tickable_chunks.is_empty() {
                let _span = tracing::trace_span!(
                    "tick_chunks",
                    count = tickable_chunks.len(),
                    total_chunks
                )
                .entered();
                let start = Instant::now();
                for holder in &tickable_chunks {
                    holder.post_process_generation();
                    if let Some(chunk_guard) = holder.try_chunk(ChunkStatus::Full) {
                        chunk_guard.tick(
                            random_tick_speed,
                            tick_count as i32,
                            &mut ready_block_ticks,
                            &mut ready_fluid_ticks,
                        );
                    }
                }
                timings.tick_chunks = start.elapsed();
            }
        }

        Self::execute_scheduled_ticks(world, ready_block_ticks, ready_fluid_ticks);

        timings
    }

    /// Scheduling tick: processes tickets, creates holders, schedules generation,
    /// runs generation tasks, and processes unloads.
    ///
    /// Runs on its own independent tick loop, separate from the game tick.
    #[instrument(level = "trace", skip(self), name = "chunk_map_scheduling_tick")]
    pub fn tick_scheduling(self: &Arc<Self>) -> ChunkMapSchedulingTimings {
        let mut timings = ChunkMapSchedulingTimings::default();

        // Only hold the ticket lock for run_all_updates — holder creation and
        // generation scheduling don't need it, and holding it blocks
        // update_player_status on the game tick.
        let changes: Vec<LevelChange> = {
            let _span = tracing::trace_span!("ticket_updates").entered();
            let start = Instant::now();
            let mut ct = self.chunk_tickets.lock();
            let result = ct.run_all_updates().to_vec();
            timings.ticket_updates = start.elapsed();
            result
        };

        let holders_to_schedule: Vec<_> = {
            let _span = tracing::trace_span!("holder_creation").entered();
            let start = Instant::now();
            let result = changes
                .iter()
                .filter_map(|change| {
                    self.update_chunk_level(change.pos, change.new_level)
                        .map(|holder| (holder, change.new_level))
                })
                .collect();
            timings.holder_creation = start.elapsed();
            result
        };

        {
            let _span = tracing::trace_span!("schedule_generation").entered();
            let start = Instant::now();
            let scheduled_count = if holders_to_schedule.len() < 100 {
                holders_to_schedule
                    .iter()
                    .filter(|(holder, level)| {
                        level.is_some_and(is_full)
                            && holder.schedule_chunk_generation_task_b(ChunkStatus::Full, self)
                    })
                    .count()
            } else {
                let self_ref = self;
                self.generation_pool.install(|| {
                    holders_to_schedule
                        .into_par_iter()
                        .filter(|(holder, level)| {
                            level.is_some_and(is_full)
                                && holder
                                    .schedule_chunk_generation_task_b(ChunkStatus::Full, self_ref)
                        })
                        .count()
                })
            };
            timings.schedule_generation = start.elapsed();
            timings.scheduled_count = scheduled_count;
        }

        {
            let _span = tracing::trace_span!("run_generation").entered();
            let start = Instant::now();
            self.run_generation_tasks_b();
            timings.run_generation = start.elapsed();
        }

        {
            let _span = tracing::trace_span!("process_unloads").entered();
            let start = Instant::now();
            self.process_unloads();
            timings.process_unloads = start.elapsed();
        }

        timings
    }

    /// Sorts and executes all ready scheduled ticks, calling block/fluid behavior callbacks.
    fn execute_scheduled_ticks(
        world: &Arc<World>,
        mut ready_block_ticks: Vec<BlockTick>,
        mut ready_fluid_ticks: Vec<FluidTick>,
    ) {
        const MAX_TICKS: usize = usize::MAX; // Vanilla uses 65_536, the lion does not concern himself with vanilla hotpatching

        if !ready_block_ticks.is_empty() {
            ready_block_ticks.sort_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| a.sub_tick_order.cmp(&b.sub_tick_order))
            });

            let block_behaviors = &*BLOCK_BEHAVIORS;
            for tick in ready_block_ticks.iter().take(MAX_TICKS) {
                let state = world.get_block_state(tick.pos);
                if state.get_block() != tick.tick_type {
                    continue;
                }
                block_behaviors
                    .get_behavior(tick.tick_type)
                    .tick(state, world, tick.pos);
            }
        }

        if !ready_fluid_ticks.is_empty() {
            ready_fluid_ticks.sort_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| a.sub_tick_order.cmp(&b.sub_tick_order))
            });

            let fluid_behaviors = &*FLUID_BEHAVIORS;
            for tick in ready_fluid_ticks.iter().take(MAX_TICKS) {
                let state = world.get_block_state(tick.pos);
                let fluid_state = state.get_fluid_state();

                // Only execute if the fluid at this location still matches the scheduled tick
                if fluid_state.fluid_id != tick.tick_type {
                    continue;
                }

                fluid_behaviors
                    .get_behavior(tick.tick_type)
                    .tick(world, tick.pos);
            }
        }
    }

    /// Saves a chunk to disk. Does not remove from `unloading_chunks`.
    #[instrument(level = "trace", skip(self, chunk_holder), fields(chunk = ?chunk_holder.get_pos()))]
    async fn save_chunk(&self, chunk_holder: &Arc<ChunkHolder>) {
        // Prepare chunk data while holding the lock, then release before async I/O
        let prepared = {
            let Some(chunk_guard) = chunk_holder.try_chunk(ChunkStatus::StructureStarts) else {
                // Chunk was at Empty stage so no need to save it
                return;
            };

            let status = chunk_holder
                .persisted_status()
                .expect("The check above confirmed it exists");

            let prepared = ChunkStorage::prepare_chunk_save(&chunk_guard);

            // Clear dirty flag while we still have the lock (only if we're actually saving)
            if prepared.is_some() {
                chunk_guard.clear_dirty();
            }

            (prepared, status)
        }; // chunk_guard dropped here

        let (prepared, status) = prepared;

        // Save chunk data if dirty
        if let Some(prepared) = prepared {
            let result = self.storage.save_chunk_data(prepared, status).await;

            if let Err(e) = result {
                tracing::error!("Error saving chunk: {e}");
            }
        }
    }

    /// Processes chunks that are pending unload.
    ///
    /// Iterates over `unloading_chunks`. For each chunk with `strong_count == 1`:
    /// - If dirty: spawn save task (keep until saved and clean)
    /// - If not dirty: release region handle and remove
    #[instrument(level = "trace", skip(self))]
    pub fn process_unloads(self: &Arc<Self>) {
        self.unloading_chunks.retain_sync(|pos, holder| {
            if Arc::strong_count(holder) == 1 {
                // Check if dirty by trying to get chunk access
                let is_dirty = holder
                    .try_chunk(ChunkStatus::StructureStarts)
                    .is_some_and(|chunk| chunk.is_dirty());

                if is_dirty {
                    // Save the chunk, keep until next tick when it's clean
                    let holder_clone = holder.clone();
                    let map_clone = self.clone();
                    self.task_tracker.spawn(async move {
                        map_clone.save_chunk(&holder_clone).await;
                    });
                    true // keep until clean
                } else if holder.try_chunk(ChunkStatus::Empty).is_none() {
                    false
                } else {
                    // Clean and no refs - release region handle and remove
                    let pos = *pos;
                    let map_clone = self.clone();
                    self.task_tracker.spawn(async move {
                        if let Err(e) = map_clone.storage.release_chunk(pos).await {
                            tracing::error!(?pos, "Error releasing chunk: {e}");
                        }
                    });
                    false // remove
                }
            } else {
                true // keep, still has refs
            }
        });
    }

    /// Updates the player's status in the chunk map.
    pub fn update_player_status(&self, player: &Player) {
        let current_chunk_pos = *player.last_chunk_pos.lock();
        let view_distance = player.view_distance();

        let new_view = PlayerChunkView::new(current_chunk_pos, view_distance);
        let mut last_view_guard = player.last_tracking_view.lock();

        if last_view_guard.as_ref() != Some(&new_view) {
            let mut chunk_tickets = self.chunk_tickets.lock();

            let world = self.world_gen_context.world();

            if let Some(last_view) = last_view_guard.as_ref() {
                if last_view.center != new_view.center
                    || last_view.view_distance != new_view.view_distance
                {
                    chunk_tickets.remove_ticket(
                        last_view.center,
                        MAX_VIEW_DISTANCE.saturating_sub(last_view.view_distance),
                    );
                    chunk_tickets.add_ticket(
                        new_view.center,
                        MAX_VIEW_DISTANCE.saturating_sub(new_view.view_distance),
                    );

                    player.send_packet(CSetChunkCenter {
                        x: new_view.center.0.x,
                        y: new_view.center.0.y,
                    });
                }

                // Track chunks for PlayerAreaMap update
                let mut added_chunks = Vec::new();
                let mut removed_chunks = Vec::new();

                // We lock here to ensure we have unique access for the duration of the diff
                let mut chunk_sender = player.chunk_sender.lock();
                let connection = &*player.connection;
                PlayerChunkView::difference(
                    last_view,
                    &new_view,
                    |pos, ctx: &mut (&mut _, &mut Vec<_>, &mut Vec<_>)| {
                        ctx.0.mark_chunk_pending_to_send(pos);
                        ctx.1.push(pos);
                    },
                    |pos, ctx: &mut (&mut _, &mut Vec<_>, &mut Vec<_>)| {
                        ctx.0.drop_chunk(connection, pos);
                        ctx.2.push(pos);
                    },
                    &mut (&mut chunk_sender, &mut added_chunks, &mut removed_chunks),
                );
                drop(chunk_sender);

                // Update the player area map with the diff
                world.player_area_map.on_player_view_change(
                    player.id,
                    &added_chunks,
                    &removed_chunks,
                );

                // Update entity tracking for this player (only check added/removed chunks)
                world.entity_tracker().on_player_view_change(
                    player,
                    &added_chunks,
                    &removed_chunks,
                );
            } else {
                chunk_tickets.add_ticket(
                    new_view.center,
                    MAX_VIEW_DISTANCE.saturating_sub(new_view.view_distance),
                );

                // Send initial chunk cache center to client
                player.send_packet(CSetChunkCenter {
                    x: new_view.center.0.x,
                    y: new_view.center.0.y,
                });

                let mut chunk_sender = player.chunk_sender.lock();
                new_view.for_each(|pos| {
                    chunk_sender.mark_chunk_pending_to_send(pos);
                });
                drop(chunk_sender);

                // First time - add all chunks in view to player area map
                world.player_area_map.on_player_join(player, &new_view);

                // Initial entity tracking for this player
                world.entity_tracker().on_player_join(player, &new_view);
            }

            *last_view_guard = Some(new_view);
        }
    }

    /// Removes a player from the chunk map.
    pub fn remove_player(&self, player: &Player) {
        // Okay to lock sync lock here cause it has low contention
        let mut last_view_guard = player.last_tracking_view.lock();
        if let Some(last_view) = last_view_guard.take() {
            drop(last_view_guard);
            let mut chunk_tickets = self.chunk_tickets.lock();
            chunk_tickets.remove_ticket(
                last_view.center,
                MAX_VIEW_DISTANCE.saturating_sub(last_view.view_distance),
            );
        }
    }

    /// Saves all dirty chunks to disk.
    ///
    /// This method should be called during graceful shutdown to ensure all
    /// modified chunks are persisted. It saves:
    /// 1. All dirty chunks in the active `chunks` map
    /// 2. All chunks pending unload in the `unloading_chunks` map
    /// 3. Closes all region file handles (flushing headers)
    ///
    /// Returns the number of chunks saved.
    #[instrument(level = "info", skip(self), name = "save_all_chunks")]
    pub async fn save_all_chunks(self: &Arc<Self>) -> io::Result<usize> {
        let mut saved_count = 0;

        // Collect all chunks from both maps
        let all_chunks: Vec<Arc<ChunkHolder>> = {
            let mut chunks = Vec::new();
            self.chunks.iter_sync(|_, holder| {
                chunks.push(holder.clone());
                true
            });
            self.unloading_chunks.iter_sync(|_, holder| {
                chunks.push(holder.clone());
                true
            });
            chunks
        };

        tracing::info!(chunk_count = all_chunks.len(), "Saving chunks");

        // Save all chunks that have data
        for holder in &all_chunks {
            let prepared = {
                let Some(chunk) = holder.try_chunk(ChunkStatus::StructureStarts) else {
                    continue;
                };
                let Some(status) = holder.persisted_status() else {
                    continue;
                };
                let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk) else {
                    continue; // Not dirty
                };
                chunk.clear_dirty();
                (prepared, status)
            };

            let (prepared, status) = prepared;
            match self.storage.save_chunk_data(prepared, status).await {
                Ok(true) => saved_count += 1,
                Ok(false) => {} // Not dirty
                Err(e) => {
                    tracing::error!(chunk = ?holder.get_pos(), "Failed to save chunk: {e}");
                }
            }
        }

        // Close all region files (flushes headers and releases file handles)
        if let Err(e) = self.storage.close_all().await {
            tracing::error!("Failed to close region files: {e}");
        }

        tracing::info!(
            saved_count,
            total_checked = all_chunks.len(),
            "Chunk save complete"
        );

        Ok(saved_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_empty_section_changes_are_drained_top_down() {
        let mut pending = PendingChunkLightUpdates::default();
        pending
            .changed_sections
            .insert(SectionPos::new(3, -2, 4), true);
        pending
            .changed_sections
            .insert(SectionPos::new(3, 5, 4), false);
        pending
            .changed_sections
            .insert(SectionPos::new(3, 0, 4), true);

        let changes = pending.empty_section_changes();
        let section_ys = changes
            .iter()
            .map(|change| change.section_pos.y())
            .collect::<Vec<_>>();

        assert_eq!(section_ys, vec![5, 0, -2]);
    }

    #[test]
    fn pending_light_updates_drain_in_first_queue_order() {
        let mut pending = PendingLightUpdates::default();
        let first_chunk = ChunkPos::new(2, 0);
        let second_chunk = ChunkPos::new(-1, 4);

        pending.queue_change(first_chunk, BlockPos::new(32, 0, 0), true, None);
        pending.queue_change(second_chunk, BlockPos::new(-16, 0, 64), true, None);
        pending.queue_change(first_chunk, BlockPos::new(33, 0, 0), true, None);

        let drained = pending.drain();
        let chunks = drained
            .iter()
            .map(|(chunk_pos, _)| *chunk_pos)
            .collect::<Vec<_>>();

        assert_eq!(chunks, vec![first_chunk, second_chunk]);
        assert_eq!(drained[0].1.changed_positions.len(), 2);
        assert!(pending.is_empty());
    }
}

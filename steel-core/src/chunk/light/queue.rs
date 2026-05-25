use steel_utils::{BlockPos, Direction};

use super::MAX_LIGHT_LEVEL;

const QUEUE_ENTRY_LEVEL_MASK: u64 = 0b1111;
const QUEUE_ENTRY_DIRECTIONS_MASK: u64 = 0b11_1111_0000;
const QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE: u64 = 1 << 10;
const QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION: u64 = 1 << 11;
const LIGHT_QUEUE_MIN_CAPACITY: usize = 512;
pub(crate) const REMOVE_TOP_SKY_SOURCE_ENTRY: LightQueueEntry =
    LightQueueEntry::decrease_all_directions(MAX_LIGHT_LEVEL);
pub(crate) const REMOVE_SKY_SOURCE_ENTRY: LightQueueEntry =
    LightQueueEntry::decrease_skip_one_direction(MAX_LIGHT_LEVEL, Direction::Up);
pub(crate) const ADD_SKY_SOURCE_ENTRY: LightQueueEntry =
    LightQueueEntry::increase_skip_one_direction(MAX_LIGHT_LEVEL, false, Direction::Up);

/// Vanilla's packed light-propagation queue entry.
///
/// `LightEngine.QueueEntry` stores the source level in bits 0..3, one
/// propagation bit per vanilla `Direction.ordinal()` in bits 4..9, and two
/// increase flags in bits 10 and 11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightQueueEntry(u64);

impl LightQueueEntry {
    /// Creates a decrease entry that propagates to all directions except one.
    #[must_use]
    pub const fn decrease_skip_one_direction(
        old_from_level: u8,
        skip_direction: Direction,
    ) -> Self {
        Self::with_level(
            Self::without_direction(QUEUE_ENTRY_DIRECTIONS_MASK, skip_direction),
            old_from_level,
        )
    }

    /// Creates a decrease entry that propagates to all directions.
    #[must_use]
    pub const fn decrease_all_directions(old_from_level: u8) -> Self {
        Self::with_level(QUEUE_ENTRY_DIRECTIONS_MASK, old_from_level)
    }

    /// Creates an increase entry sourced from a block's light emission.
    #[must_use]
    pub const fn increase_light_from_emission(new_from_level: u8, from_empty_shape: bool) -> Self {
        let mut entry = QUEUE_ENTRY_DIRECTIONS_MASK | QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION;
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(entry, new_from_level)
    }

    /// Creates an increase entry that propagates to all directions except one.
    #[must_use]
    pub const fn increase_skip_one_direction(
        new_from_level: u8,
        from_empty_shape: bool,
        skip_direction: Direction,
    ) -> Self {
        let mut entry = Self::without_direction(QUEUE_ENTRY_DIRECTIONS_MASK, skip_direction);
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(entry, new_from_level)
    }

    /// Creates an increase entry that propagates to exactly one direction.
    #[must_use]
    pub const fn increase_only_one_direction(
        new_from_level: u8,
        from_empty_shape: bool,
        direction: Direction,
    ) -> Self {
        let mut entry = 0;
        if from_empty_shape {
            entry |= QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE;
        }

        Self::with_level(Self::with_direction(entry, direction), new_from_level)
    }

    /// Creates a sky-source increase entry for selected directions.
    #[must_use]
    pub const fn increase_sky_source_in_directions(
        down: bool,
        north: bool,
        south: bool,
        west: bool,
        east: bool,
    ) -> Self {
        let mut entry = MAX_LIGHT_LEVEL as u64;
        if down {
            entry = Self::with_direction(entry, Direction::Down);
        }
        if north {
            entry = Self::with_direction(entry, Direction::North);
        }
        if south {
            entry = Self::with_direction(entry, Direction::South);
        }
        if west {
            entry = Self::with_direction(entry, Direction::West);
        }
        if east {
            entry = Self::with_direction(entry, Direction::East);
        }

        Self(entry)
    }

    /// Creates a queue entry from vanilla's packed representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns vanilla's packed representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns the source light level stored in this entry.
    #[must_use]
    pub const fn from_level(self) -> u8 {
        (self.0 & QUEUE_ENTRY_LEVEL_MASK) as u8
    }

    /// Returns true if propagation starts from an empty occlusion shape.
    #[must_use]
    pub const fn is_from_empty_shape(self) -> bool {
        self.0 & QUEUE_ENTRY_FLAG_FROM_EMPTY_SHAPE != 0
    }

    /// Returns true if this increase came from block light emission.
    #[must_use]
    pub const fn is_increase_from_emission(self) -> bool {
        self.0 & QUEUE_ENTRY_FLAG_INCREASE_FROM_EMISSION != 0
    }

    /// Returns true if this entry propagates in `direction`.
    #[must_use]
    pub const fn should_propagate_in_direction(self, direction: Direction) -> bool {
        self.0 & Self::direction_bit(direction) != 0
    }

    const fn with_level(entry: u64, level: u8) -> Self {
        Self(entry & !QUEUE_ENTRY_LEVEL_MASK | (level as u64 & QUEUE_ENTRY_LEVEL_MASK))
    }

    const fn with_direction(entry: u64, direction: Direction) -> u64 {
        entry | Self::direction_bit(direction)
    }

    const fn without_direction(entry: u64, direction: Direction) -> u64 {
        entry & !Self::direction_bit(direction)
    }

    const fn direction_bit(direction: Direction) -> u64 {
        1 << (Self::vanilla_direction_index(direction) + 4)
    }

    const fn vanilla_direction_index(direction: Direction) -> u64 {
        match direction {
            Direction::Down => 0,
            Direction::Up => 1,
            Direction::North => 2,
            Direction::South => 3,
            Direction::West => 4,
            Direction::East => 5,
        }
    }
}

/// One typed light propagation queue item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuedLightUpdate {
    /// Block position whose light should propagate.
    pub block_pos: BlockPos,
    /// Packed vanilla propagation metadata.
    pub entry: LightQueueEntry,
}

/// Array-backed FIFO used for vanilla light propagation work.
///
/// Vanilla stores alternating packed block positions and `QueueEntry` longs in
/// `LongArrayFIFOQueue`. Steel keeps typed records instead, while preserving
/// the FIFO ordering and packed queue-entry semantics that propagation depends
/// on.
#[derive(Debug)]
pub struct LightPropagationQueue {
    entries: Vec<QueuedLightUpdate>,
    read_index: usize,
}

impl LightPropagationQueue {
    /// Creates an empty propagation queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(LIGHT_QUEUE_MIN_CAPACITY),
            read_index: 0,
        }
    }

    /// Returns true when no queued work remains.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_index >= self.entries.len()
    }

    /// Returns the number of queued items that have not been dequeued yet.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len() - self.read_index
    }

    /// Adds propagation work to the back of the queue.
    pub fn enqueue(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.entries.push(QueuedLightUpdate { block_pos, entry });
    }

    /// Removes propagation work from the front of the queue.
    pub fn dequeue(&mut self) -> Option<QueuedLightUpdate> {
        if self.is_empty() {
            self.clear();
            return None;
        }

        let update = self.entries[self.read_index];
        self.read_index += 1;
        if self.is_empty() {
            self.clear();
        }

        Some(update)
    }

    /// Removes all queued work while keeping allocated storage for reuse.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.read_index = 0;
    }
}

impl Default for LightPropagationQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Vanilla's separate increase and decrease propagation queues.
#[derive(Debug, Default)]
pub struct LightPropagationQueues {
    increase: LightPropagationQueue,
    decrease: LightPropagationQueue,
}

impl LightPropagationQueues {
    /// Creates empty increase and decrease queues.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when either propagation queue contains work.
    #[must_use]
    pub fn has_work(&self) -> bool {
        !self.increase.is_empty() || !self.decrease.is_empty()
    }

    /// Enqueues decrease propagation work.
    pub fn enqueue_decrease(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.decrease.enqueue(block_pos, entry);
    }

    /// Enqueues increase propagation work.
    pub fn enqueue_increase(&mut self, block_pos: BlockPos, entry: LightQueueEntry) {
        self.increase.enqueue(block_pos, entry);
    }

    /// Dequeues decrease propagation work.
    pub fn dequeue_decrease(&mut self) -> Option<QueuedLightUpdate> {
        self.decrease.dequeue()
    }

    /// Dequeues increase propagation work.
    pub fn dequeue_increase(&mut self) -> Option<QueuedLightUpdate> {
        self.increase.dequeue()
    }

    /// Removes all increase and decrease work.
    pub fn clear(&mut self) {
        self.increase.clear();
        self.decrease.clear();
    }
}

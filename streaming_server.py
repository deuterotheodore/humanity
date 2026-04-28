"""
Streaming WebSocket Server for Agent-Based Simulations

Key design principles:
1. Simulation runs independently, just marks entities as dirty
2. Server streams updates at constant rate (not tied to tick rate)
3. Cursor-based round-robin over compact roster - O(chunk_size) per send
4. Dirty IDs only - no copying entity data, fetch at send time
5. Despawns unified with updates - dead entities wait their turn

Architecture:
    Simulation Thread          Server (Async)
    ─────────────────          ──────────────
    world.step()               
    get_dirty_ids()       →    mark dirty/dead flags (O(k))
                               
                               Send Loop (every 10ms):
                               - Advance cursor through roster
                               - Fetch current state for dirty entities
                               - Send chunk, clear dirty flags

Server is simulation-agnostic:
- Does not know about entity types (agents, cells, etc.)
- Does not access entity properties directly (no .position, no .id)
- Only handles opaque world_ids and display dicts
- Delegates all viewport logic to World methods
- Passes initialization parameters to World.create()
- Treats counts as opaque (int or array)
"""

import asyncio
import json
import websockets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any
import queue as thread_queue

from sim import World


def normalize_counts(counts):
    """
    Normalize counts to a list, handling both array and scalar inputs.
    
    This makes the server agnostic to whether the simulation returns
    a single count (int) or an array of counts.
    """
    if isinstance(counts, (int, float)):
        return [int(counts)]
    return list(counts)


# ============================================================
# Configuration
# ============================================================

# Default streaming parameters (can be adjusted dynamically)
DEFAULT_STREAM_INTERVAL_MS = 10   # Send every 10ms = 100 sends/sec
DEFAULT_ENTITIES_PER_MESSAGE = 200  # 200 entities per message
MAX_DESPAWNS_PER_MESSAGE = 500    # Limit despawns per message

# Rate limits
MIN_STREAM_INTERVAL_MS = 5        # Don't go faster than 5ms
MAX_STREAM_INTERVAL_MS = 100      # Don't go slower than 100ms
MIN_ENTITIES_PER_MESSAGE = 10     # Minimum chunk size
MAX_ENTITIES_PER_MESSAGE = 500    # Maximum chunk size

# Target: 200 entities × 100 sends/sec = 20,000 entities/sec

# ============================================================
# Streaming Buffer (per-client) - Cursor-based Round-Robin
# ============================================================

class StreamingBuffer:
    """
    Buffer for streaming entity updates to a single client.
    
    Design principles:
    1. Compact roster - no gaps in iteration, O(1) insert/remove
    2. Round-robin cursor - fair iteration over all entities
    3. Dirty flags only - no copying entity data, fetch at send time
    4. Unified despawns - dead entities wait their turn in roster
    
    Server is fully opaque to entity internals:
    - Only handles world_ids (integers)
    - Never accesses entity properties directly
    - Delegates viewport checks to World methods
    
    Memory: O(n) where n = entity count, regardless of tick backlog
    """
    
    def __init__(self):
        # Compact roster for O(1) operations
        self.roster: List[int] = []              # world_ids in send order
        self.roster_index: Dict[int, int] = {}   # world_id → position in roster
        
        # Round-robin cursor
        self.cursor: int = 0
        
        # Dirty tracking (just world_ids, no data copies!)
        self.dirty: Set[int] = set()             # Entities needing update sent
        self.dead: Set[int] = set()              # Entities needing despawn sent
        
        # Death notices - server stores these to send when cursor reaches them
        # The notice contains client-facing death info (stripped of world_id before sending)
        self.death_notices: Dict[int, dict] = {}  # world_id → death notice
        
        # Viewport filtering (None = send everything)
        # Server stores these but doesn't interpret them - just passes to World
        self.viewport_center: Optional[Any] = None
        self.viewport_radius: Optional[float] = None
        
        # Entities client knows about (for viewport exit tracking)
        self.known_entities: Set[int] = set()
        
        # Stats
        self.total_entities_sent = 0
        self.total_despawns_sent = 0
    
    def has_viewport(self) -> bool:
        """Check if viewport filtering is active."""
        return self.viewport_center is not None and self.viewport_radius is not None
    
    def on_spawn(self, world_id: int, in_viewport: bool):
        """
        Handle entity spawn. O(1) - append to roster.
        
        Args:
            world_id: Entity's globally unique world_id
            in_viewport: Whether entity is in client's viewport (pre-checked by caller)
        """
        if not in_viewport:
            return  # Don't track entities outside viewport
        
        # Add to roster if not already present
        if world_id not in self.roster_index:
            self.roster_index[world_id] = len(self.roster)
            self.roster.append(world_id)
        
        self.dirty.add(world_id)
        self.known_entities.add(world_id)
    
    def on_update(self, world_id: int, in_viewport: bool):
        """
        Handle entity update. O(1) - just mark dirty.
        
        Args:
            world_id: Entity's globally unique world_id
            in_viewport: Whether entity is in client's viewport (pre-checked by caller)
        """
        if in_viewport:
            # In viewport - ensure in roster, mark dirty
            if world_id not in self.roster_index:
                self.roster_index[world_id] = len(self.roster)
                self.roster.append(world_id)
            self.dirty.add(world_id)
            self.known_entities.add(world_id)
        else:
            # Moved out of viewport - treat as despawn for this client
            if world_id in self.known_entities:
                self.dirty.discard(world_id)
                self.dead.add(world_id)
                # Don't remove from roster yet - wait for cursor
    
    def on_death(self, world_id: int, notice: dict):
        """
        Handle entity death. O(1) - mark dead, store notice, removal happens at cursor.
        
        Args:
            world_id: Entity's world_id
            notice: Death notice dict containing client-facing fields (id, type, dead=True)
                    May contain world_id which will be stripped before sending.
        """
        if world_id in self.roster_index:
            self.dirty.discard(world_id)
            self.dead.add(world_id)
            self.death_notices[world_id] = notice
            # Don't remove from roster yet - despawn sent when cursor reaches it
    
    def _remove_from_roster(self, world_id: int):
        """
        Remove entity from roster using swap-with-last. O(1).
        """
        if world_id not in self.roster_index:
            return
        
        idx = self.roster_index.pop(world_id)
        last_idx = len(self.roster) - 1
        
        if idx < last_idx:
            # Swap with last
            last_id = self.roster[last_idx]
            self.roster[idx] = last_id
            self.roster_index[last_id] = idx
        
        self.roster.pop()
        
        # Adjust cursor if needed (if we removed before cursor position)
        if len(self.roster) > 0:
            self.cursor = self.cursor % len(self.roster)
        else:
            self.cursor = 0
    
    def pop_chunk(self, world: World, chunk_size: int) -> dict:
        """
        Extract a chunk of updates using round-robin cursor.
        
        Returns dict with 'entities' list (includes both updates and death notices).
        Death notices are sent as regular entities with 'dead': True.
        Fetches current entity state at send time via world.get_entity_display().
        
        Server never accesses entity objects directly - only receives opaque dicts.
        """
        entities_to_send = []
        
        n = len(self.roster)
        if n == 0:
            return {'entities': []}
        
        # Track how many we've scanned (avoid infinite loop if roster shrinks)
        scanned = 0
        sent = 0
        
        while scanned < n and sent < chunk_size:
            # Wrap cursor
            if self.cursor >= len(self.roster):
                self.cursor = 0
                if len(self.roster) == 0:
                    break
            
            world_id = self.roster[self.cursor]
            scanned += 1
            
            if world_id in self.dead:
                # Send death notice as a regular entity update, then remove from roster
                if world_id in self.known_entities:
                    notice = self.death_notices.pop(world_id, {}).copy()
                    notice.pop('world_id', None)  # Don't send world_id to client
                    entities_to_send.append(notice)
                    self.known_entities.discard(world_id)
                    self.total_despawns_sent += 1
                
                self.dead.discard(world_id)
                self._remove_from_roster(world_id)
                sent += 1
                # Don't advance cursor - swap-with-last put new entity here
                continue
            
            if world_id in self.dirty:
                # Fetch current state from world (not a stale copy!)
                # Server delegates everything to world.get_entity_display()
                display_dict = world.get_entity_display(
                    world_id, 
                    self.viewport_center, 
                    self.viewport_radius
                )
                
                if display_dict is not None:
                    # Entity exists and is in viewport - strip world_id before sending
                    display_dict = display_dict.copy()
                    display_dict.pop('world_id', None)
                    entities_to_send.append(display_dict)
                    self.known_entities.add(world_id)
                    sent += 1
                else:
                    # Entity doesn't exist or moved out of viewport
                    # Create a synthetic death notice if we need to tell client
                    if world_id in self.known_entities:
                        # We need the simulation to provide a death notice for this case
                        # For now, we can't send anything meaningful - just remove locally
                        self.known_entities.discard(world_id)
                    self._remove_from_roster(world_id)
                    sent += 1
                    continue  # Don't advance cursor
                
                self.dirty.discard(world_id)
            
            # Advance cursor
            self.cursor += 1
        
        # Wrap cursor for next call
        if len(self.roster) > 0:
            self.cursor = self.cursor % len(self.roster)
        
        self.total_entities_sent += len(entities_to_send) - self.total_despawns_sent
        
        return {'entities': entities_to_send}
    
    def set_viewport(self, center: Optional[Any], radius: Optional[float]):
        """Update viewport configuration"""
        self.viewport_center = center
        self.viewport_radius = radius
    
    def rebuild_roster(self, world: World):
        """
        Rebuild roster from world state. Call after viewport change or reconnect.
        
        Delegates to world.get_all_world_ids_in_viewport() to avoid
        accessing entity objects directly.
        """
        self.roster.clear()
        self.roster_index.clear()
        self.dirty.clear()
        self.dead.clear()
        self.death_notices.clear()
        self.known_entities.clear()
        self.cursor = 0
        
        # Get all world_ids that should be in roster
        # World handles viewport filtering internally
        world_ids = world.get_world_ids_in_viewport(
            self.viewport_center, 
            self.viewport_radius
        )
        
        for wid in world_ids:
            self.roster_index[wid] = len(self.roster)
            self.roster.append(wid)
            self.dirty.add(wid)
            self.known_entities.add(wid)
    
    def get_stats(self) -> dict:
        return {
            'roster_size': len(self.roster),
            'dirty': len(self.dirty),
            'dead': len(self.dead),
            'death_notices': len(self.death_notices),
            'known_entities': len(self.known_entities),
            'cursor': self.cursor,
            'total_sent': self.total_entities_sent,
            'total_despawns': self.total_despawns_sent
        }


# ============================================================
# Client Session
# ============================================================

@dataclass
class ClientSession:
    """Per-client state for WebSocket connections"""
    client_id: str
    websocket: Any
    buffer: StreamingBuffer = field(default_factory=StreamingBuffer)
    subscribed: bool = False
    send_task: Optional[asyncio.Task] = None
    last_tick_sent: int = -1
    connected: bool = True
    
    # Dynamic rate control (per-client)
    stream_interval_ms: int = DEFAULT_STREAM_INTERVAL_MS
    entities_per_message: int = DEFAULT_ENTITIES_PER_MESSAGE
    
    def adjust_rate(self, client_queue_depth: int):
        """
        Adjust streaming rate based on client feedback.
        
        If client is backing up (queue_depth > 0), slow down.
        If client is keeping up (queue_depth == 0), speed up.
        """
        if client_queue_depth > 10:
            # Client overwhelmed - slow down significantly
            self.stream_interval_ms = min(MAX_STREAM_INTERVAL_MS, self.stream_interval_ms + 5)
            self.entities_per_message = max(MIN_ENTITIES_PER_MESSAGE, self.entities_per_message - 50)
        elif client_queue_depth > 2:
            # Client falling behind - slow down a bit
            self.stream_interval_ms = min(MAX_STREAM_INTERVAL_MS, self.stream_interval_ms + 2)
        elif client_queue_depth == 0:
            # Client keeping up - try to speed up
            self.stream_interval_ms = max(MIN_STREAM_INTERVAL_MS, self.stream_interval_ms - 1)
            self.entities_per_message = min(MAX_ENTITIES_PER_MESSAGE, self.entities_per_message + 10)


# Global session registry
sessions: Dict[str, ClientSession] = {}
sessions_lock = threading.Lock()


# ============================================================
# Shared World State
# ============================================================

world: Optional[World] = None
world_lock = threading.Lock()
simulation_running = False
current_tick = 0

# Queue for deltas from simulation thread to async handlers
delta_queue: Optional[thread_queue.Queue] = None


# ============================================================
# Per-Client Send Loop
# ============================================================

async def client_send_loop(session: ClientSession):
    """
    Continuously stream updates to a single client.
    
    Runs independently of simulation - just drains the buffer at a constant rate.
    Rate is dynamically adjusted based on client feedback.
    
    NOTE: No world_lock needed - pop_chunk only calls world.get_entity_display()
    which performs thread-safe reads.
    """
    print(f"[STREAM] Starting send loop for {session.client_id}", flush=True)
    
    messages_sent = 0
    entities_sent = 0
    last_report_time = time.time()
    
    while session.connected and session.subscribed:
        try:
            loop_start = time.time()
            
            # Use session's current rate parameters
            interval = session.stream_interval_ms / 1000.0
            chunk_size = session.entities_per_message
            
            # Pop a chunk from the buffer
            if world is None:
                await asyncio.sleep(interval)
                continue
            
            chunk = session.buffer.pop_chunk(world, chunk_size)
            
            if chunk['entities']:
                msg = {
                    "type": "delta",
                    "tick": current_tick,
                    "entities": chunk['entities']
                }
                
                await session.websocket.send(json.dumps(msg))
                session.last_tick_sent = current_tick
                messages_sent += 1
                entities_sent += len(chunk['entities'])
            
            # Report server stats every 5 seconds
            now = time.time()
            if now - last_report_time >= 5.0:
                stats = session.buffer.get_stats()
                rate_info = f"interval={session.stream_interval_ms}ms, chunk={session.entities_per_message}"
                print(f"[STREAM] {session.client_id[-8:]}: {messages_sent} msgs, {entities_sent} entities, despawns={stats['total_despawns']}, roster={stats['roster_size']}, dirty={stats['dirty']}, {rate_info}", flush=True)
                messages_sent = 0
                entities_sent = 0
                last_report_time = now
            
            # Sleep for remainder of interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, interval - elapsed)  # At least 1ms sleep
            await asyncio.sleep(sleep_time)
        
        except websockets.exceptions.ConnectionClosed:
            print(f"[STREAM] Connection closed for {session.client_id}", flush=True)
            session.connected = False
            break
        except Exception as e:
            print(f"[STREAM] Error in send loop for {session.client_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.01)
    
    print(f"[STREAM] Send loop ended for {session.client_id}", flush=True)


# ============================================================
# Delta Distribution
# ============================================================

async def distribute_dirty_ids(dirty_ids: dict):
    """
    Distribute dirty IDs to all subscribed clients.
    
    OPTIMIZATION:
    - No viewport: bulk operations, O(1) per ID
    - With viewport: need individual checks via world.entity_in_viewport()
    
    Death notices are stored in the buffer and sent when cursor reaches them.
    """
    global current_tick
    current_tick = dirty_ids.get('tick', current_tick)
    
    spawned_ids = dirty_ids.get('spawned_ids', [])
    updated_ids = dirty_ids.get('updated_ids', [])
    despawn_notices = dirty_ids.get('despawn_notices', [])
    
    if world is None:
        return
    
    with sessions_lock:
        active_sessions = list(sessions.values())
    
    for session in active_sessions:
        if not session.subscribed:
            continue
        
        buf = session.buffer
        
        if not buf.has_viewport():
            # NO VIEWPORT - bulk operations, super fast
            # All entities are visible, no need to check positions
            
            # Spawns: add to roster
            for wid in spawned_ids:
                if wid not in buf.roster_index:
                    buf.roster_index[wid] = len(buf.roster)
                    buf.roster.append(wid)
                    buf.known_entities.add(wid)
            buf.dirty.update(spawned_ids)
            
            # Updates: just mark dirty
            buf.dirty.update(updated_ids)
            
            # Deaths: store notice and mark dead
            for notice in despawn_notices:
                wid = notice.get('world_id')
                if wid is not None:
                    buf.on_death(wid, notice)
            
        else:
            # VIEWPORT ACTIVE - need viewport checks via World
            # Yield frequently to avoid blocking send loops
            center = buf.viewport_center
            radius = buf.viewport_radius
            
            ops = 0
            for wid in spawned_ids:
                in_viewport = world.entity_in_viewport(wid, center, radius)
                buf.on_spawn(wid, in_viewport)
                ops += 1
                if ops % 2000 == 0:
                    await asyncio.sleep(0)
            
            for wid in updated_ids:
                in_viewport = world.entity_in_viewport(wid, center, radius)
                buf.on_update(wid, in_viewport)
                ops += 1
                if ops % 2000 == 0:
                    await asyncio.sleep(0)
            
            for notice in despawn_notices:
                wid = notice.get('world_id')
                if wid is not None:
                    # Only send death notice if client knows about this entity
                    if wid in buf.known_entities:
                        buf.on_death(wid, notice)
    
    await asyncio.sleep(0)


async def delta_distribution_loop():
    """
    Async loop that receives dirty IDs from simulation thread and distributes them.
    """
    global delta_queue
    delta_queue = thread_queue.Queue(maxsize=100)
    
    last_timing_report = time.time()
    deltas_processed = 0
    total_ids_processed = 0
    
    while True:
        try:
            # Non-blocking check for new deltas
            try:
                dirty_ids = delta_queue.get_nowait()
                
                start_time = time.time()
                await distribute_dirty_ids(dirty_ids)
                elapsed = time.time() - start_time
                
                # Track stats
                deltas_processed += 1
                n_ids = (len(dirty_ids.get('spawned_ids', [])) + 
                        len(dirty_ids.get('updated_ids', [])) + 
                        len(dirty_ids.get('despawn_notices', [])))
                total_ids_processed += n_ids
                
                # Report timing every 5 seconds
                now = time.time()
                if now - last_timing_report >= 5.0:
                    qsize = delta_queue.qsize()
                    print(f"[DIST] {deltas_processed} deltas, {total_ids_processed} IDs, queue={qsize}", flush=True)
                    deltas_processed = 0
                    total_ids_processed = 0
                    last_timing_report = now
                
                # Warn if distribution is slow
                if elapsed > 0.05:  # 50ms
                    print(f"[DIST] Slow distribution: {elapsed*1000:.1f}ms for {n_ids} IDs", flush=True)
                
                # CRITICAL: Always yield after processing a delta!
                # This lets send loops run even when deltas are backing up.
                await asyncio.sleep(0.001)  # 1ms minimum gap
                    
            except thread_queue.Empty:
                await asyncio.sleep(0.005)  # 5ms polling when idle
        
        except Exception as e:
            print(f"[DIST] Error distributing: {e}", flush=True)
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)


# ============================================================
# Message Handlers
# ============================================================

async def handle_subscribe(session: ClientSession):
    """Handle subscription - send initial snapshot, start streaming"""
    session.subscribed = True
    session.buffer = StreamingBuffer()  # Fresh buffer
    
    # Copy viewport settings if they were set before subscribe
    # (handled by handle_viewport)
    
    with world_lock:
        if world is None:
            await session.websocket.send(json.dumps({
                "type": "error",
                "message": "World not initialized"
            }))
            return
        
        # Build roster and snapshot based on viewport
        session.buffer.rebuild_roster(world)
        
        # Build snapshot - get display dicts for all entities in roster
        # Server doesn't access entity objects, just gets opaque dicts
        # Strip world_id before sending to client
        entities = []
        for wid in session.buffer.known_entities:
            display_dict = world.get_entity_display(
                wid,
                session.buffer.viewport_center,
                session.buffer.viewport_radius
            )
            if display_dict is not None:
                display_dict = display_dict.copy()
                display_dict.pop('world_id', None)  # Don't send world_id to client
                entities.append(display_dict)
        
        snapshot = {
            "type": "snapshot",
            "tick": world.tick,
            "world": {
                "width": world.width,
                "height": world.height
            },
            "entities": entities,
            "total_entities": world.entity_count()
        }
        session.last_tick_sent = world.tick
    
    await session.websocket.send(json.dumps(snapshot))
    print(f"[WS] Sent snapshot to {session.client_id}: {len(entities)} entities (roster={len(session.buffer.roster)})", flush=True)
    
    # Start the send loop if not already running
    if session.send_task is None or session.send_task.done():
        session.send_task = asyncio.create_task(client_send_loop(session))


async def handle_unsubscribe(session: ClientSession):
    """Handle unsubscription - stop streaming"""
    session.subscribed = False
    if session.send_task and not session.send_task.done():
        session.send_task.cancel()


async def handle_viewport(session: ClientSession, center: Any, radius: float):
    """Handle viewport change"""
    session.buffer.set_viewport(center, radius)
    print(f"[WS] Viewport set for {session.client_id}: center={center}, radius={radius}", flush=True)
    
    # If already subscribed, send new snapshot for the viewport
    if session.subscribed:
        await handle_subscribe(session)  # Re-send snapshot with new viewport


async def handle_flow_control(session: ClientSession, queue_depth: int):
    """Handle flow control feedback from client"""
    old_interval = session.stream_interval_ms
    old_chunk = session.entities_per_message
    
    session.adjust_rate(queue_depth)
    
    # Log significant changes
    if session.stream_interval_ms != old_interval or session.entities_per_message != old_chunk:
        print(f"[FLOW] {session.client_id[-8:]}: queue={queue_depth}, rate adjusted: {old_interval}ms/{old_chunk} → {session.stream_interval_ms}ms/{session.entities_per_message}", flush=True)


async def handle_inspect(session: ClientSession, client_id: str):
    """
    Handle inspect request - return full state of a specific entity.
    
    Unlike streaming updates which only include display data,
    inspect returns the complete entity state including internal variables.
    
    Args:
        client_id: Client-facing identifier (e.g., "42" for agent 42)
    """
    with world_lock:
        if world is None:
            await session.websocket.send(json.dumps({
                "type": "error",
                "message": "World not initialized"
            }))
            return
        
        entity_data = world.inspect_by_client_id(client_id)
    
    if entity_data is not None:
        await session.websocket.send(json.dumps({
            "type": "entity_detail",
            "entity": entity_data,
            "tick": current_tick
        }))
    else:
        await session.websocket.send(json.dumps({
            "type": "entity_detail",
            "entity": None,
            "id": client_id,
            "error": "Entity not found"
        }))


async def handle_message(session: ClientSession, message: str):
    """Route incoming WebSocket message to appropriate handler"""
    try:
        msg = json.loads(message)
        msg_type = msg.get("type")
        
        if msg_type == "subscribe":
            await handle_subscribe(session)
        
        elif msg_type == "unsubscribe":
            await handle_unsubscribe(session)
        
        elif msg_type == "viewport":
            center = msg.get("center")
            radius = msg.get("radius")
            if center is not None and radius is not None:
                await handle_viewport(session, center, radius)
        
        elif msg_type == "viewport_clear":
            session.buffer.set_viewport(None, None)
            if session.subscribed:
                await handle_subscribe(session)
        
        elif msg_type == "flow_control":
            queue_depth = msg.get("queue_depth", 0)
            await handle_flow_control(session, queue_depth)
        
        elif msg_type == "set_rate":
            # Allow client to explicitly set rate parameters
            interval = msg.get("interval_ms")
            chunk = msg.get("chunk_size")
            if interval is not None:
                session.stream_interval_ms = max(MIN_STREAM_INTERVAL_MS, min(MAX_STREAM_INTERVAL_MS, interval))
            if chunk is not None:
                session.entities_per_message = max(MIN_ENTITIES_PER_MESSAGE, min(MAX_ENTITIES_PER_MESSAGE, chunk))
            print(f"[RATE] {session.client_id[-8:]}: set to {session.stream_interval_ms}ms / {session.entities_per_message} entities", flush=True)
        
        elif msg_type == "inspect":
            # Request full state of a specific entity
            client_id = msg.get("id")
            if client_id is not None:
                await handle_inspect(session, str(client_id))
            else:
                await session.websocket.send(json.dumps({
                    "type": "error",
                    "message": "inspect requires id"
                }))
        
        else:
            await session.websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            }))
    
    except json.JSONDecodeError:
        await session.websocket.send(json.dumps({
            "type": "error",
            "message": "Invalid JSON"
        }))


# ============================================================
# WebSocket Connection Handler
# ============================================================

async def handle_connection(websocket, path):
    """Handle a WebSocket connection"""
    client_id = f"ws-{id(websocket)}"
    session = ClientSession(client_id=client_id, websocket=websocket)
    
    with sessions_lock:
        sessions[client_id] = session
    
    print(f"[WS] Client connected: {client_id}", flush=True)
    
    try:
        async for message in websocket:
            await handle_message(session, message)
    
    except websockets.exceptions.ConnectionClosed:
        pass
    
    finally:
        session.connected = False
        session.subscribed = False
        if session.send_task and not session.send_task.done():
            session.send_task.cancel()
        
        with sessions_lock:
            sessions.pop(client_id, None)
        
        print(f"[WS] Client disconnected: {client_id}", flush=True)


# ============================================================
# Stats Broadcasting
# ============================================================

# Cached simulation stats (updated by simulation loop)
# Stored as list - normalized from world.count_entities() which may return int or array
cached_entity_counts = [0]

async def broadcast_stats_loop():
    """Periodically broadcast stats to all clients"""
    while True:
        await asyncio.sleep(1.0)  # Every second
        
        histograms = world.report_statistics() if world else {}

        # Use cached values - no lock needed!
        # These are updated by simulation_loop
        # Server passes counts as-is; client interprets the array
        stats = {
            "type": "stats",
            "tick": current_tick,
            "counts": cached_entity_counts,  # Already normalized to list
            "total": sum(cached_entity_counts),
            "histograms": histograms
        }
        
        with sessions_lock:
            active_sessions = list(sessions.values())
        
        for session in active_sessions:
            if session.subscribed:
                try:
                    await session.websocket.send(json.dumps(stats))
                except:
                    pass


# ============================================================
# Simulation Loop (runs in background thread)
# ============================================================

def simulation_loop(ticks_per_second: float = 10):
    """Background thread that runs the simulation"""
    global simulation_running, world, current_tick, cached_entity_counts
    
    interval = 1.0 / ticks_per_second
    last_stats_time = time.time()
    
    print(f"[SIM] Starting simulation loop at {ticks_per_second} ticks/sec", flush=True)
    
    while simulation_running:
        start_time = time.time()
        
        with world_lock:
            if world is not None:
                world.step()
                current_tick = world.tick
                
                # Get dirty IDs (not full state!) and queue for distribution
                dirty_ids = world.get_dirty_ids()
                world.mark_clean()
                
                if delta_queue is not None:
                    try:
                        delta_queue.put_nowait(dirty_ids)
                    except thread_queue.Full:
                        print(f"[SIM] Delta queue full, dropping tick {world.tick}", flush=True)
                
                # Update cached counts (used by broadcast_stats_loop)
                # Normalize to list - simulation may return int or array
                cached_entity_counts = normalize_counts(world.count_entities())
                
                # Check if simulation has halted itself
                if world.halted:
                    print(f"[SIM] Halted at tick {world.tick}: {world.halt_reason}", flush=True)
                    simulation_running = False
                
                # Print stats every 5 seconds
                if time.time() - last_stats_time >= 5.0:
                    # Gather buffer stats
                    with sessions_lock:
                        buffer_stats = []
                        for s in sessions.values():
                            if s.subscribed:
                                bs = s.buffer.get_stats()
                                buffer_stats.append(f"{s.client_id[-8:]}: roster={bs['roster_size']}, dirty={bs['dirty']}")
                    
                    buffer_info = ", ".join(buffer_stats) if buffer_stats else "no clients"
                    print(f"[SIM] tick={world.tick}, counts={cached_entity_counts}, buffers=[{buffer_info}]", flush=True)
                    last_stats_time = time.time()
        
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
    
    print("[SIM] Simulation loop ended", flush=True)


def start_simulation(ticks_per_second: float = 10):
    """Start the simulation in a background thread"""
    global simulation_running
    simulation_running = True
    thread = threading.Thread(target=simulation_loop, args=(ticks_per_second,), daemon=True)
    thread.start()
    return thread


def stop_simulation():
    """Stop the simulation"""
    global simulation_running
    simulation_running = False


# ============================================================
# World Initialization
# ============================================================

def init_world(config: dict):
    """
    Initialize the simulation world.
    
    Passes config dict directly to World.create() - this function
    is simulation-agnostic and doesn't inspect the config contents.
    """
    global world, current_tick
    
    with world_lock:
        # Pass config through to World.create() - only it knows what's valid
        world = World.create(**config)
        current_tick = 0
    
    # Clear all client buffers
    with sessions_lock:
        for session in sessions.values():
            session.buffer = StreamingBuffer()
    
    return world


# ============================================================
# HTTP Command Handlers (for Flask integration)
# ============================================================

def handle_http_init(data: dict) -> dict:
    """
    Handle HTTP init request.
    
    Passes data dict directly to init_world without inspection.
    Only World.create() knows what parameters are valid.
    
    Returns map data (non-zero biome cells) for client to initialize terrain.
    """
    # Pass through without unpacking - simulation defines valid params
    init_world(data)
    
    # Return response - normalize counts (may be int or array from simulation)
    counts = normalize_counts(world.count_entities())
    return {
        'status': 'initialized',
        'width': world.width,
        'height': world.height,
        'counts': counts,
        'total': sum(counts),
        'tick': world.tick,
        'map': world.describe_map()  # Static map data: [[x, y, biome], ...]
    }


def handle_http_start(data: dict) -> dict:
    global simulation_running
    if world is None:
        return {'error': 'World not initialized'}
    
    if simulation_running:
        return {'status': 'already running'}
    
    tps = data.get('ticks_per_second', 10)
    start_simulation(tps)
    return {'status': 'started', 'ticks_per_second': tps}


def handle_http_stop(data: dict) -> dict:
    stop_simulation()
    return {'status': 'stopped'}


def handle_http_step(data: dict) -> dict:
    global world, current_tick
    if world is None:
        return {'error': 'World not initialized'}
    if simulation_running:
        return {'error': 'Simulation is running, stop it first'}
    
    steps = data.get('steps', 1)
    
    with world_lock:
        for _ in range(steps):
            world.step()
            current_tick = world.tick
            dirty_ids = world.get_dirty_ids()
            world.mark_clean()
            
            # Queue for distribution
            if delta_queue is not None:
                try:
                    delta_queue.put_nowait(dirty_ids)
                except thread_queue.Full:
                    pass
    
    counts = normalize_counts(world.count_entities())
    return {
        'status': 'stepped',
        'steps': steps,
        'tick': world.tick,
        'counts': counts,
        'total': sum(counts)
    }


def handle_http_stats() -> dict:
    if world is None:
        return {'error': 'World not initialized'}
    
    with world_lock:
        counts = normalize_counts(world.count_entities())
        histograms = world.report_statistics()

        return {
            'tick': world.tick,
            'counts': counts,
            'total': sum(counts),
            'running': simulation_running,
            'histograms': histograms
        }


def handle_http_viewport(center, radius: float) -> dict:
    """
    Return current state of all entities within viewport.
    
    This is the "active view" endpoint - always returns fresh data.
    Client polls at desired rate for real-time viewport updates.
    
    Delegates to world.get_viewport() - server doesn't know the topology.
    
    Parameters:
        center: Viewport center position (opaque to server)
        radius: Viewport radius
    
    Returns:
        {
            'tick': current simulation tick,
            'center': [...],  # coordinates
            'radius': radius,
            'entities': [...],  # All entities in viewport
            'count': number of entities
        }
    """
    if world is None:
        return {'error': 'World not initialized', 'entities': [], 'count': 0, 'tick': 0}
    
    try:
        return world.get_viewport(center, radius)
    except RuntimeError:
        # Dict changed during iteration - return empty result, client will retry
        return {'tick': current_tick, 'center': list(center), 'radius': radius, 'entities': [], 'count': 0}


def handle_http_inspect(client_id: str) -> dict:
    """
    Return full state of a specific entity.
    
    Unlike streaming which only sends display data, this returns
    the complete internal state.
    
    Args:
        client_id: Client-facing identifier (e.g., "42" for agent 42,
                   future: "tree_42", "cell_5_10", etc.)
    """
    if world is None:
        return {'error': 'World not initialized'}
    
    entity_data = world.inspect_by_client_id(client_id)
    
    if entity_data:
        return {
            'tick': current_tick,
            'entity': entity_data
        }
    else:
        return {
            'tick': current_tick,
            'entity': None,
            'error': f'Entity {client_id} not found'
        }


# ============================================================
# Main Entry Point
# ============================================================

async def main(host: str = "0.0.0.0", port: int = 8765):
    """Start the WebSocket server"""
    print(f"[WS] Server starting on ws://{host}:{port}", flush=True)
    print(f"[WS] Streaming config: interval={DEFAULT_STREAM_INTERVAL_MS}ms, entities/msg={DEFAULT_ENTITIES_PER_MESSAGE}", flush=True)
    
    # Start background tasks
    dist_task = asyncio.create_task(delta_distribution_loop())
    stats_task = asyncio.create_task(broadcast_stats_loop())
    
    # Start WebSocket server
    async with websockets.serve(handle_connection, host, port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    print("Starting WebSocket-only server (use combined_server.py for HTTP+WS)")
    asyncio.run(main())

# server.py Public API Reference

This document describes the functions and methods of the agent-model simulation server (currently named combined_server.py).
It serves as an intermediary between simulation code (simulation.py, imported as module) and a client accessing the simulation state via the network (http, websocket).

---

## World

The simulation is loaded as a module named World:
```python
from simulation import World
```

The following methods must exist in World (even if just as a dummy method returning null if not applicable).

---

## Core Concepts

### world_id

Every "observable" entity in the simulation must have a globally unique `world_id` (integer). This includes:
- Agents (creatures, actors, etc.)
- Map cells (if they have dynamic properties)
- Any other object that should be visible to clients

The `world_id` is distinct from domain-specific IDs:
- `agent.id` is a domain concept (agents reference each other by name)
- `agent.world_id` is an infrastructure concept (globally unique for streaming)

**Implementation:** Use a global counter in World:
```python
class World:
    def __init__(self):
        self._next_world_id = 0
    
    def _allocate_world_id(self) -> int:
        wid = self._next_world_id
        self._next_world_id += 1
        return wid
```

### Streamable Entity Interface

Any object that can be observed (streamed to clients) must implement:

```python
entity.world_id: int           # Globally unique, assigned at creation
entity.to_display_dict() -> dict  # Client-facing data (does NOT include world_id)
```

Example:
```python
class Agent:
    def __init__(self, id: int, world_id: int, x: float, y: float):
        self.id = id              # Domain ID (agent's "name")
        self.world_id = world_id  # Streaming ID (globally unique)
        self.x = x
        self.y = y
    
    def to_display_dict(self) -> dict:
        # Note: world_id is NOT included - that's internal to server
        return {
            'type': 'agent',
            'id': self.id,
            'x': self.x,
            'y': self.y,
            # ... other display properties
        }

class MapCell:
    def __init__(self, world_id: int, x: int, y: int, biome: int):
        self.world_id = world_id
        self.x = x
        self.y = y
        self.biome = biome
    
    def to_display_dict(self) -> dict:
        return {
            'type': 'cell',
            'id': -(self.y * WORLD_HEIGHT + self.x + 1),  # Client-facing ID
            'x': self.x,
            'y': self.y,
            'biome': self.biome,
        }
```

The `get_entity_display()` method in World adds `world_id` for internal tracking, then the server strips it before sending to clients.

---

## World Methods

### `create()` (classmethod)
```python
@classmethod
World.create(**kwargs) -> World
```

**Primary entry point for creating a simulation world.**

Accepts an opaque config dict from the server. Only this method knows what
parameters are valid and what they mean. The server passes the config through
without inspection.

---

### `step()`
Advance simulation one tick.

---

### `mark_clean()`
Clear dirty tracking (call after sending delta to clients).

---

### `get_dirty_ids() -> dict`
Get world_ids of changed entities since last `mark_clean()`. **Used for streaming.**

```python
{
    'tick': int,
    'spawned_ids': List[int],      # New entities (world_ids)
    'updated_ids': List[int],      # Modified entities (world_ids, excluding spawned)
    'despawn_notices': List[dict]  # Death notices for removed entities
}
```

**Death notices** are dicts containing:
- `world_id`: The entity's world_id (used by server for tracking, stripped before sending)
- `id`: The client-facing ID
- `type`: Entity type string (e.g., 'agent', 'cell')
- `dead`: True (marker for client to remove this entity)
- Any other fields the client needs for the death animation/cleanup

Example death notice:
```python
{
    'world_id': 42,      # Server uses this, strips before sending
    'id': 7,             # Client-facing agent ID
    'type': 'agent',
    'dead': True
}
```

Used by: Streaming server delta distribution

---

### `get_entity_display(world_id, center, radius) -> Optional[dict]`
**NEW** - Get display dict for a single entity, with viewport filtering.

```python
def get_entity_display(self, world_id: int, center, radius) -> Optional[dict]:
    """
    Get display dict for entity if it exists and is in viewport.
    
    Args:
        world_id: Entity's globally unique world_id
        center: Viewport center (None = no viewport filtering)
        radius: Viewport radius (None = no viewport filtering)
    
    Returns:
        Display dict (from entity.to_display_dict()) if entity exists 
        and is in viewport (or no viewport set).
        None if entity doesn't exist or is outside viewport.
    """
```

Used by: Streaming buffer `pop_chunk()`, snapshot building

---

### `entity_in_viewport(world_id, center, radius) -> bool`
**NEW** - Check if entity is within viewport.

```python
def entity_in_viewport(self, world_id: int, center, radius) -> bool:
    """
    Check if entity should be included in viewport results.
    
    Simulation decides the logic:
    - Spatial entities check their position
    - Non-spatial entities return True (always visible) or False (never in viewport)
    
    Args:
        world_id: Entity's globally unique world_id
        center: Viewport center
        radius: Viewport radius
    
    Returns:
        True if entity is in viewport, False otherwise.
        Returns False if entity doesn't exist.
    """
```

Used by: Streaming buffer viewport filtering during delta distribution

---

### `get_world_ids_in_viewport(center, radius) -> List[int]`
**NEW** - Get all world_ids that should be visible in a viewport.

```python
def get_world_ids_in_viewport(self, center, radius) -> List[int]:
    """
    Get list of all world_ids for entities in viewport.
    
    Args:
        center: Viewport center (None = return all entities)
        radius: Viewport radius (None = return all entities)
    
    Returns:
        List of world_ids for all entities that should be streamed.
    """
```

Used by: Streaming buffer `rebuild_roster()`

---

### `entity_count() -> int`
**NEW** - Get total count of streamable entities.

```python
def entity_count(self) -> int:
    """Return total number of streamable entities in the world."""
```

Used by: Snapshot building (total_entities field)

---

### `get_viewport(center, radius) -> dict`
Get all entities within a viewport region.

```python
{
    'tick': int,
    'center': List[float],  # Center position as list
    'radius': float,
    'entities': List[dict],  # to_display_dict() for each entity in viewport
    'count': int             # Number of entities
}
```

This is the simulation's implementation of viewport queries.
The server calls this without knowing topology details.

Used by: HTTP `/viewport` endpoint

---

### `inspect_entity(world_id) -> Optional[dict]`
**RENAMED** from `inspect_agent` - Get full state of a specific entity.

```python
def inspect_entity(self, world_id: int) -> Optional[dict]:
    """
    Get complete internal state of an entity.
    
    Unlike streaming which only sends display data, this returns
    the full state including internal variables.
    
    Returns None if entity not found.
    """
```

Used by: HTTP `/inspect/<world_id>`, WebSocket `inspect` command

---

### `count_agents() -> Tuple`
Get counts of agents (creatures) by type. Server passes array without unpacking:

```python
counts = world.count_agents()
return {'counts': list(counts), 'total': sum(counts)}
```

Used by: HTTP `/stats` endpoint and periodic status broadcasts

Note: This specifically counts "agents" (creatures/actors), not all entities.

---

### `report_statistics() -> dict`
Generic simulation state statistics (not interpreted by server).

---

### `report_params() -> dict` (classmethod)
Returns simulation parameters for external reporting.

Used by: HTTP `/params` endpoint. Simulation decides what to expose.

---

### `set_param(name, value) -> dict`
Set a simulation parameter at runtime.

Returns dict:
```python
{'status': 'ok', 'name': 'prey.move_cost', 'old_value': 0.1, 'new_value': 0.5}
{'status': 'error', 'message': 'Unknown parameter: foo', 'valid_params': [...]}
{'status': 'error', 'message': 'Invalid value for prey.move_cost: ...'}
```

Used by: HTTP `/set_param` endpoint.

---

### `describe_map() -> List[List[int]]`
Get static map data for initial transmission to client.

---

### `get_full_state() -> dict`
Complete world snapshot for initial sync.

```python
{
    'tick': int,
    'width': int,
    'height': int,
    'entities': List[dict],    # to_display_dict() for each entity
    'total_entities': int
}
```

Used by: WebSocket `subscribe` (initial snapshot)

---

## World Properties

```python
world.tick          # Current simulation time (int)
world.halted        # Flag if simulation has halted (bool)
world.halt_reason   # String with reason for halting
world.width         # World width (for client convenience, can be None)
world.height        # World height (for client convenience, can be None)
```

---

## Internal Implementation Notes

### Entity Storage

The simulation maintains separate collections internally but provides a unified view for streaming:

```python
class World:
    def __init__(self):
        self._next_world_id = 0
        
        # Internal storage by type (for simulation logic)
        self._agents: Dict[int, Agent] = {}      # Keyed by agent.id (domain)
        self._cells: Dict[int, MapCell] = {}     # Keyed by world_id
        
        # Lookup by world_id (for streaming)
        self._by_world_id: Dict[int, Any] = {}   # world_id → entity
        
        # Dirty tracking (uses world_ids)
        self._spawned: Set[int] = set()
        self._dirty: Set[int] = set()
        self._despawned: Set[int] = set()
```

### Dirty Tracking

When entities change, mark them dirty using world_id:

```python
def mark_dirty(self, entity):
    self._dirty.add(entity.world_id)

def spawn_agent(self, ...):
    agent = Agent(world_id=self._allocate_world_id(), ...)
    self._agents[agent.id] = agent
    self._by_world_id[agent.world_id] = agent
    self._spawned.add(agent.world_id)

def kill_agent(self, agent):
    del self._agents[agent.id]
    del self._by_world_id[agent.world_id]
    self._despawned.add(agent.world_id)
```

---

## Server

The server is completely simulation-agnostic. It:
- Only imports `World` (not entity types or constants)
- Does not inspect config parameters - passes them as opaque dict
- Does not access entity properties directly (no `.position`, no `.id`)
- Only handles `world_id` integers and opaque display dicts
- Delegates all viewport logic to World methods
- Treats agent counts as opaque arrays

### HTTP Endpoints (port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/init` | POST | Initialize world |
| `/start` | POST | Start simulation |
| `/stop` | POST | Stop simulation |
| `/step` | POST | Single step(s) |
| `/set_param` | POST | Set simulation parameter |
| `/stats` | GET | Get statistics |
| `/params` | GET | Get parameters |
| `/viewport` | GET | Get entities in viewport |
| `/inspect/<world_id>` | GET | Get full entity state |

Viewport URL format:
```
/viewport?center=50_50&r=20       ("center" is a single string to be parsed by get_viewport method) 
/viewport?cx=50&cy=50&r=20        (legacy with explicit 2d position format)
```

### WebSocket (port 8765)

**Client → Server:**
| Message | Description |
|---------|-------------|
| `subscribe` | Start receiving streaming updates |
| `unsubscribe` | Stop receiving updates |
| `viewport` | Set viewport filter for WS stream |
| `viewport_clear` | Clear viewport filter |
| `inspect` | Request full entity state (requires `world_id`) |
| `flow_control` | Report client queue depth |
| `set_rate` | Set streaming rate parameters |

**Server → Client:**
| Message | Description |
|---------|-------------|
| `snapshot` | Full state (on subscribe) - contains `entities` array |
| `delta` | Streaming updates - contains `entities` and `despawned` arrays |
| `stats` | Periodic statistics |
| `entity_detail` | Response to inspect request |

### Streaming Protocol

Delta messages contain entity updates (including death notices):
```json
{
    "type": "delta",
    "tick": 123,
    "entities": [
        {"id": 42, "type": "agent", "x": 10, "y": 20, ...},
        {"id": 500, "type": "cell", "x": 5, "y": 5, ...},
        {"id": 7, "type": "agent", "dead": true}
    ]
}
```

Death notices are regular entity updates with `"dead": true`. The client should remove the entity when it sees this.

Snapshot messages contain:
```json
{
    "type": "snapshot",
    "tick": 0,
    "world": {"width": 100, "height": 100},
    "entities": [...],
    "total_entities": 1500
}
```

---

## Migration from Previous API

### Renamed/Changed

| Old | New |
|-----|-----|
| `world.agents` (dict) | Internal: `world._agents`, `world._by_world_id` |
| `agent.position` | Not accessed by server |
| `World.is_in_viewport()` (static) | Internal only, not called by server |
| `inspect_agent(agent_id)` | `inspect_entity(world_id)` |
| Response key `"agents"` | Response key `"entities"` |
| `get_dirty_ids()["despawned_ids"]` | `get_dirty_ids()["despawn_notices"]` |
| Delta message `"despawned": [ids]` | Death notices in `"entities"` array with `dead: true` |

### New Methods Required

| Method | Purpose |
|--------|---------|
| `get_entity_display(world_id, center, radius)` | Get display dict with viewport check |
| `entity_in_viewport(world_id, center, radius)` | Check viewport membership |
| `get_world_ids_in_viewport(center, radius)` | Get all visible world_ids |
| `entity_count()` | Total streamable entity count |

### New Entity Requirements

All streamable entities must have:
- `world_id` attribute (globally unique int)
- `to_display_dict()` method (does NOT include `world_id` - that's internal)

Death notices must include:
- `world_id` (for server tracking, stripped before sending to client)
- `id` (client-facing identifier)
- `type` (entity type string)
- `dead: True`

---

## Thread Safety Notes

- Server calls `world.get_entity_display()` without holding world_lock
- These calls must be thread-safe (reading from dicts is safe in Python)
- `step()` modifies entity state - must not be called concurrently with streaming
- Display dicts are snapshots - safe to pass across threads

---

## Example Implementation

```python
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tick = 0
        self.halted = False
        self.halt_reason = ""
        
        self._next_world_id = 0
        self._agents = {}           # agent.id → Agent
        self._cells = {}            # (x,y) → MapCell
        self._by_world_id = {}      # world_id → entity
        
        self._spawned = set()
        self._dirty = set()
        self._despawn_notices = []  # List of death notice dicts
    
    def _allocate_world_id(self) -> int:
        wid = self._next_world_id
        self._next_world_id += 1
        return wid
    
    def spawn_agent(self, x, y, ...):
        agent = Agent(
            world_id=self._allocate_world_id(),
            id=len(self._agents),  # or your own ID scheme
            x=x, y=y, ...
        )
        self._agents[agent.id] = agent
        self._by_world_id[agent.world_id] = agent
        self._spawned.add(agent.world_id)
        return agent
    
    def kill_agent(self, agent):
        # Create death notice before removing
        notice = {
            'world_id': agent.world_id,
            'id': agent.id,
            'type': 'agent',
            'dead': True
        }
        self._despawn_notices.append(notice)
        
        # Remove from storage
        del self._agents[agent.id]
        del self._by_world_id[agent.world_id]
    
    def get_entity_display(self, world_id, center, radius):
        entity = self._by_world_id.get(world_id)
        if entity is None:
            return None
        
        if center is not None and radius is not None:
            if not self._is_in_viewport(entity, center, radius):
                return None
        
        d = entity.to_display_dict()
        d['world_id'] = world_id  # Add for server tracking
        return d
    
    def entity_in_viewport(self, world_id, center, radius):
        entity = self._by_world_id.get(world_id)
        if entity is None:
            return False
        return self._is_in_viewport(entity, center, radius)
    
    def _is_in_viewport(self, entity, center, radius):
        # Simulation decides how to check - might use hasattr for position
        if hasattr(entity, 'x') and hasattr(entity, 'y'):
            dx = entity.x - center[0]
            dy = entity.y - center[1]
            return (dx*dx + dy*dy) <= radius*radius
        return True  # Non-spatial entities always visible (or False, your choice)
    
    def get_world_ids_in_viewport(self, center, radius):
        if center is None or radius is None:
            return list(self._by_world_id.keys())
        return [wid for wid, entity in self._by_world_id.items()
                if self._is_in_viewport(entity, center, radius)]
    
    def entity_count(self):
        return len(self._by_world_id)
    
    def get_dirty_ids(self):
        return {
            'tick': self.tick,
            'spawned_ids': list(self._spawned),
            'updated_ids': list(self._dirty),
            'despawn_notices': self._despawn_notices.copy(),
        }
    
    def mark_clean(self):
        self._spawned.clear()
        self._dirty.clear()
        self._despawn_notices.clear()
```

import random
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import logging

TWOPI = 2 * math.pi

@dataclass
class Agent:
    id: int
    sex: int = 0            # 0 = male, 1 = female
    born: int = 0           # Tick when agent was spawned
    parent: List[int] = field(default_factory=lambda: [-1, -1])  # IDs of parents ([-1,-1] for founders)
    offspring: List[int] = field(default_factory=list)  # IDs of offspring
    #state parameters
    x: int = 0
    y: int = 0
    energy: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    trust: float = 0.0
    # Genetic parameters (inherited with mutation)
    o: float = 0.0  # Openness
    c: float = 0.0  # Conscientiousness
    e: float = 0.0  # Extraversion
    a: float = 0.0  # Agreeableness
    n: float = 0.0  # Neuroticism
    kin:  float = 0.0
    xeno: float = 0.0
    genes: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])  # Speciation genes [0-10]

    @property
    def cell(self) -> Tuple[int, int]:
        """Current cell for food/interactions (floor of position)"""
        return (int(self.x), int(self.y))
    
    # Class-level definition of phenotype fields (sent in streaming)
    # Everything else is "internal" (only sent on inspect)
    PHENOTYPE_FIELDS = ('id', 'x', 'y')
    
    @property
    def position(self) -> tuple:
        """
        Agent's position in the simulation's coordinate system.
        
        Returns an opaque position tuple - the server should not
        interpret the contents, only pass them through.
        For this 2D simulation, returns (x, y).
        """
        return (self.x, self.y)
    
    def to_display_dict(self) -> dict:
        """
        Minimal data for visualization - streamed continuously.
        Only includes phenotype fields for efficient bandwidth.
        """
        return {
            'id': self.id,
            'sex': self.sex,
            'x': self.x,
            'y': self.y,
        }

    def to_viewport_dict(self) -> dict:
        """
        Intermediate data for agents in viewport.
        """
        return {
            'id': self.id,
            'sex': self.sex,
            'x': self.x,
            'y': self.y,
            'energy': round(self.energy, 2),
            'genes': [round(g, 2) for g in self.genes],
        }

    
    def to_full_dict(self) -> dict:
        """
        Complete state - only sent on inspect request.
        Includes phenotype + internal state + genetics.
        """
        return {
            'id': self.id,
            'sex': self.sex,
            'born': self.born,
            'x': self.x,
            'y': self.y,

            'energy': round(self.energy, 2),
            'valence': round(self.valence, 2),
            'arousal': round(self.arousal, 2),
            'trust': round(self.trust, 2),
            'o': round(self.o, 2),
            'c': round(self.c, 2),
            'e': round(self.e, 2),
            'a': round(self.a, 2),
            'n': round(self.n, 2),
            'kin': round(self.kin, 2),
            'xeno': round(self.xeno, 2),
            'genes': [round(g, 2) for g in self.genes],
        }
    

class SpatialGrid:
    """
    Spatial index for O(1) queries of "which agents are in cell (x,y)?"
    
    Critical for scalability: without this, finding nearby agents is O(n).
    With this, it's O(agents_in_cell).
    
    Accepts float positions - converts to int cell coordinates internally.
    
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # cell -> set of agent IDs in that cell
        self._grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    
    def add(self, agent_id: int, x: int, y: int):
        cell = (int(x), int(y))
        self._grid[cell].add(agent_id)
    
    def remove(self, agent_id: int, x: int, y: int):
        cell = (int(x), int(y))
        self._grid[cell].discard(agent_id)
    
    def move(self, agent_id: int, old_x: int, old_y: int, new_x: int, new_y: int):
        """Update agent's cell assignment. Only modifies grid if cell changed."""
        old_cell = (int(old_x), int(old_y))
        new_cell = (int(new_x), int(new_y))
        if old_cell != new_cell:
            self._grid[old_cell].discard(agent_id)
            self._grid[new_cell].add(agent_id)
    
    def agents_at(self, x: int, y: int) -> Set[int]:
        return self._grid[(x, y)]
    
    def agents_in_region(self, x1: int, y1: int, x2: int, y2: int) -> Set[int]:
        """Get all agent IDs in a rectangular region (for AOI queries)"""
        result = set()
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                result.update(self._grid[(x, y)])
        return result

class World:
   
    # Simulation parameters
    TICK_YEARS = 0.25  # one simulation tick in human years
    FOOD_REGEN_PER_TURN = 1.0
    FOOD_CEILING = 2.0
    SEASON_STRENGTH = 0.7

    INITIAL_ENERGY = 2.0
    METABOLISM_COST = 0.05
    MAX_ENERGY = 10.0
    REPRODUCTION_THRESHOLD = 5.0
    REPRODUCTION_COST = 2.0
    MALE_INVESTMENT = 0.2
    EAT_RATE = 1.0
    INFANCY = 3
    CHILDHOOD = 7
    ADOLESCENCE = 14
    MENOPAUSE = 44
    SENESCENCE = 60
    AGE_PENALTY = 0.01
    
    # Mutation rates for genetic parameters
    GENE_MUTATION_SD = 0.1
    TRAIT_MUTATION_SD = 0.1
    
    # Default board size and founder populations
    DEFAULT_WIDTH = 100
    DEFAULT_HEIGHT = 100
    
    # Statistics logging
    STATS_LOG_INTERVAL = 10  # Print stats every N ticks (0 to disable)
    HIST_BINS = 10  # number of bins in histograms

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.tick = 0
        
        if seed is not None:
            random.seed(seed)
        self.food: Dict[Tuple[int, int], float] = {}
        for x in range(width):
            for y in range(height):
                self.food[(x, y)] = self.FOOD_CEILING
        
        # Agent management
        self._next_id = 0
        self.agents: Dict[int, Agent] = {}
        self.spatial_grid = SpatialGrid(width, height)
        
        # Delta tracking: which agents have changed since last "mark_clean()"
        self._dirty_agents: Set[int] = set()
        self._spawned_agents: Set[int] = set()  # new since last clean
        self._despawned_agents: Set[int] = set()  # removed since last clean
        self._mated_this_tick: Set[int] = set()  # agents who have mated this tick
        
        # Per-tick statistics (reset each tick, accumulated for logging)
        self._stats = {
            'births': 0,
            'deaths': 0,
        }

        # Statistics histograms, set default maxima
        self._hist_max = {
            'age': 60,      # initial defaults
            'energy': self.MAX_ENERGY,
        }
        self._histograms = {}  # filled by _update_histograms
        self._hist_tick = 0
        # How to report each property as histogram value
        self.hist_value_getters = {
            'age': lambda a: self._age_years(a),
            'energy': lambda a: a.energy,
        }
        
        # Simulation state
        self.halted = False  # Set to True when termination condition is met
        self.halt_reason: Optional[str] = None  # Why simulation halted
        
        # Deferred command execution (for commands received during step())
        self._stepping = False  # True while inside step()
        self._deferred_commands: List[Tuple[str, any]] = []  # Queue of (name, value) to execute after step
    
    def _age_years(self, agent: Agent) -> float:
        """Calculate agent's age in years."""
        return self.TICK_YEARS * (self.tick - agent.born)
        
    @classmethod
    def create(cls, **kwargs) -> 'World':
        """
        Create and initialize a world with agents.
        
        This is the main entry point for creating a simulation world.
        The server passes an opaque config dict - only this method knows
        what parameters are valid and what they mean.
        
        Arguments (all optional, with defaults):
            width: World width in cells (default: 10)
            height: World height in cells (default: 10)
            initial_pairs: Number of breeding pairs to spawn (default: 1)
            seed: Random seed for reproducibility (default: None)
        
        Returns:
            Initialized World with agents spawned and dirty state cleared
        """
        # Extract and apply defaults
        width = kwargs.get('width', cls.DEFAULT_WIDTH)
        height = kwargs.get('height', cls.DEFAULT_HEIGHT)
        seed = kwargs.get('seed', None)
        initial_pairs = kwargs.get('initial_pairs', 1)

        # Create the world
        world = cls(width, height, seed=seed)
        
        # Spawn initial breeding pairs (male + female per cell)
        for _ in range(initial_pairs):

            # Pick a random cell
            cell_x = random.randint(0, width - 1)
            cell_y = random.randint(0, height - 1)

            # Spawn male in cell
            male = world.spawn_agent(cell_x, cell_y, cls.INITIAL_ENERGY)
            male.sex = 0
                    
            # Spawn female in same cell
            female = world.spawn_agent(cell_x, cell_y, cls.INITIAL_ENERGY)
            female.sex = 1
        
        # Clear dirty state so first delta is clean
        world.mark_clean()
        
        return world
    
    def _allocate_id(self) -> int:
        """Get a unique agent ID. IDs are never reused (important for client sync)."""
        agent_id = self._next_id
        self._next_id += 1
        return agent_id
    
    def spawn_agent(self, x: int, y: int, energy: float, 
                    parent_ids: Tuple[int, int] = (-1, -1)) -> Agent:
        """
        Create a new agent in the world.
        
        Args:
            x, y: Position (cell coordinates, integers)
            energy: Initial energy
            parent_ids: Tuple of parent IDs ((-1, -1) for founders)
        
        Returns:
            The created Agent
        
        Genetic inheritance:
            - Founders (parent_ids == (-1, -1)) get defaults
            - Offspring inherit average of both parents' genes + Gaussian noise

        Sex is assigned randomly (50/50).
        """
        agent_id = self._allocate_id()
        
        # Determine genetic parameters
        parent1_id, parent2_id = parent_ids
        parent1 = self.agents.get(parent1_id) if parent1_id >= 0 else None
        parent2 = self.agents.get(parent2_id) if parent2_id >= 0 else None
        
        if parent1 and parent2:
            # Sexual reproduction: average parents' traits, then add mutation
            trait_sd = self.TRAIT_MUTATION_SD
            gene_sd = self.GENE_MUTATION_SD
            
            # Big Five personality traits
            o = max(0.0, min(1.0, (parent1.o + parent2.o) / 2 + random.gauss(0, trait_sd)))
            c = max(0.0, min(1.0, (parent1.c + parent2.c) / 2 + random.gauss(0, trait_sd)))
            e = max(0.0, min(1.0, (parent1.e + parent2.e) / 2 + random.gauss(0, trait_sd)))
            a = max(0.0, min(1.0, (parent1.a + parent2.a) / 2 + random.gauss(0, trait_sd)))
            n = max(0.0, min(1.0, (parent1.n + parent2.n) / 2 + random.gauss(0, trait_sd)))
            
            # Social traits
            kin = max(0.0, min(1.0, (parent1.kin + parent2.kin) / 2 + random.gauss(0, trait_sd)))
            xeno = max(0.0, min(1.0, (parent1.xeno + parent2.xeno) / 2 + random.gauss(0, trait_sd)))
            
            # Speciation genes: average + mutation, clamped to [0, 10]
            genes = []
            for i in range(3):
                avg_gene = (parent1.genes[i] + parent2.genes[i]) / 2
                genes.append(max(0.0, min(10.0, avg_gene + random.gauss(0, gene_sd))))

        else:
            # Founder defaults
            genes = [5.0 + 3 * (1 if x < 5 else -1), 
                     5.0 + 3 * (1 if y < 5 else -1), 
                     5.0 + 2 * (1 if x < 5 else -1)]
            o = c = e = a = n = 0.5
            kin = xeno = 0.5

        sex = random.randint(0, 1)  # 0 = male, 1 = female

        agent = Agent(
            id=agent_id,
            x=int(x),
            y=int(y),
            energy=energy,
            sex=sex,
            born=self.tick,
            parent=list(parent_ids),
            offspring=[],
            o=o,
            c=c,
            e=e,
            a=a,
            n=n,
            kin=kin,
            xeno=xeno,
            genes=genes,
        )
        self.agents[agent_id] = agent
        self.spatial_grid.add(agent_id, x, y)
        self._spawned_agents.add(agent_id)
        self._dirty_agents.add(agent_id)
        
        # Record this agent as offspring of both parents
        if parent1:
            parent1.offspring.append(agent_id)
        if parent2:
            parent2.offspring.append(agent_id)
        
        return agent
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the world."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self.spatial_grid.remove(agent_id, agent.x, agent.y)
            del self.agents[agent_id]
            self._despawned_agents.add(agent_id)
            self._dirty_agents.discard(agent_id)
            self._spawned_agents.discard(agent_id)
    
    def _mark_dirty(self, agent_id: int):
        """Mark an agent as modified (for delta updates)."""
        self._dirty_agents.add(agent_id)
    
    def get_dirty_state(self) -> dict:
        """
        Get all changes since last mark_clean().
        This is what we'll send to clients as delta updates.
        
        Uses to_display_dict() for phenotype-only data (efficient streaming).
        Full agent data is available via inspect command.
        """
        return {
            'tick': self.tick,
            'spawned': [self.agents[aid].to_display_dict() for aid in self._spawned_agents if aid in self.agents],
            'updated': [self.agents[aid].to_display_dict() for aid in self._dirty_agents - self._spawned_agents if aid in self.agents],
            'despawned': list(self._despawned_agents)
        }
    
    def get_dirty_ids(self) -> dict:
        """
        Get just the IDs of changed agents (no data copying).
        Much more efficient for streaming - actual data fetched lazily at send time.
        """
        return {
            'tick': self.tick,
            'spawned_ids': list(self._spawned_agents),
            'updated_ids': list(self._dirty_agents - self._spawned_agents),
            'despawned_ids': list(self._despawned_agents)
        }
    
    def get_full_state(self) -> dict:
        """
        Get complete world state (for initial sync or reconnection).
        
        Map cells are returned as entities alongside agents, with map=1.
        This unified approach allows the same streaming infrastructure to handle
        both agents and map data.
        """
        # Agents as entities (display dict only for streaming efficiency)
        agent_entities = [a.to_display_dict() for a in self.agents.values()]
        
        # Map cells as entities (map=1 indicates map cell)
        cell_entities = [
            {
                'id': -(x * self.height + y + 1),  # Negative IDs for cells
                'map': 1,
                'x': x,
                'y': y,
                'food': round(self.food[(x, y)], 2),
            }
            for x in range(self.width)
            for y in range(self.height)
        ]
        
        return {
            'tick': self.tick,
            'width': self.width,
            'height': self.height,
            'agents': agent_entities,
            'cells': cell_entities,
            'total_entities': len(agent_entities) + len(cell_entities)
        }
    
    def describe_map(self) -> List[List[int]]:
        """
        Get static map data for initial transmission to client.

        Returns:
            List of [x, y, biome] for all cells where biome != 0
            -- not implemented, just return empty list
        """
        return []
    
    def mark_clean(self):
        """Clear dirty tracking (call after sending delta to clients)."""
        self._dirty_agents.clear()
        self._spawned_agents.clear()
        self._despawned_agents.clear()
    
    def inspect_agent(self, agent_id: int) -> Optional[dict]:
        """
        Get full state of a specific agent (for inspect command).
        Returns None if agent doesn't exist.
        """
        if agent_id in self.agents:
            return self.agents[agent_id].to_full_dict()
        return None
    
    @staticmethod
    def distance(p1, p2) -> float:
        """
        Distance between two positions in the simulation's topology.
        
        Args:
            p1: First position (tuple/list of coordinates)
            p2: Second position (tuple/list of coordinates)
        
        Returns:
            Distance as a float
        
        Placeholder, currently unused, just return Chebyshev distance (L∞ norm).
        """
        return max(abs(a - b) for a, b in zip(p1, p2))
    
    @staticmethod
    def is_in_viewport(position, center, radius) -> bool:
        """
        Check if a position is within a viewport.
        
        This is simulation-specific - defines what "within radius" means
        for this simulation's topology.
        
        Args:
            position: Agent's position (tuple/list)
            center: Viewport center (tuple/list)
            radius: Viewport radius (float)
        
        Returns:
            True if position is within viewport
        """
        return World.distance(position, center) <= radius
    
    def get_viewport(self, center, radius) -> dict:
        """
        Get all entities (agents and map cells) within a viewport region.
        
        This is the simulation's implementation of viewport queries.
        The server calls this without knowing the topology details.
        
        Args:
            center: Viewport center position (tuple/list of coordinates)
            radius: Viewport radius
        
        Returns:
            {
                'tick': current tick,
                'center': center position as list,
                'radius': radius,
                'agents': list of entity dicts (agents + cells),
                'count': number of entities
            }
        
        "Entities" are agents and map cells
        
        """
        entities = []
        
        # Add agents in viewport
        for agent in self.agents.values():
            if self.is_in_viewport(agent.position, center, radius):
                entities.append(agent.to_viewport_dict())
        
        # Add map cells in viewport
        for (x, y), food in self.food.items():
            if self.is_in_viewport((x, y), center, radius):
                entities.append({
                    'id': -(x * self.height + y + 1),  # Negative IDs for cells
                    'map': 1,  # CELL type
                    'x': x,
                    'y': y,
                    'food': round(food, 2)
                })
        
        return {
            'tick': self.tick,
            'center': list(center),
            'radius': radius,
            'agents': entities,  # Named 'agents', as in, we treat cells as generalized "agents"
            'count': len(entities)
        }
    
    def _move_agent(self, agent: Agent, new_x: int, new_y: int) -> bool:
        """
        Move an agent to a new position.
        
        Returns:
            True if move succeeded, False if blocked (out of bounds)
        """
        if new_x == agent.x and new_y == agent.y:
            return True
        
        # Clamp to world bounds
        new_x = max(0, min(self.width - 1, new_x))
        new_y = max(0, min(self.height - 1, new_y))
        
        self.spatial_grid.move(agent.id, agent.x, agent.y, new_x, new_y)
        agent.x = new_x
        agent.y = new_y
        self._mark_dirty(agent.id)
        return True
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-directional)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        random.shuffle(neighbors)
        return neighbors
    
    # =========================================================================
    # MAIN SIMULATION STEP
    # =========================================================================
    
    def step(self):
        """
        Execute one simulation tick using cell-based processing.
        
        PHASE 1: Cell-based local interactions (no movement)
          - distribute food
          - pairwise interaction phase

        PHASE 2: trade/migration
          - interact with agents in neighboring cells under certain conditions
          - possibility to migrate to neighboring cells

        PHASE 3: Mating, birth and death
        
        Any agent-modifying commands (spawn/kill) received during step() are
        deferred and executed after the step completes.
        """
        self._stepping = True
        try:
            self._step_inner()
        finally:
            self._stepping = False
            self._process_deferred_commands()
    
    def _step_inner(self):
        """Internal step logic - called by step() with _stepping flag managed."""

        # Bookkeeping, resetting counters etc.
        self.tick += 1
        # Reset per-tick statistics
        for key in self._stats:
            self._stats[key] = 0

        # Initialize histograms
        hist_counts = {k: [0] * self.HIST_BINS for k in self._hist_max}
        hist_new_max = {k: 0.0 for k in self._hist_max}
        
        # Food regeneration
        for pos in self.food:
             
            if self.SEASON_STRENGTH == 0 or self.TICK_YEARS >= 1:
                current_ceiling = self.FOOD_CEILING
                current_regen = self.FOOD_REGEN_PER_TURN
            else:
                season_effect = 1 - 0.5 * self.SEASON_STRENGTH * (1 + math.sin((self.tick % (0.25/self.TICK_YEARS)) / (0.25/self.TICK_YEARS) * TWOPI))
                current_ceiling = self.FOOD_CEILING * season_effect
                current_regen = self.FOOD_REGEN_PER_TURN * season_effect
            
            self.food[pos] = min(current_ceiling, self.food[pos] + current_regen)
        
        # PHASE 1: Cell-based local interactions
        cells_with_agents = set()
        for agent in self.agents.values():
            cells_with_agents.add(agent.cell)
        
        for cell in cells_with_agents:
            self._process_cell_interactions(cell)

        self._mated_this_tick.clear()  # Reset mating tracker for new tick
                
        # PHASE 2: Movement with shared neighborhood per cell
        
        cell_agents = defaultdict(list)
        for agent in self.agents.values():
            cell_agents[agent.cell].append(agent)
        
        for (cx, cy), agents_here in cell_agents.items():

            for agent in agents_here:

                # Histogram updates
                for histname, get_val in self.hist_value_getters.items():
                    key = histname
                    if key in self._hist_max:
                        val = get_val(agent)
                        max_val = self._hist_max[key]
                        if max_val > 0:
                            bin_idx = int(val * self.HIST_BINS / max_val)
                            bin_idx = max(0, min(bin_idx, self.HIST_BINS - 1))
                        else:
                            bin_idx = 0
                        hist_counts[key][bin_idx] += 1
                        hist_new_max[key] = max(hist_new_max[key], val)
                # End of histogram update sequence

            # Pre-compute fertile agents in THIS cell (for quick mate check)
            fertile_males_in_cell = []
            fertile_females_in_cell = []
            for agent in agents_here:
                investment = self.MALE_INVESTMENT if agent.sex == 0 else (1 - self.MALE_INVESTMENT)
                threshold = self.REPRODUCTION_THRESHOLD * investment
                if agent.energy >= threshold:
                    age = self._age_years(agent)
                    if agent.sex == 0 and age >= self.ADOLESCENCE:
                        fertile_males_in_cell.append(agent)
                    elif age >= self.ADOLESCENCE and age < self.MENOPAUSE:
                        fertile_females_in_cell.append(agent)

            # Build cell context for optimized processing
            cell_context = {
                'fertile_males': fertile_males_in_cell,
                'fertile_females': fertile_females_in_cell,
            }

            # Process all agents with the shared context
            for agent in agents_here:
                self._process_turn(agent, cell_context)
            # End of cell loop

        # Update histogram maxima for next turn: build into temp, then atomic assign
        self._hist_max.update({k: max(v, 0.001) for k, v in hist_new_max.items() if self._hist_max[k] != -1})
        new_histograms = {k: [self._hist_max[k]] + counts for k, counts in hist_counts.items()}
        self._histograms = new_histograms  # atomic rebind
        self._hist_tick = self.tick  # record when histograms were built
        
        # Log statistics periodically
        if self.STATS_LOG_INTERVAL > 0 and self.tick % self.STATS_LOG_INTERVAL == 0:
            self._log_stats()
        
        self._check_termination()
    
    def _process_deferred_commands(self):
        """
        Process commands that were deferred during step().
        
        Called automatically at the end of step() after _stepping is set to False.
        Commands are processed in FIFO order.
        
        Note: Results from deferred commands are not returned to the original caller.
        The caller received {'status': 'deferred', ...} at call time.
        """
        while self._deferred_commands:
            name, value = self._deferred_commands.pop(0)
            # Re-call set_param - now _stepping is False so it will execute
            result = self.set_param(name, value)
            # Log if command failed (for debugging)
            if result.get('status') == 'error':
                print(f"Deferred command '{name}' failed: {result.get('message')}")
    
    def _process_cell_interactions(self, cell: Tuple[int, int]):
        """
        Process local interactions within a cell: food distribution, pairwise interaction.
        
        """
        x, y = cell
        agent_ids = list(self.spatial_grid.agents_at(x, y))
                
        agents_in_cell = []
        
        for aid in agent_ids:
            if aid not in self.agents:
                continue
            agent = self.agents[aid]
            agents_in_cell.append(agent)
       
        # 1. Pay metabolism cost and share food
        if agents_in_cell:
            for agent in agents_in_cell:
                agent.energy -= self.METABOLISM_COST 
                age = self._age_years(agent)
                if age > self.SENESCENCE:
                    agent.energy -= (self.tick - agent.born - self.SENESCENCE) * self.AGE_PENALTY
            
            hungry_agents = [a for a in agents_in_cell if a.energy < self.MAX_ENERGY]
            
            if hungry_agents:
                food_here = self.food[(x, y)]
                n_hungry = len(hungry_agents)
                total_demand = n_hungry * self.EAT_RATE
                
                # Fair distribution: if not enough food, split equally
                if food_here >= total_demand:
                    per_agent = self.EAT_RATE
                    consumed = total_demand
                else:
                    per_agent = food_here / n_hungry
                    consumed = food_here
                
                self.food[(x, y)] -= consumed
                
                for agent in hungry_agents:
                    agent.energy += per_agent
        
        # Note: Mating is handled during agent turn processing, not here
    
    # =========================================================================
    # AGENT TURN PROCESSING (called from step with pre-built neighborhood)
    # =========================================================================
    
    def _process_turn(self, agent: Agent, ctx: dict):

        if agent.energy <= 0:
            self.remove_agent(agent.id)
            self._stats['deaths'] += 1
            return

        # Try to mate if fertile
        age = self._age_years(agent)
        fertility_threshold = self.REPRODUCTION_THRESHOLD + self.REPRODUCTION_COST * (self.MALE_INVESTMENT if agent.sex == 0 else (1 - self.MALE_INVESTMENT))
        if agent.energy >= fertility_threshold and age >= self.ADOLESCENCE and (agent.sex == 0 or age < self.MENOPAUSE):
            opposite_sex_in_cell = ctx['fertile_males'] if agent.sex == 1 else ctx['fertile_females']
            mate = self._pick_mate_from_list(agent, opposite_sex_in_cell)
            if mate:
                self._mate(agent, mate)
    
    def _pick_mate_from_list(self, searcher: Agent, candidates: List[Agent]) -> Optional[Agent]:
        """
        Pick a mate from a pre-filtered list.
        
        Args:
            searcher: The agent searching
            candidates: Pre-filtered list of potential mates
        
        Returns:
            First valid mate, or None
        
        Placeholder for more sophisticated mate selection logic.
        """
        for agent in candidates:
            if agent.id != searcher.id and agent.id in self.agents:
                return agent
        return None
    
    def _mate(self, initiator: Agent, partner: Agent):
        """
        Execute mating between two agents.
        Both must be fertile (caller should verify).
        Each agent can only mate once per tick.
        """
        # Check if either has already mated this tick
        if initiator.id in self._mated_this_tick or partner.id in self._mated_this_tick:
            return
        
        threshold = self.REPRODUCTION_THRESHOLD
        male_inv = self.MALE_INVESTMENT
        initial_e = self.INITIAL_ENERGY
        repro_cost = self.REPRODUCTION_COST
        
        # Determine male and female
        if initiator.sex == 0:
            male, female = initiator, partner
        else:
            male, female = partner, initiator
        
        # Split cost by investment ratio
        male.energy -= repro_cost * male_inv
        female.energy -= (repro_cost * (1 - male_inv))
        
        # Mark both as having mated this tick
        self._mated_this_tick.add(initiator.id)
        self._mated_this_tick.add(partner.id)
        self._stats['births'] += 1
        
        # Spawn at female position
        spawn_x = female.x 
        spawn_y = female.y 
        # Clamp to world bounds
        spawn_x = max(0, min(self.width - 1, spawn_x))
        spawn_y = max(0, min(self.height - 1, spawn_y))
        
        offspring_energy = initial_e
        
        self.spawn_agent(spawn_x, spawn_y, 
                        offspring_energy, parent_ids=(male.id, female.id))
    
    # =========================================================================
    # TERMINATION AND UTILITIES
    # =========================================================================
    
    def _log_stats(self):
        """Log simulation statistics to console."""
        count = self.count_agents()
        
        print(f"[Tick {self.tick}] Population:{count} | "
              f"Births:{self._stats['births']} Deaths:{self._stats['deaths']}")
    
    def _check_termination(self):
        """Check if simulation should halt."""
        count = self.count_agents()
        
        if count == 0:
            self.halted = True
            self.halt_reason = "Population extinct"
        elif count > 10000:
            self.halted = True
            self.halt_reason = "Population exploded"
    
    def count_agents(self):
        return sum(1 for a in self.agents.values())
   
    def report_statistics(self) -> dict:
        histograms = self._histograms
    
        def sanitize(v):
            max_val = 1.0 if v[0] == -1 else v[0]
            return [round(max_val, 2)] + v[1:]
    
        return {k: sanitize(v) for k, v in histograms.items()}

    @classmethod
    def report_params(cls) -> dict:
        """
        Report simulation parameters.
        
        This is the simulation's own decision about what parameters to expose.
        Server should call this rather than accessing constants directly.
        """
        return {
                'food_per_turn': cls.FOOD_REGEN_PER_TURN,
                'food_ceiling': cls.FOOD_CEILING,
                'season_strength': cls.SEASON_STRENGTH,
                'map_width': cls.DEFAULT_WIDTH,
                'map_height': cls.DEFAULT_HEIGHT,  
                'histo_bins': cls.HIST_BINS,    
                'initial_energy': cls.INITIAL_ENERGY,
                'metabolism_cost': cls.METABOLISM_COST,
                'reproduction_threshold': cls.REPRODUCTION_THRESHOLD,
                'reproduction_cost': cls.REPRODUCTION_COST,
                'eat_rate': cls.EAT_RATE,
                'senescence': cls.SENESCENCE,
                'age_penalty': cls.AGE_PENALTY,
                'max_energy': cls.MAX_ENERGY,    
                'male_investment': cls.MALE_INVESTMENT,
            }

    # Mapping from JSON param names to class attribute names
    _PARAM_MAP = {
        'food_per_turn': 'FOOD_REGEN_PER_TURN',
        'food_ceiling': 'FOOD_CEILING',
        'season_strength': 'SEASON_STRENGTH',
        'map_width': 'DEFAULT_WIDTH',
        'map_height': 'DEFAULT_HEIGHT',
        'histo_bins': 'HIST_BINS',    
        'initial_energy': 'INITIAL_ENERGY',
        'metabolism_cost': 'METABOLISM_COST',
        'reproduction_threshold': 'REPRODUCTION_THRESHOLD',
        'reproduction_cost': 'REPRODUCTION_COST',
        'eat_rate': 'EAT_RATE',
        'senescence': 'SENESCENCE',
        'age_penalty': 'AGE_PENALTY',
        'max_energy': 'MAX_ENERGY',
        'male_investment': 'MALE_INVESTMENT',
    }
    
    # Commands that operate on the world instance
    _COMMANDS = {'kill_agent', 'clone_agent', 'boost_agent', 'breeding_pair'}
    
    @classmethod
    def _set_class_param(cls, name: str, value) -> dict:
        """
        Internal classmethod to set class-level simulation parameters.
        Called by set_param() when name is in _PARAM_MAP.
        """
        attr_name = cls._PARAM_MAP[name]
        
        # Get current value to determine type
        try:
            current_value = getattr(cls, attr_name)
        except AttributeError:
            return {
                'status': 'error',
                'message': f'Internal error: attribute {attr_name} not found'
            }
        
        # Convert value to appropriate type
        try:
            if isinstance(current_value, int):
                new_value = int(value)
            elif isinstance(current_value, float):
                new_value = float(value)
            else:
                new_value = value
        except (ValueError, TypeError) as e:
            return {
                'status': 'error',
                'message': f'Invalid value for {name}: {e}'
            }
        
        # Validate value (basic sanity checks)
        if isinstance(new_value, (int, float)):
            if new_value < 0 and 'cost' not in name and 'strength' not in name:
                return {
                    'status': 'error',
                    'message': f'Value for {name} cannot be negative: {new_value}'
                }
        
        # Set the new value
        setattr(cls, attr_name, new_value)
        
        return {
            'status': 'ok',
            'name': name,
            'old_value': current_value,
            'new_value': new_value
        }
    
    def set_param(self, name: str, value) -> dict:
        """
        Set a simulation parameter or execute a command.
        
        Args:
            name: Parameter name in dotted format (e.g., 'prey.move_cost') or command name
            value: New value for parameters, or command-specific data
        
        Commands:
            kill_agent: value=agent_id - despawn the agent with this id
            boost_agent: value=agent_id - double the agent's energy
        
        Returns:
            {'status': 'ok', ...} on success
            {'status': 'deferred', ...} for deferred execution
            {'status': 'error', 'message': '...'} on failure
        """
        # Check if this is a class-level parameter
        if name in self._PARAM_MAP:
            return self._set_class_param(name, value)
        
        # Check if this is a command
        if name in self._COMMANDS:
            
            # Commands that modify the agents dict must be deferred during step()
            # to avoid "dictionary changed size during iteration" errors
            if name in ('kill_agent') and self._stepping:
                self._deferred_commands.append((name, value))
                return {
                    'status': 'deferred',
                    'command': name,
                    'message': 'Command queued for execution after current step completes'
                }
            
            if name == 'kill_agent':
                try:
                    agent_id = int(value)
                except (ValueError, TypeError):
                    return {'status': 'error', 'message': f'kill_agent requires integer agent id, got: {value}'}
                
                if agent_id not in self.agents:
                    return {'status': 'error', 'message': f'Agent {agent_id} not found'}
                
                self.remove_agent(agent_id)
                self._stats['deaths'] += 1
                return {'status': 'ok', 'command': name, 'agent_id': agent_id}

            elif name == 'boost_agent':
                try:
                    agent_id = int(value)
                except (ValueError, TypeError):
                    return {'status': 'error', 'message': f'boost_agent requires integer agent id, got: {value}'}
                
                if agent_id not in self.agents:
                    return {'status': 'error', 'message': f'Agent {agent_id} not found'}
                
                agent = self.agents[agent_id]
                old_energy = agent.energy
                agent.energy *= 2
                self._mark_dirty(agent_id)
                
                return {
                    'status': 'ok', 
                    'command': name, 
                    'agent_id': agent_id,
                    'old_energy': round(old_energy, 2),
                    'new_energy': round(agent.energy, 2)
                }
        
        # Unknown parameter or command
        return {
            'status': 'error',
            'message': f'Unknown parameter or command: {name}',
            'valid_params': list(self._PARAM_MAP.keys()),
            'valid_commands': list(self._COMMANDS)
        }

def run_simulation(
    width: int = 10,
    height: int = 10,
    initial_pairs: int = 2,
    max_ticks: int = 2000,
    seed: Optional[int] = None
):
    """Run the simulation and print statistics."""
    
    # Use World.create() for initialization
    world = World.create(
        width=width,
        height=height,
        initial_pairs=initial_pairs,
        seed=seed
    )
    
    print(f"Starting simulation: {width}x{height} world, {initial_pairs} founder pairs")
    print("-" * 50)
    
    for tick in range(max_ticks):
        world.step()
        count = world.count_agents()
        
        # Print status every 10 ticks
        if tick % 10 == 0:
            print(f"Tick {world.tick:4d}: Pop={count:4d}")
        
        # Check if simulation halted itself
        if world.halted:
            print(f"\n*** {world.halt_reason} at tick {world.tick} ***")
            break
    else:
        print(f"\n*** Simulation completed {max_ticks} ticks ***")
    
    count = world.count_agents()
    print(f"Final state: Pop={count}")
    
    # Demo: show delta tracking
    print("\n--- Delta tracking demo ---")
    world.halted = False  # Reset for demo
    world.mark_clean()  # Clear dirty state
    world.step()  # Do one more step
    delta = world.get_dirty_state()
    print(f"After 1 tick: {len(delta['spawned'])} spawned, {len(delta['updated'])} updated, {len(delta['despawned'])} despawned")


if __name__ == "__main__":
    run_simulation(seed=42)

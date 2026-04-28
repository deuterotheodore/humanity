import random
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import logging

# Import decision logic module (can be swapped for different strategies)
import decide as decide_module

TWOPI = 2 * math.pi

@dataclass
class Agent:
    id: int
    sex: int = 0            # 0 = male, 1 = female
    born: int = 0           # Tick when agent was spawned
    parent: List[int] = field(default_factory=lambda: [-1, -1])  # IDs of parents ([-1,-1] for founders)
    offspring: List[int] = field(default_factory=list)  # IDs of offspring
    kinship: Dict[int, float] = field(default_factory=dict)  # {agent_id: relatedness coefficient}
    pregnant: bool = False  # True if carrying unborn offspring (females only)
    # State parameters
    x: int = 0
    y: int = 0
    energy: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    trust: float = 0.0
    # Genetic parameters - phenotypes (inherited with mutation, diploid)
    o: float = 0.5   # Openness
    c: float = 0.5   # Conscientiousness
    e: float = 0.5   # Extraversion
    a: float = 0.5   # Agreeableness
    n: float = 0.5   # Neuroticism
    kin: float = 0.5   # Kin altruism tendency (trait, not to confuse with kinship dict)
    xeno: float = 0.5  # Xenophilia/xenophobia tendency
    # Diploid variances (|allele1 - allele2|, used only at procreation)
    vo: float = 0.0
    vc: float = 0.0
    ve: float = 0.0
    va: float = 0.0
    vn: float = 0.0
    vkin: float = 0.0
    vxeno: float = 0.0
    # Speciation genes [0-10] - haploid (averaged over many loci)
    genes: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])

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
            'vo': round(self.vo, 2),
            'vc': round(self.vc, 2),
            've': round(self.ve, 2),
            'va': round(self.va, 2),
            'vn': round(self.vn, 2),
            'vkin': round(self.vkin, 2),
            'vxeno': round(self.vxeno, 2),
            'genes': [round(g, 2) for g in self.genes],
            'kinship': {k: round(v, 3) for k, v in self.kinship.items()},
            'n_kin': len(self.kinship),
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
    EAT_RATE = 1.0  # DEPRECATED - food distribution now handled via PD game
    INFANCY = 3
    CHILDHOOD = 7
    ADOLESCENCE = 14
    ADULTHOOD = 21
    MENOPAUSE = 44
    SENESCENCE = 60
    SENESCENCE_DEATH_RATE = 0.005  # Per-tick death probability = (age - SENESCENCE) * this (~2%/year)
    GESTATION_TICKS = 3  # int(0.75 / TICK_YEARS) - 9 months gestation
    
    # Twin probabilities
    P_ID_TWINS = 0.005  # Identical twins (0.5%)
    P_FR_TWINS = 0.015  # Fraternal twins (1.5%)
    
    # Mutation rates for genetic parameters
    GENE_MUTATION_SD = 0.1
    TRAIT_MUTATION_SD = 0.1
    
    # Kinship tracking threshold (cousin-tier = 0.125, half-cousin = 0.0625)
    RELATEDNESS_THRESHOLD = 0.0625
    # Mate selection: strong suppression above this kinship (cousin = 0.125)
    CONSANGUINITY_TOLERANCE = 0.15
    
    # Migration parameters
    P_MIGRATION = 0.1  # Probability of migration from overpopulated cell
    FEM_MIGRATION_RATIO = 0.5  # Female migration probability multiplier
    
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
        
        # Prisoner's dilemma interaction history
        # Key: (id1, id2) with id1 < id2 (normalized)
        # Value: [recent1, recent2, n_pre, sum1_pre, sum2_pre]
        #   recent1/2: last 3 actions (oldest first), lists of up to 3 elements
        #   n_pre: count of turns before the last 3
        #   sum1/2_pre: cooperation sums for pre-history (avg = sum/n_pre)
        self._interaction_history: Dict[Tuple[int, int], List] = {}
        # Reverse index: agent_id -> set of (id1, id2) keys they participate in
        self._interaction_index: Dict[int, Set[Tuple[int, int]]] = {}
        
        # Per-tick statistics (reset each tick, accumulated for logging)
        self._stats = {
            'births': 0,
            'deaths': 0,
        }

        # Statistics histograms, set default maxima
        self._hist_max = {
            'age': 60,      # initial defaults
            'energy': self.MAX_ENERGY,
            'kin': 20,      # number of tracked kin relations
        }
        self._histograms = {}  # filled by _update_histograms
        self._hist_tick = 0
        # How to report each property as histogram value
        self.hist_value_getters = {
            'age': lambda a: max(0, self._age_years(a)),  # Clamp negative (unborn) to 0
            'energy': lambda a: a.energy,
            'kin': lambda a: len(a.kinship),
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
    
    def _metabolism_multiplier(self, agent: Agent) -> float:
        """
        Calculate metabolism/eating rate multiplier based on age, sex, and pregnancy.
        
        Returns 0 for unborn agents (their consumption is in mother's pregnancy multiplier).
        
        Multipliers (cumulative):
        - Age 0 to ADULTHOOD: 0.5 to 1.0 (linear)
        - Age > SENESCENCE: 0.8
        - Male: 1.3
        - Pregnant: 1.2
        """
        age = self._age_years(agent)
        
        # Unborn agents don't consume
        if age < 0:
            return 0.0
        
        # Base age multiplier
        if age < self.ADULTHOOD:
            # Linear interpolation from 0.5 at age 0 to 1.0 at ADULTHOOD
            age_mult = 0.5 + 0.5 * (age / self.ADULTHOOD)
        elif age > self.SENESCENCE:
            age_mult = 0.8
        else:
            age_mult = 1.0
        
        # Sex multiplier (male = 1.3)
        sex_mult = 1.3 if agent.sex == 0 else 1.0
        
        # Pregnancy multiplier
        preg_mult = 1.2 if agent.pregnant else 1.0
        
        return age_mult * sex_mult * preg_mult
    
    # =========================================================================
    # PRISONER'S DILEMMA GAME
    # =========================================================================
    
    def _decide(self, own: Agent, opp: Agent, history: List, n_pre: int, kinship: float, distance: float) -> int:
        """
        Decide whether to cooperate (1) or defect (0) in prisoner's dilemma.
        
        Delegates to decide_module.decide() which computes cooperation probability
        based on personality traits, history, kinship, and genetic distance.
        
        Args:
            own: The agent making the decision
            opp: The opponent agent
            history: [
                [own_t-1, own_t-2, own_t-3],  # own recent actions (most recent first)
                [opp_t-1, opp_t-2, opp_t-3],  # opponent recent actions (most recent first)
                [own_avg, opp_avg]             # pre-history averages, or []
            ]
            n_pre: Number of interactions before the last 3 (relationship duration)
            kinship: Relatedness coefficient between agents
            distance: Squared Euclidean genetic distance
        
        Returns:
            1 for cooperate, 0 for defect
        """
        return decide_module.decide(own, opp, history, n_pre, kinship, distance)
    
    def _genetic_distance(self, a1: Agent, a2: Agent) -> float:
        """Squared Euclidean distance between agents' genes arrays."""
        return sum((g1 - g2) ** 2 for g1, g2 in zip(a1.genes, a2.genes))
    
    def _get_history_for_decide(self, own_id: int, opp_id: int) -> Tuple[List, int]:
        """
        Get interaction history formatted for decide() from own's perspective.
        
        Returns:
            (history, n_pre) where history is:
            [
                [own_t-1, own_t-2, own_t-3],  # own's last 3 actions (most recent first)
                [opp_t-1, opp_t-2, opp_t-3],  # opponent's last 3 actions (most recent first)
                [own_avg, opp_avg]             # averages for turns before last 3, or []
            ]
            and n_pre is the count of pre-history turns (relationship duration beyond last 3)
        """
        key = (min(own_id, opp_id), max(own_id, opp_id))
        
        if key not in self._interaction_history:
            return [[], [], []], 0
        
        # Storage format: [recent1, recent2, n_pre, sum1_pre, sum2_pre]
        recent1, recent2, n_pre, sum1, sum2 = self._interaction_history[key]
        
        # Determine which is own vs opp based on key ordering
        if own_id < opp_id:
            own_recent, opp_recent = recent1, recent2
            own_sum, opp_sum = sum1, sum2
        else:
            own_recent, opp_recent = recent2, recent1
            own_sum, opp_sum = sum2, sum1
        
        if not own_recent:
            return [[], [], []], 0
        
        # Recent actions (reversed: most recent first)
        own_recent_out = list(reversed(own_recent))
        opp_recent_out = list(reversed(opp_recent))
        
        # Pre-history averages
        if n_pre > 0:
            pre_avg = [own_sum / n_pre, opp_sum / n_pre]
        else:
            pre_avg = []
        
        return [own_recent_out, opp_recent_out, pre_avg], n_pre
    
    def _record_interaction(self, id1: int, id2: int, action1: int, action2: int):
        """
        Record one turn of prisoner's dilemma interaction.
        
        Storage format: [recent1, recent2, n_pre, sum1_pre, sum2_pre]
        - recent1/2: last 3 actions (oldest first), lists of up to 3 elements
        - n_pre: count of turns before the last 3
        - sum1/2_pre: cooperation sums for pre-history (avg = sum/n_pre)
        
        Args:
            id1, id2: Agent IDs (order doesn't matter, will be normalized)
            action1: id1's action (0=defect, 1=cooperate)
            action2: id2's action
        """
        # Normalize key so id1 < id2
        if id1 > id2:
            id1, id2 = id2, id1
            action1, action2 = action2, action1
        
        key = (id1, id2)
        
        if key not in self._interaction_history:
            # [recent1, recent2, n_pre, sum1_pre, sum2_pre]
            self._interaction_history[key] = [[], [], 0, 0, 0]
            # Update reverse index
            self._interaction_index.setdefault(id1, set()).add(key)
            self._interaction_index.setdefault(id2, set()).add(key)
        
        entry = self._interaction_history[key]
        recent1, recent2, n_pre, sum1, sum2 = entry
        
        # If recent buffer is full, move oldest to pre-history
        if len(recent1) == 3:
            oldest1 = recent1.pop(0)
            oldest2 = recent2.pop(0)
            n_pre += 1
            sum1 += oldest1
            sum2 += oldest2
            entry[2] = n_pre
            entry[3] = sum1
            entry[4] = sum2
        
        # Append new actions
        recent1.append(action1)
        recent2.append(action2)
    
    def _cleanup_interaction_history(self, agent_id: int):
        """Remove all interaction history entries for a dead agent."""
        if agent_id not in self._interaction_index:
            return
        
        for key in list(self._interaction_index[agent_id]):
            # Remove from main history
            if key in self._interaction_history:
                del self._interaction_history[key]
            # Remove from other agent's index
            other_id = key[0] if key[1] == agent_id else key[1]
            if other_id in self._interaction_index:
                self._interaction_index[other_id].discard(key)
        
        del self._interaction_index[agent_id]
    
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
                    parent_ids: Tuple[int, int] = (-1, -1),
                    born_tick: int = None,
                    _twin_of: Agent = None) -> Agent:
        """
        Create a new agent in the world.
        
        Args:
            x, y: Position (cell coordinates, integers)
            energy: Initial energy
            parent_ids: Tuple of parent IDs ((-1, -1) for founders)
            born_tick: Tick when agent is "born" (default: current tick)
                       Set to future tick for gestation (agent exists but age < 0)
            _twin_of: Internal - if set, this is a fraternal twin of the given agent
        
        Returns:
            The created Agent (primary twin if twins spawned)
        
        Genetic inheritance (diploid for personality traits):
            - Founders get homozygous defaults (variance = 0)
            - Offspring inherit one allele from each parent (randomly chosen)
            - Mutation applied to each inherited allele
            - Phenotype = mean of two alleles
            - Speciation genes remain haploid (simple averaging)

        Sex is assigned randomly (50/50).
        
        Twins:
            - P_ID_TWINS chance of identical twins (same genetics, kinship=1.0)
            - P_FR_TWINS chance of fraternal twins (independent genetics, kinship=0.5)
        """
        if born_tick is None:
            born_tick = self.tick
        agent_id = self._allocate_id()
        
        # Determine genetic parameters
        parent1_id, parent2_id = parent_ids
        parent1 = self.agents.get(parent1_id) if parent1_id >= 0 else None
        parent2 = self.agents.get(parent2_id) if parent2_id >= 0 else None
        
        if parent1 and parent2:
            # Sexual reproduction with diploid inheritance
            trait_sd = self.TRAIT_MUTATION_SD
            gene_sd = self.GENE_MUTATION_SD
            
            # Big Five personality traits (diploid)
            o, vo = self._inherit_diploid(parent1.o, parent1.vo, parent2.o, parent2.vo, trait_sd)
            c, vc = self._inherit_diploid(parent1.c, parent1.vc, parent2.c, parent2.vc, trait_sd)
            e, ve = self._inherit_diploid(parent1.e, parent1.ve, parent2.e, parent2.ve, trait_sd)
            a, va = self._inherit_diploid(parent1.a, parent1.va, parent2.a, parent2.va, trait_sd)
            n, vn = self._inherit_diploid(parent1.n, parent1.vn, parent2.n, parent2.vn, trait_sd)
            
            # Social traits (diploid)
            kin_trait, vkin = self._inherit_diploid(parent1.kin, parent1.vkin, parent2.kin, parent2.vkin, trait_sd)
            xeno, vxeno = self._inherit_diploid(parent1.xeno, parent1.vxeno, parent2.xeno, parent2.vxeno, trait_sd)
            
            # Speciation genes: haploid (simple averaging + mutation)
            genes = []
            for i in range(3):
                avg_gene = (parent1.genes[i] + parent2.genes[i]) / 2
                genes.append(max(0.0, min(10.0, avg_gene + random.gauss(0, gene_sd))))

        else:
            # Founder defaults (homozygous: variance = 0)
            genes = [5.0 + 3 * (1 if x < 5 else -1), 
                     5.0 + 3 * (1 if y < 5 else -1), 
                     5.0 + 2 * (1 if x < 5 else -1)]
            o = c = e = a = n = 0.5
            vo = vc = ve = va = vn = 0.0
            kin_trait = xeno = 0.5
            vkin = vxeno = 0.0

        sex = random.randint(0, 1)  # 0 = male, 1 = female

        agent = Agent(
            id=agent_id,
            x=int(x),
            y=int(y),
            energy=energy,
            sex=sex,
            born=born_tick,
            parent=list(parent_ids),
            offspring=[],
            o=o, vo=vo,
            c=c, vc=vc,
            e=e, ve=ve,
            a=a, va=va,
            n=n, vn=vn,
            kin=kin_trait, vkin=vkin,
            xeno=xeno, vxeno=vxeno,
            genes=genes,
        )
        self.agents[agent_id] = agent
        self.spatial_grid.add(agent_id, x, y)
        self._spawned_agents.add(agent_id)
        self._dirty_agents.add(agent_id)
        
        # Record this agent as offspring of both parents and establish kinship
        if parent1 and parent2:
            parent1.offspring.append(agent_id)
            parent2.offspring.append(agent_id)
            self._establish_kinship(agent, parent1, parent2)
        
        # Handle fraternal twin kinship (sibling relationship already established)
        if _twin_of is not None:
            # Fraternal twin - kinship 0.5 (full sibling)
            agent.kinship[_twin_of.id] = 0.5
            _twin_of.kinship[agent.id] = 0.5
        
        # Check for twins (only for non-founders and not already a twin spawn)
        if parent1 and parent2 and _twin_of is None:
            twin_roll = random.random()
            
            if twin_roll < self.P_ID_TWINS:
                # Identical twin - clone with same genetics
                twin_id = self._allocate_id()
                twin = Agent(
                    id=twin_id,
                    x=int(x),
                    y=int(y),
                    energy=energy,
                    sex=sex,  # Same sex
                    born=born_tick,
                    parent=list(parent_ids),
                    offspring=[],
                    o=o, vo=vo,
                    c=c, vc=vc,
                    e=e, ve=ve,
                    a=a, va=va,
                    n=n, vn=vn,
                    kin=kin_trait, vkin=vkin,
                    xeno=xeno, vxeno=vxeno,
                    genes=list(genes),  # Copy the list
                )
                self.agents[twin_id] = twin
                self.spatial_grid.add(twin_id, x, y)
                self._spawned_agents.add(twin_id)
                self._dirty_agents.add(twin_id)
                parent1.offspring.append(twin_id)
                parent2.offspring.append(twin_id)
                self._establish_kinship(twin, parent1, parent2)
                
                # Identical twins have kinship 1.0
                agent.kinship[twin_id] = 1.0
                twin.kinship[agent_id] = 1.0
                
            elif twin_roll < self.P_ID_TWINS + self.P_FR_TWINS:
                # Fraternal twin - independent genetics
                self.spawn_agent(x, y, energy, parent_ids, born_tick, _twin_of=agent)
        
        return agent
    
    def _inherit_diploid(self, p1_pheno: float, p1_var: float, 
                         p2_pheno: float, p2_var: float, 
                         mutation_sd: float,
                         min_val: float = 0.0, max_val: float = 1.0) -> Tuple[float, float]:
        """
        Diploid inheritance: each parent contributes one randomly-chosen allele.
        
        Args:
            p1_pheno, p1_var: Parent 1's phenotype and diploid variance
            p2_pheno, p2_var: Parent 2's phenotype and diploid variance
            mutation_sd: Standard deviation for mutation noise
            min_val, max_val: Bounds for allele values
        
        Returns:
            (phenotype, variance) tuple for offspring
        
        The two alleles are reconstructed as pheno ± var/2, then one is
        randomly chosen from each parent, mutated, and combined.
        """
        # Reconstruct alleles from phenotype ± variance/2
        p1_hi, p1_lo = p1_pheno + p1_var / 2, p1_pheno - p1_var / 2
        p2_hi, p2_lo = p2_pheno + p2_var / 2, p2_pheno - p2_var / 2
        
        # Each parent contributes one allele (with mutation)
        from_p1 = (p1_hi if random.random() < 0.5 else p1_lo) + random.gauss(0, mutation_sd)
        from_p2 = (p2_hi if random.random() < 0.5 else p2_lo) + random.gauss(0, mutation_sd)
        
        # Clamp alleles before combining
        from_p1 = max(min_val, min(max_val, from_p1))
        from_p2 = max(min_val, min(max_val, from_p2))
        
        phenotype = (from_p1 + from_p2) / 2
        variance = abs(from_p1 - from_p2)
        
        return phenotype, variance
    
    def _establish_kinship(self, child: Agent, parent1: Agent, parent2: Agent):
        """
        Calculate and store kinship relations for a newborn.
        
        Simplified "known kinship" model:
        - Parents are always r=0.5 to child
        - Full siblings are always r=0.5 to child
        - Inherit from parents' kinship at half the value
        - Only propagate entries > 2×threshold (so halved value meets threshold)
        - For shared relatives: use min(1.2 * max, sum) to recognize some
          consanguinity while still enforcing decay over ~3 generations
        
        This gives: parent=0.5, full sibling=0.5, grandparent=0.25, half-sibling≥0.25
        
        Relations are stored bidirectionally and pruned below RELATEDNESS_THRESHOLD.
        """
        threshold = self.RELATEDNESS_THRESHOLD
        inherit_threshold = 2 * threshold  # 0.125 - only propagate if half will meet threshold
        new_kin = {}
        
        # Parents always at 0.5
        new_kin[parent1.id] = 0.5
        new_kin[parent2.id] = 0.5
        
        # Child's parent set for sibling detection
        child_parents = {parent1.id, parent2.id}
        
        # Collect contributions from both parents
        contributions = {}  # x_id -> [r_from_p1/2, r_from_p2/2]
        
        for x_id, r_p1 in parent1.kinship.items():
            if x_id == parent2.id:
                continue
            if r_p1 > inherit_threshold:
                contributions[x_id] = [r_p1 / 2, 0]
        
        for x_id, r_p2 in parent2.kinship.items():
            if x_id == parent1.id:
                continue
            if r_p2 > inherit_threshold:
                if x_id in contributions:
                    contributions[x_id][1] = r_p2 / 2
                else:
                    contributions[x_id] = [0, r_p2 / 2]
        
        # Combine contributions: min(1.2 * max, sum) for consanguinity with decay
        for x_id, (c1, c2) in contributions.items():
            # Check if X is a full sibling (same parents as child)
            if x_id in self.agents:
                x_agent = self.agents[x_id]
                x_parents = {x_agent.parent[0], x_agent.parent[1]}
                if x_parents == child_parents:
                    new_kin[x_id] = 0.5  # Full sibling override
                    continue
            
            if c1 > 0 and c2 > 0:
                # Both parents related to X - use capped formula
                new_kin[x_id] = min(1.2 * max(c1, c2), c1 + c2)
            else:
                # Only one parent related
                new_kin[x_id] = c1 + c2
        
        # Store in child and update relatives' records (bidirectional)
        for x_id, r in new_kin.items():
            if r >= threshold:
                child.kinship[x_id] = r
                if x_id in self.agents:
                    self.agents[x_id].kinship[child.id] = r
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the world and clean up kinship records.
        
        If the agent is a mother, all her dependent children die too:
        - Unborn (age < 0): fetus cannot survive without mother
        - Infants (0 <= age <= INFANCY): too young to survive independently
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # If mother dies, kill dependent children (unborn + infants)
            if agent.sex == 1:  # Female
                dependents_to_kill = []
                for child_id in agent.offspring:
                    child = self.agents.get(child_id)
                    if child:
                        child_age = self._age_years(child)
                        if child_age <= self.INFANCY:  # Includes unborn (age < 0)
                            dependents_to_kill.append(child_id)
                # Kill dependents (recursively calls remove_agent)
                for dependent_id in dependents_to_kill:
                    if dependent_id in self.agents:  # May already be removed
                        self.remove_agent(dependent_id)
                        self._stats['deaths'] += 1
            
            # Check agent still exists (may have been removed as infant of another dying mother)
            if agent_id not in self.agents:
                return
                
            # Remove from relatives' kinship dicts
            for relative_id in agent.kinship:
                if relative_id in self.agents:
                    self.agents[relative_id].kinship.pop(agent_id, None)
            # Clean up interaction history
            self._cleanup_interaction_history(agent_id)
            # Remove from spatial grid and tracking
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
    
    def _get_moore_neighborhood(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get cell and all valid neighboring cells (self + 8-directional Moore neighborhood)."""
        cells = [(x, y)]  # Include self
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cells.append((nx, ny))
        return cells
    
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

        # Clear pregnancy flags for females whose children are now born
        for agent in self.agents.values():
            if agent.pregnant and agent.sex == 1:
                # Check if all offspring are now born (age >= 0)
                all_born = True
                for child_id in agent.offspring:
                    child = self.agents.get(child_id)
                    if child and child.born > self.tick:
                        all_born = False
                        break
                if all_born:
                    agent.pregnant = False

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
        
        # =====================================================================
        # PHASE 1: Cell-based local interactions (metabolism, PD, food distribution)
        # =====================================================================
        cells_with_agents = set()
        for agent in self.agents.values():
            cells_with_agents.add(agent.cell)
        
        for cell in cells_with_agents:
            self._process_cell_interactions(cell)

        # =====================================================================
        # PHASE 2: Mating (two-phase: males apply, females choose)
        # =====================================================================
        self._mated_this_tick.clear()
        
        # Build cell->agents map and identify fertile agents
        cell_agents = defaultdict(list)
        fertile_males = []
        fertile_females = []
        fertile_female_ids = set()  # For O(1) lookup
        
        for agent in self.agents.values():
            cell_agents[agent.cell].append(agent)
            
            # Check fertility
            investment = self.MALE_INVESTMENT if agent.sex == 0 else (1 - self.MALE_INVESTMENT)
            threshold = self.REPRODUCTION_THRESHOLD + self.REPRODUCTION_COST * investment
            if agent.energy >= threshold:
                age = self._age_years(agent)
                if agent.sex == 0 and age >= self.ADOLESCENCE:
                    fertile_males.append(agent)
                elif agent.sex == 1 and age >= self.ADOLESCENCE and age < self.MENOPAUSE and not agent.pregnant:
                    fertile_females.append(agent)
                    fertile_female_ids.add(agent.id)
        
        # Phase 2a: Males apply to females in their Moore neighborhood
        # Males cannot apply to their own mother (incest taboo)
        suitors: Dict[int, List[Agent]] = defaultdict(list)  # female_id -> [male suitors]
        
        for male in fertile_males:
            # Scan own cell + neighbors
            for cell in self._get_moore_neighborhood(male.x, male.y):
                for agent in cell_agents.get(cell, []):
                    if agent.id in fertile_female_ids:  # O(1) set lookup
                        # Incest taboo: male cannot apply to own mother
                        if agent.id in male.parent:
                            continue
                        suitors[agent.id].append(male)
        
        # Phase 2b: Females choose among suitors
        for female in fertile_females:
            if female.id not in suitors or not suitors[female.id]:
                continue
            if female.id in self._mated_this_tick:
                continue
            
            # Filter out males who already mated
            available_suitors = [m for m in suitors[female.id] 
                                if m.id not in self._mated_this_tick and m.id in self.agents]
            if not available_suitors:
                continue
            
            # Female chooses using _pick_suitor (considers kinship, prefers high energy)
            chosen_male = self._pick_suitor(female, available_suitors)
            if chosen_male:
                self._mate(chosen_male, female)

        # =====================================================================
        # PHASE 3: Migration from overpopulated cells
        # =====================================================================
        # Recalculate cell_agents after potential deaths
        cell_agents = defaultdict(list)
        for agent in self.agents.values():
            cell_agents[agent.cell].append(agent)
        
        # Cell carrying capacity based on food regeneration
        cell_capacity = self.FOOD_REGEN_PER_TURN / (2 * self.METABOLISM_COST)
        
        for (cx, cy), agents_here in list(cell_agents.items()):
            if len(agents_here) > cell_capacity:
                # Find neighboring cells with fewer agents
                neighbors = self._get_neighbors(cx, cy)
                
                for agent in agents_here:
                    # Young children don't migrate independently - they follow mother
                    agent_age = self._age_years(agent)
                    if agent_age <= self.INFANCY:
                        continue
                    
                    # Migration probability scaled for females
                    migration_prob = self.P_MIGRATION
                    if agent.sex == 1:  # Female
                        migration_prob *= self.FEM_MIGRATION_RATIO
                    
                    if random.random() < migration_prob:
                        # Find least populated neighbor
                        best_neighbor = None
                        best_pop = len(agents_here)
                        
                        for nx, ny in neighbors:
                            neighbor_pop = len(cell_agents.get((nx, ny), []))
                            if neighbor_pop < best_pop:
                                best_pop = neighbor_pop
                                best_neighbor = (nx, ny)
                        
                        if best_neighbor:
                            # Migrate
                            old_cell = agent.cell
                            self._move_agent(agent, best_neighbor[0], best_neighbor[1])
                            # Update cell_agents for subsequent migration decisions
                            cell_agents[old_cell].remove(agent)
                            cell_agents[best_neighbor].append(agent)
                            
                            # If female, move young children (age <= INFANCY) with her
                            if agent.sex == 1:
                                for child_id in agent.offspring:
                                    child = self.agents.get(child_id)
                                    if child:
                                        child_age = self._age_years(child)
                                        if child_age <= self.INFANCY:  # Includes unborn (age < 0)
                                            child_old_cell = child.cell
                                            self._move_agent(child, best_neighbor[0], best_neighbor[1])
                                            # Update cell_agents if child was in tracked cell
                                            if child_old_cell in cell_agents and child in cell_agents[child_old_cell]:
                                                cell_agents[child_old_cell].remove(child)
                                                cell_agents[best_neighbor].append(child)

        # =====================================================================
        # PHASE 4: Death check and histogram collection
        # =====================================================================
        for agent in list(self.agents.values()):
            # Death check: starvation
            if agent.energy <= 0:
                self.remove_agent(agent.id)
                self._stats['deaths'] += 1
                continue
            
            # Death check: senescence (sudden death probability increases with age)
            age = self._age_years(agent)
            if age > self.SENESCENCE:
                death_prob = (age - self.SENESCENCE) * self.SENESCENCE_DEATH_RATE
                if random.random() < death_prob:
                    self.remove_agent(agent.id)
                    self._stats['deaths'] += 1
                    continue
            
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
        Process local interactions within a cell: metabolism, feeding, and prisoner's dilemma.
        
        Flow:
        1. All agents pay metabolism (scaled by age/sex/pregnancy)
        2. Starved agents (energy <= 0) are removed
        3. Children (age < CHILDHOOD) eat first from cell food
        4. Adults play prisoner's dilemma, accumulate scores
        5. Remaining food scaled by cooperation level (avg_score / 2.5)
        6. Adults receive scaled food proportional to their PD scores
        7. Excess food (above MAX_ENERGY) stays in cell
        
        PD scores: CC=3,3  CD=5,0  DC=0,5  DD=1,1
        Baseline 2.5 = expected value if random play
        """
        x, y = cell
        agent_ids = list(self.spatial_grid.agents_at(x, y))
        
        # Gather agents in cell
        agents_in_cell = []
        for aid in agent_ids:
            if aid not in self.agents:
                continue
            agents_in_cell.append(self.agents[aid])
        
        if not agents_in_cell:
            return
        
        # =====================================================
        # STEP 1: Pay metabolism cost
        # =====================================================
        for agent in agents_in_cell:
            mult = self._metabolism_multiplier(agent)
            if mult > 0:  # Skip unborn (mult = 0)
                agent.energy -= self.METABOLISM_COST * mult
        
        # =====================================================
        # STEP 2: Remove starved agents
        # =====================================================
        starved = [a for a in agents_in_cell if a.energy <= 0]
        for agent in starved:
            self.remove_agent(agent.id)
            self._stats['deaths'] += 1
        agents_in_cell = [a for a in agents_in_cell if a.energy > 0]
        
        if not agents_in_cell:
            return
        
        # =====================================================
        # STEP 3: Separate children and adults
        # =====================================================
        children = []
        adults = []
        for agent in agents_in_cell:
            age = self._age_years(agent)
            if age < 0:  # Unborn - don't participate
                continue
            elif age < self.CHILDHOOD:
                children.append(agent)
            else:
                adults.append(agent)
        
        food_here = self.food[(x, y)]
        
        # =====================================================
        # STEP 4: Feed children first
        # =====================================================
        for child in children:
            if food_here <= 0:
                break
            deficit = self.MAX_ENERGY - child.energy
            if deficit > 0:
                # Children get up to their metabolism cost worth of food
                mult = self._metabolism_multiplier(child)
                child_need = min(deficit, self.METABOLISM_COST * mult * 2)  # 2x metabolism as buffer
                child_gets = min(child_need, food_here)
                child.energy += child_gets
                food_here -= child_gets
        
        # =====================================================
        # STEP 5: Adults play prisoner's dilemma
        # =====================================================
        if len(adults) < 2:
            # Solo adult just eats from remaining food
            if adults and food_here > 0:
                agent = adults[0]
                deficit = self.MAX_ENERGY - agent.energy
                agent.energy += min(deficit, food_here)
                food_here -= min(deficit, food_here)
            self.food[(x, y)] = food_here
            return
        
        # Initialize score tracking
        agent_scores = {a.id: 0 for a in adults}
        total_score = 0
        n_games = 0
        
        # Each pair plays once
        for i, own in enumerate(adults):
            for opp in adults[i+1:]:
                # Get history and parameters
                history_own, n_pre = self._get_history_for_decide(own.id, opp.id)
                history_opp, _ = self._get_history_for_decide(opp.id, own.id)
                kinship = own.kinship.get(opp.id, 0.0)
                distance = self._genetic_distance(own, opp)
                
                # Both decide simultaneously
                own_action = self._decide(own, opp, history_own, n_pre, kinship, distance)
                opp_action = self._decide(opp, own, history_opp, n_pre, kinship, distance)
                
                # Record interaction
                self._record_interaction(own.id, opp.id, own_action, opp_action)
                
                # Compute payoffs: CC=3,3  CD=5,0  DC=0,5  DD=1,1
                if own_action == 1 and opp_action == 1:
                    own_score, opp_score = 3, 3
                elif own_action == 1 and opp_action == 0:
                    own_score, opp_score = 0, 5
                elif own_action == 0 and opp_action == 1:
                    own_score, opp_score = 5, 0
                else:  # Both defect
                    own_score, opp_score = 1, 1
                
                agent_scores[own.id] += own_score
                agent_scores[opp.id] += opp_score
                total_score += own_score + opp_score
                n_games += 1
        
        # =====================================================
        # STEP 6: Scale food by cooperation level and distribute
        # =====================================================
        if n_games > 0 and total_score > 0:
            avg_score = total_score / (2 * n_games)  # Per-agent average
            coop_mult = avg_score / 2.5  # 2.5 is neutral baseline
            scaled_pool = food_here * coop_mult  # Effective value to distribute
            
            # Distribute proportionally to scores
            total_distributed = 0
            for agent in adults:
                score = agent_scores[agent.id]
                if score > 0:
                    share = scaled_pool * (score / total_score)
                    deficit = self.MAX_ENERGY - agent.energy
                    actual_gain = min(share, deficit)
                    agent.energy += actual_gain
                    total_distributed += actual_gain
            
            # Physical food consumed proportional to utilization
            # High cooperation: more value per food unit (efficient)
            # Low cooperation: less value per food unit (wasteful)
            if scaled_pool > 0:
                utilization = total_distributed / scaled_pool
            else:
                utilization = 0
            food_consumed = food_here * utilization
            food_here -= food_consumed
        
        # =====================================================
        # STEP 7: Leave remaining food in cell
        # =====================================================
        self.food[(x, y)] = food_here
    
    # =========================================================================
    # AGENT TURN PROCESSING (called from step with pre-built neighborhood)
    # =========================================================================
    
    def _process_turn(self, agent: Agent, ctx: dict):
        """
        Process individual agent turn.
        
        Currently a placeholder - death and mating handled in main step phases.
        Reserved for future per-agent logic (movement decisions, interactions, etc.)
        """
        pass
    
    # Class-level tracking for mate score statistics
    _mate_score_min: float = float('inf')
    _mate_score_max: float = float('-inf')
    _mate_score_count: int = 0
    _mate_score_sum: float = 0.0
    MATE_SCORE_REPORT_INTERVAL: int = 100  # Print stats every N mate selections
    
    # Scoring weights (calibrated for ~3 expected contribution each at typical values)
    # Typical: energy=10, e=0.5, xeno=0.5, kin=0.5, genetic_dist_sq=3, trait_dist_sq=0.24, r=0.1
    MATE_WEIGHT_ENERGY: float = 1.0
    MATE_WEIGHT_EXTRAVERSION: float = 12.0    # 0.5*0.5*12 = 3.0
    MATE_WEIGHT_GENETIC: float = 2.0          # 3*0.5*2 = 3.0
    MATE_WEIGHT_TRAIT: float = 25.0           # 0.24*0.5*25 = 3.0
    MATE_WEIGHT_KIN_ALTRUISM: float = 60.0    # 0.1*0.5*60 = 3.0
    MATE_INCEST_DECAY: float = 10.0  # exp(-10*(0.5-0.15)) ≈ 0.03 (3%) for r=0.5
    
    def _pick_mate_from_list(self, searcher: Agent, candidates: List[Agent]) -> Optional[Agent]:
        """
        Pick a mate from a pre-filtered list using weighted preference scoring.
        
        Args:
            searcher: The female agent searching (uses her traits for preferences)
            candidates: Pre-filtered list of potential male suitors
        
        Returns:
            Best scoring mate, or None if no valid candidates
        
        Scoring system:
        1. If ALL suitors have kinship > 0.15, pick highest energy (bottleneck survival hack)
        2. Otherwise, compute weighted score:
           - Energy: raw suitor energy
           - Extraversion match: suitor.e weighted by female.e (high female.e favors extroverts)
           - Genetic similarity: negative squared genetic distance, weighted by -xeno (xenophilia)
           - Trait similarity: negative squared [o,c,a] distance, inversely weighted by female.e
           - Kin altruism: kinship bonus for 0 < r < 0.15, weighted by female.kin
           - Incest taboo: exponential score depression for r > 0.15
        """
        # Filter to valid candidates
        valid = [a for a in candidates if a.id != searcher.id and a.id in self.agents]
        if not valid:
            return None
        
        # Compute kinship values for all candidates
        def get_kinship(suitor: Agent) -> float:
            return searcher.kinship.get(suitor.id, 0.0)
        
        kinship_values = [get_kinship(s) for s in valid]
        min_kinship = min(kinship_values)
        
        # Early game / bottleneck hack: if ALL suitors are highly related,
        # just pick the one with highest energy to ensure population survival.
        # Only applies to adult females (age > 21) - younger females defer mating.
        if min_kinship > self.CONSANGUINITY_TOLERANCE:
            female_age = self._age_years(searcher)
            if female_age > self.ADULTHOOD:
                return max(valid, key=lambda s: s.energy)
            else:
                # Young female with only high-kinship suitors: defer mating
                return None
        
        # Otherwise, use full scoring system
        def compute_score(suitor: Agent) -> float:
            r = get_kinship(suitor)
            
            # 1. Energy component (later: males may boost via "persuasion" expenditure)
            score_energy = suitor.energy * self.MATE_WEIGHT_ENERGY
            
            # 2. Extraversion match: high female.e means she strongly favors extroverted males
            #    Score contribution: suitor.e * female.e (both normalized 0-1)
            score_extraversion = suitor.e * searcher.e * self.MATE_WEIGHT_EXTRAVERSION
            
            # 3. Genetic similarity: negative squared distance, weighted by xenophobia
            #    High xeno (xenophobic) → stronger penalty for genetic distance
            #    Low xeno (xenophilic) → tolerates or prefers genetic diversity
            genetic_dist_sq = sum((sg - fg) ** 2 for sg, fg in zip(suitor.genes, searcher.genes))
            # xeno=0 means xenophilic (no penalty), xeno=1 means xenophobic (full penalty)
            score_genetic = -genetic_dist_sq * searcher.xeno * self.MATE_WEIGHT_GENETIC
            
            # 4. Trait similarity [o,c,a]: negative squared distance
            #    Inversely weighted by female.e: low extraversion → cares about trait similarity
            trait_dist_sq = (
                (suitor.o - searcher.o) ** 2 +
                (suitor.c - searcher.c) ** 2 +
                (suitor.a - searcher.a) ** 2
            )
            # (1 - female.e) weight: introverts care more about similar personalities
            score_trait = -trait_dist_sq * (1.0 - searcher.e) * self.MATE_WEIGHT_TRAIT
            
            # 5. Kin altruism bonus for moderate relatedness (0 < r < 0.15)
            #    High kin altruism → boost for related but not incestuous mates
            if 0 < r < self.CONSANGUINITY_TOLERANCE:
                # Linear bonus scaled by kinship and kin altruism trait
                score_kin = r * searcher.kin * self.MATE_WEIGHT_KIN_ALTRUISM
            else:
                score_kin = 0.0
            
            # 6. Incest taboo: exponential depression for r > 0.15
            if r > self.CONSANGUINITY_TOLERANCE:
                incest_penalty = math.exp(-self.MATE_INCEST_DECAY * (r - self.CONSANGUINITY_TOLERANCE))
            else:
                incest_penalty = 1.0
            
            # Total score with incest penalty applied multiplicatively
            raw_score = score_energy + score_extraversion + score_genetic + score_trait + score_kin
            final_score = raw_score * incest_penalty
            
            return final_score
        
        # Score all candidates and pick best
        scored = [(s, compute_score(s)) for s in valid]
        best_suitor, best_score = max(scored, key=lambda x: x[1])
        
        # Track score statistics for calibration
        self._mate_score_count += 1
        self._mate_score_sum += best_score
        if best_score < self._mate_score_min:
            self._mate_score_min = best_score
        if best_score > self._mate_score_max:
            self._mate_score_max = best_score
        
        # Periodically report score statistics
        if self._mate_score_count % self.MATE_SCORE_REPORT_INTERVAL == 0:
            avg = self._mate_score_sum / self._mate_score_count
            print(f"[Mate Score Stats] n={self._mate_score_count} "
                  f"min={self._mate_score_min:.3f} max={self._mate_score_max:.3f} "
                  f"avg={avg:.3f} (current={best_score:.3f})")
        
        return best_suitor
    
    def _pick_suitor(self, female: Agent, suitors: List[Agent]) -> Optional[Agent]:
        """
        Female selects a mate from available suitors.
        
        Selection rules:
        1. Absolutely reject own sons (incest taboo - should be pre-filtered)
        2. If ANY suitor has kinship < CONSANGUINITY_TOLERANCE: reject all high-kinship suitors
        3. If ALL suitors have high kinship: forced to pick one (prevents extinction)
        4. Among acceptable suitors, pick by effective_energy (energy * incest_penalty)
        
        Incest penalty: exp(-10 * (r - threshold)) for r > threshold, else 1.0
        
        Args:
            female: The female agent choosing
            suitors: List of male agents applying
            
        Returns:
            Chosen male agent, or None if no acceptable suitors (only if all are sons)
        """
        # Filter out sons (absolute taboo)
        candidates = [m for m in suitors if m.id not in female.offspring]
        
        if not candidates:
            return None
        
        # Compute kinship for each candidate
        def get_kinship(male):
            return female.kinship.get(male.id, 0.0)
        
        # Check if any candidate has acceptable kinship
        has_acceptable = any(get_kinship(m) < self.CONSANGUINITY_TOLERANCE for m in candidates)
        
        if has_acceptable:
            # Filter to only acceptable candidates
            candidates = [m for m in candidates if get_kinship(m) < self.CONSANGUINITY_TOLERANCE]
            # Pick highest energy (no penalty needed, all are acceptable)
            return max(candidates, key=lambda m: m.energy)
        else:
            # All candidates are high-kinship - forced to pick one
            # Use effective energy with incest penalty
            def effective_energy(male):
                r = get_kinship(male)
                penalty = math.exp(-10 * (r - self.CONSANGUINITY_TOLERANCE))
                return male.energy * penalty
            
            return max(candidates, key=effective_energy)
    
    def _mate(self, initiator: Agent, partner: Agent):
        """
        Execute mating between two agents.
        Both must be fertile (caller should verify).
        Each agent can only mate once per tick.
        
        Spawns offspring immediately (at conception) but sets born_tick
        to current tick + GESTATION_TICKS. The offspring exists but has
        negative age until born.
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
        
        # Mark female as pregnant
        female.pregnant = True
        
        # Spawn at female position with future birth date
        spawn_x = female.x 
        spawn_y = female.y 
        # Clamp to world bounds
        spawn_x = max(0, min(self.width - 1, spawn_x))
        spawn_y = max(0, min(self.height - 1, spawn_y))
        
        offspring_energy = initial_e
        born_tick = self.tick + self.GESTATION_TICKS
        
        self.spawn_agent(spawn_x, spawn_y, 
                        offspring_energy, parent_ids=(male.id, female.id),
                        born_tick=born_tick)
    
    # =========================================================================
    # TERMINATION AND UTILITIES
    # =========================================================================
    
    def _log_stats(self):
        """Log simulation statistics and histograms to console."""
        count = self.count_agents()
        
        print(f"\n[Tick {self.tick}] Population:{count} | "
              f"Births:{self._stats['births']} Deaths:{self._stats['deaths']}")
        
        # Print histograms
        if self._histograms:
            for name in ['age', 'energy', 'kin']:
                if name in self._histograms:
                    hist = self._histograms[name]
                    max_val = hist[0]
                    counts = hist[1:]
                    total = sum(counts)
                    if total > 0:
                        # Format as simple bar using characters
                        bar_width = 40
                        max_count = max(counts) if counts else 1
                        bars = []
                        for c in counts:
                            bar_len = int(c / max_count * 5) if max_count > 0 else 0
                            bars.append('█' * bar_len + '░' * (5 - bar_len))
                        bin_width = max_val / len(counts)
                        print(f"  {name:6s} [0-{max_val:.1f}]: {' '.join(f'{c:3d}' for c in counts)}  |{'|'.join(bars)}|")
        
        # Run kinship diagnostic every 100 ticks
        if self.tick % 100 == 0:
            self._kinship_diagnostic()
    
    def _kinship_diagnostic(self):
        """
        Sanity check: verify kinship values for known relationships.
        
        For each agent with living parent(s), find siblings through parent's
        offspring list. Track mean and variance of kinship values for:
        - Full siblings (both parents identical)
        - Half-siblings (one parent identical)
        
        Expected values: full=0.50 (fixed), half≥0.25 (with consanguinity boost)
        """
        full_sib_r = []  # Kinship values for full sibling pairs
        half_sib_r = []  # Kinship values for half sibling pairs
        
        checked_pairs = set()  # Avoid counting pairs twice
        
        for agent in self.agents.values():
            # Need at least one living parent to find siblings
            p1_id, p2_id = agent.parent
            p1 = self.agents.get(p1_id) if p1_id >= 0 else None
            p2 = self.agents.get(p2_id) if p2_id >= 0 else None
            
            if not p1 and not p2:
                continue
            
            # Collect potential siblings from both parents' offspring
            potential_siblings = set()
            if p1:
                potential_siblings.update(p1.offspring)
            if p2:
                potential_siblings.update(p2.offspring)
            
            for sib_id in potential_siblings:
                if sib_id == agent.id:
                    continue
                if sib_id not in self.agents:
                    continue
                
                # Avoid counting same pair twice
                pair_key = (min(agent.id, sib_id), max(agent.id, sib_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                sib = self.agents[sib_id]
                sib_p1, sib_p2 = sib.parent
                
                # Check if full or half siblings
                agent_parents = {p1_id, p2_id}
                sib_parents = {sib_p1, sib_p2}
                shared_parents = agent_parents & sib_parents - {-1}
                
                r = agent.kinship.get(sib_id, 0)
                r_reverse = sib.kinship.get(agent.id, 0)
                
                if len(shared_parents) == 2:
                    if r < 0.9:  # Exclude identical twins (r=1.0)
                        full_sib_r.append(r)
                        if abs(r - 0.5) > 0.01:  # Not ~0.5
                            print(f"[SIB ANOMALY] {agent.id} & {sib_id}: r={r:.3f}, "
                                  f"agent.parent={agent.parent}, sib.parent={[sib_p1, sib_p2]}")
                        if abs(r - r_reverse) > 0.001:  # Asymmetric
                            print(f"[SIB ASYMMETRY] {agent.id}→{sib_id}: {r:.3f}, "
                                  f"{sib_id}→{agent.id}: {r_reverse:.3f}")
                elif len(shared_parents) == 1:
                    half_sib_r.append(r)
        
        # Calculate statistics
        def stats(values):
            if not values:
                return 0, 0, 0
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            else:
                variance = 0
            return n, mean, variance ** 0.5  # Return std dev instead of variance
        
        n_full, mean_full, std_full = stats(full_sib_r)
        n_half, mean_half, std_half = stats(half_sib_r)
        
        print(f"  Kinship check: full-sibs n={n_full} r={mean_full:.3f}±{std_full:.3f} (expect=0.50) | "
              f"half-sibs n={n_half} r={mean_half:.3f}±{std_half:.3f} (expect≥0.25)")
    
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
                'adulthood': cls.ADULTHOOD,
                'senescence': cls.SENESCENCE,
                'senescence_death_rate': cls.SENESCENCE_DEATH_RATE,
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
        'adulthood': 'ADULTHOOD',
        'senescence': 'SENESCENCE',
        'senescence_death_rate': 'SENESCENCE_DEATH_RATE',
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

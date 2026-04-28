import random
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import logging

# Import decision logic module (can be swapped for different strategies)
#import decide_optimized as decide_module
import decide_cython as decide_module

TWOPI = 2 * math.pi

@dataclass
class Agent:
    world_id: int
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
    hap: float = 0.5
    trust: float = 0.5
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
    # Social bonds
    spouse: int = -1  # ID of spouse (-1 = unmarried), set on first mating with unmarried partner
    # Cultural traits [0-10] - inherited from mother, updated via cooperation and spousal assimilation
    culture: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])

    @property
    def cell(self) -> Tuple[int, int]:
        """Current cell for food/interactions (floor of position)"""
        return (int(self.x), int(self.y))
       
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
        Note: world_id is added by World.get_entity_display(), not here.
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
            'born': self.born,
            'x': self.x,
            'y': self.y,
            'parent': self.parent,
            'spouse': self.spouse,
            'energy': round(self.energy, 2),
            'o': round(self.o, 2),
            'c': round(self.c, 2),
            'e': round(self.e, 2),
            'a': round(self.a, 2),
            'n': round(self.n, 2),
            'hap': round(self.hap, 2),
            'genes': [round(g, 2) for g in self.genes],
            'culture': [round(c, 2) for c in self.culture],
        }

    
    def to_full_dict(self) -> dict:
        """
        Complete state - only sent on inspect request.
        Includes phenotype + internal state + genetics.

        currently unused (and not sending diploid variations or full kinship dict, 'pregnant' boolean):
            'hap': round(self.hap, 2),
            'trust': round(self.trust, 2),
            'kinship': {k: round(v, 3) for k, v in self.kinship.items()},
            'vo': round(self.vo, 2),
            'vc': round(self.vc, 2),
            've': round(self.ve, 2),
            'va': round(self.va, 2),
            'vn': round(self.vn, 2),
            'vkin': round(self.vkin, 2),
            'vxeno': round(self.vxeno, 2),

        """
        return {
            'id': self.id,
            'sex': self.sex,
            'born': self.born,
            'x': self.x,
            'y': self.y,
            'parent': self.parent,
            'offspring': self.offspring,
            'spouse': self.spouse,
            'energy': round(self.energy, 2),
            'o': round(self.o, 2),
            'c': round(self.c, 2),
            'e': round(self.e, 2),
            'a': round(self.a, 2),
            'n': round(self.n, 2),
            'kin': round(self.kin, 2),
            'xeno': round(self.xeno, 2),
            'hap': round(self.hap, 2),
            'trust': round(self.trust, 2),
            'genes': [round(g, 2) for g in self.genes],
            'culture': [round(c, 2) for c in self.culture],
            'n_kin': len(self.kinship),
        }

@dataclass
class Map:
    world_id: int
    x: int
    y: int
    pd: List[int]
    food: float

    @property
    def position(self) -> tuple:
        """Map cell position for viewport filtering."""
        return (self.x, self.y)
    
    def to_display_dict(self) -> dict:
        """
        Minimal data for visualization - streamed continuously.
        """
        return {
            'map': 1,  # Marker so client knows this is a cell, not an agent
            'x': self.x,
            'y': self.y,
            'pd': list(self.pd),  # Copy to avoid mutation issues
        }

    def to_viewport_dict(self) -> dict:
        """
        Data for cells in viewport (same as display for now).
        """
        return {
            'map': 1,
            'x': self.x,
            'y': self.y,
            'pd': list(self.pd),
        }

    
    def to_full_dict(self) -> dict:
        return {
            'map': 1,
            'x': self.x,
            'y': self.y,
            'pd': list(self.pd),
            'food': round(self.food, 2),
        }



@dataclass
class NeighborhoodCache:
    """Pre-computed neighborhood data, built once per tick."""
    tick: int
    cell_fertile_females: Dict[Tuple[int,int], List[int]] = field(default_factory=lambda: defaultdict(list))
    cell_fertile_males: Dict[Tuple[int,int], List[int]] = field(default_factory=lambda: defaultdict(list))
    cell_adult_males: Dict[Tuple[int,int], List[int]] = field(default_factory=lambda: defaultdict(list))
    cell_adults: Dict[Tuple[int,int], List[int]] = field(default_factory=lambda: defaultdict(list))
    moore_fertile_females: Dict[Tuple[int,int], List[int]] = field(default_factory=dict)
    moore_fertile_males: Dict[Tuple[int,int], List[int]] = field(default_factory=dict)
    moore_adult_males: Dict[Tuple[int,int], List[int]] = field(default_factory=dict)
    fertile_female_ids: Set[int] = field(default_factory=set)


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
    FOOD_REGEN_PER_TURN = 3.0
    FOOD_CEILING = 3.0
    SEASON_STRENGTH = 0.7

    INITIAL_ENERGY = 0.8
    METABOLISM_COST = 0.06
    MAX_ENERGY = 10.0
    REPRODUCTION_THRESHOLD = 1.6
    REPRODUCTION_COST = 0.8
    MALE_INVESTMENT = 0.05
#    EAT_RATE = 1.0  # DEPRECATED - food distribution now handled via PD game
    INFANCY = 3
    CHILDHOOD = 7
    ADOLESCENCE = 14
    ADULTHOOD = 21
    MENOPAUSE = 44
    SENESCENCE = 60
    GESTATION_TICKS = int(0.75 / TICK_YEARS) # 9 months gestation
    
    # Accident/death parameters (children under ADOLESCENCE are exempt)
    ACCIDENT_BASE_RATE = 0.002      # Base per-tick accident rate for adult females (~0.8%/year)
    ACCIDENT_MALE_MULT = 1.5        # Males have 1.5x accident rate
    ACCIDENT_YOUNG_MALE_MULT = 2.0  # Young males (ADOLESCENCE-ADULTHOOD) have additional 2x multiplier
    ACCIDENT_OPENNESS_FACTOR = 0.002  # Additional rate per unit openness (max +0.002 at o=1.0)
    ACCIDENT_SENESCENCE_RATE = 0.005  # Additional per-tick rate = (age - SENESCENCE) * this
    
    # Twin probabilities
    P_ID_TWINS = 0.003  # Identical twins chance
    P_FR_TWINS = 0.014  # Fraternal twins (only if not id_twins, we don't do triplets)
    
    # Mutation rates for genetic parameters
    GENE_MUTATION_SD = 1.0 # range [0,10]
    TRAIT_MUTATION_SD = 0.05 # range [0,1]
    
    # Kinship tracking threshold (cousin-tier = 0.125, half-cousin = 0.0625)
    RELATEDNESS_THRESHOLD = 0.0625
    # Mate selection: incest taboo, strong suppression above this kinship (cousin = 0.125)
    CONSANGUINITY_TOLERANCE = 0.15
    
    # Culture assimilation rate (how much culture shifts per cooperative interaction or spousal tick)
    CULTURE_ASSIMILATION_RATE = 0.05
    # Female cultural malleability multiplier (females more responsive to cultural pressure)
    FEMALE_CULTURE_MULT = 2.0
    # Extraversion dampening (extroverts comfortable with diversity, don't need alignment)
    CULTURE_EXTRAVERSION_DAMP = 0.8  # high E (1.0) → 0.7x rate
    # Neuroticism culture shock threshold and dampening
    CULTURE_SHOCK_THRESHOLD = 2.5  # Euclidean distance where neuroticism resistance kicks in
    CULTURE_NEUROTICISM_DAMP = 0.5  # high N (1.0) past threshold → 0.5x rate
    # Conscientiousness spousal assimilation bonus
    CULTURE_SPOUSAL_C_BONUS = 0.5  # high C (1.0) in spousal → 1.5x rate
    # Maximum assimilation step per interaction (Euclidean distance)
    CULTURE_MAX_STEP = 0.05
    # Cultural tt: random exploration weighted by Openness
    CULTURE_DRIFT_RATE = 0.2  # baseline random drift per tick per dimension
    # Cultural repulsion: CD outcome pushes wronged agent away, weighted by Neuroticism
    #this was probably a stupid idea, disabled for now.
    CULTURE_REPULSION_RATE = 0.00  # Max repulsion step per CD interaction
    
    # Migration parameters
    P_MIGRATION = 0.1  # Probability of migration from overpopulated cell
    FEM_MIGRATION_RATIO = 0.3  # Female migration probability multiplier
    P_MATE_SEEKING_MIGRATION = 0.20  # Base probability for unmarried males seeking mates
    
    # Default board size and founder populations
    DEFAULT_WIDTH = 10
    DEFAULT_HEIGHT = 10
    
    # Statistics logging
    STATS_LOG_INTERVAL = 100  # Print stats every N ticks (0 to disable)
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
        
        # Population counters: [male_count, female_count]
        self._population = [0, 0]
        
        # Per-cell PD outcome tracking: [cc, cd, dd] counts per tick
        self.pd_games: Dict[Tuple[int, int], List[int]] = {}
        for x in range(width):
            for y in range(height):
                self.pd_games[(x, y)] = [0, 0, 0]
        
        # Dead agents storage for genealogical reconstruction
        # Flat array of integers: [id, born, death_tick, parent0, parent1, ...]
        # 5 integers per agent, optimized for memory in long simulations
        self.dead_agents: List[int] = []
        
        # Delta tracking: which entities have changed since last "mark_clean()"
        self._next_world_id = 0
        self._dirty_entities: Set[int] = set()      # world_ids of modified entities
        self._spawned_entities: Set[int] = set()    # world_ids of new entities
        self._despawn_notices: List[dict] = []      # death notices for removed entities
        self._by_world_id: Dict[int, Any] = {}      # world_id → entity lookup
        
        # Map cells - created once, never despawn
        self._cells: Dict[Tuple[int, int], Map] = {}
        for x in range(width):
            for y in range(height):
                cell_world_id = self._allocate_world_id()
                cell = Map(
                    world_id=cell_world_id,
                    x=x,
                    y=y,
                    pd=[0, 0, 0],
                    food=self.FOOD_CEILING
                )
                self._cells[(x, y)] = cell
                self._by_world_id[cell_world_id] = cell
                self._spawned_entities.add(cell_world_id)
        
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
            'deaths_starvation': 0,
            'deaths_accident': 0,
            'matings_spousal': 0,      # between existing spouses
            'matings_unm': 0,      # both unmarried (creates marriage)
            'matings_adulterous': 0,   # at least one already married to another
            'matings_incestuous': 0,   # kinship > 0.15
        }

        # Statistics histograms, set default maxima
        self._hist_max = {
            'age': 60,      # initial defaults
            'energy': self.MAX_ENERGY,
            'kin': 20,      # number of tracked kin relations
            # Personality traits (OCEAN) - all 0-1 range
            'o': 1.0,       # Openness
            'c': 1.0,       # Conscientiousness
            'e': 1.0,       # Extraversion
            'a': 1.0,       # Agreeableness
            'n': 1.0,       # Neuroticism
            # Other agent traits - all 0-1 range
            'kin_trait': 1.0,  # Kin altruism tendency
            'xeno': 1.0,    # Xenophilia/xenophobia
            'hap': 1.0,     # Happiness
            'trust': 1.0,   # Trust
            # PD game scores (dynamic max)
            'score': 30,    # Initial estimate, will auto-adjust
        }
        self._histograms = {}  # filled by _update_histograms
        self._hist_tick = 0
        self._tick_agent_scores = {}  # Per-tick PD scores, reset each step
        # How to report each property as histogram value
        self.hist_value_getters = {
            'age': lambda a: max(0, self._age_years(a)),  # Clamp negative (unborn) to 0
            'energy': lambda a: a.energy,
            'kin': lambda a: len(a.kinship),
            # Personality traits
            'o': lambda a: a.o,
            'c': lambda a: a.c,
            'e': lambda a: a.e,
            'a': lambda a: a.a,
            'n': lambda a: a.n,
            # Other traits
            'kin_trait': lambda a: a.kin,
            'xeno': lambda a: a.xeno,
            'hap': lambda a: a.hap,
            'trust': lambda a: a.trust,
            # PD score (0 if not played this tick)
            'score': lambda a: self._tick_agent_scores.get(a.id, 0),
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
    
    def _decide(self, own: Agent, opp: Agent, history: List, n_pre: int, kinship: float, distance: float, cultural_distance: float) -> int:
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
            cultural_distance: Squared Euclidean cultural distance
        
        Returns:
            1 for cooperate, 0 for defect
        """
        return decide_module.decide(own, opp, history, n_pre, kinship, distance)

    #optimized version
    def _decide_fast(self, own: Agent, opp: Agent, history: List, n_pre: int, kinship: float, distance: float, cultural_distance: float) -> int:

        flat = decide_module.agent_to_flat(own, history, n_pre)
        return 1 if random.random() < decide_module.compute_coop_prob_fast(*flat, kinship, distance) else 0

    def _decide_cython(self, own: Agent, opp: Agent, history: List, n_pre: int, kinship: float, distance: float, cultural_distance: float) -> int:

        own_history = history[0] if len(history) > 0 else []
        opp_history = history[1] if len(history) > 1 else []
        pre_history = history[2] if len(history) > 2 else []    
        own_h0 = own_history[0] if len(own_history) > 0 else -1
        own_h1 = own_history[1] if len(own_history) > 1 else -1
        own_h2 = own_history[2] if len(own_history) > 2 else -1
        opp_h0 = opp_history[0] if len(opp_history) > 0 else -1
        opp_h1 = opp_history[1] if len(opp_history) > 1 else -1
        opp_h2 = opp_history[2] if len(opp_history) > 2 else -1
        own_avg = pre_history[0] if len(pre_history) > 0 else -1.0
        opp_avg = pre_history[1] if len(pre_history) > 1 else -1.0
 
        #update to iclude own.sex, opp.sex !
        return decide_module.decide_cython(
            own.o, own.c, own.e, own.a, own.n,
            own.kin, own.xeno,
            own.hap, own.trust,
            own_h0, own_h1, own_h2,
            opp_h0, opp_h1, opp_h2,
            own_avg, opp_avg,
            n_pre,
            kinship, distance, cultural_distance,
            own.sex, opp.sex, 
            self._age_years(own), self._age_years(opp),
            random.random(),
            sigmoid_mode=0
        )
    
    def _genetic_distance(self, a1: Agent, a2: Agent) -> float:
        """Squared Euclidean distance between agents' genes arrays."""
        return sum((g1 - g2) ** 2 for g1, g2 in zip(a1.genes, a2.genes))

    def _cultural_distance(self, a1: Agent, a2: Agent) -> float:
        """Squared Euclidean cultural distance between agents' culture arrays."""
        return sum((g1 - g2) ** 2 for g1, g2 in zip(a1.culture, a2.culture))
    
    def _assimilate_culture(self, agent: Agent, other: Agent, rate_mult: float = 1.0):
        """
        Move agent's culture toward other's culture.
        
        Args:
            agent: The agent whose culture is being modified
            other: The agent whose culture is being assimilated toward
            rate_mult: External multiplier for assimilation rate
        
        Trait effects:
        - Agreeableness: increases rate (accommodating)
        - Female sex: 2x multiplier (cultural enforcers/transmitters)
        - Age: younger = more malleable; no assimilation toward children
        - Extraversion: decreases rate (comfortable with diversity)
        - Neuroticism: decreases rate past distance threshold (culture shock)
        - Conscientiousness: increases rate for spousal interaction only
        
        Constraints:
        - Culture values clamped to [0, 10]
        - Maximum step capped at CULTURE_MAX_STEP Euclidean distance
        """
        agent_age = self._age_years(agent)
        other_age = self._age_years(other)
        
        # No assimilation toward children - adults don't learn culture from kids
        if other_age < self.ADOLESCENCE:
            return
        
        # Calculate cultural distance for neuroticism threshold
        cultural_dist_sq = self._cultural_distance(agent, other)
        
        # Base rate
        rate = self.CULTURE_ASSIMILATION_RATE * rate_mult
        
        # Agreeableness: increases rate
        rate *= agent.a
        
        # Female multiplier
        if agent.sex == 1:
            rate *= self.FEMALE_CULTURE_MULT
        
        # Age-based malleability: full until ADULTHOOD, then linear decay to 0.3 at SENESCENCE
        if agent_age < self.ADULTHOOD:
            age_mult = 1.0
        elif agent_age < self.SENESCENCE:
            decay = 0.7 * (agent_age - self.ADULTHOOD) / (self.SENESCENCE - self.ADULTHOOD)
            age_mult = 1.0 - decay
        else:
            age_mult = 0.3  # Floor for elderly
        rate *= age_mult
        
        #Teenage subcultures:
        if agent_age < self.ADULTHOOD and other_age < self.ADULTHOOD:
            rate *= 2.0
        else:
            # Age difference: younger learns more from older (+2% per year, capped)
            age_diff = other_age - agent_age
            diff_mult = max(0.5, min(1.5, 1.0 + 0.02 * age_diff))
            rate *= diff_mult
        
        # Extraversion dampening: extroverts don't need cultural alignment
        rate *= (1.0 - agent.e * self.CULTURE_EXTRAVERSION_DAMP)
        
        # Neuroticism culture shock: resistance past threshold
        if cultural_dist_sq > self.CULTURE_SHOCK_THRESHOLD:
            rate *= (1.0 - agent.n * self.CULTURE_NEUROTICISM_DAMP)
        
        # Conscientiousness bonus for spousal assimilation only
        if agent.spouse == other.id:
            rate *= (1.0 + agent.c * self.CULTURE_SPOUSAL_C_BONUS)
        
        # Calculate proposed deltas
        deltas = [rate * (other.culture[i] - agent.culture[i]) for i in range(len(agent.culture))]
        
        # Cap step at CULTURE_MAX_STEP Euclidean distance
        step_dist = math.sqrt(sum(d * d for d in deltas))
        if step_dist > self.CULTURE_MAX_STEP:
            scale = self.CULTURE_MAX_STEP / step_dist
            deltas = [d * scale for d in deltas]
        
        # Apply deltas and clamp to [0, 10]
        for i in range(len(agent.culture)):
            agent.culture[i] = max(0.0, min(10.0, agent.culture[i] + deltas[i]))
    
    def _repel_culture(self, agent: Agent, betrayer: Agent):
        """
        Move agent's culture away from betrayer's culture after being wronged in CD outcome.
        
        Repulsion is weighted by agent's neuroticism (resentful, holds grudges).
        Only applies to adults (children don't develop cultural grudges).
        
        Args:
            agent: The agent who cooperated but was betrayed
            betrayer: The agent who defected
        """
        agent_age = self._age_years(agent)
        if agent_age < self.ADOLESCENCE:
            return

        # Repulsion rate scaled by neuroticism (disabled)
        rate = self.CULTURE_REPULSION_RATE * agent.n
        
        # Move away from betrayer (negative direction)
        for i in range(len(agent.culture)):
            delta = agent.culture[i] - betrayer.culture[i]  # Direction away
            # Normalize: if delta is 0, no movement; otherwise scale to rate
            if abs(delta) > 0.001:
                step = rate * (delta / abs(delta))  # ±rate based on direction
            else:
                # Cultures identical on this dimension - random direction
                step = rate * (1 if random.random() > 0.5 else -1)
            agent.culture[i] = max(0.0, min(10.0, agent.culture[i] + step))
    
    def _drift_culture(self, agent: Agent):
        """
        Apply random cultural drift, weighted by agent's openness.
        
        Open agents are more exploratory and creative, leading to cultural innovation.
        Only applies to adults (children's culture is shaped by others, not self-generated).
        """
        agent_age = self._age_years(agent)
        if agent_age < self.ADOLESCENCE:
            return
        
        # cultural drift triggered by openness
        if random.random() < agent.o:
            rate = self.CULTURE_DRIFT_RATE
            #Teenagers drive cultural innovation
            if agent_age < self.ADULTHOOD:
                rate *= 2.5
        
            for i in range(len(agent.culture)):
                drift = random.gauss(0, rate)
                agent.culture[i] = max(0.0, min(10.0, agent.culture[i] + drift))
    
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
        needs to become versatile (specify not only number of breeding pairs, but also positions, properties, ...)
        
        Returns:
            Initialized World with agents spawned and dirty state cleared
        """
        # Extract or use defaults
        width = kwargs.get('width', cls.DEFAULT_WIDTH)
        height = kwargs.get('height', cls.DEFAULT_HEIGHT)
        seed = kwargs.get('seed', None)
        initial_pairs = kwargs.get('initial_pairs', 1)

        # Create the world
        world = cls(width, height, seed=seed)

	#for now ignore initial_pairs and set defaults here        

        male = world.spawn_agent(1, 1, 2, sex=0)
        male = world.spawn_agent(1, 1, 2, sex=0)
        female = world.spawn_agent(1, 1, 2, sex=1)
        female = world.spawn_agent(1, 1, 2, sex=1)
        female = world.spawn_agent(1, 1, 2, sex=1)

        male = world.spawn_agent(8, 8, 2, sex=0)
        male = world.spawn_agent(8, 8, 2, sex=0)
        female = world.spawn_agent(8, 8, 2, sex=1)
        female = world.spawn_agent(8, 8, 2, sex=1)
        female = world.spawn_agent(8, 8, 2, sex=1)


        # Clear dirty state so first delta is clean
        world.mark_clean()
        
        return world
    
    def _allocate_id(self) -> int:
        """Get a unique agent ID. IDs are never reused (important for client sync)."""
        agent_id = self._next_id
        self._next_id += 1
        return agent_id

    def _allocate_world_id(self) -> int:
        world_id = self._next_world_id
        self._next_world_id +=1
        return world_id
    
    def spawn_agent(self, x: int, y: int, energy: float, 
                    parent_ids: Tuple[int, int] = (-1, -1),
                    born_tick: int = -20,
                    _twin_of: Agent = None,
                    sex: Optional[int] = None) -> Agent:
        """
        Create a new agent in the world.
        
        Args:
            x, y: Position (cell coordinates, integers)
            energy: Initial energy
            parent_ids: Tuple of parent IDs ((-1, -1) for founders)
            born_tick: Tick when agent is "born" (default: current tick)
                       Set to future tick for gestation (agent exists but age < 0)
            _twin_of: Internal - if set, this is a fraternal twin of the given agent
            sex: Optional sex (0=male, 1=female). Random if not specified.
        
        Returns:
            The created Agent (primary twin if twins spawned)
        
        Genetic inheritance (diploid for personality traits):
            - Founders get homozygous defaults (variance = 0)
            - Offspring inherit one allele from each parent (randomly chosen)
            - Mutation applied to each inherited allele
            - Phenotype = mean of two alleles
            - Speciation genes remain haploid (simple averaging)

        Sex is assigned randomly (50/50) unless specified.
        
        Twins:
            - P_ID_TWINS chance of identical twins (same genetics, kinship=1.0)
            - P_FR_TWINS chance of fraternal twins (independent genetics, kinship=0.5)
        """
        if born_tick is None:
            born_tick = self.tick
        agent_id = self._allocate_id()
        world_id = self._allocate_world_id()
        
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
            
            # Culture: cloned from mother (parent2)
            culture = list(parent2.culture)

        else:
            # Founder defaults (homozygous: variance = 0)
            genes = [5.0 + 3 * (1 if x < 5 else -1), 
                     5.0 + 3 * (1 if y < 5 else -1), 
                     5.0]
            o = c = e = a = n = 0.5
            vo = vc = ve = va = vn = 0.1
            kin_trait = 0.6
            xeno = 0.6
            vkin = vxeno = 0.05
            # Founder culture: same pattern as genes
            culture = [5.0 + 3 * (1 if x < 5 else -1), 
                       5.0 + 3 * (1 if y < 5 else -1), 
                       5.0]

        if sex is None:
            sex = random.randint(0, 1)  # 0 = male, 1 = female

        agent = Agent(
            world_id=world_id,
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
            culture=culture,
        )
        self.agents[agent_id] = agent
        self._by_world_id[agent.world_id] = agent
        self.spatial_grid.add(agent_id, x, y)
        self._spawned_entities.add(world_id)
        self._dirty_entities.add(world_id)
        self._population[sex] += 1
        
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
                twin_world_id = self._allocate_world_id()
                twin = Agent(
                    world_id=twin_world_id,
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
                    culture=list(culture),  # Copy the list
                )
                self.agents[twin_id] = twin
                self._by_world_id[twin.world_id] = twin
                self.spatial_grid.add(twin_id, x, y)
                self._spawned_entities.add(twin_world_id)
                self._dirty_entities.add(twin_world_id)
                self._population[sex] += 1  # Same sex as primary twin
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
            world_id = agent.world_id
            
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
            
            # Happiness hit for death of spouse
            if agent.spouse != -1 and agent.spouse in self.agents:
                spouse = self.agents[agent.spouse]
                # Check interaction history to see how much surviving spouse cooperated
                # Normalize key so lower id is first
                if spouse.id < agent_id:
                    key = (spouse.id, agent_id)
                    spouse_idx = 0  # spouse's actions are in recent1
                else:
                    key = (agent_id, spouse.id)
                    spouse_idx = 1  # spouse's actions are in recent2
                
                coop_count = 0
                if key in self._interaction_history:
                    entry = self._interaction_history[key]
                    recent = entry[spouse_idx]  # recent actions list (last 3)
                    coop_count = sum(recent)  # count cooperations in recent history only
                
                # Happiness loss: baseline 0.05 per cooperation (0-3), times (1+n)
                # Depression (trust < 0.5) amplifies loss
                if coop_count > 0:
                    trust_mult = (1.5 - spouse.trust) if spouse.trust < 0.5 else 1.0
                    hap_loss = 0.2 * coop_count * (1 + spouse.n) * trust_mult
                    spouse.hap = max(0.0, min(1.0, spouse.hap - hap_loss))
                
                spouse.spouse = -1
            
            # Happiness hit for death of child (to mother)
            # Only if child age < ADOLESCENCE
            child_age = self._age_years(agent)
            if child_age < self.ADOLESCENCE:
                # Find mother (parent[1] is mother by convention)
                mother_id = agent.parent[1]
                if mother_id != -1 and mother_id in self.agents:
                    mother = self.agents[mother_id]
                    # Happiness loss: 0.1 * (1 + n), amplified by depression
                    trust_mult = (1.5 - mother.trust) if mother.trust < 0.5 else 1.0
                    hap_loss = 0.2 * (1 + mother.n) * trust_mult
                    mother.hap = max(0.0, min(1.0, mother.hap - hap_loss))
                
            # Remove from relatives' kinship dicts
            for relative_id in agent.kinship:
                if relative_id in self.agents:
                    self.agents[relative_id].kinship.pop(agent_id, None)
            # Clean up interaction history
            self._cleanup_interaction_history(agent_id)
            # Remove from spatial grid and tracking
            self.spatial_grid.remove(agent_id, agent.x, agent.y)
            # Record in dead_agents for genealogical reconstruction
            self.dead_agents.extend([
                agent.id,
                agent.born,
                self.tick,
                agent.parent[0],
                agent.parent[1]
            ])
            self._population[agent.sex] -= 1
            del self.agents[agent_id]
            del self._by_world_id[agent.world_id]
            self._dirty_entities.discard(agent.world_id)
            self._spawned_entities.discard(agent.world_id)
            notice = {
                'world_id': agent.world_id,
                'id': agent.id,
                'type': 'agent',
                'dead': True
            }
            self._despawn_notices.append(notice)
            if (agent_id % 10000 == 0):
                print(f"{agent_id} {self._family_tree(agent_id, 5)}")
    
    def _family_tree(self, target_id: int, n: int) -> list:
        """
        Reconstruct family tree for a dead agent up to n generations.
        
        Traverses dead_agents to find all relatives within n relationship hops
        and builds a nested tree structure for genealogical analysis.
        
        Args:
            target_id: ID of the agent to build tree for
            n: Maximum generational distance (relationship hops)
               i=1: parents and children
               i=2: grandparents, siblings, grandchildren
               i=3: great-grandparents, uncles, nephews, etc.
        
        Returns:
            [] if n==0 or agent not found in dead_agents
            [ancestry, posterity] where:
                ancestry = [father, father_tree, mother, mother_tree]
                posterity = [child1, child1_tree, child2, child2_tree, ...]
            Each *_tree follows the same [ancestry, posterity] structure.
            -1 sentinel replaces tree when agent already appears elsewhere
            (pedigree collapse).
        """
        if n == 0:
            return []
        
        # Build indices from flat dead_agents array
        # Format: [id, born, death_tick, parent0, parent1, ...] (5 ints per agent)
        agents = {}  # id -> (born, death, parent0, parent1)
        children = defaultdict(list)  # parent_id -> [child_ids]
        
        for i in range(0, len(self.dead_agents), 5):
            aid = self.dead_agents[i]
            born = self.dead_agents[i + 1]
            death = self.dead_agents[i + 2]
            p0 = self.dead_agents[i + 3]
            p1 = self.dead_agents[i + 4]
            agents[aid] = (born, death, p0, p1)
            if p0 != -1:
                children[p0].append(aid)
            if p1 != -1:
                children[p1].append(aid)
        
        if target_id not in agents:
            return []
        
        # BFS to find all agents within n relationship hops
        distances = {target_id: 0}
        queue = [target_id]
        head = 0
        while head < len(queue):
            aid = queue[head]
            head += 1
            d = distances[aid]
            if d >= n:
                continue
            
            entry = agents.get(aid)
            if entry is None:
                continue
            
            p0, p1 = entry[2], entry[3]
            for p in (p0, p1):
                if p != -1 and p not in distances:
                    distances[p] = d + 1
                    queue.append(p)
            for c in children.get(aid, []):
                if c not in distances:
                    distances[c] = d + 1
                    queue.append(c)
        
        # Track seen agents for pedigree collapse detection
        seen = set()
        
        def build_tree(aid):
            """Build [ancestry, posterity] for agent aid."""
            if aid not in distances:
                return []
            
            if aid in seen:
                return -1  # Pedigree collapse
            seen.add(aid)
            
            entry = agents.get(aid)
            if entry is None:
                return []
            
            p0, p1 = entry[2], entry[3]
            
            # Ancestry: [father, father_tree, mother, mother_tree]
            ancestry = []
            for p in (p0, p1):
                if p != -1 and p in distances:
                    ancestry.extend([p, build_tree(p)])
            
            # Posterity: [child1, child1_tree, ...]
            posterity = []
            for c in children.get(aid, []):
                if c in distances:
                    posterity.extend([c, build_tree(c)])
            
            if not ancestry and not posterity:
                return []
            
            return [ancestry, posterity]
        
        return build_tree(target_id)
    
    def _mark_dirty(self, agent_id: int):
        """Mark an agent as modified (for delta updates)."""
        self._dirty_entities.add(self.agents[agent_id].world_id)
    
    def get_dirty_state(self) -> dict:
        """
        Get all changes since last mark_clean().
        This is what we'll send to clients as delta updates.
        
        Uses to_display_dict() for phenotype-only data (efficient streaming).
        Full agent data is available via inspect command.
        """
        return {
            'tick': self.tick,
            'spawned': [self.agents[aid].to_display_dict() for aid in self._spawned_entities if aid in self.agents],
            'updated': [self.agents[aid].to_display_dict() for aid in self._dirty_entities - self._spawned_entities if aid in self.agents],
            'despawned': list(self._despawned_entities)
        }
    
    def get_dirty_ids(self) -> dict:
        """
        Get just the IDs of changed agents (no data copying).
        Much more efficient for streaming - actual data fetched lazily at send time.
        """
        return {
            'tick': self.tick,
            'spawned_ids': list(self._spawned_entities),
            'updated_ids': list(self._dirty_entities - self._spawned_entities),
            'despawn_notices': list(self._despawn_notices),
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
        
        # Map cells as entities
        cell_entities = [cell.to_display_dict() for cell in self._cells.values()]
        
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
        self._dirty_entities.clear()
        self._spawned_entities.clear()
        self._despawn_notices.clear()
    
    def inspect_entity(self, world_id: int) -> Optional[dict]:
        """
        Get full state of a specific entity (for inspect command).
        Returns None if entity doesn't exist.
        """
        entity = self._by_world_id.get(world_id)
        if entity is not None:
            return entity.to_full_dict()
        return None

    def inspect_by_client_id(self, client_id: str) -> Optional[dict]:
        """
        Inspect entity by client-facing identifier.
        
        Agents: numeric ID (e.g., "42")
        Cells: "cell_x_y" format (e.g., "cell_5_10")
        
        Args:
            client_id: String identifier from client
        
        Returns:
            Full dict for the entity, or None if not found
        """

        # for future use: for "inspect cell" use format: "cell_x_y"
        # if client_id.startswith("cell_"): # fails when client_id is interpreted as int
        #    try:
        #        parts = client_id[5:].split("_")
        #        x, y = int(parts[0]), int(parts[1])
        #        cell = self._cells.get((x, y))
        #        if cell:
        #            return cell.to_full_dict()
        #    except (ValueError, IndexError):
        #        pass
        #    return None
        
        # Otherwise assume it's an agent ID
        try:
            agent_id = int(client_id)
        except ValueError:
            return None
        
        agent = self.agents.get(agent_id)
        if agent:
            return agent.to_full_dict()
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
                'entities': list of entity dicts (agents + cells),
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
        for cell in self._cells.values():
            if self.is_in_viewport(cell.position, center, radius):
                entities.append(cell.to_viewport_dict())
        
        return {
            'tick': self.tick,
            'center': list(center),
            'radius': radius,
            'entities': entities,
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

        self._stepping = True
        try:
            self._step_inner()
        finally:
            self._stepping = False
            self._process_deferred_commands()

    def _step_inner(self):
        """
        Internal step logic with neighborhood cache and collect-then-execute pattern.

        PHASE 1: Intra-cell interactions (metabolism, PD, food)
        PHASE 2: Build neighborhood cache

        PHASE 2a: Mating (males apply using cache, females choose)
        PHASE 2b: Collect inter-cell intentions (migration, trade, raid)
        PHASE 3: Execute inter-cell intentions
        PHASE 4: Death check and histograms
        """

        # Bookkeeping
        self.tick += 1
        for key in self._stats:
            self._stats[key] = 0
        
        # Reset PD game counters for this tick
        # First, copy last turn's PD data to Map cells and mark dirty if changed
        for pos, counts in self.pd_games.items():
            cell = self._cells[pos]
            if cell.pd != counts:
                cell.pd = counts.copy()
                self._dirty_entities.add(cell.world_id)
        
        for pos in self.pd_games:
            self.pd_games[pos] = [0, 0, 0]

        # Reset per-tick score tracking for histogram
        self._tick_agent_scores = {}

        # Clear pregnancy flags for females whose children are now born
        for agent in self.agents.values():
            if agent.pregnant and agent.sex == 1:
                all_born = all(
                    self.agents.get(child_id) is None or self.agents[child_id].born <= self.tick
                    for child_id in agent.offspring
                )
                if all_born:
                    agent.pregnant = False
                    #childbirth mortality? (but how will we handle the orphan?)
                    # (can't change dictionary size during iteration)

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
                if season_effect > 1 and random.random()<0.02:
                    season_effect = 0 #failed harvest (crisis)
		
                current_ceiling = self.FOOD_CEILING * season_effect
                current_regen = self.FOOD_REGEN_PER_TURN * season_effect

            self.food[pos] = min(current_ceiling, self.food[pos] + current_regen)


        # =========================================================================
        # PHASE 1: Intra-cell interactions (metabolism, PD, food distribution)
        # =========================================================================
        cells_with_agents = set()
        for agent in self.agents.values():
            cells_with_agents.add(agent.cell)

        for cell in cells_with_agents:
            self._process_cell_interactions(cell)

        # =========================================================================
        # PHASE 2: preparation for neighboring-cell mechanics: Build neighborhood cache
        # =========================================================================
        cache = self._build_neighborhood_cache()

        # =========================================================================
        # PHASE 2a: Mating (using cache for suitor collection)
        # =========================================================================

        # Build cell->agents map for migration (need current state post-feeding)
        cell_agents = defaultdict(list)
        for agent in self.agents.values():
            cell_agents[agent.cell].append(agent)

        # Phase 2a: Males apply to females using pre-computed neighborhoods
        suitors: Dict[int, List] = defaultdict(list)  # female_id -> [male agents]

        for cell in cells_with_agents:
            # Get fertile males in this cell
            for male_id in cache.cell_fertile_males.get(cell, []):
                if male_id not in self.agents:
                    continue
                male = self.agents[male_id]

                # Look at fertile females in Moore neighborhood (pre-computed)
                for female_id in cache.moore_fertile_females.get(cell, []):
                    # Incest taboo: male cannot apply to own mother
                    if female_id in male.parent:
                        continue
                    suitors[female_id].append(male)

        # Phase 2b: Females choose among suitors
        for female_id in cache.fertile_female_ids:
            if female_id not in self.agents:
                continue
            if female_id not in suitors or not suitors[female_id]:
                continue

            female = self.agents[female_id]

            # Filter to males still fertile (energy may have dropped from earlier matings)
            male_threshold = self.REPRODUCTION_THRESHOLD + self.REPRODUCTION_COST * self.MALE_INVESTMENT
            available_suitors = [m for m in suitors[female_id] if m.id in self.agents and m.energy >= male_threshold]
            if not available_suitors:
                continue

            # Female chooses
            chosen_male = self._pick_suitor(female, available_suitors)
            if chosen_male:
                self._mate(chosen_male, female)

        # =========================================================================
        # PHASE 2b: Collect inter-cell intentions (trade/raid not implemented)
        # =========================================================================
        migration_intentions = self._collect_migration_intentions(cache, cell_agents)
        trade_intentions = self._collect_trade_intentions(cache)
        raid_intentions = self._collect_raid_intentions(cache)

        # =========================================================================
        # PHASE 3: Execute inter-cell intentions
        # =========================================================================
        self._execute_migration_intentions(migration_intentions, cell_agents)
        self._execute_trade_intentions(trade_intentions)
        self._execute_raid_intentions(raid_intentions)

        # =========================================================================
        # PHASE 4: Death check and histogram collection
        # =========================================================================
        for agent in list(self.agents.values()):
            # Death check: starvation
            if agent.energy <= 0:
                self.remove_agent(agent.id)
                self._stats['deaths'] += 1
                self._stats['deaths_starvation'] += 1
                continue

            # Death check: accidents (includes senescence)
            # Children under ADOLESCENCE are exempt from accidents
            age = self._age_years(agent)
            if age >= self.ADOLESCENCE:
                # Base accident rate
                accident_prob = self.ACCIDENT_BASE_RATE
                
                # Male multiplier
                if agent.sex == 0:  # Male
                    accident_prob *= self.ACCIDENT_MALE_MULT
                    # Young male additional multiplier
                    if age < self.ADULTHOOD:
                        accident_prob *= self.ACCIDENT_YOUNG_MALE_MULT
                
                # Openness factor (additive)
                accident_prob += self.ACCIDENT_OPENNESS_FACTOR * agent.o
                
                # Senescence factor (additive, increases with age past SENESCENCE)
                if age > self.SENESCENCE:
                    accident_prob += (age - self.SENESCENCE) * self.ACCIDENT_SENESCENCE_RATE
                
                if random.random() < accident_prob:
                    self.remove_agent(agent.id)
                    self._stats['deaths'] += 1
                    self._stats['deaths_accident'] += 1
                    continue

            # Infants (age < INFANCY) copy emotional state from mother
            # This builds trust/happiness during formative years
            if age >= 0 and age < self.INFANCY:
                mother_id = agent.parent[1]
                if mother_id != -1 and mother_id in self.agents:
                    mother = self.agents[mother_id]
                    agent.hap = mother.hap
                    agent.trust = 0.8*mother.hap + 0.2*mother.trust

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

        # Update histogram maxima
        self._hist_max.update({k: max(v, 0.001) for k, v in hist_new_max.items() if self._hist_max[k] != -1})
        new_histograms = {k: [self._hist_max[k]] + counts for k, counts in hist_counts.items()}
        self._histograms = new_histograms
        self._hist_tick = self.tick

        # =========================================================================
        # PHASE 5: Spousal culture assimilation and cultural drift
        # =========================================================================
        for agent in self.agents.values():
            if agent.spouse != -1 and agent.spouse in self.agents:
                spouse = self.agents[agent.spouse]
                # Bidirectional assimilation handled by iterating all agents
                # Each agent assimilates toward their spouse once per tick
                self._assimilate_culture(agent, spouse)
            
            # Cultural drift: random exploration weighted by openness
            self._drift_culture(agent)

        # Sync food levels to Map cells (for inspect calls)
        # Note: food isn't in display_dict, so no need to mark cells dirty
        for pos, food_level in self.food.items():
            self._cells[pos].food = food_level

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
        # (we do this again under "PHASE 4" for agents who starved during turn(?))
        # =====================================================
        starved = [a for a in agents_in_cell if a.energy <= 0]
        for agent in starved:
            self.remove_agent(agent.id)
            self._stats['deaths'] += 1
            self._stats['deaths_starvation'] += 1
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
                cultural_distance = self._cultural_distance(own, opp)
                
                # Both decide simultaneously
                #optimization: pick _decide, _decide_fast or _decide_cython depending on imported "decide" module
                own_action = self._decide_cython(own, opp, history_own, n_pre, kinship, distance, cultural_distance)
                opp_action = self._decide_cython(opp, own, history_opp, n_pre, kinship, distance, cultural_distance)
                
                # Record interaction
                self._record_interaction(own.id, opp.id, own_action, opp_action)
                
                # Compute payoffs: CC=3,3  CD=5,0  DC=0,5  DD=1,1
                if own_action == 1 and opp_action == 1:
                    own_score, opp_score = 3, 3
                    self.pd_games[(x, y)][0] += 1  # CC
                    # Culture assimilation on mutual cooperation
                    # Each agent moves toward the other's culture, weighted by agreeableness
                    self._assimilate_culture(own, opp)
                    self._assimilate_culture(opp, own)
                elif own_action == 1 and opp_action == 0:
                    own_score, opp_score = 0, 5
                    self.pd_games[(x, y)][1] += 1  # CD
                    # Own cooperated but was betrayed - cultural repulsion
                    #self._repel_culture(own, opp)
                elif own_action == 0 and opp_action == 1:
                    own_score, opp_score = 5, 0
                    self.pd_games[(x, y)][1] += 1  # CD
                    # Opp cooperated but was betrayed - cultural repulsion
                    #self._repel_culture(opp, own)
                else:  # Both defect
                    own_score, opp_score = 1, 1
                    self.pd_games[(x, y)][2] += 1  # DD
                
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
            
            # Distribute proportionally to scores, update emotional state
            total_distributed = 0
            n_opponents = len(adults) - 1  # Each adult played against all others
            for agent in adults:
                score = agent_scores[agent.id]
                
                # Store score for histogram collection
                self._tick_agent_scores[agent.id] = score
                
                # Update emotional state based on PD outcomes
                self._update_emotional_state(agent, score, n_opponents)
                
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
        #self.food[(x, y)] = food_here
        #or: uneaten food is wasted
        self.food[(x, y)] = 0        
    
    def _update_emotional_state(self, agent: Agent, score: int, n_games: int):
        """
        Update agent's happiness (hap) and trust based on this tick's outcomes.
        
        Happiness is short-term satisfaction, more malleable.
        Trust is long-term "world trust", becomes less malleable with age.
        
        Args:
            agent: The agent to update
            score: Total PD score this tick
            n_games: Number of PD games played (opponents faced)
        """
        if n_games == 0:
            return
        
        avg_score = score / n_games
        
        # Energy factor: negative for hungry agents
        # Hunger baseline at 2.0 
        energy_factor = 0 if agent.energy > 2.0 else  (agent.energy - 2.0) / 2.0
        
        # Score factor: how well did PD games go?
        # Neutral baseline at 2.0 (between mutual defection=1 and cooperation=3)
        # CC=3 -> +0.25, DD=1 -> -0.25, random~2.25 -> +0.06
        score_factor = (avg_score - 2.25) / 4.0
        
        # Neuroticism: asymmetric penalty for negative outcomes
        # Neurotic agents feel bad outcomes more strongly (up to 1.5x)
        if energy_factor < 0:
            energy_factor *= (1 + 0.5 * agent.n)
        if score_factor < 0:
            score_factor *= (1 + 0.5 * agent.n)
        
        # Trust affects interpretation, but weakly to avoid feedback loops
        # High trust slightly buffers negative outcomes
        # Low trust slightly buffers positive outcomes
        trust_deviation = agent.trust - 0.5
        if trust_deviation > 0 and score_factor < 0:
            score_factor *= (1 - trust_deviation * 0.1)  # High trust dampens bad by up to 15%
        elif trust_deviation < 0 and score_factor > 0:
            score_factor *= (1 + trust_deviation * 0.1)  # Low trust dampens good by up to 15%
        
        # Update happiness: combine factors, scale to small delta
        hap_delta = (energy_factor + score_factor) * 0.04
        agent.hap = max(0.0, min(1.0, agent.hap + hap_delta))
	#regression to the mean
        if (agent.hap < 0.45):
            agent.hap += 0.01
        elif (agent.hap > 0.65):
            agent.hap -= 0.01*(0.5+agent.n)
	
        # Update trust: moves toward happiness, but slowly (scale factor in trust_rate!)
        # Rate depressed by age and conscientiousness
        # Young (age 7) low-C agent: rate ~ 0.5
        # Old (age 40) high-C agent: rate ~ 0.04
        age = max(7.0, self._age_years(agent))  # Children don't update trust much
        trust_rate =  min(1, 7 /age / (1 + agent.c)) * 0.05
        trust_delta = (agent.hap - agent.trust) * trust_rate 
        agent.trust = max(0.0, min(1.0, agent.trust + trust_delta))
    
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

    # Mate choice logic parameters
    # Scoring weights (calibrated for ~3 expected contribution each at typical values)
    # Typical: energy=10, e=0.5, xeno=0.5, kin=0.5, genetic_dist_sq=3, trait_dist_sq=0.24, r=0.1
    MATE_WEIGHT_ENERGY: float = 1.0
    MATE_WEIGHT_EXTRAVERSION: float = 1.0    # contributes to score: 0.5*0.5*n
    MATE_WEIGHT_GENETIC: float = 0.5          # ? 3*0.5*n  
    MATE_WEIGHT_TRAIT: float = 20.0           # 0.24*0.5*n  
    MATE_WEIGHT_KIN_ALTRUISM: float = 30.0    # 0.1*0.5*n  
    MATE_WEIGHT_SPOUSE: float = 15.0          # Spouse bonus weighted by conscientiousness
    MATE_INCEST_DECAY: float = 10.0  # exp(-10*(0.5-0.15)) ≈ 0.03 (3%) for r=0.5
    
    def _pick_mate_from_list(self, bride: Agent, candidates: List[Agent]) -> Optional[Agent]:
        """
        Pick a mate from a pre-filtered list using weighted preference scoring.
        
        Args:
            bride: The female agent searching (uses her traits for preferences)
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
        valid = [a for a in candidates if a.id != bride.id and a.id in self.agents]
        if not valid:
            return None
        
        # Compute kinship values for all candidates
        def get_kinship(suitor: Agent) -> float:
            return bride.kinship.get(suitor.id, 0.0)
        
        kinship_values = [get_kinship(s) for s in valid]
        min_kinship = min(kinship_values)
        
        # Early game / bottleneck hack: if ALL suitors are highly related,
        # just pick the one with highest energy to ensure population survival.
        # Only applies to adult females (age > 21) - younger females defer mating.
        if min_kinship > self.CONSANGUINITY_TOLERANCE:
            female_age = self._age_years(bride)
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
            #    Score contribution: (suitor.e-.5) * female.e (both normalized 0-1)
            # this may lead to runaway selection for high extraversion, tune with care
            score_extraversion = (bride.e if bride.e>0.5 else 0) * (suitor.e-0.5) *  self.MATE_WEIGHT_EXTRAVERSION
            
            # 3. Genetic similarity: negative squared distance, weighted by xenophobia
            #    High xeno (xenophobic) → stronger penalty for genetic distance
            #    Low xeno (xenophilic) → tolerates or prefers genetic diversity
            genetic_dist_sq = sum((sg - fg) ** 2 for sg, fg in zip(suitor.genes, bride.genes))
            # xeno=0 means xenophilic (bonus), xeno=0.25 neutral, xeno=1 means xenophobic (full penalty)
            score_genetic = -genetic_dist_sq * (bride.xeno-0.25) * self.MATE_WEIGHT_GENETIC
            
            # 4. Trait similarity [o,c,a]: negative squared distance
            #    Inversely weighted by female.e: low extraversion → cares about trait similarity
            trait_dist_sq = (
                (suitor.o - bride.o) ** 2 +
                (suitor.c - bride.c) ** 2 +
                (suitor.a - bride.a) ** 2
            )
            # (1 - female.e) weight: introverts care more about similar personalities
            score_trait = -trait_dist_sq * (1.0 - bride.e) * self.MATE_WEIGHT_TRAIT
            
            # 5. Kin altruism bonus for moderate relatedness (0 < r < 0.15)
            #    High kin altruism → boost for related but not incestuous mates
            if 0 < r < self.CONSANGUINITY_TOLERANCE:
                # Linear bonus scaled by kinship and kin altruism trait
                score_kin = r * bride.kin * self.MATE_WEIGHT_KIN_ALTRUISM
            else:
                score_kin = 0.0
            
            # 6. Spouse bonus: if already married to this suitor, weighted by conscientiousness
            if bride.spouse == suitor.id:
                score_spouse = bride.c * self.MATE_WEIGHT_SPOUSE
            # already has husband
            elif bride.spouse > 0:
                score_spouse = -bride.c * self.MATE_WEIGHT_SPOUSE
            #suitor is married
            elif suitor.spouse > 0:
                score_spouse = -bride.c * self.MATE_WEIGHT_SPOUSE/2
            else:
                score_spouse = 0.0
            
            # 7. Incest taboo: exponential depression for r > 0.15
            if r > self.CONSANGUINITY_TOLERANCE:
                incest_penalty = math.exp(-self.MATE_INCEST_DECAY * (r - self.CONSANGUINITY_TOLERANCE))
            else:
                incest_penalty = 1.0
            
            # Total score with incest penalty applied multiplicatively
            raw_score = score_energy + score_extraversion + score_genetic + score_trait + score_kin + score_spouse
            final_score = raw_score * incest_penalty
            
            return final_score
        
        # Score all candidates and pick best
        scored = [(s, compute_score(s)) for s in valid]
        best_suitor, best_score = max(scored, key=lambda x: x[1])
                
        return best_suitor
    
    def _pick_suitor(self, female: Agent, suitors: List[Agent]) -> Optional[Agent]:
        """
        Female selects a mate from available suitors using weighted preference scoring.
        
        Selection rules:
        1. Absolutely reject own sons (incest taboo)
        2. If ANY suitor has kinship < CONSANGUINITY_TOLERANCE: reject all high-kinship suitors
           (unless female is past ADULTHOOD, then forced emergency incest is allowed)
        3. Score acceptable suitors using multi-factor preference system
        
        Scoring factors:
        - Energy (resource provision)
        - Extraversion match (extroverted females prefer extroverted males)
        - Genetic similarity (weighted by xenophobia trait)
        - Trait similarity (weighted inversely by extraversion)
        - Kin altruism bonus for moderate relatedness
        - Spouse loyalty bonus/adultery penalty (weighted by conscientiousness)
        - Incest penalty for high kinship
        
        Args:
            female: The female agent choosing
            suitors: List of male agents applying
            
        Returns:
            Chosen male agent, or None if no acceptable suitors
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
        
        female_age = self._age_years(female)
        
        if has_acceptable:
            # Filter to only acceptable candidates
            candidates = [m for m in candidates if get_kinship(m) < self.CONSANGUINITY_TOLERANCE]
        elif female_age <= self.ADULTHOOD:
            # Young female with only high-kinship suitors: defer mating
            return None
        # else: older female forced to accept high-kinship candidates (emergency incest)
        
        # Score each candidate
        def compute_score(suitor: Agent) -> float:
            r = get_kinship(suitor)
            
            # 1. Energy component
            score_energy = suitor.energy * self.MATE_WEIGHT_ENERGY
            
            # 2. Extraversion match: high female.e means she favors extroverted males
            score_extraversion = suitor.e * female.e * self.MATE_WEIGHT_EXTRAVERSION
            
            # 3. Genetic similarity: negative squared distance, weighted by xenophobia
            genetic_dist_sq = sum((sg - fg) ** 2 for sg, fg in zip(suitor.genes, female.genes))
            score_genetic = -genetic_dist_sq * female.xeno * self.MATE_WEIGHT_GENETIC
            
            # 4. Trait similarity [o,c,a]: negative squared distance
            trait_dist_sq = (
                (suitor.o - female.o) ** 2 +
                (suitor.c - female.c) ** 2 +
                (suitor.a - female.a) ** 2
            )
            score_trait = -trait_dist_sq * (1.0 - female.e) * self.MATE_WEIGHT_TRAIT
            
            # 5. Kin altruism bonus for moderate relatedness (0 < r < threshold)
            if 0 < r < self.CONSANGUINITY_TOLERANCE:
                score_kin = r * female.kin * self.MATE_WEIGHT_KIN_ALTRUISM
            else:
                score_kin = 0.0
            
            # 6. Spouse loyalty / adultery penalty (weighted by conscientiousness)
            if female.spouse == suitor.id:
                # Already married to this suitor - loyalty bonus
                score_spouse = female.c * self.MATE_WEIGHT_SPOUSE
            elif female.spouse != -1:
                # Female is married to someone else - adultery penalty
                score_spouse = -female.c * self.MATE_WEIGHT_SPOUSE
            elif suitor.spouse != -1:
                # Suitor is married to someone else - smaller penalty
                score_spouse = -female.c * self.MATE_WEIGHT_SPOUSE / 2
            else:
                score_spouse = 0.0
            
            # 7. Incest penalty for high kinship (multiplicative)
            if r > self.CONSANGUINITY_TOLERANCE:
                incest_penalty = math.exp(-self.MATE_INCEST_DECAY * (r - self.CONSANGUINITY_TOLERANCE))
            else:
                incest_penalty = 1.0
            
            raw_score = score_energy + score_extraversion + score_genetic + score_trait + score_kin + score_spouse
            return raw_score * incest_penalty
        
        # Score all candidates and pick best
        scored = [(c, compute_score(c)) for c in candidates]
        best_candidate, best_score = max(scored, key=lambda x: x[1])
                
        return best_candidate
    
    def _mate(self, initiator: Agent, partner: Agent):
        """
        Execute mating between two agents.
        Male fertility is re-checked since he may have mated earlier this tick.
        
        Spawns offspring immediately (at conception) but sets born_tick
        to current tick + GESTATION_TICKS. The offspring exists but has
        negative age until born.
        """

        threshold = self.REPRODUCTION_THRESHOLD
        male_inv = self.MALE_INVESTMENT
        initial_e = self.INITIAL_ENERGY
        repro_cost = self.REPRODUCTION_COST
        
        # Determine male and female
        if initiator.sex == 0:
            male, female = initiator, partner
        else:
            male, female = partner, initiator
        # Re-check male fertility (may have mated already this tick)
        male_threshold = self.REPRODUCTION_THRESHOLD + self.REPRODUCTION_COST * self.MALE_INVESTMENT
        if male.energy < male_threshold:
            return  # No longer fertile

        # Track mating type for statistics
        # Check kinship for incest tracking
        kinship = male.kinship.get(female.id, 0.0)
        if kinship > 0.15:
            self._stats['matings_incestuous'] += 1
        
        # Categorize by marital status
        if male.spouse == female.id and female.spouse == male.id:
            # Already married to each other
            self._stats['matings_spousal'] += 1
        elif male.spouse == -1 and female.spouse == -1:
            # Both unmarried - will create marriage
            self._stats['matings_unm'] += 1
        else:
            # At least one married to someone else
            self._stats['matings_adulterous'] += 1

        # Split cost by investment ratio
        male.energy -= repro_cost * male_inv
        female.energy -= (repro_cost * (1 - male_inv))
        
        # Establish marriage if both unmarried
        if male.spouse == -1 and female.spouse == -1:
            male.spouse = female.id
            female.spouse = male.id
        
        self._stats['births'] += 1
        
        # Happiness boost for both parents on birth of child
        # Base 0.05 for male, 0.1 for female (2x multiplier)
        # Depressed if trust < 0.5 (low trust = depression)
        male_boost = 0.06
        female_boost = 0.12
        if male.trust < 0.5:
            male_boost *= (0.5 + male.trust)  # ranges from 0.5x to 1x
        if female.trust < 0.5:
            female_boost *= (0.5 + female.trust)
        male.hap = min(1.0, male.hap + male_boost)
        female.hap = min(1.0, female.hap + female_boost)
        
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
        counts = self.count_entities()
        total = sum(counts)
        
        # Calculate total matings this tick
        total_matings = (self._stats['matings_spousal'] + 
                        self._stats['matings_unm'] + 
                        self._stats['matings_adulterous'])
        
        # Build death breakdown string
        deaths_total = self._stats['deaths']
        if deaths_total > 0:
            death_str = (f"Deaths:{deaths_total} "
                        f"(starve:{self._stats['deaths_starvation']} "
                        f"accident:{self._stats['deaths_accident']})")
        else:
            death_str = "Deaths:0"
        
        print(f"\n[Tick {self.tick}] Population:{total} (M:{counts[0]} F:{counts[1]}) | "
              f"Births:{self._stats['births']} {death_str}")
        
        # Print mating breakdown if any matings occurred
        if total_matings > 0:
            print(f"  Matings: {total_matings} total | "
                  f"spousal:{self._stats['matings_spousal']} "
                  f"out-of-wedlock:{self._stats['matings_unm']} "
                  f"adulterous:{self._stats['matings_adulterous']} "
                  f"incestuous:{self._stats['matings_incestuous']}")
        
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
                        print(f"  {name:10s} [0-{max_val:.1f}]: {' '.join(f'{c:3d}' for c in counts)}  |{'|'.join(bars)}|")
        
        #  kinship diagnostic every 100 ticks (debugging)
        #if self.tick % 100 == 0:
        #    self._kinship_diagnostic()
    
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
                
                if len(shared_parents) == 2:
                    full_sib_r.append(r) # includes identical twins with r=1.0
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
        counts = self.count_entities()
        total = sum(counts)
        
        if total == 0:
            self.halted = True
            self.halt_reason = "Population extinct"
        elif total > 10000:
            self.halted = True
            self.halt_reason = "Population exploded"
    
    def count_entities(self) -> List[int]:
        """Return population counts as [male_count, female_count]."""
        return list(self._population)

    def entity_count(self) -> int:
        """Return total number of streamable entities."""
        return len(self._by_world_id)

    def get_entity_display(self, world_id: int, center, radius) -> Optional[dict]:
        """
        Get display dict for entity if it exists and is in viewport.
        Returns None if entity doesn't exist or is outside viewport.
        """
        entity = self._by_world_id.get(world_id)
        if entity is None:
            return None
        
        # Check viewport if specified
        if center is not None and radius is not None:
            if not self.is_in_viewport(entity.position, center, radius):
                return None
        
        d = entity.to_display_dict()
        d['world_id'] = world_id  # Add for server tracking (server strips before sending)
        return d

    def entity_in_viewport(self, world_id: int, center, radius) -> bool:
        """Check if entity is within viewport."""
        entity = self._by_world_id.get(world_id)
        if entity is None:
            return False
        return self.is_in_viewport(entity.position, center, radius)

    def get_world_ids_in_viewport(self, center, radius) -> List[int]:
        """Get all world_ids for entities in viewport."""
        if center is None or radius is None:
            return list(self._by_world_id.keys())
        return [wid for wid, entity in self._by_world_id.items()
                if self.is_in_viewport(entity.position, center, radius)]

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
                'world.food_per_turn': cls.FOOD_REGEN_PER_TURN,
                'world.food_ceiling': cls.FOOD_CEILING,
                'world.season_strength': cls.SEASON_STRENGTH,
                'world.map_width': cls.DEFAULT_WIDTH,
                'world.map_height': cls.DEFAULT_HEIGHT,  
                'stats.histo_bins': cls.HIST_BINS,    
                'agents.initial_energy': cls.INITIAL_ENERGY,
                'agents.metabolism_cost': cls.METABOLISM_COST,
                'agents.reproduction_threshold': cls.REPRODUCTION_THRESHOLD,
                'agents.reproduction_cost': cls.REPRODUCTION_COST,
                'agents.adulthood': cls.ADULTHOOD,
                'agents.senescence': cls.SENESCENCE,
                'agents.accident_base_rate': cls.ACCIDENT_BASE_RATE,
                'agents.accident_senescence_rate': cls.ACCIDENT_SENESCENCE_RATE,
                'agents.max_energy': cls.MAX_ENERGY,    
                'offspring.male_investment': cls.MALE_INVESTMENT,
                'offspring.gene_mutation':  cls.GENE_MUTATION_SD,
                'offspring.trait_mutation':  cls.TRAIT_MUTATION_SD,
                'identity.assimilation_rate':  cls.CULTURE_ASSIMILATION_RATE,                
                'identity.female_multiplier':  cls.FEMALE_CULTURE_MULT,
                'identity.spouse_multiplier':  cls.CULTURE_SPOUSAL_C_BONUS,
                'identity.extraversion_damp':  cls.CULTURE_EXTRAVERSION_DAMP,
                'identity.neuroticism_damp':  cls.CULTURE_NEUROTICISM_DAMP,
                'identity.culture_shock_threshold':  cls.CULTURE_SHOCK_THRESHOLD,
                'identity.max_assim_step':  cls.CULTURE_MAX_STEP,
                'identity.drift_rate':  cls.CULTURE_DRIFT_RATE,
                'migration.rate':  cls.P_MIGRATION,
                'migration.fem_multiplier':  cls.FEM_MIGRATION_RATIO,
                'migration.mate_seeking':  cls.P_MATE_SEEKING_MIGRATION,                
            }

    # Mapping from JSON param names to class attribute names
    _PARAM_MAP = {
        'world.food_per_turn': 'FOOD_REGEN_PER_TURN',
        'world.food_ceiling': 'FOOD_CEILING',
        'world.season_strength': 'SEASON_STRENGTH',
        'world.map_width': 'DEFAULT_WIDTH',
        'world.map_height': 'DEFAULT_HEIGHT',
        'stats.histo_bins': 'HIST_BINS',    
        'agents.initial_energy': 'INITIAL_ENERGY',
        'agents.metabolism_cost': 'METABOLISM_COST',
        'agents.reproduction_threshold': 'REPRODUCTION_THRESHOLD',
        'agents.reproduction_cost': 'REPRODUCTION_COST',
        'agents.adulthood': 'ADULTHOOD',
        'agents.senescence': 'SENESCENCE',
        'agents.accident_base_rate': 'ACCIDENT_BASE_RATE',
        'agents.accident_senescence_rate': 'ACCIDENT_SENESCENCE_RATE',
        'agents.max_energy': 'MAX_ENERGY',
        'offspring.male_investment': 'MALE_INVESTMENT',
        'offspring.gene_mutation':  'GENE_MUTATION_SD',
        'offspring.trait_mutation':  'TRAIT_MUTATION_SD',
        'identity.assimilation_rate':  'CULTURE_ASSIMILATION_RATE',                
        'identity.female_multiplier':  'FEMALE_CULTURE_MULT',
        'identity.spouse_multiplier':  'CULTURE_SPOUSAL_C_BONUS',
        'identity.extraversion_damp':  'CULTURE_EXTRAVERSION_DAMP',
        'identity.neuroticism_damp':  'CULTURE_NEUROTICISM_DAMP',
        'identity.culture_shock_threshold':  'CULTURE_SHOCK_THRESHOLD',
        'identity.max_assim_step':  'CULTURE_MAX_STEP',
        'identity.drift_rate':  'CULTURE_DRIFT_RATE',
        'migration.rate':  'P_MIGRATION',
        'migration.fem_multiplier':  'FEM_MIGRATION_RATIO',
        'migration.mate_seeking':  'P_MATE_SEEKING_MIGRATION',                
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

    #optimization helpers

    def _build_neighborhood_cache(self) -> NeighborhoodCache:
        """
        Build neighborhood cache at start of tick.

        Categorizes all agents and pre-joins Moore neighborhoods.
        Call once at tick start, reuse throughout tick.
        """
        cache = NeighborhoodCache(tick=self.tick)

        cells_with_agents = set()

        # Pass 1: Categorize agents by cell
        for agent in self.agents.values():
            cell = agent.cell
            cells_with_agents.add(cell)

            age = self._age_years(agent)
            if age < 0:
                continue  # unborn, skip

            # Check fertility
            investment = self.MALE_INVESTMENT if agent.sex == 0 else (1 - self.MALE_INVESTMENT)
            threshold = self.REPRODUCTION_THRESHOLD + self.REPRODUCTION_COST * investment
            is_fertile = agent.energy >= threshold

            if agent.sex == 0:  # male
                if age >= self.ADOLESCENCE:
                    cache.cell_adult_males[cell].append(agent.id)
                    if is_fertile:
                        cache.cell_fertile_males[cell].append(agent.id)
            else:  # female
                if is_fertile and age >= self.ADOLESCENCE and age < self.MENOPAUSE and not agent.pregnant:
                    cache.cell_fertile_females[cell].append(agent.id)
                    cache.fertile_female_ids.add(agent.id)

            # All non-infant agents count as adults for migration purposes
            if age >= self.CHILDHOOD:
                cache.cell_adults[cell].append(agent.id)

        # Pass 2: Pre-join Moore neighborhoods for cells with agents
        for cell in cells_with_agents:
            moore_cells = self._get_moore_neighborhood(cell[0], cell[1])

            # Fertile females in Moore neighborhood
            females = []
            for mc in moore_cells:
                females.extend(cache.cell_fertile_females.get(mc, []))
            cache.moore_fertile_females[cell] = females

            # Fertile males in Moore neighborhood
            males = []
            for mc in moore_cells:
                males.extend(cache.cell_fertile_males.get(mc, []))
            cache.moore_fertile_males[cell] = males

            # Adult males in Moore neighborhood (for future trade)
            adult_males = []
            for mc in moore_cells:
                adult_males.extend(cache.cell_adult_males.get(mc, []))
            cache.moore_adult_males[cell] = adult_males

        return cache

    def _collect_migration_intentions(self, cache: NeighborhoodCache,
                                    cell_agents: Dict[Tuple[int,int], List]) -> Dict[int, Tuple[int,int]]:
        """
        Collect migration intentions from agents.

        Two migration drivers:
        1. Overpopulation pressure: agents in cells over carrying capacity may migrate
           to less populated neighbors.
        2. Mate-seeking: unmarried males (ADOLESCENCE <= age < SENESCENCE) may
           migrate to cells with more fertile females, weighted by openness.

        Returns dict of agent_id -> target_cell.
        """
        intentions = {}

        # Cell carrying capacity based on food regeneration
        cell_capacity = self.FOOD_REGEN_PER_TURN / (2 * self.METABOLISM_COST)

        for (cx, cy), agents_here in cell_agents.items():
            neighbors = self._get_neighbors(cx, cy)
            if not neighbors:
                continue

            is_overpopulated = len(agents_here) > cell_capacity

            for agent in agents_here:
                agent_age = self._age_years(agent)
                
                # Young children don't migrate independently
                if agent_age <= self.INFANCY:
                    continue

                # Check for mate-seeking migration (unmarried males, adolescent to pre-senescent)
                is_unmarried_male = (
                    agent.sex == 0 and
                    agent.spouse == -1 and
                    self.ADOLESCENCE <= agent_age < self.SENESCENCE
                )

                if is_unmarried_male:
                    # Mate-seeking migration: probability scaled by openness
                    mate_seek_prob = self.P_MATE_SEEKING_MIGRATION * agent.o
                    
                    if random.random() < mate_seek_prob:
                        # Find neighbor with most fertile females
                        best_neighbor = None
                        best_female_count = 0
                        
                        for nx, ny in neighbors:
                            # Count fertile females in that cell (not Moore neighborhood)
                            female_count = len(cache.cell_fertile_females.get((nx, ny), []))
                            if female_count > best_female_count:
                                best_female_count = female_count
                                best_neighbor = (nx, ny)
                        
                        # Only migrate if there's a cell with fertile females
                        if best_neighbor and best_female_count > 0:
                            intentions[agent.id] = best_neighbor
                            continue  # Don't also consider overpopulation migration

                # Overpopulation-based migration (original logic)
                if is_overpopulated:
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
                            intentions[agent.id] = best_neighbor

        return intentions

    def _execute_migration_intentions(self, intentions: Dict[int, Tuple[int,int]],
                                    cell_agents: Dict[Tuple[int,int], List]):
        """
        Execute collected migration intentions.

        Processes in arbitrary order (could sort by motivation if we add scoring later).
        Updates cell_agents as migrations happen.
        """
        for agent_id, target_cell in intentions.items():
            if agent_id not in self.agents:
                continue

            agent = self.agents[agent_id]
            old_cell = agent.cell

            # Execute move
            self._move_agent(agent, target_cell[0], target_cell[1])

            # Update cell_agents tracking
            if old_cell in cell_agents and agent in cell_agents[old_cell]:
                cell_agents[old_cell].remove(agent)
            if target_cell not in cell_agents:
                cell_agents[target_cell] = []
            cell_agents[target_cell].append(agent)

            # If female, move young children with her
            if agent.sex == 1:
                for child_id in agent.offspring:
                    child = self.agents.get(child_id)
                    if child:
                        child_age = self._age_years(child)
                        if child_age <= self.INFANCY:
                            child_old_cell = child.cell
                            self._move_agent(child, target_cell[0], target_cell[1])
                            # Update cell_agents for child
                            if child_old_cell in cell_agents and child in cell_agents[child_old_cell]:
                                cell_agents[child_old_cell].remove(child)
                            cell_agents[target_cell].append(child)

    def _collect_trade_intentions(self, cache: NeighborhoodCache) -> Dict[int, List[int]]:
        """
        Collect trade intentions from adult males.

        SCAFFOLDING - returns empty dict for now.
        Future: males decide which neighboring males to trade with.
        """
        # TODO: Implement trade partner selection
        # Available data:
        #   cache.moore_adult_males[cell] - adult males in Moore neighborhood
        #   self._get_history_for_decide() - past interactions
        #   agent.kinship - relatedness
        #   self._genetic_distance() - genetic distance
        return {}

    def _execute_trade_intentions(self, intentions: Dict[int, List[int]]):
        """
        Execute trade interactions between agents.

        SCAFFOLDING - does nothing for now.
        Future: matched pairs play PD, apply payoffs.
        """
        # TODO: Implement trade execution
        # Likely pattern:
        #   1. Find bilateral matches (both want to trade with each other)
        #   2. Play PD game for each matched pair
        #   3. Record interactions, apply payoffs
        pass


    def _collect_raid_intentions(self, cache: NeighborhoodCache) -> Dict[int, Tuple[int,int]]:
        """
        Collect raid intentions.

        SCAFFOLDING - returns empty dict for now.
        Future: agents decide whether to raid neighboring cells.
        """
        # TODO: Implement raid decision logic
        return {}


    def _execute_raid_intentions(self, intentions: Dict[int, Tuple[int,int]]):
        """
        Execute raid actions.

        SCAFFOLDING - does nothing for now.
        Future: resolve raids, transfer resources, possible combat.
        """
        # TODO: Implement raid execution
        pass

#end of World class

def run_simulation(
    width: int = 10,
    height: int = 10,
    initial_pairs: int = 2,
    max_ticks: int = 4000,
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
        counts = world.count_entities()
        total = sum(counts)
        
        # Print status every n ticks
        if tick % 20 == 0:
            print(f"Tick {world.tick:4d}: Pop={total:4d} (M:{counts[0]} F:{counts[1]})")
        
        # Check if simulation halted itself
        if world.halted:
            print(f"\n*** {world.halt_reason} at tick {world.tick} ***")
            break
    else:
        print(f"\n*** Simulation completed {max_ticks} ticks ***")
    
    counts = world.count_entities()
    total = sum(counts)
    print(f"Final state: Pop={total} (M:{counts[0]} F:{counts[1]})")
    
    # Demo: show delta tracking
    print("\n--- Delta tracking demo ---")
    world.halted = False  # Reset for demo
    world.mark_clean()  # Clear dirty state
    world.step()  # Do one more step
    delta = world.get_dirty_state()
    print(f"After 1 tick: {len(delta['spawned'])} spawned, {len(delta['updated'])} updated, {len(delta['despawned'])} despawned")


if __name__ == "__main__":
    run_simulation()

"""
Prisoner's Dilemma decision logic module (v2).

This module computes cooperation probability based on agent personality traits,
interaction history, kinship, genetic distance, and cultural distance.

Changes from v1:
- Proper geometric weighting of pre-history based on n_pre
- Cultural distance as separate dimension from genetic distance  
- Simplified global state: hap (happiness), trust (world trust)
- Cleaner separation of identity modifiers

Agent objects are expected to have these attributes:
- Personality: o, c, e, a, n (Big Five, range [0,1])
- Social traits: kin, xeno (range [0,1])
- Global state: hap, trust (range [0,1], 0.5 = neutral)
"""

import math
import random
from typing import List, Optional, Tuple


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x: float) -> float:
    """Map score to probability [0, 1]. Score ~0 → 0.5, >5 → ~1, <-5 → ~0."""
    return 1.0 / (1.0 + math.exp(-x))


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def geometric_series_sum(decay: float, start: int, count: int) -> float:
    """
    Sum of geometric series: decay^start + decay^(start+1) + ... + decay^(start+count-1)
    
    = decay^start * (1 - decay^count) / (1 - decay)  if decay != 1
    = count                                           if decay == 1
    """
    if count <= 0:
        return 0.0
    if abs(decay - 1.0) < 1e-9:
        return float(count)
    return (decay ** start) * (1 - decay ** count) / (1 - decay)


def compute_weights(decay_rate: float, recent_depth: int, n_pre: int) -> Tuple[List[float], float]:
    """
    Compute normalized weights for recent history and pre-history.
    
    The weighting scheme:
    - t-1: decay^0
    - t-2: decay^1  
    - t-3: decay^2
    - Pre-history (t-4 to t-(3+n_pre)): sum of decay^3 through decay^(2+n_pre)
    
    Args:
        decay_rate: Per-turn decay factor
        recent_depth: Number of recent history slots (1, 2, or 3)
        n_pre: Number of turns in pre-history (0 if none)
    
    Returns:
        (recent_weights, pre_history_weight) - both normalized so total = 1
    """
    # Raw weights for recent history
    recent_raw = [decay_rate ** i for i in range(recent_depth)]
    
    # Raw weight for pre-history: geometric sum from decay^recent_depth to decay^(recent_depth+n_pre-1)
    pre_raw = geometric_series_sum(decay_rate, recent_depth, n_pre) if n_pre > 0 else 0.0
    
    # Normalize
    total = sum(recent_raw) + pre_raw
    if total < 1e-9:
        # Edge case: everything decayed to nothing
        return [1.0 / recent_depth] * recent_depth, 0.0
    
    recent_weights = [w / total for w in recent_raw]
    pre_weight = pre_raw / total
    
    return recent_weights, pre_weight


# =============================================================================
# CONTEXTUAL VALENCE EVALUATION
# =============================================================================

def contextual_valence(action: int, prior_context: int) -> float:
    """
    Evaluate an action given what preceded it.
    
    Args:
        action: 1 (cooperated) or 0 (defected)
        prior_context: 1, 0, or -1 (no context)
    
    Returns:
        Valence score roughly in [-1.5, +1.5]
    """
    if prior_context == -1:  # No prior context
        return 0.5 if action == 1 else -0.8
    
    if action == 1:  # Cooperation
        if prior_context == 0:
            return 1.5   # Forgiveness - cooperated after being wronged
        else:
            return 1.0   # Mutual cooperation - expected good behavior
    else:  # Defection
        if prior_context == 0:
            return -0.3  # Retaliation - defected after being wronged
        else:
            return -1.5  # Betrayal - defected after receiving cooperation


def evaluate_opponent_actions(
    own_history: List[int],
    opp_history: List[int],
    own_avg: Optional[float],
    opp_avg: Optional[float],
    n: float,
    recent_depth: int,
    n_pre: int
) -> Tuple[List[float], Optional[float]]:
    """
    Evaluate opponent actions with N-based negativity bias.
    
    Returns:
        (recent_valences, pre_history_valence)
        - recent_valences: List of contextualized valences for recent slots
        - pre_history_valence: Single valence for pre-history average (or None)
    """
    
    def apply_n_bias(valence: float) -> float:
        """N-based negativity bias: negative events amplified, positive discounted."""
        if valence < 0:
            return valence * (1 + 0.6 * n)
        else:
            return valence * (1 - 0.4 * n)
    
    recent_valences = []
    
    # Evaluate recent history
    for i in range(recent_depth):
        opp_action = opp_history[i] if i < len(opp_history) else None
        
        if opp_action is None:
            recent_valences.append(0.0)
            continue
        
        # What did I do before this opponent action? (opponent at t-i reacted to my t-(i+1))
        if i + 1 < len(own_history):
            prior_context = own_history[i + 1]
        elif n_pre > 0 and own_avg is not None:
            # Look back to pre-history average
            prior_context = 1 if own_avg > 0.5 else 0
        else:
            prior_context = -1  # No context
        
        valence = contextual_valence(opp_action, prior_context)
        recent_valences.append(apply_n_bias(valence))
    
    # Evaluate pre-history average
    pre_valence = None
    if n_pre > 0 and opp_avg is not None:
        # For pre-history, use own_avg as context (both are averages over same period)
        prior_context = 1 if (own_avg is not None and own_avg > 0.5) else -1
        opp_action_binary = 1 if opp_avg > 0.5 else 0
        pre_valence = contextual_valence(opp_action_binary, prior_context)
        pre_valence = apply_n_bias(pre_valence)
    
    return recent_valences, pre_valence


# =============================================================================
# PRINCIPLED ADJUSTMENT (CONSCIENTIOUSNESS)
# =============================================================================

def compute_principled_adjustment(
    c: float,
    own_history: List[int],
    opp_history: List[int],
    own_avg: Optional[float],
    opp_avg: Optional[float],
    recent_weights: List[float],
    pre_weight: float,
    recent_depth: int,
    n_pre: int
) -> float:
    """
    Conscientiousness-based principled adjustment.
    C makes agent sensitive to justification of own behavior.
    """
    adjustment = 0.0
    
    def evaluate_own_action(own_action: int, prior_opp: Optional[int]) -> float:
        """Return adjustment based on whether own action was principled."""
        if own_action == 0:  # I defected
            if prior_opp is not None:
                if prior_opp == 0:
                    return 0.8 * c   # Justified retaliation
                else:
                    return -2.5 * c  # Unprovoked defection
            else:
                return -0.5 * c      # No context, slight disapproval
        else:  # I cooperated
            if prior_opp is not None and prior_opp == 0:
                return 0.4 * c       # Forgiveness - principled mercy
        return 0.0
    
    # Evaluate recent history
    for i in range(recent_depth):
        own_action = own_history[i] if i < len(own_history) else None
        if own_action is None:
            continue
        
        # What did opponent do before my action at i? (I reacted to opp at i+1)
        prior_opp = None
        if i + 1 < len(opp_history):
            prior_opp = opp_history[i + 1]
        elif n_pre > 0 and opp_avg is not None:
            prior_opp = 1 if opp_avg > 0.5 else 0
        
        adjustment += recent_weights[i] * evaluate_own_action(own_action, prior_opp)
    
    # Evaluate pre-history average behavior
    if n_pre > 0 and own_avg is not None and pre_weight > 0:
        own_binary = 1 if own_avg > 0.5 else 0
        # For pre-history, context is itself (long-term pattern)
        prior_opp = 1 if (opp_avg is not None and opp_avg > 0.5) else None
        adjustment += pre_weight * evaluate_own_action(own_binary, prior_opp)
    
    return adjustment


# =============================================================================
# IDENTITY MODIFIERS (Kin, Genetic Distance, Cultural Distance)
# =============================================================================

def compute_kinship_bonus(own, kinship: float) -> float:
    """
    Kin selection: Hamilton's rule approximation.
    
    Args:
        own: Agent with 'kin' trait (sensitivity to kinship)
        kinship: Genealogical relatedness coefficient [0, 0.5]
                 (0.5 = sibling/parent, 0.25 = grandparent, 0.125 = cousin)
    
    Returns:
        Cooperation bonus (always >= 0)
    """
    # kin trait modulates sensitivity to kinship coefficient
    # Quadratic in kinship to emphasize close relatives
    return own.kin * kinship * 4.0


def compute_genetic_distance_penalty(own, genetic_distance: float) -> float:
    """
    Ethnic/racial ingroup preference penalty.
    
    This represents innate, relatively fixed out-group bias based on
    perceived genetic/ethnic distance.
    
    Args:
        own: Agent with 'xeno' and 'o' (openness) traits
        genetic_distance: Genetic/ethnic distance [0, ~2]
    
    Returns:
        Cooperation penalty (always >= 0)
    """
    # Openness slightly reduces genetic distance sensitivity
    # (but effect is limited - this is a more "primitive" response)
    effective_xeno = own.xeno * (1 - 0.2 * own.o)
    return effective_xeno * genetic_distance


def compute_cultural_distance_penalty(own, cultural_distance: Optional[float]) -> float:
    """
    Cultural out-group penalty.
    
    Unlike genetic distance, cultural distance is:
    - More malleable (can change with exposure)
    - More strongly modulated by personality traits
    - Affected by both cognitive (O) and social (E, A) factors
    
    The model:
    - Base penalty scales with xeno (general out-group wariness)
    - Openness (O) strongly reduces penalty (curiosity about other cultures)
    - Extraversion (E) moderately reduces penalty (social engagement)
    - Agreeableness (A) slightly reduces penalty (accommodation)
    
    Args:
        own: Agent with xeno, o, e, a traits
        cultural_distance: Cultural distance [0, ~2], or None to skip
    
    Returns:
        Cooperation penalty (always >= 0)
    """
    if cultural_distance is None or cultural_distance <= 0:
        return 0.0
    
    # Personality-based modulation of cultural sensitivity
    # O has strongest effect (intellectual openness to different worldviews)
    # E has moderate effect (willingness to engage across cultural lines)
    # A has mild effect (general accommodation)
    openness_factor = 1 - 0.5 * own.o      # O=1 → 50% reduction
    extraversion_factor = 1 - 0.25 * own.e  # E=1 → 25% reduction
    agreeableness_factor = 1 - 0.15 * own.a # A=1 → 15% reduction
    
    # Combined reduction (multiplicative, so high O+E+A gives strong reduction)
    personality_modifier = openness_factor * extraversion_factor * agreeableness_factor
    
    # xeno provides base sensitivity, personality modulates it
    effective_sensitivity = own.xeno * personality_modifier
    
    return effective_sensitivity * cultural_distance


def compute_identity_modifier(
    own, 
    kinship: float, 
    genetic_distance: float, 
    cultural_distance: Optional[float] = None
) -> float:
    """
    Combined identity-based cooperation modifier.
    
    Returns:
        Net modifier (positive = more cooperative, negative = less)
    """
    kin_bonus = compute_kinship_bonus(own, kinship)
    genetic_penalty = compute_genetic_distance_penalty(own, genetic_distance)
    cultural_penalty = compute_cultural_distance_penalty(own, cultural_distance)
    
    return kin_bonus - genetic_penalty - cultural_penalty


# =============================================================================
# GLOBAL STATE MODIFIER (hap, trust)
# =============================================================================

def compute_state_modifier(own) -> float:
    """
    Compute cooperation modifier from global emotional/trust state.
    
    Applied at the very end before sigmoid, as a nudge on the score
    already computed from per-partner interaction history.
    
    Parameters (agent attributes):
        hap: [0,1] happiness/life-satisfaction, 0.5 = neutral
        trust: [0,1] generalized world trust, 0.5 = neutral
    
    Personality modulation:
        N > 0.5: negative deviations (< 0.5) weighted more severely
        C > 0.5: both effects reduced, but hap reduced more than trust
                 (principled agents less swayed by mood, but still heed trust)
    """
    hap_dev = own.hap - 0.5      # range: [-0.5, 0.5]
    trust_dev = own.trust - 0.5  # range: [-0.5, 0.5]
    
    n_adj = own.n - 0.5  # range: [-0.5, 0.5]
    c_adj = own.c - 0.5
    
    # N-based asymmetry: high N amplifies negative, dampens positive
    if hap_dev < 0:
        hap_dev *= (1 + n_adj)      # n=1 → 1.5x, n=0.5 → 1x, n=0 → 0.5x
    else:
        hap_dev *= (1 - 0.5 * n_adj)  # n=1 → 0.75x, n=0.5 → 1x, n=0 → 1.25x
    
    if trust_dev < 0:
        trust_dev *= (1 + n_adj)
    else:
        trust_dev *= (1 - 0.5 * n_adj)
    
    # C-based dampening: high C reduces influence of both, but hap more so
    hap_weight = 1 - 1.2 * c_adj    # c=1 → 0.4, c=0.5 → 1.0, c=0 → 1.6
    trust_weight = 1 - 0.6 * c_adj  # c=1 → 0.7, c=0.5 → 1.0, c=0 → 1.3
    
    # Base scale: at neutral C/N and extreme hap/trust, effect is ±1.5 on score
    # (shifts p from 0.5 to ~0.82 or ~0.18)
    base_scale = 3.0
    
    return base_scale * (hap_weight * hap_dev + trust_weight * trust_dev)


# =============================================================================
# EXTRAVERSION MODIFIER
# =============================================================================

def apply_extraversion(score: float, e: float) -> float:
    """E amplifies decisiveness: pushes score away from 0."""
    multiplier = 0.6 + 0.8 * e
    return score * multiplier


# =============================================================================
# MAIN DECISION FUNCTIONS
# =============================================================================

def compute_coop_prob(
    own,
    opp,
    history: List[List],
    n_pre: int,
    kinship: float,
    genetic_distance: float,
    cultural_distance: Optional[float] = None
) -> float:
    """
    Compute cooperation probability for 'own' agent facing 'opp' agent.
    
    Args:
        own: The agent making the decision (needs o,c,e,a,n,kin,xeno,hap,trust)
        opp: The opponent agent (currently unused, but available for extensions)
        history: [
            [own_t-1, own_t-2, own_t-3],  # own recent actions (1=coop, 0=defect)
            [opp_t-1, opp_t-2, opp_t-3],  # opponent recent actions
            [own_avg, opp_avg]             # pre-history averages [0,1], or empty
        ]
        n_pre: Number of interactions before the last 3 (relationship duration)
        kinship: Genealogical relatedness coefficient [0, 0.5]
        genetic_distance: Genetic/ethnic distance [0, ~2]
        cultural_distance: Cultural distance [0, ~2], or None to skip cultural mechanics
    
    Returns:
        Cooperation probability [0, 1]
    """
    o, c, e, a, n = own.o, own.c, own.e, own.a, own.n
    
    own_history = history[0] if len(history) > 0 else []
    opp_history = history[1] if len(history) > 1 else []
    pre_history = history[2] if len(history) > 2 else []
    
    own_avg = pre_history[0] if len(pre_history) > 0 else None
    opp_avg = pre_history[1] if len(pre_history) > 1 else None
    
    # ============================================
    # FIRST MOVE (no history)
    # ============================================
    if len(own_history) == 0 or (len(own_history) > 0 and own_history[0] is None):
        # A dominates with quadratic effect
        score = 8.0 * a * a - 2.0
        
        # O influence: open to giving benefit of doubt
        score += 0.5 * o
        
        # C bonus: theory of mind - anticipates reciprocity
        score += 2.5 * c
        
        # Apply E as decisiveness multiplier
        score = apply_extraversion(score, e)
        
        # Identity modifier
        score += compute_identity_modifier(own, kinship, genetic_distance, cultural_distance)
        
        # Global state modifier (hap, trust)
        score += compute_state_modifier(own)
        
        return sigmoid(score)
    
    # ============================================
    # DETERMINE ACTUAL HISTORY DEPTH
    # ============================================
    recent_depth = min(len(own_history), 3)
    
    # Validate n_pre: only use if we have averages
    effective_n_pre = n_pre if (n_pre > 0 and own_avg is not None and opp_avg is not None) else 0
    
    # ============================================
    # TIME DECAY WEIGHTS
    # ============================================
    decay_rate = 0.4 + 0.8 * c - 0.5 * n
    decay_rate = clamp(decay_rate, 0.1, 1.1)
    
    recent_weights, pre_weight = compute_weights(decay_rate, recent_depth, effective_n_pre)
    
    # ============================================
    # OPPONENT ACTION EVALUATION (with N-asymmetry)
    # ============================================
    recent_valences, pre_valence = evaluate_opponent_actions(
        own_history, opp_history, own_avg, opp_avg, n, recent_depth, effective_n_pre
    )
    
    # ============================================
    # RECIPROCITY TERM
    # ============================================
    reciprocity_strength = 4.0
    
    opp_term = reciprocity_strength * sum(
        w * v for w, v in zip(recent_weights, recent_valences)
    )
    if pre_valence is not None and pre_weight > 0:
        opp_term += reciprocity_strength * pre_weight * pre_valence
    
    # ============================================
    # CONSCIENTIOUSNESS TERM (principled behavior)
    # ============================================
    principled_term = compute_principled_adjustment(
        c, own_history, opp_history, own_avg, opp_avg,
        recent_weights, pre_weight, recent_depth, effective_n_pre
    )
    
    # ============================================
    # AGREEABLENESS BASELINE
    # ============================================
    baseline = 4.0 * a - 1.5
    
    # ============================================
    # COMBINE SCORE
    # ============================================
    score = baseline + opp_term + principled_term
    
    # Apply extraversion (decisiveness multiplier)
    score = apply_extraversion(score, e)
    
    # ============================================
    # IDENTITY MODIFIER (kin, genetic, cultural)
    # ============================================
    score += compute_identity_modifier(own, kinship, genetic_distance, cultural_distance)
    
    # ============================================
    # GLOBAL STATE MODIFIER (hap, trust)
    # ============================================
    score += compute_state_modifier(own)
    
    return sigmoid(score)


def decide(
    own, 
    opp, 
    history: List[List], 
    n_pre: int, 
    kinship: float, 
    genetic_distance: float,
    cultural_distance: Optional[float] = None
) -> int:
    """
    Make a cooperation decision (stochastic).
    
    Returns:
        1 (cooperate) or 0 (defect)
    """
    prob = compute_coop_prob(own, opp, history, n_pre, kinship, genetic_distance, cultural_distance)
    return 1 if random.random() < prob else 0

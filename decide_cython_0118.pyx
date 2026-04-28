# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython implementation of Prisoner's Dilemma decision logic (v2).

Changes from v1:
- Proper geometric weighting of pre-history based on n_pre
- Cultural distance as separate dimension from genetic distance
- Simplified global state: hap (happiness), trust (world trust)
- N-based asymmetry and C-based dampening for state modifier

To compile:
    pip install cython
    python setup_cython.py build_ext --inplace

This produces a .so (Linux) or .pyd (Windows) binary that can be imported.
"""

from libc.math cimport exp


cdef double fast_sigmoid(double x) nogil:
    """
    Simple linear approximation: 0.5 + 0.1*x, clamped to [0, 1].
    Matches the scoring scheme exactly: -5→0%, 0→50%, +5→100%.
    """
    if x <= -5.0:
        return 0.0
    if x >= 5.0:
        return 1.0
    return 0.5 + x * 0.1


cdef double fast_sigmoid_smooth(double x) nogil:
    """
    Slightly more accurate piecewise approximation.
    Uses 3 segments for better fit. Max error ~0.04.
    """
    if x <= -5.0:
        return 0.0
    if x >= 5.0:
        return 1.0
    if x < -2.0:
        return (x + 5.0) * 0.04  # 0.0 at -5, 0.12 at -2
    if x > 2.0:
        return 0.88 + (x - 2.0) * 0.04  # 0.88 at 2, 1.0 at 5
    return 0.5 + x * 0.19  # slope 0.19 in middle region


cdef double accurate_sigmoid(double x) nogil:
    """Original sigmoid using exp."""
    return 1.0 / (1.0 + exp(-x))


cdef double clamp(double val, double lo, double hi) nogil:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


cdef double contextual_valence(int action, int prior_context) nogil:
    if prior_context == -1:
        return 0.5 if action == 1 else -0.8
    if action == 1:
        return 1.5 if prior_context == 0 else 1.0
    else:
        return -0.3 if prior_context == 0 else -1.5


cdef double geometric_series_sum(double decay, int start, int count) nogil:
    """
    Sum of geometric series: decay^start + decay^(start+1) + ... + decay^(start+count-1)
    """
    cdef double decay_start, decay_count
    if count <= 0:
        return 0.0
    if decay > 0.999 and decay < 1.001:  # decay ≈ 1
        return <double>count
    decay_start = 1.0
    cdef int i
    for i in range(start):
        decay_start *= decay
    decay_count = 1.0
    for i in range(count):
        decay_count *= decay
    return decay_start * (1.0 - decay_count) / (1.0 - decay)


cdef double compute_identity_modifier(
    double kin_trait, double xeno_trait, double o, double e, double a,
    double kinship, double genetic_distance, double cultural_distance
) nogil:
    """
    Combined identity-based cooperation modifier.
    
    genetic_distance: fixed ethnic/racial distance
    cultural_distance: malleable cultural distance (< 0 means undefined/skip)
    """
    cdef double kin_bonus, genetic_penalty, cultural_penalty
    cdef double openness_factor, extraversion_factor, agreeableness_factor
    cdef double personality_modifier, effective_sensitivity
    
    # Kin bonus
    kin_bonus = kin_trait * kinship * 4.0
    
    # Genetic distance penalty (O slightly reduces)
    genetic_penalty = xeno_trait * (1.0 - 0.2 * o) * genetic_distance
    
    # Cultural distance penalty (only if defined, i.e. >= 0)
    cultural_penalty = 0.0
    if cultural_distance >= 0.0:
        openness_factor = 1.0 - 0.5 * o
        extraversion_factor = 1.0 - 0.25 * e
        agreeableness_factor = 1.0 - 0.15 * a
        personality_modifier = openness_factor * extraversion_factor * agreeableness_factor
        effective_sensitivity = xeno_trait * personality_modifier
        cultural_penalty = effective_sensitivity * cultural_distance
    
    return kin_bonus - genetic_penalty - cultural_penalty


cdef double compute_state_modifier(double hap, double trust, double n, double c) nogil:
    """
    Compute cooperation modifier from global hap/trust state.
    
    Applied at the very end before sigmoid, as a nudge on the score.
    
    Parameters:
        hap: [0,1] happiness/life-satisfaction, 0.5 = neutral
        trust: [0,1] generalized world trust, 0.5 = neutral
        n: neuroticism trait
        c: conscientiousness trait
    
    Personality modulation:
        N > 0.5: negative deviations (< 0.5) weighted more severely
        C > 0.5: both effects reduced, but hap reduced more than trust
    """
    cdef double hap_dev, trust_dev, n_adj, c_adj
    cdef double hap_weight, trust_weight
    cdef double base_scale = 3.0
    
    hap_dev = hap - 0.5      # range: [-0.5, 0.5]
    trust_dev = trust - 0.5  # range: [-0.5, 0.5]
    
    n_adj = n - 0.5  # range: [-0.5, 0.5]
    c_adj = c - 0.5
    
    # N-based asymmetry: high N amplifies negative, dampens positive
    if hap_dev < 0:
        hap_dev = hap_dev * (1.0 + n_adj)      # n=1 → 1.5x, n=0.5 → 1x, n=0 → 0.5x
    else:
        hap_dev = hap_dev * (1.0 - 0.5 * n_adj)  # n=1 → 0.75x, n=0.5 → 1x, n=0 → 1.25x
    
    if trust_dev < 0:
        trust_dev = trust_dev * (1.0 + n_adj)
    else:
        trust_dev = trust_dev * (1.0 - 0.5 * n_adj)
    
    # C-based dampening: high C reduces influence of both, but hap more so
    hap_weight = 1.0 - 1.2 * c_adj    # c=1 → 0.4, c=0.5 → 1.0, c=0 → 1.6
    trust_weight = 1.0 - 0.6 * c_adj  # c=1 → 0.7, c=0.5 → 1.0, c=0 → 1.3
    
    return base_scale * (hap_weight * hap_dev + trust_weight * trust_dev)


cpdef double compute_coop_prob_cython(
    double o, double c, double e, double a, double n,
    double kin_trait, double xeno_trait,
    double hap, double trust,
    int own_h0, int own_h1, int own_h2,
    int opp_h0, int opp_h1, int opp_h2,
    double own_avg, double opp_avg,
    int n_pre,
    double kinship, double genetic_distance, double cultural_distance,
    int sigmoid_mode
) nogil:
    """
    Cython-optimized cooperation probability.
    
    Args:
        o, c, e, a, n: Big Five traits [0, 1]
        kin_trait, xeno_trait: social traits [0, 1]
        hap: happiness/life-satisfaction [0, 1], 0.5 = neutral
        trust: world trust [0, 1], 0.5 = neutral
        own_h0, own_h1, own_h2: own recent actions (-1 for missing)
        opp_h0, opp_h1, opp_h2: opponent recent actions (-1 for missing)
        own_avg, opp_avg: pre-history averages (-1.0 for missing)
        n_pre: number of interactions in pre-history
        kinship: genealogical relatedness [0, 0.5]
        genetic_distance: genetic/ethnic distance [0, ~2]
        cultural_distance: cultural distance [0, ~2], or < 0 to skip
        sigmoid_mode: 0=linear (fastest), 1=smooth, 2=accurate (exp)
    
    Returns:
        Cooperation probability [0, 1]
    """
    cdef double score, baseline, opp_term, principled, v
    cdef double e_mult, id_mod, state_mod
    cdef double decay_rate, w0, w1, w2, pre_weight, total, pre_raw
    cdef double c_justified, c_unprovoked, c_none, c_forgive
    cdef double neg_mult, pos_mult
    cdef int recent_depth, prior
    cdef bint has_pre_history
    cdef int effective_n_pre
    
    # Pre-compute common values
    e_mult = 0.6 + 0.8 * e
    id_mod = compute_identity_modifier(
        kin_trait, xeno_trait, o, e, a,
        kinship, genetic_distance, cultural_distance
    )
    state_mod = compute_state_modifier(hap, trust, n, c)
    
    # First move check
    if own_h0 == -1:
        score = 8.0 * a * a - 2.0 + 0.5 * o + 2.5 * c
        score = score * e_mult
        score = score + id_mod + state_mod
        if sigmoid_mode == 0:
            return fast_sigmoid(score)
        elif sigmoid_mode == 1:
            return fast_sigmoid_smooth(score)
        return accurate_sigmoid(score)
    
    # Determine recent history depth (1, 2, or 3)
    recent_depth = 1
    if own_h1 != -1:
        recent_depth = 2
    if own_h2 != -1:
        recent_depth = 3
    
    # Validate n_pre: only use if we have averages
    has_pre_history = n_pre > 0 and own_avg >= 0.0 and opp_avg >= 0.0
    effective_n_pre = n_pre if has_pre_history else 0
    
    # Compute decay rate
    decay_rate = clamp(0.4 + 0.8 * c - 0.5 * n, 0.1, 1.1)
    
    # Compute weights with proper geometric series for pre-history
    # Recent: w0 = decay^0, w1 = decay^1, w2 = decay^2
    # Pre-history: sum of decay^recent_depth to decay^(recent_depth + n_pre - 1)
    w0 = 1.0
    w1 = decay_rate
    w2 = decay_rate * decay_rate
    
    if effective_n_pre > 0:
        pre_raw = geometric_series_sum(decay_rate, recent_depth, effective_n_pre)
    else:
        pre_raw = 0.0
    
    # Compute total for normalization based on actual depth
    if recent_depth == 1:
        total = w0 + pre_raw
    elif recent_depth == 2:
        total = w0 + w1 + pre_raw
    else:  # recent_depth == 3
        total = w0 + w1 + w2 + pre_raw
    
    # Normalize
    w0 = w0 / total
    w1 = w1 / total
    w2 = w2 / total
    pre_weight = pre_raw / total
    
    # Negativity bias multipliers
    neg_mult = 1.0 + 0.6 * n
    pos_mult = 1.0 - 0.4 * n
    
    # Evaluate opponent actions
    opp_term = 0.0
    
    # Slot 0 (most recent)
    if recent_depth >= 1:
        if recent_depth >= 2:
            prior = own_h1
        elif has_pre_history:
            prior = 1 if own_avg > 0.5 else 0
        else:
            prior = -1
        v = contextual_valence(opp_h0, prior)
        v = v * neg_mult if v < 0 else v * pos_mult
        opp_term = opp_term + w0 * v
    
    # Slot 1
    if recent_depth >= 2:
        if recent_depth >= 3:
            prior = own_h2
        elif has_pre_history:
            prior = 1 if own_avg > 0.5 else 0
        else:
            prior = -1
        v = contextual_valence(opp_h1, prior)
        v = v * neg_mult if v < 0 else v * pos_mult
        opp_term = opp_term + w1 * v
    
    # Slot 2
    if recent_depth >= 3:
        if has_pre_history:
            prior = 1 if own_avg > 0.5 else 0
        else:
            prior = -1
        v = contextual_valence(opp_h2, prior)
        v = v * neg_mult if v < 0 else v * pos_mult
        opp_term = opp_term + w2 * v
    
    # Pre-history slot
    if has_pre_history and pre_weight > 0.0:
        prior = 1 if own_avg > 0.5 else 0
        v = contextual_valence(1 if opp_avg > 0.5 else 0, prior)
        v = v * neg_mult if v < 0 else v * pos_mult
        opp_term = opp_term + pre_weight * v
    
    opp_term = opp_term * 4.0  # reciprocity_strength
    
    # Principled adjustment (conscientiousness)
    c_justified = 0.8 * c
    c_unprovoked = 2.5 * c
    c_none = 0.5 * c
    c_forgive = 0.4 * c
    principled = 0.0
    
    # Slot 0
    if recent_depth >= 1:
        if recent_depth >= 2:
            prior = opp_h1
        elif has_pre_history:
            prior = 1 if opp_avg > 0.5 else 0
        else:
            prior = -1
        
        if own_h0 == 0:
            if prior != -1:
                principled = principled + w0 * (c_justified if prior == 0 else -c_unprovoked)
            else:
                principled = principled - w0 * c_none
        elif own_h0 == 1 and prior == 0:
            principled = principled + w0 * c_forgive
    
    # Slot 1
    if recent_depth >= 2:
        if recent_depth >= 3:
            prior = opp_h2
        elif has_pre_history:
            prior = 1 if opp_avg > 0.5 else 0
        else:
            prior = -1
        
        if own_h1 == 0:
            if prior != -1:
                principled = principled + w1 * (c_justified if prior == 0 else -c_unprovoked)
            else:
                principled = principled - w1 * c_none
        elif own_h1 == 1 and prior == 0:
            principled = principled + w1 * c_forgive
    
    # Slot 2
    if recent_depth >= 3:
        if has_pre_history:
            prior = 1 if opp_avg > 0.5 else 0
        else:
            prior = -1
        
        if own_h2 == 0:
            if prior != -1:
                principled = principled + w2 * (c_justified if prior == 0 else -c_unprovoked)
            else:
                principled = principled - w2 * c_none
        elif own_h2 == 1 and prior == 0:
            principled = principled + w2 * c_forgive
    
    # Pre-history slot (evaluate own_avg behavior)
    if has_pre_history and pre_weight > 0.0:
        # For pre-history, context is opp_avg (long-term pattern)
        prior = 1 if opp_avg > 0.5 else 0
        if own_avg <= 0.5:  # treated as defection
            if prior == 0:
                principled = principled + pre_weight * c_justified
            else:
                principled = principled - pre_weight * c_unprovoked
        else:  # treated as cooperation
            if prior == 0:
                principled = principled + pre_weight * c_forgive
    
    # Combine score
    baseline = 4.0 * a - 1.5
    score = baseline + opp_term + principled
    score = score * e_mult
    score = score + id_mod + state_mod
    
    if sigmoid_mode == 0:
        return fast_sigmoid(score)
    elif sigmoid_mode == 1:
        return fast_sigmoid_smooth(score)
    return accurate_sigmoid(score)


cpdef int decide_cython(
    double o, double c, double e, double a, double n,
    double kin_trait, double xeno_trait,
    double hap, double trust,
    int own_h0, int own_h1, int own_h2,
    int opp_h0, int opp_h1, int opp_h2,
    double own_avg, double opp_avg,
    int n_pre,
    double kinship, double genetic_distance, double cultural_distance,
    double rand_val,
    int sigmoid_mode
):
    """
    Make stochastic decision. Pass random value as rand_val.
    
    Args:
        (see compute_coop_prob_cython for parameter descriptions)
        rand_val: pre-generated random value in [0, 1)
        sigmoid_mode: 0=linear (fastest), 1=smooth, 2=accurate
    
    Returns:
        1 (cooperate) or 0 (defect)
    """
    cdef double prob = compute_coop_prob_cython(
        o, c, e, a, n,
        kin_trait, xeno_trait,
        hap, trust,
        own_h0, own_h1, own_h2,
        opp_h0, opp_h1, opp_h2,
        own_avg, opp_avg,
        n_pre,
        kinship, genetic_distance, cultural_distance,
        sigmoid_mode
    )
    return 1 if rand_val < prob else 0

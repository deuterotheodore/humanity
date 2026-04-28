# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython implementation of Prisoner's Dilemma decision logic (v3).

Changes from v2:
- State modifier simplified to just happiness (slight linear effect)
- Identity modifier: kin_trait/xeno_trait centered at 0.5, outgroup preference weaker slope
- Genetic/cultural distance: squared Euclidean, capped at 75, scaled by /25
- Cultural preference = average of kin_trait and xeno_trait
- First move: superlinear C boost above 0.5
- N-based asymmetry centered at 0.5
- Principled term: forgiveness = A * (1-C), restructured valences
- Deviousness term: (1-E) * (1-C) enables strategic exploitation
- Demographic modifier: age-tier based effects (child/juvenile/adult/elderly)
- Pre-history: avgs in [-1, 1], nPre > 0 activates

To compile:
    pip install cython
    python setup_cython.py build_ext --inplace
"""

from libc.math cimport exp, fmin, fmax


cdef double fast_sigmoid(double x) nogil:
    """
    Simple linear approximation: 0.5 + 0.1*x, clamped to [0, 1].
    """
    if x <= -5.0:
        return 0.0
    if x >= 5.0:
        return 1.0
    return 0.5 + x * 0.1


cdef double fast_sigmoid_smooth(double x) nogil:
    """
    Slightly more accurate piecewise approximation.
    """
    if x <= -5.0:
        return 0.0
    if x >= 5.0:
        return 1.0
    if x < -2.0:
        return (x + 5.0) * 0.04
    if x > 2.0:
        return 0.88 + (x - 2.0) * 0.04
    return 0.5 + x * 0.19


cdef double accurate_sigmoid(double x) nogil:
    """Original sigmoid using exp."""
    return 1.0 / (1.0 + exp(-x))


cdef double clamp(double val, double lo, double hi) nogil:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


cdef double contextual_valence(int action, double prior_context) nogil:
    """
    Compute valence of opponent action given prior context.
    prior_context can be discrete (-1, +1) or continuous [-1, +1].
    Returns valence interpolated based on prior.
    
    action: 1 = cooperate, 0 = defect
    prior_context: -1 = I defected, +1 = I cooperated, or continuous [-1, 1]
                   -2 = no context (sentinel)
    """
    cdef double t, v_after_defect, v_after_coop
    
    # No context case
    if prior_context < -1.5:  # sentinel for no context
        if action == 1:
            return 0.5  # cooperation without context
        else:
            return -0.8  # defection without context
    
    # Map prior_context from [-1, 1] to t in [0, 1]
    # t = 0 means I defected, t = 1 means I cooperated
    t = (prior_context + 1.0) / 2.0
    
    if action == 1:  # opponent cooperated
        v_after_defect = 1.5   # forgiveness/reciprocation
        v_after_coop = 1.0     # mutual cooperation
        return v_after_defect * (1.0 - t) + v_after_coop * t
    else:  # opponent defected
        v_after_defect = -0.3  # mutual defection (retaliation)
        v_after_coop = -1.5    # betrayal
        return v_after_defect * (1.0 - t) + v_after_coop * t


cdef double contextual_valence_continuous(double opp_avg, double own_avg) nogil:
    """
    Compute expected valence from continuous pre-history averages.
    Averages are in [-1, 1]: -1 = always defect, +1 = always coop.
    """
    cdef double p_opp_coop, p_own_coop
    cdef double p_cc, p_cd, p_dc, p_dd
    cdef double v_cc, v_cd, v_dc, v_dd
    
    # Map averages from [-1, 1] to probabilities [0, 1]
    p_opp_coop = (opp_avg + 1.0) / 2.0
    p_own_coop = (own_avg + 1.0) / 2.0
    
    p_cc = p_opp_coop * p_own_coop
    p_cd = p_opp_coop * (1.0 - p_own_coop)
    p_dc = (1.0 - p_opp_coop) * p_own_coop
    p_dd = (1.0 - p_opp_coop) * (1.0 - p_own_coop)
    
    v_cc = 1.0   # mutual cooperation
    v_cd = 1.5   # opponent cooperated after I defected (forgiveness)
    v_dc = -1.5  # opponent defected after I cooperated (betrayal)
    v_dd = -0.3  # mutual defection
    
    return p_cc * v_cc + p_cd * v_cd + p_dc * v_dc + p_dd * v_dd


cdef double geometric_series_sum(double decay, int start, int count) nogil:
    """
    Sum of geometric series: decay^start + decay^(start+1) + ... + decay^(start+count-1)
    """
    cdef double decay_start, decay_count
    cdef int i
    
    if count <= 0:
        return 0.0
    if decay > 0.999 and decay < 1.001:  # decay ≈ 1
        return <double>count
    
    decay_start = 1.0
    for i in range(start):
        decay_start *= decay
    decay_count = 1.0
    for i in range(count):
        decay_count *= decay
    return decay_start * (1.0 - decay_count) / (1.0 - decay)


cdef double compute_identity_modifier(
    double kin_trait, double xeno_trait, double o,
    double kinship, double genetic_distance, double cultural_distance
) nogil:
    """
    Identity-based cooperation modifier.
    
    kin_trait, xeno_trait: [0, 1], 0.5 = neutral
    kinship: relatedness coefficient [0, 1]
    genetic_distance, cultural_distance: squared Euclidean [0, 75], cultural < 0 = skip
    """
    cdef double modifier = 0.0
    cdef double kin_effect, xeno_effect, cult_effect
    cdef double openness_mod, cult_openness_mod
    cdef double cultural_pref
    cdef double capped_gen_dist, capped_cult_dist
    
    # === KIN ALTRUISM ===
    kin_effect = kin_trait - 0.5  # range [-0.5, 0.5]
    if kin_effect >= 0:
        # Ingroup preference: higher relatedness -> more cooperation
        modifier = modifier + kin_effect * 2.0 * kinship * 3.0
    else:
        # Outgroup preference: weaker slope
        modifier = modifier + kin_effect * 2.0 * kinship * 1.5
    
    # === XENOPHOBIA ===
    capped_gen_dist = fmin(genetic_distance, 75.0)
    xeno_effect = xeno_trait - 0.2
    openness_mod = 1.0 - 0.3 * o
    if xeno_effect >= 0:
        # Ingroup preference: genetic distance -> less cooperation
        # Max effect: 0.5 * 2 * 1 * 75 / 25 = 3.0
        modifier = modifier - xeno_effect * 4.0 * openness_mod * capped_gen_dist / 25.0
    else:
        # Outgroup preference: weaker slope
        modifier = modifier - xeno_effect * 4.0 * capped_gen_dist / 50.0
    
    # === CULTURAL DISTANCE ===
    # Cultural preference = average of kin_trait and xeno_trait
    if cultural_distance >= 0:
        capped_cult_dist = fmin(cultural_distance, 75.0)
        cultural_pref = (kin_trait + xeno_trait) / 2.0
        cult_effect = cultural_pref - 0.5
        cult_openness_mod = 1.0 - 0.3 * o  # O reduces cultural effect 
        
        if cult_effect >= 0:
            modifier = modifier - cult_effect * 4.0 * cult_openness_mod * capped_cult_dist / 25.0
        else:
            # Outgroup preference: weaker slope
            modifier = modifier - cult_effect * 4.0 * capped_cult_dist / 50.0
    
    return modifier


cdef double compute_state_modifier(double hap) nogil:
    """
    Global state modifier: happiness only (world trust reserved).
    Slight linear impact centered at 0.5.
    """
    return (hap - 0.5) * 0.8  # ±0.4 at extremes


cdef int get_age_tier(int age) nogil:
    """0 = child, 1 = juvenile, 2 = adult, 3 = elderly"""
    if age < 14:
        return 0
    if age < 21:
        return 1
    if age <= 65:
        return 2
    return 3


cdef double compute_demographic_modifier(
    int own_sex, int opp_sex,  # 0 = male, 1 = female
    int own_age, int opp_age,
    double c, double n
) nogil:
    """
    Age-tier based demographic effects.
    Effects scaled by C (except F-F penalty scaled by N).
    """
    cdef double modifier = 0.0
    cdef int own_tier, opp_tier
    cdef bint same_sex
    cdef double c_scale, n_scale, c_scale_inverse
    cdef double age_diff, age_progress, baseline, c_mod
    
    own_tier = get_age_tier(own_age)
    opp_tier = get_age_tier(opp_age)
    same_sex = own_sex == opp_sex
    c_scale = 0.5 + c  # range [0.5, 1.5]
    
    # === CHILDREN (tier 0) ===
    if own_tier == 0:
        # Cooperation boost for older opponents (capped at 10 years diff)
        age_diff = fmin(<double>(opp_age - own_age), 10.0)
        if age_diff > 0:
            modifier = modifier + (age_diff / 10.0) * 1.2 * c_scale
    
    # === JUVENILES (tier 1) ===
    elif own_tier == 1:
        # Boost for opposite sex juveniles
        if opp_tier == 1 and not same_sex:
            modifier = modifier + 2.0 * c_scale
        
        # Rebelliousness toward adults/elderly fades with age
        if opp_tier == 2 or opp_tier == 3:
            age_progress = (<double>(own_age - 14)) / 7.0  # 0 at 14, 1 at 21
            baseline = -0.80 * (1.0 - age_progress)  # fades from -0.80 to 0
            c_mod = (c - 0.5) * 1.6  # -0.80 at C=0, +0.80 at C=1
            modifier = modifier + baseline + c_mod
    
    # === ADULTS (tier 2) ===
    elif own_tier == 2:
        # Boost for elderly and children
        if opp_tier == 3 or opp_tier == 0:
            modifier = modifier + 1.0 * c_scale
        
        # Boost for adult males playing against adult females
        if own_sex == 0 and opp_sex == 1 and opp_tier == 2:
            modifier = modifier + 0.8 * c_scale
        
        # Reduction for adult females playing adult females (scaled by N)
        if own_sex == 1 and opp_sex == 1 and opp_tier == 2:
            n_scale = 0.5 + n
            modifier = modifier - 1.0 * n_scale
    
    # === ELDERLY (tier 3) ===
    # No specific modifiers (generosity comes through A parameter)
    
    return modifier


cdef double apply_neuroticism_bias(double valence, double n_effect) nogil:
    """
    Apply N-based asymmetry to valence.
    n_effect = N - 0.5, range [-0.5, 0.5]
    N=0: dampens negative (0.4x), boosts positive (1.4x)
    N=0.5: neutral (1.0x)
    N=1: amplifies negative (1.6x), dampens positive (0.6x)
    """
    if valence < 0:
        return valence * (1.0 + 1.2 * n_effect)
    else:
        return valence * (1.0 - 0.8 * n_effect)


cdef double compute_deviousness_term(
    double e, double c,
    int own_h0, int own_h1, int own_h2,
    int opp_h0, int opp_h1, int opp_h2,
    double w0, double w1, double w2,
    int recent_depth
) nogil:
    """
    Deviousness = (1-E) * (1-C) enables strategic exploitation.
    Only considers recent history (devious agents are opportunistic).
    """
    cdef double deviousness = (1.0 - e) * (1.0 - c)
    cdef double mutual_coop_weight = 0.0
    cdef double mutual_defect_weight = 0.0
    cdef double sucker_exploit_weight = 0.0
    cdef double recent_total, exploit_trust, reset_effect, sucker_effect
    
    if deviousness < 0.05:
        return 0.0
    
    # Count patterns in recent history
    if recent_depth >= 1:
        if own_h0 == 1 and opp_h0 == 1:
            mutual_coop_weight = mutual_coop_weight + w0
        elif own_h0 == 0 and opp_h0 == 0:
            mutual_defect_weight = mutual_defect_weight + w0
        elif own_h0 == 0 and opp_h0 == 1:
            sucker_exploit_weight = sucker_exploit_weight + w0
    
    if recent_depth >= 2:
        if own_h1 == 1 and opp_h1 == 1:
            mutual_coop_weight = mutual_coop_weight + w1
        elif own_h1 == 0 and opp_h1 == 0:
            mutual_defect_weight = mutual_defect_weight + w1
        elif own_h1 == 0 and opp_h1 == 1:
            sucker_exploit_weight = sucker_exploit_weight + w1
    
    if recent_depth >= 3:
        if own_h2 == 1 and opp_h2 == 1:
            mutual_coop_weight = mutual_coop_weight + w2
        elif own_h2 == 0 and opp_h2 == 0:
            mutual_defect_weight = mutual_defect_weight + w2
        elif own_h2 == 0 and opp_h2 == 1:
            sucker_exploit_weight = sucker_exploit_weight + w2
    
    # Normalize to recent-only weights
    recent_total = 0.0
    if recent_depth >= 1:
        recent_total = recent_total + w0
    if recent_depth >= 2:
        recent_total = recent_total + w1
    if recent_depth >= 3:
        recent_total = recent_total + w2
    
    if recent_total > 0:
        mutual_coop_weight = mutual_coop_weight / recent_total
        mutual_defect_weight = mutual_defect_weight / recent_total
        sucker_exploit_weight = sucker_exploit_weight / recent_total
    
    # Compute effects
    exploit_trust = -mutual_coop_weight * (1.0 + 0.3 * mutual_coop_weight)
    reset_effect = mutual_defect_weight * 0.4
    sucker_effect = -sucker_exploit_weight * 1.2
    
    return deviousness * (exploit_trust + reset_effect + sucker_effect) * 3.5


cpdef double compute_coop_prob_cython(
    double o, double c, double e, double a, double n,
    double kin_trait, double xeno_trait,
    double hap, double trust,
    int own_h0, int own_h1, int own_h2,
    int opp_h0, int opp_h1, int opp_h2,
    double own_avg, double opp_avg,
    int n_pre,
    double kinship, double genetic_distance, double cultural_distance,
    int own_sex, int opp_sex,
    int own_age, int opp_age,
    int sigmoid_mode
) nogil:
    """
    Cython-optimized cooperation probability.
    
    Args:
        o, c, e, a, n: Big Five traits [0, 1]
        kin_trait, xeno_trait: social traits [0, 1], 0.5 = neutral
        hap: happiness [0, 1], 0.5 = neutral
        trust: world trust [0, 1], 0.5 = neutral (currently unused, reserved)
        own_h0, own_h1, own_h2: own recent actions (1=coop, 0=defect, -1=missing)
        opp_h0, opp_h1, opp_h2: opponent recent actions
        own_avg, opp_avg: pre-history averages in [-1, 1]
        n_pre: number of interactions in pre-history (0 = no pre-history)
        kinship: relatedness coefficient [0, 1]
        genetic_distance: squared Euclidean distance [0, 75]
        cultural_distance: squared Euclidean distance [0, 75], or < 0 to skip
        own_sex, opp_sex: 0 = male, 1 = female
        own_age, opp_age: age in years [7, 80]
        sigmoid_mode: 0=linear (fastest), 1=smooth, 2=accurate (exp)
    
    Returns:
        Cooperation probability [0, 1]
    """
    cdef double score, baseline, opp_term, principled_term, devious_term
    cdef double transparency_mult, id_mod, state_mod, demo_mod
    cdef double decay_rate, w0, w1, w2, pre_weight, total, pre_raw
    cdef double n_effect, v, prior_context
    cdef double forgiveness_tendency
    cdef double c_above_half, first_move_score
    cdef double p_own_coop, p_opp_coop
    cdef double p_cd, p_dd, p_dc, p_cc
    cdef double v_cd, v_dd, v_dc, v_cc
    cdef double t, v_retaliation, v_unprovoked, v_after_defect, v_after_coop
    cdef int recent_depth
    cdef bint has_pre_history
    
    # Pre-compute modifiers
    id_mod = compute_identity_modifier(
        kin_trait, xeno_trait, o,
        kinship, genetic_distance, cultural_distance
    )
    state_mod = compute_state_modifier(hap)
    demo_mod = compute_demographic_modifier(own_sex, opp_sex, own_age, opp_age, c, n)
    
    # Transparency multiplier
    transparency_mult = 0.5 + 0.8 * e
    
    # ============================================
    # FIRST MOVE (no history)
    # ============================================
    if own_h0 == -1:
        first_move_score = 8.0 * a * a - 2.0
        first_move_score = first_move_score + 1.5 * c
        
        # Superlinear C boost above 0.5
        c_above_half = fmax(0.0, c - 0.5)
        first_move_score = first_move_score + 6.0 * c_above_half * c_above_half * 4.0
        
        first_move_score = first_move_score * transparency_mult
        first_move_score = first_move_score + id_mod + state_mod + demo_mod
        
        if sigmoid_mode == 0:
            return fast_sigmoid(first_move_score)
        elif sigmoid_mode == 1:
            return fast_sigmoid_smooth(first_move_score)
        return accurate_sigmoid(first_move_score)
    
    # ============================================
    # DETERMINE HISTORY DEPTH
    # ============================================
    recent_depth = 1
    if own_h1 != -1:
        recent_depth = 2
    if own_h2 != -1:
        recent_depth = 3
    
    # Pre-history active when nPre > 0
    has_pre_history = n_pre > 0 and recent_depth == 3
    
    # ============================================
    # COMPUTE WEIGHTS
    # ============================================
    decay_rate = clamp(0.4 + 0.8 * c - 0.5 * n, 0.1, 1.1)
    
    w0 = 1.0
    w1 = decay_rate
    w2 = decay_rate * decay_rate
    
    if has_pre_history:
        pre_raw = geometric_series_sum(decay_rate, recent_depth, n_pre)
    else:
        pre_raw = 0.0
    
    # Compute total for normalization
    if recent_depth == 1:
        total = w0 + pre_raw
    elif recent_depth == 2:
        total = w0 + w1 + pre_raw
    else:
        total = w0 + w1 + w2 + pre_raw
    
    # Normalize
    w0 = w0 / total
    w1 = w1 / total
    w2 = w2 / total
    pre_weight = pre_raw / total
    
    # N-based asymmetry centered at 0.5
    n_effect = n - 0.5
    
    # ============================================
    # OPPONENT ACTION EVALUATION
    # ============================================
    opp_term = 0.0
    
    # Slot 0 (most recent)
    if recent_depth >= 1:
        if recent_depth >= 2:
            prior_context = <double>own_h1 * 2.0 - 1.0  # convert 0/1 to -1/+1
        elif has_pre_history:
            prior_context = own_avg
        else:
            prior_context = -2.0  # sentinel for no context
        
        v = contextual_valence(opp_h0, prior_context)
        v = apply_neuroticism_bias(v, n_effect)
        opp_term = opp_term + w0 * v
    
    # Slot 1
    if recent_depth >= 2:
        if recent_depth >= 3:
            prior_context = <double>own_h2 * 2.0 - 1.0
        elif has_pre_history:
            prior_context = own_avg
        else:
            prior_context = -2.0
        
        v = contextual_valence(opp_h1, prior_context)
        v = apply_neuroticism_bias(v, n_effect)
        opp_term = opp_term + w1 * v
    
    # Slot 2
    if recent_depth >= 3:
        if has_pre_history:
            prior_context = own_avg
        else:
            prior_context = -2.0
        
        v = contextual_valence(opp_h2, prior_context)
        v = apply_neuroticism_bias(v, n_effect)
        opp_term = opp_term + w2 * v
    
    # Pre-history slot
    if has_pre_history and pre_weight > 0.0:
        v = contextual_valence_continuous(opp_avg, own_avg)
        v = apply_neuroticism_bias(v, n_effect)
        opp_term = opp_term + pre_weight * v
    
    opp_term = opp_term * 4.0  # reciprocity_strength
    
    # ============================================
    # PRINCIPLED TERM (conscientiousness + forgiveness)
    # ============================================
    forgiveness_tendency = a * (1.0 - c)
    principled_term = 0.0
    
    # Helper values for interpolation
    # own defection: vRetaliation (after opp D) vs vUnprovoked (after opp C)
    # own cooperation: vAfterDefect vs vAfterCoop
    
    # Slot 0
    if recent_depth >= 1:
        if recent_depth >= 2:
            prior_context = <double>opp_h1 * 2.0 - 1.0
        elif has_pre_history:
            prior_context = opp_avg
        else:
            prior_context = -2.0
        
        if prior_context > -1.5:  # has context
            t = (prior_context + 1.0) / 2.0  # 0 = opp defected, 1 = opp cooperated
            
            if own_h0 == 0:  # I defected
                v_retaliation = 0.6 * c
                v_unprovoked = -2.5 * c
                principled_term = principled_term + w0 * (v_retaliation * (1.0 - t) + v_unprovoked * t)
            else:  # I cooperated
                v_after_defect = forgiveness_tendency * 1.2 - c * 0.3
                v_after_coop = 0.1
                principled_term = principled_term + w0 * (v_after_defect * (1.0 - t) + v_after_coop * t)
    
    # Slot 1
    if recent_depth >= 2:
        if recent_depth >= 3:
            prior_context = <double>opp_h2 * 2.0 - 1.0
        elif has_pre_history:
            prior_context = opp_avg
        else:
            prior_context = -2.0
        
        if prior_context > -1.5:
            t = (prior_context + 1.0) / 2.0
            
            if own_h1 == 0:
                v_retaliation = 0.6 * c
                v_unprovoked = -2.5 * c
                principled_term = principled_term + w1 * (v_retaliation * (1.0 - t) + v_unprovoked * t)
            else:
                v_after_defect = forgiveness_tendency * 1.2 - c * 0.3
                v_after_coop = 0.1
                principled_term = principled_term + w1 * (v_after_defect * (1.0 - t) + v_after_coop * t)
    
    # Slot 2
    if recent_depth >= 3:
        if has_pre_history:
            prior_context = opp_avg
        else:
            prior_context = -2.0
        
        if prior_context > -1.5:
            t = (prior_context + 1.0) / 2.0
            
            if own_h2 == 0:
                v_retaliation = 0.6 * c
                v_unprovoked = -2.5 * c
                principled_term = principled_term + w2 * (v_retaliation * (1.0 - t) + v_unprovoked * t)
            else:
                v_after_defect = forgiveness_tendency * 1.2 - c * 0.3
                v_after_coop = 0.1
                principled_term = principled_term + w2 * (v_after_defect * (1.0 - t) + v_after_coop * t)
    
    # Pre-history slot
    if has_pre_history and pre_weight > 0.0:
        p_own_coop = (own_avg + 1.0) / 2.0
        p_opp_coop = (opp_avg + 1.0) / 2.0
        
        p_cd = p_own_coop * (1.0 - p_opp_coop)       # I coop after opp defect
        p_dd = (1.0 - p_own_coop) * (1.0 - p_opp_coop)  # I defect after opp defect
        p_dc = (1.0 - p_own_coop) * p_opp_coop       # I defect after opp coop
        p_cc = p_own_coop * p_opp_coop              # I coop after opp coop
        
        v_cd = forgiveness_tendency * 1.2 - c * 0.3
        v_dd = 0.6 * c
        v_dc = -2.5 * c
        v_cc = 0.1
        
        principled_term = principled_term + pre_weight * (p_cd * v_cd + p_dd * v_dd + p_dc * v_dc + p_cc * v_cc)
    
    # ============================================
    # AGREEABLENESS BASELINE
    # ============================================
    baseline = 4.0 * a - 1.5
    
    # ============================================
    # DEVIOUSNESS TERM
    # ============================================
    devious_term = compute_deviousness_term(
        e, c,
        own_h0, own_h1, own_h2,
        opp_h0, opp_h1, opp_h2,
        w0, w1, w2,
        recent_depth
    )
    
    # ============================================
    # COMBINE SCORE
    # ============================================
    # Transparent behavior modulated by E
    score = baseline + opp_term + principled_term
    score = score * transparency_mult
    
    # Devious term added separately (not dampened by low E)
    score = score + devious_term
    score = score + id_mod + state_mod + demo_mod
    
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
    int own_sex, int opp_sex,
    int own_age, int opp_age,
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
        own_sex, opp_sex,
        own_age, opp_age,
        sigmoid_mode
    )
    return 1 if rand_val < prob else 0

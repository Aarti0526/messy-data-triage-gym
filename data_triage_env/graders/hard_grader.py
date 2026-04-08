from data_triage_env.graders.easy_grader import score as _base_score

def score(agent_df, manifest):
    # Hard mode: 20x penalty for mistakes, and cap score at 0.5 if the trap is broken
    return _base_score(
        agent_df, 
        manifest, 
        penalty_multiplier=20.0, 
        max_score_if_trap_broken=0.5
    )

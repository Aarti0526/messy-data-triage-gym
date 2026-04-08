from data_triage_env.graders.easy_grader import score as _base_score

def score(agent_df, manifest):
    return _base_score(agent_df, manifest, penalty_multiplier=15.0)

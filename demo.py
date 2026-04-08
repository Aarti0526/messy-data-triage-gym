import numpy as np
import pandas as pd
from data_triage_env.engine.dataset_factory import generate_clean
from data_triage_env.engine.corruptor import CORRUPT_FNS
from data_triage_env.engine.executor import run_action
from data_triage_env.models import DataAction
from data_triage_env.graders.easy_grader import score as score_easy

def show_demo_easy_fixed():
    seed = 42
    print("=== FULL LIFECYCLE DEMO: EASY TASK ===\n")
    
    # 1. Clean Data
    clean_df = generate_clean("easy", seed)
    print("✨ 1. THE GROUND TRUTH (Clean Data):")
    print(clean_df.head(10).to_string())
    print("-" * 60)
    
    # 2. Corrupt Data
    dirty_df, manifest = CORRUPT_FNS["easy"](clean_df.copy(), np.random.default_rng(seed))
    
    print("\n🐛 2. THE CORRUPTED DATA (What the AI sees initially):")
    print(dirty_df.head(10).to_string())
    
    # Calculate initial score (should be low)
    initial_score = score_easy(dirty_df, manifest)
    print(f"\n=> Initial Score: {initial_score:.2f} / 1.0")
    print("-" * 60)
    
    # 3. Apply Fix 1: Impute 'price' median
    print("\n🤖 3. AGENT ACTION 1: Impute 'price' with median")
    action1 = DataAction(action="impute", column="price", params={"strategy": "median"})
    dirty_df, obs1, msg1 = run_action(dirty_df, action1)
    
    print("\nData after Action 1:")
    print(dirty_df.head(10).to_string())
    score1 = score_easy(dirty_df, manifest)
    print(f"=> Score after Action 1: {score1:.2f} / 1.0")
    print("-" * 60)
    
    # 4. Apply Fix 2: Cast 'quantity' to fix the "N/A" strings
    print("\n🤖 4. AGENT ACTION 2: Cast 'quantity' to numeric (Int64 ignores NaNs) and then impute")
    # Actually, executor's cast pushes errors to NaN, so "N/A" becomes NaN.
    action2_a = DataAction(action="cast", column="quantity", params={"dtype": "float64"})
    dirty_df, obs2a, msg2a = run_action(dirty_df, action2_a)
    
    action2_b = DataAction(action="impute", column="quantity", params={"strategy": "median"})
    dirty_df, obs2b, msg2b = run_action(dirty_df, action2_b)
    
    # Finally cast back to Int64 to match clean data perfectly!
    action2_c = DataAction(action="cast", column="quantity", params={"dtype": "int64"})
    dirty_df, obs2c, msg2c = run_action(dirty_df, action2_c)
    
    print("\nData after Action 2 (Cast & Impute):")
    print(dirty_df.head(10).to_string())
    
    final_score = score_easy(dirty_df, manifest)
    print(f"\n✅ FINAL SCORE: {final_score:.2f} / 1.0")
    print("-" * 60)
    print("Agent successfully cleaned the data!")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    show_demo_easy_fixed()

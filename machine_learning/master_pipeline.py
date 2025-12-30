# master_pipeline.py
# Usage:
# source /work/Bachelor/env/bin/activate
# cd Bachelor
# python master_pipeline.py
#
# FAMD DATA LEAKAGE PREVENTION:
# This pipeline ensures FAMD is trained WITHOUT the outcome variable to prevent data leakage.
# Verification steps:
# 1. After splitting X and y, verify outcome not in X.columns
# 2. Save only X (features) to features_for_famd.parquet
# 3. After FAMD training, verify outcome not in famd.num_cols_ or famd.cat_cols_
# 4. If existing FAMD object is corrupted, delete and regenerate
# 5. FAMD multi-seed training uses parallel processing (n_jobs=-1) for speed

from scripts.data_preparation import (
    load_data,
    split_features_outcome,
    save_features_for_famd,
    apply_encodings_and_save,
    reload_with_metadata,
)

from scripts.famd_pipeline import run_famd, run_famd_with_seeds
from scripts.famd_visualization import log_mean_contributions_wandb, build_mean_contrib_from_parquet, visualize_famd
import joblib
import numpy as np
# NOTE: visualization disabled inside sweeps for stability
# from scripts.famd_visualization import visualize_famd

from scripts.train_model import train_and_evaluate_model
import wandb
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Paths & settings  ✅ YOU MAY EDIT THESE
# -------------------------------
RAW_PATH = "/work/Bachelor/data/processed/ml_encoded_df.parquet"
FAMD_INPUT_PATH = "/work/Bachelor/data/processed/features_for_famd.parquet"
FAMD_OUTPUT_PATH = "/work/Bachelor/outputs/models/df_famd.parquet"
FAMD_OBJ_PATH = "/work/Bachelor/outputs/models/famd_object.joblib"
FAMD_CONTRIB_PATH = "/work/Bachelor/outputs/models/famd_contributions.parquet"
MODEL_OUTPUT_DIR = "/work/Bachelor/outputs/models"

OUTCOME = "rotter_score_2"        # ✅ continuous variable (regression)
WANDB_PROJECT = "bachelor_project"
WANDB_ENTITY = None               # Set to your username if using team workspace, else None
MODEL_TYPE = "regression"          # continuous outcome

# Optional interim files (if you maintain an external encoding CSV)
INTERIM_PARQUET = "/work/Bachelor/data/interim/ml_ready_data.parquet"
INTERIM_ENCODINGS = "/work/Bachelor/data/interim/ml_ready_encodings.csv"

# -------------------------------
# Step 1: Initialize W&B (env file > key file > interactive)
# -------------------------------
# Preferred secure options: set the `WANDB_API_KEY` env var, or create a
# local key file at `~/.wandb_api_key` (file should be chmod 600). Avoid
# hard-coding the key into source files or committing it to version control.

# Allow optional entity via environment as well
WANDB_ENTITY = os.getenv("WANDB_ENTITY", WANDB_ENTITY)

# Look for API key in environment first, then a user-local key file
_wandb_key = os.getenv("WANDB_API_KEY")
if not _wandb_key:
    _key_file = os.path.expanduser("~/.wandb_api_key")
    if os.path.exists(_key_file):
        with open(_key_file, "r") as _f:
            _wandb_key = _f.read().strip()

# Fail fast if no key found (avoids interactive prompt during headless runs)
if not _wandb_key:
    raise SystemExit(
        "W&B API key not found. Set the environment variable WANDB_API_KEY or create ~/.wandb_api_key with your key (chmod 600).\n"
        "Example: export WANDB_API_KEY=\"<your_key>\"\n"
        "See README.md for full instructions."
    )

# Non-interactive login using provided key
wandb.login(key=_wandb_key)
print("W&B: logged in using WANDB_API_KEY / ~/.wandb_api_key")

# Create project if it doesn't exist (data prep logged once)
print(f"Initializing W&B project: {WANDB_PROJECT}")


# Only generate ml_encoded_df.parquet if it does not exist, but interim files do
if not os.path.exists(RAW_PATH):
    if os.path.exists(INTERIM_PARQUET) and os.path.exists(INTERIM_ENCODINGS):
        print("ml_encoded_df.parquet not found, but interim files detected — applying encodings and writing processed parquet...")
        apply_encodings_and_save(INTERIM_PARQUET, INTERIM_ENCODINGS, RAW_PATH)
    else:
        raise SystemExit(
            f"Neither {RAW_PATH} nor interim files found. Please provide data in one of these locations."
        )


# Load and prepare data
df = load_data(RAW_PATH)
df = reload_with_metadata(RAW_PATH)
print("Columns in loaded DataFrame:", df.columns.tolist())
if OUTCOME not in df.columns:
    raise SystemExit(f"Outcome variable '{OUTCOME}' not found in loaded DataFrame. Columns present: {df.columns.tolist()}")

# Exclude outcome AND ID columns (CASEID_1979 should never be used as a feature)
X, y = split_features_outcome(df, OUTCOME, exclude_cols=['CASEID_1979'])

# CRITICAL: Verify outcome variable and ID columns were properly excluded from features
if OUTCOME in X.columns:
    raise SystemExit(f"ERROR: Outcome variable '{OUTCOME}' still in features after split! Check split_features_outcome function.")
if 'CASEID_1979' in X.columns:
    raise SystemExit(f"ERROR: ID column 'CASEID_1979' still in features after split! This will cause data leakage.")
print(f"✓ Verified: Outcome '{OUTCOME}' and ID column 'CASEID_1979' excluded from features (X shape: {X.shape}, y shape: {y.shape})")

# Ensure categorical columns are Categorical dtype for FAMD consistency
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols) > 0:
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype('category')
save_features_for_famd(X, FAMD_INPUT_PATH)
print(f"✓ Saved features (without outcome) to {FAMD_INPUT_PATH}")

# Sanity check: require at least one categorical column (pipeline expects mixed data)
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
if len(cat_cols) == 0:
    raise SystemExit(
        "No categorical columns detected after loading data.\n"
        "This pipeline expects mixed-type data (categorical + numeric).\n"
        "Check your encodings CSV and interim parquet, or run apply_encodings_and_save manually.\n"
        "If you intentionally have no categorical variables, remove this check in master_pipeline.py."
    )

# Log data info to W&B (short run to record dataset stats)
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="data_prep", reinit=True)
wandb.log({"data_rows": int(len(X)), "n_features": int(X.shape[1]), "target_mean": float(y.mean()), "target_std": float(y.std())})
wandb.finish()

# -------------------------------
# Pre-sweep FAMD stability analysis (run once)
# -------------------------------
if not os.path.exists(FAMD_OBJ_PATH):
    print("Running pre-sweep FAMD stability analysis (multiple seeds with parallel processing)...")
    seeds = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    famd_stability = run_famd_with_seeds(
        input_path=FAMD_INPUT_PATH,
        n_components=50,
        seeds=seeds,
        contributions_aggregate_path=FAMD_CONTRIB_PATH,
        top_k=10,
        n_jobs=-1,  # Use all available CPU cores
    )

    mean_contrib = famd_stability['mean_contributions']
    per_seed = famd_stability['per_seed_contributions']
    seeds_list = famd_stability.get('seeds', list(seeds))

    # choose the seed whose contributions are closest to the mean (Frobenius norm)
    distances = []
    for df in per_seed:
        # ensure aligned columns/index same as mean_contrib
        aligned = df.reindex(index=mean_contrib.index, columns=mean_contrib.columns).fillna(0.0)
        dist = np.linalg.norm((aligned.values - mean_contrib.values))
        distances.append(dist)

    best_idx = int(np.argmin(distances))
    best_seed = int(seeds_list[best_idx])
    print(f"Chosen representative FAMD seed: {best_seed} (seed index {best_idx})")

    # Fit and save a deterministic FAMD object with that seed and save transformed features
    famd_result = run_famd(
        input_path=FAMD_INPUT_PATH,
        output_path=FAMD_OUTPUT_PATH,
        famd_obj_path=FAMD_OBJ_PATH,
        n_components=50,
        contributions_path=FAMD_CONTRIB_PATH,
        seed=best_seed,
    )
    print(f"Saved representative FAMD object to {FAMD_OBJ_PATH} and features to {FAMD_OUTPUT_PATH}")
    
    # CRITICAL: Verify FAMD object doesn't contain outcome variable
    import joblib
    famd_check = joblib.load(FAMD_OBJ_PATH)
    if hasattr(famd_check, 'num_cols_') and OUTCOME in famd_check.num_cols_:
        raise SystemExit(f"ERROR: FAMD object contains outcome variable '{OUTCOME}' in num_cols_! Data leakage detected.")
    if hasattr(famd_check, 'cat_cols_') and OUTCOME in famd_check.cat_cols_:
        raise SystemExit(f"ERROR: FAMD object contains outcome variable '{OUTCOME}' in cat_cols_! Data leakage detected.")
    print(f"✓ Verified: FAMD object trained without outcome variable")
    
    # Log mean contributions and top-k plots to W&B for quick inspection
    try:
        mc = build_mean_contrib_from_parquet(FAMD_CONTRIB_PATH)
        log_mean_contributions_wandb(mc, project=WANDB_PROJECT, entity=WANDB_ENTITY, name="famd_stability", top_k=10)
    except Exception as e:
        print(f"Warning: failed to log FAMD visualizations to W&B: {e}")
else:
    print(f"Found existing FAMD object at {FAMD_OBJ_PATH}; will reuse it for sweep runs.")
    
    # CRITICAL: Verify existing FAMD object doesn't contain outcome variable
    import joblib
    famd_check = joblib.load(FAMD_OBJ_PATH)
    if hasattr(famd_check, 'num_cols_') and OUTCOME in famd_check.num_cols_:
        print(f"⚠️  WARNING: Existing FAMD object contains outcome variable '{OUTCOME}'!")
        print(f"Deleting corrupted FAMD object and regenerating...")
        os.remove(FAMD_OBJ_PATH)
        if os.path.exists(FAMD_OUTPUT_PATH):
            os.remove(FAMD_OUTPUT_PATH)
        raise SystemExit("Deleted corrupted FAMD files. Please re-run master_pipeline.py to regenerate them correctly.")
    print(f"✓ Verified: Existing FAMD object is clean (no outcome variable)")

# -------------------------------
# Step 2: Sweep configuration
# -------------------------------
sweep_config = {
    "method": "random",
    "metric": {"name": "cv_r2_mean", "goal": "maximize"},
    "parameters": {
        "model_name": {"values": ["random_forest", "xgboost", "lightgbm", "catboost", "ridge", "elasticnet"]},
        "famd_components": {"values": [10, 20, 50]},
        "n_estimators": {"values": [100, 200]},
        "max_depth": {"values": [5, 10, 15, None]},
        "learning_rate": {"values": [0.01, 0.1]},
        "alpha": {"values": [0.001, 0.01, 0.1, 1.0, 10.0]},
        "l1_ratio": {"values": [0.1, 0.5, 0.9]},
        "l2_leaf_reg": {"values": [3, 10, 20] },
        "thread_count": {"values": [64] },
    },
}


# -------------------------------
# Step 3: Sweep training function
# -------------------------------
def sweep_train(config=None):
    # When running inside `wandb.agent` the agent provides the sweep's
    # project/entity; passing `project`/`entity` here causes a benign
    # W&B warning "Ignoring project". Let the agent set project/entity
    # and only pass the config (reinit True to allow multiple runs).
    wandb.init(config=config, reinit=True)
    cfg = wandb.config

    try:
        model_name = getattr(cfg, 'model_name', 'random_forest')

        # If using CatBoost, prefer raw features and provide categorical columns
        if model_name == 'catboost':
            # Convert categorical columns to string for CatBoost only
            X_cb = X.copy()
            cat_cols = X_cb.select_dtypes(include=['category', 'object']).columns.tolist()
            if len(cat_cols) > 0:
                for c in cat_cols:
                    X_cb[c] = X_cb[c].astype(str)


            # Extract l2_leaf_reg and thread_count if present, else default
            l2_leaf_reg = getattr(cfg, 'l2_leaf_reg', 10)
            thread_count = getattr(cfg, 'thread_count', 4)

            result = train_and_evaluate_model(
                X=X_cb,
                y=y,
                model_name='catboost',
                model_type=MODEL_TYPE,
                config={
                    'n_estimators': cfg.n_estimators,
                    'max_depth': cfg.max_depth,
                    'learning_rate': cfg.learning_rate,
                    'l2_leaf_reg': l2_leaf_reg,
                    'thread_count': thread_count
                },
                categorical_cols=cat_cols,
                output_dir=MODEL_OUTPUT_DIR
            )
            wandb.log({'note': 'Used raw features with CatBoost', 'n_cat_cols': len(cat_cols), 'l2_leaf_reg': l2_leaf_reg, 'thread_count': thread_count})

        else:
            # --- Use precomputed FAMD object if present, otherwise fit one for this run ---
            if os.path.exists(FAMD_OBJ_PATH):
                famd = joblib.load(FAMD_OBJ_PATH)
                # CRITICAL: Convert categorical columns to STRING dtype for FAMD transform
                # FAMD expects str, not pandas category dtype
                X_for_famd = X.copy()
                cat_cols = X_for_famd.select_dtypes(include=["object", "category"]).columns.tolist()
                if len(cat_cols) > 0:
                    for c in cat_cols:
                        X_for_famd[c] = X_for_famd[c].astype(str)
                X_famd = famd.transform(X_for_famd)
                X_famd.columns = [f"FAMD_{i+1}" for i in range(X_famd.shape[1])]
                # Log that we reused the precomputed FAMD
                wandb.log({'famd_used_precomputed': True})
            else:
                famd_result = run_famd(
                    input_path=FAMD_INPUT_PATH,
                    output_path=FAMD_OUTPUT_PATH,
                    famd_obj_path=FAMD_OBJ_PATH,
                    n_components=cfg.famd_components,
                    contributions_path=FAMD_CONTRIB_PATH
                )
                X_famd = famd_result['X_famd']

            # --- Train model with model-specific config ---
            # Build config based on model type
            if model_name in ['ridge', 'elasticnet']:
                # Linear models use alpha and l1_ratio
                model_config = {
                    'alpha': getattr(cfg, 'alpha', 1.0),
                }
                if model_name == 'elasticnet':
                    model_config['l1_ratio'] = getattr(cfg, 'l1_ratio', 0.5)
            else:
                # Tree-based models use n_estimators, max_depth, learning_rate
                model_config = {
                    'n_estimators': cfg.n_estimators,
                    'max_depth': cfg.max_depth,
                    'learning_rate': cfg.learning_rate
                }
            
            result = train_and_evaluate_model(
                X=X_famd,
                y=y,
                model_name=model_name,
                model_type=MODEL_TYPE,
                config=model_config,
                output_dir=MODEL_OUTPUT_DIR
            )

        print(f"Sweep completed successfully for model={model_name}, famd={getattr(cfg, 'famd_components', None)}, "
              f"n_est={cfg.n_estimators}, max_depth={cfg.max_depth}")

    except Exception as e:
        print(f"Error in sweep_train: {e}")
        raise

    finally:
        wandb.finish()

# -------------------------------
# Step 4: Run sweep
# -------------------------------
if __name__ == "__main__":
    print(f"Starting W&B sweep with 10 iterations...")
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train, count=10)
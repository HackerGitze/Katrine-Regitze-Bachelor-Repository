# followup_master_pipeline.py
"""
A follow-up pipeline for detailed hyperparameter sweep and final training for CatBoost and ElasticNet.
Builds on the structure of master_pipeline.py, but focuses on these two models with expanded hyperparameter grids,
W&B logging, learning curves, and SHAP plots.

Usage:
    python followup_master_pipeline.py elasticnet  # Run ElasticNet sweep
    python followup_master_pipeline.py catboost    # Run CatBoost sweep
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import wandb
import logging
from scripts.data_preparation import load_data, split_features_outcome, reload_with_metadata
from scripts.famd_pipeline import run_famd, run_famd_with_seeds
from scripts.train_model import train_and_evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Parse command line argument for model selection
if len(sys.argv) > 1:
    SELECTED_MODEL = sys.argv[1].lower()
    if SELECTED_MODEL not in ['lightgbm', 'catboost']:
        raise ValueError(f"Invalid model: {SELECTED_MODEL}. Choose 'lightgbm' or 'catboost'")
else:
    raise ValueError("Please specify model: python followup_master_pipeline.py [lightgbm|catboost]")

# Paths & settings
RAW_PATH = "/work/Bachelor/data/processed/ml_encoded_df.parquet"
FAMD_OBJ_PATH = "/work/Bachelor/outputs/models/famd_object.joblib"
MODEL_OUTPUT_DIR = "/work/Bachelor/outputs/models"
WANDB_PROJECT = "bachelor_followup"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
OUTCOME = "rotter_score_2"
MODEL_TYPE = "regression"


# Optional interim files
INTERIM_PARQUET = "/work/Bachelor/data/interim/ml_ready_data.parquet"
INTERIM_ENCODINGS = "/work/Bachelor/data/interim/ml_ready_classifications.csv"

# Only generate ml_encoded_df.parquet if it does not exist, but interim files do
if not os.path.exists(RAW_PATH):
    if os.path.exists(INTERIM_PARQUET) and os.path.exists(INTERIM_ENCODINGS):
        print("ml_encoded_df.parquet not found, but interim files detected — applying encodings and writing processed parquet...")
        from scripts.data_preparation import apply_encodings_and_save
        apply_encodings_and_save(INTERIM_PARQUET, INTERIM_ENCODINGS, RAW_PATH)
    else:
        raise SystemExit(
            f"Neither {RAW_PATH} nor interim files found. Please provide data in one of these locations."
        )
df = load_data(RAW_PATH)
df = reload_with_metadata(RAW_PATH)

# CRITICAL: Exclude CASEID_1979 to prevent data leakage
X, y = split_features_outcome(df, OUTCOME, exclude_cols=['CASEID_1979'])

# Verify CASEID_1979 was excluded
if 'CASEID_1979' in X.columns:
    raise ValueError("ERROR: CASEID_1979 still in features! This causes data leakage.")
print(f"✓ Verified: CASEID_1979 excluded from features")


# Ensure categorical columns are Categorical dtype for FAMD consistency
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols) > 0:
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype('category')

# If FAMD object does not exist, compute default FAMD components (e.g., 50)
DEFAULT_FAMD_COMPONENTS = 50
if not os.path.exists(FAMD_OBJ_PATH):
    print(f"FAMD object {FAMD_OBJ_PATH} not found. Computing new FAMD components...")
    from scripts.famd_pipeline import run_famd
    FAMD_INPUT_PATH = RAW_PATH
    FAMD_OUTPUT_PATH = f"/work/Bachelor/outputs/models/df_famd_{DEFAULT_FAMD_COMPONENTS}.parquet"
    famd_result = run_famd(
        input_path=FAMD_INPUT_PATH,
        output_path=FAMD_OUTPUT_PATH,
        famd_obj_path=FAMD_OBJ_PATH,
        n_components=DEFAULT_FAMD_COMPONENTS
    )

# Define separate sweep configs for each model
lightgbm_sweep_config = {
    "method": "bayes",
    "metric": {"name": "cv_r2_mean", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"values": [100, 200, 300, 400]},
        "max_depth": {"values": [4, 6, 8, 10]},
        "learning_rate": {"min": 0.01, "max": 0.2},
        "num_leaves": {"values": [15, 31, 63]},
        "min_child_samples": {"values": [10, 20, 30]},
        "famd_components": {"values": [20, 30, 40, 50]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
    }
}

catboost_sweep_config = {
    "method": "bayes",
    "metric": {"name": "cv_r2_mean", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"values": [100, 200, 300]},  # Reduced max to save memory
        "max_depth": {"values": [4, 6, 8]},  # Reduced max depth (12 uses too much GPU RAM)
        "learning_rate": {"min": 0.01, "max": 0.3},
        "l2_leaf_reg": {"min": 1.0, "max": 50.0},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
    }
}

# Select config based on command line argument
sweep_config = lightgbm_sweep_config if SELECTED_MODEL == 'lightgbm' else catboost_sweep_config
print(f"\n{'='*60}")
print(f"Running Bayesian hyperparameter sweep for: {SELECTED_MODEL.upper()}")
print(f"{'='*60}\n")

# Sweep training function
def sweep_train(config=None):
    wandb.init(config=config, reinit=True, project=f"{WANDB_PROJECT}_{SELECTED_MODEL}", entity=WANDB_ENTITY)
    cfg = wandb.config
    
    if SELECTED_MODEL == 'catboost':
        X_cb = X.copy()
        cat_cols_cb = X_cb.select_dtypes(include=['category', 'object']).columns.tolist()
        for c in cat_cols_cb:
            X_cb[c] = X_cb[c].astype(str)
        
        # Check if CatBoost has GPU support
        try:
            import subprocess
            subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Force GPU usage - if it fails, fall back to CPU
            try:
                from catboost import CatBoostRegressor
                # Create a minimal test to verify GPU works
                import numpy as np
                test_X = np.random.rand(100, 10)
                test_y = np.random.rand(100)
                test_model = CatBoostRegressor(task_type='GPU', devices='0', iterations=2, verbose=0)
                test_model.fit(test_X, test_y)
                use_gpu = True
                gpu_note = 'GPU-accelerated'
                print(f"  ✓ CatBoost GPU test successful - using GPUs 0 and 1")
            except Exception as e:
                use_gpu = False
                gpu_note = 'CPU (GPU test failed)'
                print(f"  ⚠️  GPU test failed, falling back to CPU: {str(e)[:100]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            use_gpu = False
            gpu_note = 'CPU (no GPU detected)'
            print(f"  Using CPU (no GPU hardware detected)")
        
        # Configure CatBoost based on GPU availability
        catboost_config = {
            'n_estimators': cfg.n_estimators,
            'max_depth': cfg.max_depth,
            'learning_rate': cfg.learning_rate,
            'l2_leaf_reg': cfg.l2_leaf_reg,
            'verbose': 100,  # Show progress every 100 iterations to see GPU activity
        }
        
        if use_gpu:
            catboost_config['task_type'] = 'GPU'
            catboost_config['devices'] = '0'  # Use single GPU (avoids multi-GPU memory issues)
            catboost_config['gpu_ram_part'] = 0.5  # Use only 50% of GPU memory (very conservative)
            catboost_config['pinned_memory_size'] = 1024 * 1024 * 1024  # 1GB pinned memory limit
        else:
            catboost_config['task_type'] = 'CPU'
        
        print(f"  DEBUG: Passing config to train_and_evaluate_model: {catboost_config}")
        
        result = train_and_evaluate_model(
            X=X_cb,
            y=y,
            model_name='catboost',
            model_type=MODEL_TYPE,
            config=catboost_config,
            categorical_cols=cat_cols_cb,
            output_dir=MODEL_OUTPUT_DIR
        )
        wandb.log({'note': f'CatBoost run ({gpu_note})', 'n_cat_cols': len(cat_cols_cb), 'use_gpu': use_gpu})
    else:
        # LightGBM with FAMD (dimensionality reduction from 3301 features)
        n_famd = getattr(cfg, 'famd_components', 50)
        from scripts.famd_pipeline import run_famd
        
        # Check if FAMD object exists, otherwise train it
        FAMD_OBJ_PATH_N = f"/work/Bachelor/outputs/models/famd_object_{n_famd}.joblib"
        if os.path.exists(FAMD_OBJ_PATH_N):
            # Load existing FAMD and transform
            famd = joblib.load(FAMD_OBJ_PATH_N)
            X_for_famd = X.copy()
            # Convert categoricals to string for FAMD consistency
            cat_cols_famd = X_for_famd.select_dtypes(include=["object", "category"]).columns.tolist()
            if len(cat_cols_famd) > 0:
                for c in cat_cols_famd:
                    X_for_famd[c] = X_for_famd[c].astype(str)
            X_famd = famd.transform(X_for_famd)
            X_famd.columns = [f"FAMD_{i+1}" for i in range(X_famd.shape[1])]
        else:
            # Train new FAMD
            from scripts.data_preparation import save_features_for_famd
            FAMD_INPUT_PATH = f"/work/Bachelor/data/processed/features_for_famd_{n_famd}.parquet"
            FAMD_OUTPUT_PATH = f"/work/Bachelor/outputs/models/df_famd_{n_famd}.parquet"
            # Save features with string-converted categoricals
            save_features_for_famd(X, FAMD_INPUT_PATH)
            famd_result = run_famd(
                input_path=FAMD_INPUT_PATH,
                output_path=FAMD_OUTPUT_PATH,
                famd_obj_path=FAMD_OBJ_PATH_N,
                n_components=n_famd
            )
            X_famd = famd_result['X_famd']
        
        result = train_and_evaluate_model(
            X=X_famd,
            y=y,
            model_name='lightgbm',
            model_type=MODEL_TYPE,
            config={
                'n_estimators': cfg.n_estimators,
                'max_depth': cfg.max_depth,
                'learning_rate': cfg.learning_rate,
                'num_leaves': cfg.num_leaves,
                'min_child_samples': cfg.min_child_samples,
                'random_state': 42,
                'verbose': -1
            },
            output_dir=MODEL_OUTPUT_DIR
        )
        wandb.log({'note': f'LightGBM run (FAMD {n_famd} components from 3301 features)'})
    # Only log serializable results (not model object)
    if 'metrics' in result:
        wandb.log(result['metrics'])
    if 'cv_results' in result:
        wandb.log({'cv_results': result['cv_results']})
    wandb.finish()

# Launch sweep
# With Bayesian optimization, different counts for different models
if SELECTED_MODEL == 'lightgbm':
    run_count = 50  # LightGBM: 7 hyperparameters, moderate exploration
else:
    run_count = 60  # CatBoost: 4 hyperparameters, more exploration needed

print(f"Starting {run_count} Bayesian optimization runs for {SELECTED_MODEL}...")
sweep_id = wandb.sweep(sweep_config, project=f"{WANDB_PROJECT}_{SELECTED_MODEL}", entity=WANDB_ENTITY)
wandb.agent(sweep_id, function=sweep_train, count=run_count)

# After sweep: Train best model on full data, plot learning curves and SHAP
# (Pseudo-code, to be filled in after sweep results are available)
# best_config = ... # Load from W&B or set manually
# model = train_and_evaluate_model(..., config=best_config, ...)
# plot_learning_curve(model, X, y, ...)
# plot_shap(model, X, ...)

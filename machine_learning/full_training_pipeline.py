"""
Full Training Pipeline - Final model training with optimized hyperparameters

This pipeline trains final models using the BEST hyperparameters found by 
followup_master_pipeline.py (Bayesian optimization with W&B).

Workflow:
1. Run hyperparameter sweeps first:
   python followup_master_pipeline.py lightgbm  # 50 runs
   python followup_master_pipeline.py catboost  # 60 runs

2. Get best hyperparameters from W&B:
   - Go to https://wandb.ai
   - Project: bachelor_followup_lightgbm (sort by cv_r2_mean, descending)
   - Project: bachelor_followup_catboost (sort by cv_r2_mean, descending)
   - Copy the hyperparameters from the best run

3. Run this pipeline with those hyperparameters:
   
   python full_training_pipeline.py lightgbm --n_estimators 300 --max_depth 8 ...
   python full_training_pipeline.py catboost --iterations 350 --depth 8 ...
   
   Or programmatically:
   from full_training_pipeline import main
   results = main(
       model='lightgbm',
       model_params={'n_estimators': 300, 'max_depth': 8, ...}
   )

This pipeline does NOT do hyperparameter tuning - it uses your optimized params.

Usage:
    python full_training_pipeline.py lightgbm  # Train LightGBM only
    python full_training_pipeline.py catboost  # Train CatBoost only
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, learning_curve
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np

# Import your custom SHAP plotting function
from scripts.train_model import custom_shap_summary_plot

# ========================================================================
# EDIT OPTIMIZED HYPERPARAMETERS HERE (from W&B sweeps)
# ========================================================================

OPTIMIZED_LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'famd_components': 40,  # From W&B sweep
    'random_state': 42,
    'verbose': -1
}

OPTIMIZED_CATBOOST_PARAMS = {
    'iterations': 350,
    'depth': 8,
    'learning_rate': 0.08,
    'l2_leaf_reg': 5,
    'random_state': 42,
    'verbose': 0
}

# ========================================================================

def ensure_file(path, compute_fn, *args, **kwargs):
    if not os.path.exists(path):
        print(f"File {path} not found. Generating...")
        return compute_fn(*args, **kwargs)
    else:
        print(f"File {path} found. Loading...")
        return joblib.load(path) if path.endswith('.joblib') else pd.read_parquet(path)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, n_splits=5, use_gpu=False):
    """
    Standard ML training pipeline:
    1. Cross-validation on training set
    2. Train final model on full training set
    3. Evaluate on holdout test set
    4. Generate diagnostics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Step 1: Cross-validation on training set ONLY
    print(f"\nStep 1: {n_splits}-Fold Cross-Validation on Training Set...")
    y_train_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    # When using GPU learners, avoid parallel CV workers that each try to use the GPU.
    cv_n_jobs = 1 if use_gpu else -1
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=kfold.split(X_train, y_train_binned),
        scoring=scoring,
        return_train_score=True,
        n_jobs=cv_n_jobs
    )
    
    cv_r2_mean = cv_results['test_r2'].mean()
    cv_r2_std = cv_results['test_r2'].std()
    cv_rmse_mean = np.sqrt(-cv_results['test_neg_mse'].mean())
    cv_rmse_std = np.sqrt(cv_results['test_neg_mse'].std())
    cv_mae_mean = -cv_results['test_neg_mae'].mean()
    
    print(f"  CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    print(f"  CV MAE: {cv_mae_mean:.4f}")
    
    # Step 2: Train final model on full training set
    print(f"\nStep 2: Training final model on full training set ({len(X_train)} samples)...")
    if isinstance(model, CatBoostRegressor):
        model.fit(X_train, y_train, verbose=0)
    else:
        # LGBMRegressor verbose is set in constructor, not in fit()
        model.fit(X_train, y_train)
    print(f"  Model trained successfully")
    
    # Step 3: Evaluate on holdout test set
    print(f"\nStep 3: Evaluating on holdout test set ({len(X_test)} samples)...")
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"  Holdout R²: {test_r2:.4f}")
    print(f"  Holdout RMSE: {test_rmse:.4f}")
    print(f"  Holdout MAE: {test_mae:.4f}")
    
    # Step 4: Generate diagnostics
    print(f"\nStep 4: Generating diagnostics...")
    
    # Learning curve using CV (proper method)
    # Use single-process learning curve when running on GPU to avoid GPU contention
    lc_n_jobs = 1 if use_gpu else -1
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=kfold.split(X_train, y_train_binned),
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='r2',
        n_jobs=lc_n_jobs
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes_abs, test_scores.mean(axis=1), 'o-', label='CV Score', linewidth=2)
    plt.fill_between(train_sizes_abs, 
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.2)
    plt.fill_between(train_sizes_abs,
                     test_scores.mean(axis=1) - test_scores.std(axis=1),
                     test_scores.mean(axis=1) + test_scores.std(axis=1),
                     alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title(f'{model_name} Learning Curve (Cross-Validation)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{model_name.lower()}_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Learning curve saved to outputs/{model_name.lower()}_learning_curve.png")
    
    # SHAP analysis on holdout test set
    try:
        print(f"  Generating SHAP summary plot...")
        custom_shap_summary_plot(model, X_train, X_test, model_type=model_name.lower(), 
                                feature_names=X_train.columns)
        print(f"  ✓ SHAP plot saved to outputs/{model_name.lower()}_shap_summary.png")
    except Exception as e:
        print(f"  ⚠️  SHAP plot generation failed: {e}")
        print(f"  Continuing without SHAP plot...")
    
    # Save model (always happens even if SHAP fails)
    model_path = f'outputs/{model_name.lower()}_final_model.joblib'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    metrics = {
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'cv_mae_mean': cv_mae_mean,
        'holdout_r2': test_r2,
        'holdout_rmse': test_rmse,
        'holdout_mae': test_mae
    }
    
    return model, metrics


def main(
    model='lightgbm',
    encoded_df_path='/work/Bachelor/data/processed/ml_encoded_df.parquet',
    famd_obj_path='/work/Bachelor/outputs/models/famd_object.joblib',
    famd_features_path='/work/Bachelor/outputs/models/famd_features.parquet',
    target_col='rotter_score_2',
    model_params=None,
    test_size=0.2,
    random_state=42,
    n_splits=5,
    use_gpu=False
):
    """
    Full training pipeline using OPTIMIZED hyperparameters from W&B sweeps.
    
    This pipeline does NOT do hyperparameter tuning - it uses the best parameters
    found by your followup_master_pipeline.py Bayesian optimization sweeps.
    
    Standard ML workflow:
    1. Load preprocessed data (FAMD features)
    2. Split into train/holdout FIRST (stratified by target quintiles)
    3. For selected model:
       - Cross-validate on training set (using optimized hyperparameters)
       - Train final model on full training set
       - Evaluate on holdout test set
       - Generate diagnostics (learning curves, SHAP) and save
    
    Args:
        model: Model to train ('lightgbm' or 'catboost')
        model_params: Dict with optimized hyperparameters from W&B sweep
                     LightGBM example: {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05,
                                       'num_leaves': 31, 'min_child_samples': 20, 'famd_components': 40, 'random_state': 42}
                     CatBoost example: {'iterations': 350, 'depth': 8, 'learning_rate': 0.08,
                                       'l2_leaf_reg': 5, 'random_state': 42, 'verbose': 0}
        
        Note: For LightGBM, include 'famd_components' in model_params (e.g., 20, 30, 40, or 50)
              This will load the FAMD model with that many components from your sweep
    
    Note: Get best hyperparameters from W&B dashboard:
          - bachelor_followup_lightgbm project (sort by cv_r2_mean)
          - bachelor_followup_catboost project (sort by cv_r2_mean)
    """
    print("="*60)
    print(f"FULL TRAINING PIPELINE - {model.upper()}")
    print("Using optimized hyperparameters from W&B sweeps")
    print("="*60)
    
    # Ensure encoded data and FAMD object
    def compute_encoded_df():
        from scripts.data_preparation import main as prep_main
        prep_main()  # Should save ml_encoded_df.parquet
        return pd.read_parquet(encoded_df_path)

    def compute_famd():
        from scripts.famd_pipeline import run_famd
        run_famd(encoded_df_path, famd_features_path, famd_obj_path, n_components=20, seed=random_state)
        return joblib.load(famd_obj_path)

    print("\nStep 1: Loading preprocessed data...")
    df = ensure_file(encoded_df_path, compute_encoded_df)
    y = df[target_col]
    
    # For LightGBM, load FAMD features with specified n_components
    # For CatBoost, use raw features (handles categoricals natively)
    if model == 'lightgbm':
        # Extract famd_components from model_params (will be removed before model init)
        n_famd = model_params.get('famd_components', 50) if model_params else 50
        
        # Load or compute FAMD with specified components
        famd_obj_path_n = f"/work/Bachelor/outputs/models/famd_object_{n_famd}.joblib"
        famd_features_path_n = f"/work/Bachelor/outputs/models/df_famd_{n_famd}.parquet"
        
        if os.path.exists(famd_features_path_n):
            print(f"  Loading FAMD features ({n_famd} components)...")
            X_famd = pd.read_parquet(famd_features_path_n)
        else:
            print(f"  Computing FAMD features ({n_famd} components)...")
            from famd_pipeline import run_famd
            # Prepare data for FAMD
            X_for_famd = df.drop(columns=[target_col])
            cat_cols = X_for_famd.select_dtypes(include=['object', 'category']).columns.tolist()
            for c in cat_cols:
                X_for_famd[c] = X_for_famd[c].astype(str)
            
            famd_input_path = f"/work/Bachelor/data/processed/features_for_famd_{n_famd}.parquet"
            X_for_famd.to_parquet(famd_input_path)
            
            from scripts.famd_pipeline import run_famd
            run_famd(
                input_path=famd_input_path,
                output_path=famd_features_path_n,
                famd_obj_path=famd_obj_path_n,
                n_components=n_famd
            )
            X_famd = pd.read_parquet(famd_features_path_n)
        
        X = X_famd
        print(f"  Data loaded: {X.shape[0]} samples, {X.shape[1]} FAMD features")
    else:  # catboost
        print(f"  Using raw features for CatBoost...")
        X = df.drop(columns=[target_col])
        print(f"  Data loaded: {X.shape[0]} samples, {X.shape[1]} raw features")
    
    # CRITICAL: Verify no ID columns - drop them with a warning rather than failing
    id_check_cols = ['CASEID_1979']
    found_id_cols = [c for c in id_check_cols if c in X.columns]
    if found_id_cols:
        print(f"WARNING: Found potential ID columns in features which may cause data leakage: {found_id_cols}")
        print("Dropping these columns from features before training.")
        try:
            X = X.drop(columns=found_id_cols)
            # record what we dropped for reproducibility
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/dropped_id_columns.txt', 'w') as fh:
                fh.write('\n'.join(found_id_cols))
            print(f"Dropped ID columns and recorded to outputs/dropped_id_columns.txt")
        except Exception as e:
            raise RuntimeError(f"Failed to drop ID columns {found_id_cols}: {e}")
    
    # Step 2: Split into train/holdout FIRST (stratified by target quintiles)
    print(f"\nStep 2: Creating train/holdout split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_binned
    )
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Holdout test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Holdout set will remain unseen until final evaluation")
    
    # Use hardcoded optimized hyperparameters from top of file
    if model_params is None:
        if model == 'lightgbm':
            model_params = OPTIMIZED_LIGHTGBM_PARAMS.copy()
            print(f"\n✓ Using optimized LightGBM hyperparameters (hardcoded):")
        else:  # catboost
            model_params = OPTIMIZED_CATBOOST_PARAMS.copy()
            print(f"\n✓ Using optimized CatBoost hyperparameters (hardcoded):")
        
        # Display params
        for k, v in model_params.items():
            if k not in ['random_state', 'verbose']:
                print(f"   {k}: {v}")
    else:
        print(f"\n✓ Using provided {model.upper()} hyperparameters: {model_params}")
        # Ensure required params are set
        if 'random_state' not in model_params:
            model_params['random_state'] = random_state
        if model == 'lightgbm' and 'verbose' not in model_params:
            model_params['verbose'] = -1
        elif model == 'catboost' and 'verbose' not in model_params:
            model_params['verbose'] = 0
    
    # Create model instance
    # Remove famd_components from model_params (not a model hyperparameter)
    model_params_clean = {k: v for k, v in model_params.items() if k != 'famd_components'}
    
    if model == 'lightgbm':
        model_instance = LGBMRegressor(**model_params_clean)
        model_name = 'LightGBM'
    else:  # catboost
        # If requested, enable GPU for CatBoost
        if use_gpu:
            # CatBoost uses task_type='GPU' to enable GPU training
            model_params_clean.setdefault('task_type', 'GPU')
            # Optionally specify devices (e.g., '0') if needed
            # model_params_clean.setdefault('devices', '0')
        model_instance = CatBoostRegressor(**model_params_clean)
        model_name = 'CatBoost'
    
    # Train and evaluate the selected model
    trained_model, metrics = train_and_evaluate_model(
        model_instance, model_name, X_train, X_test, y_train, y_test, n_splits, use_gpu=use_gpu
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{model_name}:")
    print(f"  CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(f"  Holdout R²: {metrics['holdout_r2']:.4f}")
    print(f"  Holdout RMSE: {metrics['holdout_rmse']:.4f}")
    print(f"  Holdout MAE: {metrics['holdout_mae']:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({model_name: metrics}).T
    metrics_path = f'outputs/{model}_final_metrics.csv'
    metrics_df.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Model and diagnostics saved to outputs/ directory")
    
    return {
        'model': trained_model,
        'metrics': metrics,
        'model_type': model
    }

if __name__ == "__main__":
    # Parse command line argument for model selection
    if len(sys.argv) > 1:
        selected_model = sys.argv[1].lower()
        if selected_model not in ['lightgbm', 'catboost']:
            print(f"Error: Invalid model '{selected_model}'")
            print("Usage: python full_training_pipeline.py [lightgbm|catboost]")
            sys.exit(1)
    else:
        print("Error: No model specified")
        print("Usage: python full_training_pipeline.py [lightgbm|catboost]")
        sys.exit(1)
    
    # Determine if user requested GPU (e.g.,: python full_training_pipeline.py catboost --use-gpu)
    use_gpu_flag = '--use-gpu' in sys.argv
    if use_gpu_flag:
        print('INFO: GPU mode requested (use_gpu=True)')
    # Run pipeline for selected model
    results = main(model=selected_model, use_gpu=use_gpu_flag)

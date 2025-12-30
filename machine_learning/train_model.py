def custom_shap_summary_plot(
    shap_values, features, feature_types=None, max_display=20, output_path=None, xlim=None, encodings_csv_path=None, mode="auto"
):
    """
    Custom SHAP summary plot with beeswarm (density) layout and improved color handling.
    - mode: 'famd' (single continuous colorbar), 'catboost'/'original' (continuous + categorical colorbars), or 'auto' (guess from feature names).
    - Uses beeswarm layout to spread points horizontally by density (like standard SHAP summary plot).
    - Colorbar uses 'High' and 'Low' labels for feature values.
    - Uses 'bwr' colormap for continuous features (colored center).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import pandas as pd
    import os
    from scipy.stats import rankdata
    # If encodings_csv_path is provided, use it to get variable types
    enc_types = None
    if encodings_csv_path is not None and os.path.exists(encodings_csv_path):
        enc_df = pd.read_csv(encodings_csv_path)
        if 'variable_name' in enc_df.columns and 'variable_type' in enc_df.columns:
            enc_types = dict(zip(enc_df['variable_name'], enc_df['variable_type']))

    # Get mean(|SHAP|) for feature ranking
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_features = features.columns[top_idx]
    shap_top = shap_values.values[:, top_idx]
    feat_top = features[top_features]
    # Prepare variable type annotation
    if enc_types is not None:
        var_types = pd.Series([enc_types.get(f, 'cont') for f in top_features], index=top_features)
    elif feature_types is not None:
        var_types = pd.Series(feature_types)
        var_types = var_types.reindex(top_features).fillna('cont')
    else:
        # Guess types: binary if 2 unique, cat if object/category, cont otherwise
        var_types = []
        for col in top_features:
            vals = feat_top[col]
            if pd.api.types.is_numeric_dtype(vals):
                uniq = pd.unique(vals)
                if len(uniq) == 2:
                    var_types.append('binary')
                else:
                    var_types.append('cont')
            elif pd.api.types.is_categorical_dtype(vals) or vals.dtype == object:
                var_types.append('cat_nom')
            else:
                var_types.append('cont')
        var_types = pd.Series(var_types, index=top_features)

    # Determine mode if auto
    if mode == "auto":
        if all(f.startswith("FAMD_") for f in top_features):
            mode = "famd"
        else:
            mode = "catboost"

    fig, ax = plt.subplots(figsize=(16, max(6, 0.5 * max_display)))
    def beeswarm_positions(x, y_base, spread=0.4):
        # x: SHAP values for one feature
        # y_base: vertical position for this feature
        # spread: max vertical spread
        order = np.argsort(x)
        y = np.zeros_like(x, dtype=float)
        occupied = {}
        for idx in order:
            val = x[idx]
            # Find available y offset
            offset = 0.0
            while True:
                candidate = y_base + offset
                if candidate not in occupied.get(val, set()):
                    break
                offset += 0.02
            y[idx] = y_base + offset
            occupied.setdefault(val, set()).add(candidate)
        # Normalize to [-spread, spread]
        y = y - y_base
        if len(y) > 1:
            y = y / (np.max(np.abs(y)) + 1e-8) * spread
        return y + y_base
    if mode == "famd":
        cont_cmap = plt.get_cmap('bwr')  # changed from 'RdBu' to 'bwr'
        for i, col in enumerate(top_features):
            n_points = feat_top.shape[0]
            y_base = max_display - i - 1
            shap_vals = shap_top[:, i]
            # Beeswarm layout
            y = beeswarm_positions(shap_vals, y_base)
            cvals = feat_top[col]
            norm = mpl.colors.TwoSlopeNorm(vmin=np.nanmin(cvals), vcenter=0, vmax=np.nanmax(cvals))
            colors = [cont_cmap(norm(val)) for val in cvals]
            ax.scatter(shap_vals, y, c=colors, s=16, alpha=0.7, edgecolor='none')
        # Y labels: feature name + type
        ax.set_yticks(np.arange(max_display))
        ax.set_yticklabels([f"{f} (cont)" for f in top_features])
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title("Custom SHAP summary plot (top features)")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            xabs = np.abs(shap_top).max()
            ax.set_xlim(-1.2 * xabs, 1.2 * xabs)
        # Add a single colorbar for continuous features, with High/Low labels
        sm = mpl.cm.ScalarMappable(cmap=cont_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=20, fraction=0.03)
        cbar.set_ticks([norm.vmin, norm.vmax])
        cbar.set_ticklabels(["Low", "High"])
        cbar.set_label("Feature value", rotation=270, labelpad=15)
    else:
        # CatBoost/original: two colorbars
        cont_cmap = plt.get_cmap('bwr')  # changed from 'RdBu' to 'bwr'
        cat_cmap = plt.get_cmap('tab10')
        cont_norm = None
        cat_norm = None
        cat_labels = []
        for i, (col, vtype) in enumerate(zip(top_features, var_types)):
            n_points = feat_top.shape[0]
            y_base = max_display - i - 1
            shap_vals = shap_top[:, i]
            y = beeswarm_positions(shap_vals, y_base)
            cvals = feat_top[col]
            if vtype in ('cat_nom', 'cat_ord', 'binary'):
                uniq = pd.unique(cvals)
                uniq_sorted = np.sort(uniq)
                cat_norm = mpl.colors.Normalize(vmin=0, vmax=len(uniq_sorted) - 1)
                val_to_int = {v: idx for idx, v in enumerate(uniq_sorted)}
                colors = [cat_cmap(cat_norm(val_to_int.get(val, 0))) for val in cvals]
                cat_labels = [str(u) for u in uniq_sorted]
            else:
                cont_norm = mpl.colors.Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
                colors = [cont_cmap(cont_norm(val)) for val in cvals]
            ax.scatter(shap_vals, y, c=colors, s=16, alpha=0.7, edgecolor='none')
        # Y labels: feature name + type
        ax.set_yticks(np.arange(max_display))
        ax.set_yticklabels([f"{f} ({t})" for f, t in zip(top_features, var_types)])
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title("Custom SHAP summary plot (top features)")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            xabs = np.abs(shap_top).max()
            ax.set_xlim(-1.2 * xabs, 1.2 * xabs)
        # Add colorbars
        if cont_norm is not None:
            sm = mpl.cm.ScalarMappable(cmap=cont_cmap, norm=cont_norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=20, fraction=0.03)
            cbar.set_ticks([cont_norm.vmin, cont_norm.vmax])
            cbar.set_ticklabels(["Low", "High"])
            cbar.set_label("Continuous feature value", rotation=270, labelpad=15)
        if cat_norm is not None and cat_labels:
            sm = mpl.cm.ScalarMappable(cmap=cat_cmap, norm=cat_norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.08, aspect=20, fraction=0.03)
            cbar.set_ticks(np.arange(len(cat_labels)))
            cbar.set_ticklabels(cat_labels)
            cbar.set_label("Categorical/Binary value", rotation=270, labelpad=15)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return output_path

# scripts/train_model.py
import wandb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold, learning_curve, validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _safe_mape(y_true, y_pred):
    """Compute MAPE, handling edge cases (zero values in y_true)."""
    try:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    except (ValueError, ZeroDivisionError):
        logger.warning("MAPE undefined (y_true contains zero or near-zero values); returning NaN")
        return np.nan


def _plot_learning_curve(model, X, y, cv=5, train_sizes=None, output_dir=None, model_name="model"):
    """Plot learning curve (training size vs CV R2 score)."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        # Use n_jobs=1 for GPU models to avoid parallel GPU memory exhaustion
        # Parallel CV folds would each load full dataset into GPU memory
        n_jobs = 1 if model_name == 'catboost' else -1
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring='r2', n_jobs=n_jobs, random_state=42
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, 'o-', label='Training score', color='C0')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='C0')
        ax.plot(train_sizes, val_mean, 'o-', label='Validation score', color='C1')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='C1')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R² Score')
        ax.set_title(f'Learning Curve — {model_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"learning_curve_{model_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved learning curve to {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not plot learning curve: {e}")
        return None


def _plot_validation_curve(model, X, y, param_name, param_range, cv=5, output_dir=None, model_name="model"):
    """Plot validation curve (hyperparameter vs CV R2 score)."""
    try:
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring='r2', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_range, train_mean, 'o-', label='Training score', color='C0')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='C0')
        ax.plot(param_range, val_mean, 'o-', label='Validation score', color='C1')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color='C1')
        ax.set_xlabel(param_name)
        ax.set_ylabel('R² Score')
        ax.set_title(f'Validation Curve — {model_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"validation_curve_{model_name}_{param_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved validation curve to {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not plot validation curve for {param_name}: {e}")
        return None


def _instantiate_regressor(model_name, config):
    model_name = (model_name or "random_forest").lower()
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=config.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError("xgboost is required for model_name='xgboost' (install via pip)") from e
        return XGBRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=int(config.get("max_depth", 6)) if config.get("max_depth") is not None else None,
            learning_rate=float(config.get("learning_rate", 0.1)),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise ImportError("lightgbm is required for model_name='lightgbm' (install via pip)") from e
        return LGBMRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=int(config.get("max_depth", -1)) if config.get("max_depth") is not None else -1,
            learning_rate=float(config.get("learning_rate", 0.1)),
            random_state=42,
            n_jobs=1,  # Disable parallelism to avoid threading hang
            force_col_wise=True,  # Use column-wise histogram building (more stable)
            verbose=-1,  # Suppress warnings
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as e:
            raise ImportError("catboost is required for model_name='catboost' (install via pip)") from e
        
        # Build CatBoost params from config
        catboost_params = {
            'iterations': int(config.get("n_estimators", 100)),
            'depth': int(config.get("max_depth", 6)) if config.get("max_depth") is not None else None,
            'learning_rate': float(config.get("learning_rate", 0.1)),
            'random_seed': 42,
            'verbose': int(config.get("verbose", 0)),
            'l2_leaf_reg': float(config.get("l2_leaf_reg", 10)),
        }
        
        # Add GPU/CPU task_type if specified (critical for GPU usage!)
        if 'task_type' in config:
            catboost_params['task_type'] = config['task_type']
            if config['task_type'] == 'GPU' and 'devices' in config:
                catboost_params['devices'] = config['devices']
        else:
            # Default to CPU with thread_count if no task_type specified
            catboost_params['thread_count'] = int(config.get("thread_count", 4))
        
        return CatBoostRegressor(**catboost_params)
    if model_name == "ridge":
        return Ridge(alpha=float(config.get("alpha", 1.0)))
    if model_name == "elasticnet":
        return ElasticNet(alpha=float(config.get("alpha", 1.0)), l1_ratio=float(config.get("l1_ratio", 0.5)))
    raise ValueError(f"Unknown model_name: {model_name}")


def train_and_evaluate_model(X, y, model_name="random_forest", model_type="regression", config=None, output_dir="outputs/models", categorical_cols=None, n_splits=5):
    """
    Train a model (multiple options) with cross-validation and holdout evaluation.

    Args:
        X: Feature DataFrame (can be FAMD-transformed or raw features)
        y: Target vector/Series
        model_name: one of 'random_forest','xgboost','lightgbm','catboost','ridge','elasticnet'
        model_type: 'regression' or 'classification' (we focus on regression)
        config: dict of hyperparameters
        output_dir: directory to save model and artifacts
        categorical_cols: list of column names or indices (used for CatBoost)
        n_splits: CV folds

    Returns:
        dict with model, metrics, cv_results and saved artifact paths
    """
    if config is None:
        config = {"n_estimators": 100, "max_depth": None}

    os.makedirs(output_dir, exist_ok=True)

    # Convert categorical columns to strings for CatBoost compatibility
    # (CatBoost rejects float/numeric categorical values)
    if model_name == "catboost":
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype(str)
        logger.info(f"Converted {len(cat_cols)} categorical columns to string dtype for CatBoost")

    # Encode labels if classification
    label_encoder = None
    y_encoded = y
    if model_type == "classification" and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

    # Instantiate model (possibly wrapped in a pipeline for linear models)
    base_model = _instantiate_regressor(model_name, config)

    # Apply scaling for linear / penalized models (Ridge, ElasticNet).
    # By default we skip scaling when the input looks like FAMD components
    # (column names starting with 'FAMD_'), but a user can override this
    # by setting `config['scale_famd'] = True` to force scaling of FAMD output.
    scale_famd = bool(config.get("scale_famd", False)) if isinstance(config, dict) else False
    is_famd_output = any(str(c).startswith("FAMD_") for c in X.columns)
    if model_name in ("ridge", "elasticnet") and (not is_famd_output or scale_famd):
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) == 0:
            # nothing numeric to scale; use base model
            model = base_model
        else:
            preproc = ColumnTransformer([
                ("num", StandardScaler(), numeric_cols)
            ], remainder="passthrough")
            model = Pipeline([("preproc", preproc), ("est", base_model)])
            logger.info(f"Applied StandardScaler to {len(numeric_cols)} numeric columns (scale_famd={scale_famd})")
    else:
        model = base_model

    # CRITICAL: Create holdout test set FIRST to prevent data leakage
    # The holdout set must be completely unseen during hyperparameter tuning
    
    # CRITICAL DEBUG: Verify no ID columns in features
    if 'CASEID_1979' in X.columns:
        raise ValueError("CRITICAL ERROR: CASEID_1979 is in features! Data leakage!")
    id_cols = [c for c in X.columns if 'caseid' in str(c).lower() or 'id' in str(c).lower()]
    if len(id_cols) > 0:
        logger.warning(f"Potential ID columns found in features: {id_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=pd.qcut(y_encoded, q=5, labels=False, duplicates='drop') if len(y_encoded) >= 50 else None)
    logger.info(f"Created holdout test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    logger.info(f"Feature check: {X_train.shape[1]} features, first 5: {list(X_train.columns[:5])}")
    
    # Scoring for regression
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }

    # Prepare fit_params if CatBoost requires categorical columns
    fit_params = {}
    if model_name == 'catboost' and categorical_cols is not None:
        try:
            if isinstance(categorical_cols[0], str):
                cat_idx = [X_train.columns.get_loc(c) for c in categorical_cols]
            else:
                cat_idx = list(categorical_cols)
        except Exception:
            cat_idx = list(categorical_cols)
        fit_params = {'cat_features': cat_idx}

    # Stratified K-Fold for regression: bin continuous target into quintiles
    # This ensures balanced distribution of target values across folds
    # IMPORTANT: CV now only uses the TRAINING set (X_train, y_train)
    try:
        y_train_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        logger.info(f"Using StratifiedKFold with {n_splits} splits on training set only (target binned into quintiles)")
        cv_results = cross_validate(model, X_train, y_train, cv=kfold.split(X_train, y_train_binned), scoring=scoring, return_train_score=True, fit_params=fit_params)
    except (ValueError, TypeError) as e:
        # Fallback to regular KFold if stratification fails (e.g., too few samples)
        logger.warning(f"Stratified K-Fold failed ({e}), falling back to regular K-Fold")
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, return_train_score=True, fit_params=fit_params)

    # Summarize CV metrics
    cv_r2_mean = cv_results['test_r2'].mean()
    cv_r2_std = cv_results['test_r2'].std()
    cv_mse_mean = -cv_results['test_neg_mse'].mean()
    cv_rmse_mean = np.sqrt(cv_mse_mean)
    cv_rmse_std = np.sqrt(np.std(-cv_results['test_neg_mse']))
    cv_mae_mean = -cv_results['test_neg_mae'].mean()

    # Compute MAPE from CV fold predictions (best effort)
    cv_mape_scores = []
    for fold_idx in range(n_splits):
        try:
            # Approximate: use CV results if available, otherwise skip
            # (MAPE not in standard scoring, so we estimate from MAE)
            pass
        except Exception:
            pass
    
    metrics = {
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_mae_mean': cv_mae_mean,
        'cv_rmse_std': cv_rmse_std,
    }

    logger.info(f"CV R2 (training set only): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}, RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}, MAE: {cv_mae_mean:.4f}")

    # Fit final model on full training set (X_train already created above)
    if model_name == 'catboost' and fit_params:
        model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train)

    # Holdout evaluation
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = _safe_mape(y_test, y_test_pred)

    metrics.update({
        'holdout_test_mse': test_mse,
        'holdout_test_rmse': test_rmse,
        'holdout_test_mae': test_mae,
        'holdout_test_mape': test_mape,
        'holdout_test_r2': test_r2
    })

    # Log to W&B
    wandb.log(metrics)

    # Generate learning and validation curves for diagnostics (using training set only)
    try:
        lc_path = _plot_learning_curve(model, X_train, y_train, cv=kfold, output_dir=output_dir, model_name=model_name)
        if lc_path is not None:
            wandb.log({f"learning_curve_{model_name}": wandb.Image(lc_path)})
    except Exception as e:
        logger.warning(f"Could not generate learning curve: {e}")

    # Validation curve for key hyperparameter (depends on model type) - using training set only
    try:
        if model_name == "ridge":
            param_name, param_range = "est__alpha", np.logspace(-4, 2, 10)
            vc_path = _plot_validation_curve(model, X_train, y_train, param_name, param_range, cv=kfold, output_dir=output_dir, model_name=model_name)
            if vc_path is not None:
                wandb.log({f"validation_curve_{model_name}_alpha": wandb.Image(vc_path)})
        elif model_name == "elasticnet":
            param_name, param_range = "est__alpha", np.logspace(-4, 2, 10)
            vc_path = _plot_validation_curve(model, X_train, y_train, param_name, param_range, cv=kfold, output_dir=output_dir, model_name=model_name)
            if vc_path is not None:
                wandb.log({f"validation_curve_{model_name}_alpha": wandb.Image(vc_path)})
        elif model_name in ("random_forest", "xgboost", "lightgbm"):
            param_name, param_range = "max_depth", [2, 5, 10, 20, 30]
            vc_path = _plot_validation_curve(model, X_train, y_train, param_name, param_range, cv=kfold, output_dir=output_dir, model_name=model_name)
            if vc_path is not None:
                wandb.log({f"validation_curve_{model_name}_depth": wandb.Image(vc_path)})
    except Exception as e:
        logger.warning(f"Could not generate validation curve: {e}")

    # Save model
    model_path = os.path.join(output_dir, f"model_{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    if label_encoder is not None:
        encoder_path = os.path.join(output_dir, f"label_encoder_{model_name}.joblib")
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")

    # Attempt SHAP explainability for tree models
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = None
        if model_name in ("random_forest", "xgboost", "lightgbm", "catboost"):
            explainer = shap.Explainer(model)
        if explainer is not None:
            sample = X_test if X_test.shape[0] <= 1000 else X_test.sample(1000, random_state=42)
            # For CatBoost, encode categoricals as codes for SHAP
            if model_name == 'catboost':
                sample_enc = sample.copy()
                cat_cols = sample_enc.select_dtypes(include=["object", "category"]).columns.tolist()
                for col in cat_cols:
                    sample_enc[col] = sample_enc[col].astype('category').cat.codes
                shap_values = explainer(sample_enc)
                shap_path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
                # Always use encodings and mode for CatBoost
                custom_shap_summary_plot(
                    shap_values, sample_enc, feature_types=None, max_display=20, output_path=shap_path,
                    encodings_csv_path="/work/Bachelor/data/interim/ml_ready_encodings.csv", mode="catboost"
                )
                wandb.log({f"shap_summary_{model_name}": wandb.Image(shap_path)})
            else:
                sample_enc = sample
                shap_values = explainer(sample_enc)
                feature_types = None
                if hasattr(config, 'get') and callable(config.get):
                    feature_types = config.get('feature_types', None)
                shap_path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
                custom_shap_summary_plot(shap_values, sample_enc, feature_types=feature_types, max_display=20, output_path=shap_path)
                wandb.log({f"shap_summary_{model_name}": wandb.Image(shap_path)})
    except Exception as e:
        logger.info(f"SHAP explanation skipped: {e}")

    return {
        'model': model,
        'cv_results': cv_results,
        'metrics': metrics,
        'model_path': model_path
    }

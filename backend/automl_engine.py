import pandas as pd
import numpy as np
import joblib
import optuna
import shap
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

class FullAutoMLEngine:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.raw_df = df
        self.target_column = target_column
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' missing from dataset.")
            
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.task_type = self._detect_task()
        self.preprocessor = self._build_preprocessor()

    def _detect_task(self) -> str:
        if self.y.dtype == 'object' or self.y.nunique() < 20:
            return "classification"
        return "regression"

    def _build_preprocessor(self) -> ColumnTransformer:
        num_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ]
        )

    def extract_3d_pca(self) -> List[Dict[str, Any]]:
        X_trans = self.preprocessor.fit_transform(self.X)
        pca = PCA(n_components=3, random_state=42)
        reduced = pca.fit_transform(X_trans)
        
        points = []
        labels = self.y.astype(str).tolist()
        for idx in range(min(len(reduced), 400)):
            points.append({
                "x": float(reduced[idx, 0]),
                "y": float(reduced[idx, 1]),
                "z": float(reduced[idx, 2]),
                "label": labels[idx]
            })
        return points

    def benchmark_and_tune(self, n_trials: int = 15) -> Tuple[Pipeline, Dict[str, Any]]:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if self.task_type == "classification" else KFold(n_splits=3, shuffle=True, random_state=42)
        metric = "f1_weighted" if self.task_type == "classification" else "neg_root_mean_squared_error"

        if self.task_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForest": RandomForestClassifier(n_estimators=40, random_state=42),
                "LightGBM": LGBMClassifier(verbosity=-1, random_state=42)
            }
        else:
            models = {
                "Ridge": Ridge(),
                "RandomForest": RandomForestRegressor(n_estimators=40, random_state=42),
                "LightGBM": LGBMRegressor(verbosity=-1, random_state=42)
            }

        leaderboard = {}
        for name, model in models.items():
            pipe = Pipeline([('preprocessor', self.preprocessor), ('model', model)])
            score = np.mean(cross_val_score(pipe, self.X, self.y, cv=cv, scoring=metric))
            leaderboard[name] = float(score)

        best_model_name = max(leaderboard, key=leaderboard.get)
        X_trans = self.preprocessor.fit_transform(self.X)

        def objective(trial):
            if "LightGBM" in best_model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 25, 120),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 45),
                    'random_state': 42,
                    'verbose': -1
                }
                est = LGBMClassifier(**params) if self.task_type == "classification" else LGBMRegressor(**params)
            elif "RandomForest" in best_model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 30, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'random_state': 42
                }
                est = RandomForestClassifier(**params) if self.task_type == "classification" else RandomForestRegressor(**params)
            else:
                c_val = trial.suggest_float('C', 0.1, 5.0, log=True) if self.task_type == "classification" else trial.suggest_float('alpha', 0.1, 5.0, log=True)
                est = LogisticRegression(C=c_val, max_iter=500) if self.task_type == "classification" else Ridge(alpha=c_val)

            return np.mean(cross_val_score(est, X_trans, self.y, cv=cv, scoring=metric))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        if "LightGBM" in best_model_name:
            final_est = LGBMClassifier(**study.best_params, verbose=-1, random_state=42) if self.task_type == "classification" else LGBMRegressor(**study.best_params, verbose=-1, random_state=42)
        elif "RandomForest" in best_model_name:
            final_est = RandomForestClassifier(**study.best_params, random_state=42) if self.task_type == "classification" else RandomForestRegressor(**study.best_params, random_state=42)
        else:
            final_est = LogisticRegression(**study.best_params, max_iter=500) if self.task_type == "classification" else Ridge(**study.best_params)

        final_pipe = Pipeline([('preprocessor', self.preprocessor), ('model', final_est)])
        final_pipe.fit(self.X, self.y)

        shap_summary = {}
        try:
            if "LightGBM" in best_model_name or "RandomForest" in best_model_name:
                fitted_model = final_pipe.named_steps['model']
                explainer = shap.TreeExplainer(fitted_model)
                shap_vals = explainer.shap_values(X_trans[:100])
                if isinstance(shap_vals, list):
                    shap_vals = np.abs(shap_vals[0]).mean(axis=0)
                else:
                    shap_vals = np.abs(shap_vals).mean(axis=0)
                feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
                top_indices = np.argsort(shap_vals)[-8:][::-1]
                shap_summary = {feature_names[i]: float(shap_vals[i]) for i in top_indices}
        except Exception:
            shap_summary = {"status": "Fallback to native importance"}

        metrics = {
            "task_type": self.task_type,
            "leaderboard": leaderboard,
            "best_model": best_model_name,
            "tuned_score": float(study.best_value),
            "best_params": study.best_params,
            "shap_importance": shap_summary
        }
        return final_pipe, metrics
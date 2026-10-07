import io
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import optuna

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI(title="DataPulse Universal AutoML Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "online", "engine": "DataPulse AutoML"}

@app.post("/api/v1/execute")
async def execute_pipeline(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    # 1. Read input dataset
    try:
        contents = await file.read()
        # Handle standard CSV or tab/comma text files
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), sep=r'\s+')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")

    # Clean header whitespaces
    df.columns = df.columns.str.strip()
    target_clean = target_column.strip()

    if target_clean not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Target column '{target_clean}' not found in uploaded file. Available columns: {list(df.columns)}"
        )

    # 2. Drop rows where target is missing
    df = df.dropna(subset=[target_clean])
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 10 valid rows.")

    y = df[target_clean]
    X = df.drop(columns=[target_clean])

    # 3. Detect Task Type (Classification vs. Regression)
    is_numeric = pd.api.types.is_numeric_dtype(y)
    unique_count = y.nunique()

    if not is_numeric or unique_count <= 10 or (unique_count / len(y) < 0.05 and unique_count < 20):
        task_type = "classification"
        # Convert target to string categories for consistent class handling
        y = y.astype(str)
        classes = sorted(y.unique().tolist())
    else:
        task_type = "regression"
        y = y.astype(float)
        classes = []

    # 4. Partition Numerical vs Categorical features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Fit preprocessor
    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    # 5. Compute Orthogonal 3D PCA Space
    n_components = min(3, X_processed.shape[1], X_processed.shape[0])
    pca = PCA(n_components=3 if X_processed.shape[1] >= 3 else n_components)
    
    if X_processed.shape[1] >= 3:
        coords = pca.fit_transform(X_processed)
        pc1, pc2, pc3 = coords[:, 0], coords[:, 1], coords[:, 2]
    elif X_processed.shape[1] == 2:
        coords = pca.fit_transform(X_processed)
        pc1, pc2, pc3 = coords[:, 0], coords[:, 1], np.zeros(len(coords))
    else:
        pc1 = X_processed[:, 0]
        pc2 = np.zeros(len(X_processed))
        pc3 = np.zeros(len(X_processed))

    # Standardize 3D point cloud scale to fit Three.js canvas nicely
    max_abs = max(np.max(np.abs(pc1)), np.max(np.abs(pc2)), np.max(np.abs(pc3)), 1e-5)
    pc1_norm = (pc1 / max_abs) * 3.0
    pc2_norm = (pc2 / max_abs) * 3.0
    pc3_norm = (pc3 / max_abs) * 3.0

    points_3d = []
    # Cap visualization at 400 sample points to keep rendering fast
    max_pts = min(400, len(df))
    for i in range(max_pts):
        points_3d.append({
            "x": float(pc1_norm[i]),
            "y": float(pc2_norm[i]),
            "z": float(pc3_norm[i]),
            "label": str(y.iloc[i])
        })

    # 6. Model Tournament (Cross-Validation)
    cv_splits = 3 if len(df) < 50 else 5
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scoring = "f1_weighted"
        candidates = {
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42)
        }
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scoring = "r2"
        candidates = {
            "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
            "RidgeRegression": Ridge(),
            "DecisionTree": DecisionTreeRegressor(random_state=42)
        }

    leaderboard = {}
    for name, model in candidates.items():
        pipe = Pipeline([('prep', preprocessor), ('estimator', model)])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            leaderboard[name] = float(np.mean(scores))
        except Exception:
            leaderboard[name] = 0.0

    best_model_name = max(leaderboard, key=leaderboard.get)

    # 7. Bayesian Optimization with Optuna on Winning Model
    def objective(trial):
        if task_type == "classification":
            n_est = trial.suggest_int("n_estimators", 20, 120)
            max_d = trial.suggest_int("max_depth", 3, 15)
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
        else:
            n_est = trial.suggest_int("n_estimators", 20, 120)
            max_d = trial.suggest_int("max_depth", 3, 15)
            clf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)

        pipe = Pipeline([('prep', preprocessor), ('estimator', clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Fit final pipeline with optimal hyperparameters
    best_params = study.best_params
    if task_type == "classification":
        final_estimator = RandomForestClassifier(**best_params, random_state=42)
    else:
        final_estimator = RandomForestRegressor(**best_params, random_state=42)

    final_pipeline = Pipeline([
        ('prep', preprocessor),
        ('estimator', final_estimator)
    ])
    final_pipeline.fit(X, y)

    # 8. Serialize .pkl Artifact
    artifact_path = os.path.join(ARTIFACT_DIR, "best_pipeline.pkl")
    joblib.dump(final_pipeline, artifact_path)

    return {
        "metadata": {
            "task_type": task_type,
            "rows": len(df),
            "features": X.shape[1],
            "target": target_clean
        },  
        "metrics": {
            "leaderboard": leaderboard,
            "best_model": best_model_name,
            "tuned_score": float(study.best_value)
        },
        "points_3d": points_3d,
        "download_url": "/api/v1/download"
    }

@app.get("/api/v1/download")
def download_artifact():
    artifact_path = os.path.join(ARTIFACT_DIR, "best_pipeline.pkl")
    if not os.path.exists(artifact_path):
        raise HTTPException(status_code=404, detail="Pipeline artifact not found.")
    return FileResponse(artifact_path, media_type="application/octet-stream", filename="datapulse_pipeline.pkl")



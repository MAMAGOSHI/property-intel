# modeling.py — trains the model, evaluates it, saves it, explains it

import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from property_intel.exceptions import ModelError

logger = logging.getLogger("property_intel")


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Trains a Linear Regression model to predict property price.
    Evaluates on test set only — never on training data.
    Saves the model to disk with joblib.
    Returns metrics, the model, and feature importances.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if "price" not in df.columns:
        raise ModelError("price column missing from DataFrame")

    # separate features (X) from target (y)
    # drop price_per_sqm too — it is derived from price, using it would be cheating
    drop_cols = ["price", "price_per_sqm"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["price"]

    logger.info(f"Training with {X.shape[1]} features on {len(X)} rows")

    # split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    # train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate on TEST set only
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info(f"RMSE: R{rmse:,.0f}  |  MAE: R{mae:,.0f}  |  R2: {r2:.3f}")

    # save model and feature names together
    model_path = Path(config["model"]["model_path"])
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        model_path
    )
    logger.info(f"Model saved to {model_path}")

    # feature importance — absolute coefficient size
    importance = pd.Series(model.coef_, index=X.columns)
    importance = importance.abs().sort_values(ascending=False)

    return {
        "model":         model,
        "rmse":          rmse,
        "mae":           mae,
        "r2":            r2,
        "importance":    importance,
        "feature_names": list(X.columns),
        "y_test":        y_test,
        "y_pred":        y_pred,
    }


def explain_model(importance: pd.Series) -> str:
    """Plain-English explanation of what drives property prices."""
    top = importance.head(5)
    lines = ["Top factors driving predicted price:\n"]
    for feature, value in top.items():
        lines.append(f"  {feature}: coefficient magnitude R{value:,.0f}")
    lines.append(
        "\nA one-unit increase in that feature is associated with "
        "that rand change in predicted price, all else equal."
    )
    return "\n".join(lines)
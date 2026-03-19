import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
EXPERIMENTS_DIR = os.path.dirname(__file__)


def log_experiment(
    name: str,
    config: dict,
    features: list,
    metrics: dict,
    notes: str = "",
) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name}.json"
    filepath = os.path.join(EXPERIMENTS_DIR, filename)

    safe_metrics = {k: v for k, v in metrics.items() if k not in ("equity_curve", "trades")}

    experiment = {
        "name": name,
        "timestamp": timestamp,
        "config": config,
        "features": features,
        "metrics": safe_metrics,
        "notes": notes,
    }

    with open(filepath, "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    logger.info(f"Experiment logged: {filepath}")

    # MLflow logging
    try:
        import mlflow
        mlflow_dir = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "mlruns")
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        mlflow.set_experiment("quant_trading_ai")

        with mlflow.start_run(run_name=f"{name}_{timestamp}"):
            # Log hyperparameters
            for k, v in config.items():
                mlflow.log_param(k, v)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("notes", notes)

            # Log metrics
            for k, v in safe_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log experiment JSON as artifact
            mlflow.log_artifact(filepath)

        logger.info(f"MLflow run logged: {name}")
    except Exception as e:
        logger.debug(f"MLflow logging skipped: {e}")

    return filepath

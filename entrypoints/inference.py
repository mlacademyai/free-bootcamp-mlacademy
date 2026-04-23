import time
from pathlib import Path

import pandas as pd
import yaml
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_inference() -> None:
    project_path = Path(__file__).resolve().parent.parent
    bootstrap_project(project_path)
    configure_project("free_bootcamp_mlacademy")

    # Load parameters from the configuration file
    params_path = project_path / "conf" / "base" / "parameters.yml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Load catalog to get data paths
    catalog_path = project_path / "conf" / "base" / "catalog.yml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    runner_config = params["pipeline_runner"]

    # Clear predictions file to start fresh
    predictions_path = project_path / catalog["predictions_with_timestamps"]["filepath"]
    if predictions_path.exists():
        predictions_path.unlink()
        print("Cleared previous predictions")

    # Load inference data from catalog
    inference_data_path = project_path / catalog["inference_data"]["filepath"]
    data = pd.read_parquet(inference_data_path)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.reset_index(drop=True)

    # Parse config
    batch_size = runner_config["batch_size"]
    first_timestamp = pd.to_datetime(runner_config["first_timestamp"])
    num_steps = runner_config["num_steps_inference"]
    interval_seconds = runner_config["inference_interval_seconds"]

    # Find first index matching the start timestamp
    first_idx = data[data["datetime"] >= first_timestamp].index[0]

    print(f"Starting inference: {num_steps} steps, batch_size={batch_size}")

    for step in range(num_steps):
        current_idx = first_idx + step
        batch_start = max(0, current_idx - batch_size + 1)
        batch_end = current_idx + 1

        batch = data.iloc[batch_start:batch_end].copy()

        # Save batch to catalog location
        batch_path = project_path / catalog["inference_batch"]["filepath"]
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch.to_parquet(batch_path, index=False)

        # Run pipeline
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="inference")

        print(f"[{step + 1}/{num_steps}] Prediction saved")

        if step < num_steps - 1:
            time.sleep(interval_seconds)

    print("Inference loop completed!")

if __name__ == "__main__":
    run_inference()
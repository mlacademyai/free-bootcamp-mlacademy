from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_training() -> None:
    project_path = Path(__file__).resolve().parent.parent
    bootstrap_project(project_path)
    configure_project("free_bootcamp_mlacademy")

    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="training")
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    run_training()
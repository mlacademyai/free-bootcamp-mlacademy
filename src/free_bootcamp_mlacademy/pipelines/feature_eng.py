from kedro.pipeline import Pipeline, node
from .nodes import rename_columns, get_features, load_data

def create_feature_eng_pipeline() -> Pipeline:
    return Pipeline(
        [
        node(
            func=rename_columns,
            inputs=["input_data", "params:feature_engineering.rename_columns"],
            outputs="renamed_data",
        ),
        node(
            func=get_features,
            inputs=["renamed_data", "params:feature_engineering.lag_params"],
            outputs=["features", "timestamps"],
        )
    ])


def load_training_data() -> Pipeline:
    """Load training data pipeline."""
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="train_data",
                outputs=["input_data", "last_timestamp"],
            ),
        ]
    )

def load_inference_data() -> Pipeline:
    """Load inference batch pipeline with timestamp extraction."""
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="inference_batch",
                outputs=["input_data", "last_timestamp"],
            ),
        ]
    )

def feat_eng_pipeline_training(**kwargs) -> Pipeline:
    """Feature engineering pipeline for training."""
    return load_training_data() + create_feature_eng_pipeline()


def feat_eng_pipeline_inference(**kwargs) -> Pipeline:
    """Feature engineering pipeline for inference."""
    return load_inference_data() + create_feature_eng_pipeline()
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

data_version = "Data_Clustering_v1"
model_version = "Clustering_V1"


def preprocess_clustering_data(project_root: Path) -> pd.DataFrame:
    """
    Loads raw CSV, removes NaNs in numeric columns, scales numeric features,
    saves preprocessed data, and returns cleaned DataFrame.
    """
    data_path = project_root / "data" / data_version / "LLM data_aggregate_8.25.25 data_updated 10.8.25.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    numeric_cols = [
        'FSR', 'BIS', 'SRS.Raw', 'TDNorm_avg_PE',
        'overall_avg_PE', 'TDnorm_concept_learning', 'overall_concept_learning'
    ]

    df_filtered = df.dropna(subset=numeric_cols).reset_index(drop=True)
    print(f"After dropping NaNs: {df_filtered.shape}")

    scaler = StandardScaler()
    df_filtered[numeric_cols] = scaler.fit_transform(df_filtered[numeric_cols])

    preprocessed_path = project_root / "data" / data_version / "data_preprocessed_Clustering_V1.csv"
    df_filtered.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed data → {preprocessed_path}")

    results_dir = project_root / "Results" / model_version / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "original_shape": df.shape,
        "filtered_shape": df_filtered.shape,
        "dropped_rows": int(df.shape[0] - df_filtered.shape[0]),
        "numeric_columns": numeric_cols
    }
    with open(results_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return df_filtered


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    preprocess_clustering_data(project_root)

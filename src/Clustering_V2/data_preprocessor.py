import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

data_version = "Data_Clustering_v2"
model_version = "Clustering_V2"


def preprocess_clustering_data(project_root: Path) -> pd.DataFrame:
    """
    Loads raw CSV, removes NaNs in numeric columns, scales selective features,
    skips already-scaled ones, keeps only scaled versions, and saves cleaned DataFrame.
    """
    data_path = project_root / "data" / data_version / "LLM data_aggregate_8.25.25 data_updated 10.8.25.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # Define feature sets
    already_scaled = ['TDnorm_concept_learning', 'overall_concept_learning']
    to_scale = ['FSR', 'BIS', 'SRS.Raw', 'TDNorm_avg_PE', 'overall_avg_PE']

    # Drop rows with missing values in any of these
    df_filtered = df.dropna(subset=to_scale + already_scaled).reset_index(drop=True)
    print(f"After dropping NaNs: {df_filtered.shape}")

    # Scale only the "to_scale" features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[to_scale])

    # Create new DataFrame with "_scaled" suffixes
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{c}_scaled" for c in to_scale])

    # Keep the already scaled columns as-is
    df_final = pd.concat([df_filtered[already_scaled], scaled_df, df_filtered[['td_or_asd']]], axis=1)

    # Save preprocessed version
    preprocessed_path = project_root / "data" / data_version / "data_preprocessed_Clustering_V2.csv"
    df_final.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed data → {preprocessed_path}")

    # Log preprocessing summary
    results_dir = project_root / "Results" / model_version / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "original_shape": df.shape,
        "filtered_shape": df_filtered.shape,
        "dropped_rows": int(df.shape[0] - df_filtered.shape[0]),
        "scaled_columns": [f"{c}_scaled" for c in to_scale],
        "already_scaled_columns": already_scaled,
        "final_columns": list(df_final.columns)
    }
    with open(results_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return df_final


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    preprocess_clustering_data(project_root)

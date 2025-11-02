import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

data_version = "Data_Clustering_v1"
model_version = "Clustering_V1"


def preprocess_clustering_data(project_root: Path) -> pd.DataFrame:
    """
    Loads raw CSV, removes NaNs, scales only specific numeric features (excluding concept learning),
    drops original unscaled versions, and saves the cleaned DataFrame for Clustering_V1.
    """
    # === Load Data ===
    data_path = project_root / "data" / data_version / "LLM data_aggregate_8.25.25 data_updated 10.8.25.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # === Feature groups ===
    to_scale = ['FSR', 'BIS', 'SRS.Raw', 'TDNorm_avg_PE', 'overall_avg_PE']
    already_scaled = ['TDnorm_concept_learning', 'overall_concept_learning']

    # === Drop NaNs across required columns ===
    df_filtered = df.dropna(subset=to_scale + already_scaled).reset_index(drop=True)
    print(f"After dropping NaNs: {df_filtered.shape}")

    # === Scale selected features ===
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[to_scale])

    # Create scaled columns
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{c}_scaled" for c in to_scale])

    # Combine: scaled features + untouched concept-learning + target
    df_final = pd.concat([scaled_df, df_filtered[already_scaled + ['td_or_asd']]], axis=1)

    # === Save cleaned data ===
    preprocessed_path = project_root / "data" / data_version / "data_preprocessed_Clustering_V1.csv"
    df_final.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed data → {preprocessed_path}")

    # === Save preprocessing metadata ===
    results_dir = project_root / "Results" / model_version / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "original_shape": df.shape,
        "filtered_shape": df_filtered.shape,
        "dropped_rows": int(df.shape[0] - df_filtered.shape[0]),
        "scaled_columns": [f"{c}_scaled" for c in to_scale],
        "untouched_columns": already_scaled,
        "final_columns": list(df_final.columns)
    }
    with open(results_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved preprocessing metadata → {results_dir / 'preprocessing_info.json'}")

    return df_final


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running Clustering_V1 Preprocessing ===\n")
    preprocess_clustering_data(project_root)


import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

# ----- App / Layout -----
st.set_page_config(
    page_title="TD/ASD Classification - Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Paths -----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
DATA_ROOT = PROJECT_ROOT / "data"
# Ensure Root/src is on sys.path so Model_V1, Model_V2, etc. can be imported
SRC_ROOT = CURRENT_DIR.parent  # this goes up from Demo/ → src/
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODEL_OVERVIEW_MD = \
    {

    "V1": """
## Model Overview (V1)

Baseline **XGBoost** classifier on core NLP features.

### Key Features
- Token/char stats, sentiment, readability
- Basic linguistic cohesion features
- SHAP + LIME for explainability

### Pipeline
1. Preprocess text
2. Feature engineering (core NLP)
3. XGBoost classification
4. Explainability (global + local)
5. Visualization
""",

    "V2": """
## Model Overview (V2)

Adds **characteristic-based** features (11 traits) extracted via LLM.

### Key Features
- All V1 features
- + 11 characteristic scores from LLM
- Improved SHAP global consistency

### Pipeline
1. Preprocess + LLM traits
2. Feature engineering (core + traits)
3. XGBoost classification
4. Explainability
5. Visualization
""",

    "V3": """
## Model Overview (V3)

Refined traits + stronger regularization; calibrated probabilities.

### Key Features
- All V2 features
- Probability calibration (Platt/Isotonic)
- Better class balance handling

### Pipeline
1. Preprocess + refined traits
2. Feature engineering
3. XGBoost + calibration
4. Explainability
5. Visualization
""",

    "V4": """
## Model Overview (V4)

Feature stability checks + cross-validation artifacts.

### Key Features
- All V3 features
- CV-based feature stability
- Robustness to outliers

### Pipeline
1. Preprocess + traits
2. Stable feature set via CV
3. XGBoost
4. Explainability
5. Visualization
""",

    "V5": """
## Model Overview (V5)

Production-ready: thresholds, drift checks, and richer explainability.

### Key Features
- All V4 features
- Threshold tuning for F1/Recall
- Data drift monitors (WIP)

### Pipeline
1. Preprocess + traits
2. Final features + thresholds
3. XGBoost
4. Explainability
5. Visualization + monitoring
"""
}

MODEL_INFO = {
    "V1": "Baseline TD/ASD classifier using core NLP and linguistic features.",
    "V2": "Enhanced model with characteristic-based features extracted via LLM.",
    "V3": "Advanced TD/ASD Classification using XGBoost and NLP features.",
    "V4": "Cross-validated model with feature stability and improved explainability.",
    "V5": "Production-ready model with calibrated thresholds and drift detection."
}

def get_model_info(version: str) -> str:
    """Return sidebar caption text for a given model version."""
    return MODEL_INFO.get(version, "Experimental TD/ASD model.")

def _discover_models() -> list[str]:
    """Return available model versions by scanning Results/* that look like V{number}."""
    if not RESULTS_ROOT.exists():
        return ["V1", "V2", "V3", "V4", "V5"]
    versions = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("V") and p.name[1:].isdigit():
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4", "V5"]

def get_overview_md(version: str) -> str:
    """Return Model MD Card based on Model Version Selection."""
    return MODEL_OVERVIEW_MD.get(version, MODEL_OVERVIEW_MD["V1"])

# Initialize session state and model selection
if "page" not in st.session_state:
    st.session_state.page = "home"

if "model_version" not in st.session_state:
    st.session_state.model_version = "V1"

if "available_models" not in st.session_state:
    st.session_state.available_models = _discover_models()

class DemoApp:
    def __init__(self):
        self.model_version = st.session_state.model_version
        self.results_dir = RESULTS_ROOT / self.model_version
        self.data_dir = DATA_ROOT / self.model_version
        self.viz_dir = self.results_dir / "visualizations"
        self.predictor = None

    # ----- Model Version Selection -----
    def set_model_version(self, version: str):
        """Update the app to a new model version and refresh result paths."""
        self.model_version = version
        st.session_state.model_version = version
        self.results_dir = RESULTS_ROOT / version
        self.data_dir = DATA_ROOT / self.model_version
        self.viz_dir = self.results_dir / "visualizations"

    # ----- Data / Model loading  -----
    def load_results(self) -> bool:
        # Example check: existence of directory or an expected file
        # Adjust to your real artifacts (e.g., metrics.json, model.pkl, etc.)

        try:
            version = self.model_version
            explainability_path = self.results_dir / f'explainability_analysis_{version}.json'
            training_results_path = self.results_dir / f'training_results_{version}.json'
            test_results_path = self.results_dir / f'test_results_{version}.json'

            with open(explainability_path, 'r') as f:
                self.explainability_data = json.load(f)

            with open(training_results_path, 'r') as f:
                self.training_results = json.load(f)


            try:
                with open(test_results_path, 'r') as f:
                    self.test_results = json.load(f)
            except:
                self.test_results = None

            return True

        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            st.error(f"Results directory: {self.results_dir}")
            st.error(f"Directory exists: {self.results_dir.exists()}")
            if self.results_dir.exists():
                st.error(f"Files in directory: {list(self.results_dir.glob('*.json'))}")
            return False



    def initialize_predictor(self):
        # Load model + artifacts for the selected version here

        return

    # ----- Pages -----
    def show_home_page(self):
        st.title("🧠 TD/ASD Classification Model")
        st.markdown("### Advanced Text Analysis for Autism Spectrum Disorder Classification")

        # ---- Model selector ON HOME PAGE ----
        st.markdown("#### Select a Model Version")
        selected = st.selectbox(
            "Choose which model version to explore:",
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.model_version)
            if st.session_state.model_version in st.session_state.available_models else 0,
            key="model_version_home_select"
        )

        # Apply selection immediately if changed
        if selected != self.model_version:
            self.set_model_version(selected)
            st.success(f"Model version set to **{self.model_version}**.")


        st.markdown("### Checklist for Specific Files Needed for Downstream Analysis")

        data = [
            ["Processed Training Data", f"data/Data_{self.model_version}",
             f"LLM_data_train_preprocessed_{self.model_version}", "", ""],

            ["Processed Test Data", f"data/Data_{self.model_version}",
             f"LLM_data_test_preprocessed_{self.model_version}", "", ""],

            ["Training Results", f"Results/{self.model_version}",
             f"training_results_{self.model_version}", "", ""],

            ["Test Results", f"Results/{self.model_version}",
             f"test_results_{self.model_version}", "", ""],

            ["Model Pkl File", f"Results/{self.model_version}",
             f"xgboost_model_{self.model_version}", "", ""]
        ]

        df = pd.DataFrame(data, columns=["Files", "Directory", "Filename", "Status", "Instruction"])

        for i, row in df.iterrows():
            if i in [0, 1]:
                ext = ".csv"
                script_hint = "run data_preprocesor.py"
            elif i in [2, 3]:
                ext = ".json"
                if i==2:
                    script_hint = "run train.py"
                if i==3:
                    script_hint = "run predict.py"
            else:
                ext = ".pkl"
                script_hint = "run train.py"

            path = PROJECT_ROOT / row["Directory"] / (row["Filename"] + ext)
            exists = path.exists()


            df.loc[i, "Status"] = "✅ Found" if exists else "❌ Missing"
            df.loc[i, "Instruction"] = "Proceed" if exists else script_hint


        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown(get_overview_md(self.model_version))

        if st.button("🚀 Explore Model Results"):
            st.session_state.page = "results"
            st.rerun()

    def show_results_page(self):
        st.header(f"📊 Results — {self.model_version}")
        st.caption(f"Reading from: `Results/{self.model_version}`")
        # TODO: load and display figures/metrics specific to self.model_version
        st.info("Results page ")

        tab1, tab2, tab3 = st.tabs(["Features List","Training Performance", "Test Performance"])

        with tab1:
            st.subheader("Selected Feature List")
            if self.training_results:
                features_selected = self.training_results['feature_names']
                st.info(f"Number of Features : {self.training_results['n_features']}")
                st.json(features_selected)
            else:
                st.info("Training Report not found. Run train.py to generate a training report.")

        with tab2:
            st.subheader("Model Training Metrics")
            if self.training_results:
                st.markdown("##### Cross Validation Scores - Accuracy")
                st.json(self.training_results['cv_scores'])
            else:
                st.info("Training Report not found. Run train.py to generate a training report.")

        with tab3:
            st.subheader("Model Performance Metrics")

            col1, col2, col3, = st.columns(3)

            with col1:
                if self.test_results:
                    st.info("TD Class")
                    st.json(self.test_results['classification_report']['0'])
                else:
                    st.info("Classification report not available for Test Set. Run predict.py to generate Test Metrics.")

            with col2:
                if self.test_results:
                    st.info("ASD Class")
                    st.json(self.test_results['classification_report']['1'])
                else:
                    st.info("Classification report not available for Test Set. Run predict.py to generate Test Metrics.")

            with col3:
                if self.test_results:
                    st.info("Macro Avg")
                    st.json(self.test_results['classification_report']['macro avg'])
                else:
                    st.info("Classification report not available for Test Set. Run predict.py to generate Test Metrics.")

            cm = np.array(self.test_results['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['TD', 'ASD'], yticklabels=['TD', 'ASD'], ax=ax)
            ax.set_title(f'Confusion Matrix - Accuracy {self.test_results["accuracy"]:.4f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)

            # img = Image.open(self.viz_dir/f"confusion_matrix_{self.model_version}.png")
            # st.image(img,use_container_width=True)



    def show_prediction_page(self):
        st.header(f"🔮 Predictions — {self.model_version}")
        # TODO: wire predictor for the selected version
        st.info("Make predictions on your data")

    def show_explainability_page(self):
        st.header(f"🔍 Explainability — {self.model_version}")
        st.caption(f"Using artifacts from: `Results/{self.model_version}`")
        # TODO: show SHAP/LIME global & local explanations tied to self.model_version
        st.info("Explainability visualizations (global + local) ")

        # ---- Select explainability type ----
        explain_type = st.radio("Select explainability method", ["SHAP", "LIME"], horizontal=True)

        # ---- Row selection for local explainability ----
        row_idx = st.number_input("Select a row index to explain", min_value=0, value=0, step=1)
        top_n = st.slider("Top features (for SHAP only)", 3, 20, 10)

        st.divider()

        # ---- Dynamic import of explainers ----
        from importlib import import_module

        try:
            # Import path: Model_V1.shap_explain  or  Model_V1.lime_explain
            model_folder = f"Model_{self.model_version}"
            module_name = "shap_explain" if explain_type == "SHAP" else "lime_explain"
            full_module_path = f"{model_folder}.{module_name}"

            expl = import_module(full_module_path)
            st.success(f"Loaded {full_module_path}")
        except Exception as e:
            st.error(f"❌ Could not import {module_name}.py for model {self.model_version}")
            st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
            st.exception(e)
            return

        # ---- Global explainability ----
        if st.button(f"Run {explain_type} Global Explainability"):
            with st.spinner(f"Running {explain_type} global explainability..."):
                results = expl.explain_global()
            st.success("Global explainability generated.")
            for output in results.get("outputs", []):
                if output.endswith(".png"):
                    st.image(output, caption=f"{explain_type} Global Plot")
                elif output.endswith(".json"):
                    st.markdown("**Top global features:**")
                    with open(output, "r") as f:
                        st.json(dict(list(json.load(f).items())[:10]))

        st.divider()

        # ---- Local explainability ----
        if st.button(f"Run {explain_type} Local Explainability"):
            with st.spinner(f"Running {explain_type} local explainability for row {row_idx}..."):
                if explain_type == "SHAP":
                    local_results = expl.explain_local(indices=[row_idx], top_n=top_n)
                else:
                    local_results = expl.explain_local(indices=[row_idx])
            if not local_results:
                st.warning("No local explanation generated.")
                return
            res = local_results[0]

            # Display visualization
            if "plot" in res:
                st.image(res["plot"], caption=f"{explain_type} Local Plot (Row {row_idx})")
            elif "png" in res:
                st.image(res["png"], caption=f"{explain_type} Local Plot (Row {row_idx})")

            # Dynamic Markdown summary
            if explain_type == "SHAP":
                st.markdown(f"""
                    **Dynamic Explanation (SHAP)**  
                    - **Row:** {row_idx}  
                    - **Base value:** `{res.get('base_value', 0):.3f}`  
                    - Red (positive) bars increase ASD probability; blue (negative) bars increase TD probability.  
                    - The final prediction depends on the sum of all SHAP contributions added to the base value.
                """)
            else:
                st.markdown(f"""
                    **Dynamic Explanation (LIME)**  
                    - **Row:** {row_idx}  
                    - Green bars push prediction toward ASD, red bars toward TD.  
                    - The length of each bar shows how strongly that feature influenced this instance.  
                    - Check the generated `.html` file ({res['html']}) for an interactive visualization.
                """)

        st.markdown("---")
        st.info("This section updates dynamically based on the selected model and row.")

    def show_model_and_data_page(self):
        st.header(f"🔍 Explainability — {self.model_version}")
        st.caption(f"Using artifacts from: `Results/{self.model_version}`")
        # TODO: show SHAP/LIME global & local explanations tied to self.model_version
        st.info("Explainability visualizations (global + local) ")

def main():
    app = DemoApp()

    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": "home",
        "Model & Data" : "overview",
        "📊 Model Results": "results",
        "🔍 xAI - Explainability": "explainability",
        "🔮 Test Your Data": "predictions",
        "Cluster Analysis": "clustering",
        "Experiments": "experiments",

    }

    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("###### Note: These fields update when new page is selected.")

    st.sidebar.markdown("### Model Version (current)")
    st.sidebar.info(st.session_state.model_version)

    st.sidebar.markdown("### Model Info")
    st.sidebar.caption(get_model_info(st.session_state.model_version))

    # Ensure artifacts exist for selected version
    if not app.load_results():
        st.error(f"Could not find results for **{st.session_state.model_version}** at `Results/{st.session_state.model_version}`.")
        st.stop()

    # Route to selected page
    if st.session_state.page == "home":
        app.show_home_page()
    elif st.session_state.page == "results":
        app.show_results_page()
    elif st.session_state.page == "predictions":
        app.show_prediction_page()
    elif st.session_state.page == "explainability":
        app.show_explainability_page()

if __name__ == "__main__":
    main()

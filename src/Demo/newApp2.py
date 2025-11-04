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
from importlib import import_module

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
SRC_ROOT = CURRENT_DIR.parent
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

- Data Source : Data_V4
- Train-Test Split : 80-20 Stratified Split with Seed 42
- Key Change : Generated Slope Features

""",

    "V7-1": """
## Model Overview (V7.1)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : 80-20 Stratified Split with Seed 42
- Baseline Model for FSR overlap Test
""",

    "V7-2": """
## Model Overview (V7.2)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : FSR Overlap Region is Training Set
- Baseline Model for FSR overlap Test
""",

    "V7-3": """
## Model Overview (V7.3)

- Data Source : Data_V7
- Features Added : TDNorm PE and Concept Learning
- Train-Test Split : FSR Non-Overlap Region is Training Set
- Baseline Model for FSR overlap Test
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
        if p.is_dir() and p.name.upper().startswith("V") and p.name[1].isdigit():
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4", "V5"]

def _discover_culstering_models() -> list[str]:
    """Return available clustering model versions by scanning Results/* that look like Clustering_V{number}."""
    if not RESULTS_ROOT.exists():
        return ["Clustering_V1", "Clustering_V2"]
    versions = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("CLUSTERING"):
            versions.append(p.name.upper())
    return versions or ["Clustering_V1", "Clustering_V2"]

def _discover_datasets() -> list[str]:
    """Return available datasets by scanning data/* that look like Data_v{number}."""
    if not DATA_ROOT.exists():
        return ["V1", "V2", "V3",]
    versions = []
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir() and p.name.upper().startswith("DATA_V"):
            versions.append(p.name.upper())
    return versions or ["V1", "V2", "V3", "V4"]


def get_overview_md(version: str) -> str:
    """Return Model MD Card based on Model Version Selection."""
    return MODEL_OVERVIEW_MD.get(version, MODEL_OVERVIEW_MD["V1"])

# Initialize session state and model selection
if "page" not in st.session_state:
    st.session_state.page = "home"

if "model_version" not in st.session_state:
    st.session_state.model_version = "V7.1"

if "data_version" not in st.session_state:
    st.session_state.data_version = "DATA_V7.1"

if "available_models" not in st.session_state:
    st.session_state.available_models = _discover_models()

if "available_datasets" not in st.session_state:
    st.session_state.available_datasets = _discover_datasets()

class DemoApp:
    def __init__(self):
        self.model_version = st.session_state.model_version
        self.data_version = st.session_state.data_version
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

    def set_data_version(self, version: str):
        """Update the app to a new data version and refresh paths."""
        self.data_version = version
        st.session_state.data_version = version
        self.data_dir = DATA_ROOT / self.data_version


    # ----- Data / Model loading  -----
    def load_results(self) -> bool:

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
        st.header(f"📊 About the Project : 🧠 TD/ASD Classification ")

        # =======================
        # Project Overview
        # =======================

        st.markdown("""
            ### Project Overview
            This project aims to investigate how **interpretable,
            multi-modal machine learning models** can enhance the
            diagnosis and understand of **Autism Spectrum Disorder
            (ASD)**. By integrating **behavioral** and **linguistic
            data**, the study aims to move beyond traditional, subjective,
            assessments toward more objective, data-driven approaches.
            The proposed models seek to improve **diagnosis precision**
            by identifying subtle patterns across multiple data types, enable
            **subtype discovery** through clustering of behavioral and linguistic
            traits, and enhance **generalizability** by developing frameworks that
            perform robustly across diverse populations and datasets. Ultimately,
            this research contributes to more accurate, transparent, and inclusive
            ASD prediction and analysis.

            ---
            """)

        # ==========================================
        # 🎯 Key Objectives | 🧰 Tools & Techniques
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Key Objectives")
            st.markdown("""
            - 🧩 Develop interpretable machine learning models  
            - 🗣️ Analyze linguistic and behavioral patterns  
            - 🧠 Identify subtypes of ASD using clustering  
            - 📈 Improve diagnostic generalization across datasets
            """)

        with col2:
            st.subheader("🧰 Tools & Techniques")
            st.markdown("""
            - **LLM Agent:** Qwen  
            - **NLP:** BERT  
            - **Classification Model:** XGBoost  
            - **Dimension Reduction:** PCA, AGAE  
            - **Clustering Model:** KMeans, Gaussian Mixture Model, HDBSCAN  
            - **Explainability Method:** SHAP, LIME
            """)

        # =======================
        # 🧭 App Navigation Guide
        # =======================
        st.markdown("---")
        st.subheader("🧭 App Navigation Guide")

        st.markdown("""
        Use the tabs at the top to explore different parts of the project:

        - **🏠 Home** – Project overview, key objectives, tools 
        - **📈 Data & Model** – Data versions, Data dictionary, EDA, Models selection, Confusion matrix, F1-scores
        - **📊 Model Results** – Train accuracy, Test accuracy
        - **🧩 Cluster Analysis** – Unsupervised clustering (KMeans, GMM, HDBSCAN) and ASD subtype discovery  
        - **🔍 xAI - Explainability** – SHAP / LIME explanations for model predictions and important features
        - **🔮 Test Your Data** - Upload and test your own dataset with the preferred model
        - **🔬 Experiments** - Experiment and compare different combinations of features

        """)


    def show_results_page(self):
        st.header(f"📊 Results — {self.model_version}")
        st.caption(f"Reading from: `Results/{self.model_version}`")
        # TODO: load and display figures/metrics specific to self.model_version
        # st.info("Results page ")


        # ---- Dynamic import of explainers ----
        try:
            # Import path: Model_Vx.shap_explain  or  Model_Vx.lime_explain
            model_folder = f"Model_{self.model_version}"
            module_name = "visualization"
            full_module_path = f"{model_folder}.{module_name}"

            viz = import_module(full_module_path)
            # st.success(f"Loaded {full_module_path}")
        except Exception as e:
            st.error(f"❌ Could not import {module_name}.py for model {self.model_version}")
            st.info(f"Tried path: {full_module_path.replace('.', '/')}.py")
            st.exception(e)
            return

        visualizer = viz.ModelVisualizerInteractive()

        tab1, tab2, tab3 = st.tabs(["Training Performance", "Test Performance", "Feature Visualizations"])

        with tab1:

                if self.training_results:
                    st.plotly_chart(visualizer.fig_cv_scores(self.training_results), use_container_width=True)
                    st.plotly_chart(visualizer.fig_overall_performance(self.training_results), use_container_width=True)
                else:
                    st.info("Training Report not found. Run train.py to generate a training report.")


        with tab2:
            if self.test_results:
                st.plotly_chart(visualizer.fig_confusion_matrix(self.test_results), use_container_width=True)

                st.plotly_chart(visualizer.fig_classification_report(self.test_results), use_container_width=True)

            else:
                st.info("Test Report not found. Run predict.py to generate a training report.")

        with tab3:
            if self.explainability_data:
                st.plotly_chart(visualizer.fig_feature_importance_by_target(self.explainability_data), use_container_width=True)

            else:
                st.info("Test Report not found. Run predict.py to generate a training report.")


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
        try:
            # Import path: Model_Vx.shap_explain  or  Model_Vx.lime_explain
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
        st.title("Data and Model Cards")
        st.markdown("### Autism Spectrum Disorder Classification Using Trial and Text Features")

        tab1, tab2 = st.tabs(["📂 Data Version", "🧮 Model Version"])

        with tab1:
            # ---- Data selector ----
            st.markdown("## Data Overview")

            st.markdown("""
                    The data is provided by The George Washington University - Department of Psychological 
                    & Brain Sciences. There are 1,119 participants in this data aging from 8 to 12 years old.

                    - **Data V1:** Raw data at trial level (187,187 observations and 19 variables)
                    - **Data V2:** Aggregated data with slope features (2,648 observations and 11 variables)  
                    - **Data V3:** Aggregated data with concept learning features (1,119 observations and 10 variables)""")

            st.markdown("#### Select a Data Version to View")
            selected = st.selectbox(
                "Choose which Data version to explore:",
                options=st.session_state.available_datasets,
                index=st.session_state.available_datasets.index(st.session_state.data_version)
                if st.session_state.available_datasets in st.session_state.available_datasets else 0,
                key="data_version_select"
            )

            # Apply selection immediately if changed
            if selected != self.data_version:
                self.set_data_version(selected)
                st.success(f"Data version set to **{self.data_version}**.")
                st.rerun()
            # TODO: add data dictionary for selected data version
            # TODO: add EDA for selected data version

        with tab2:
            # ---- Model selector ----
            st.markdown("#### Select a Model Version")
            selected = st.selectbox(
                "Choose which model version to explore:",
                options=st.session_state.available_models,
                index=st.session_state.available_models.index(st.session_state.model_version)
                if st.session_state.model_version in st.session_state.available_models else 0,
                key="model_version_select"
            )

            # Apply selection immediately if changed
            if selected != self.model_version:
                self.set_model_version(selected)
                st.success(f"Model version set to **{self.model_version}**.")
                st.rerun()

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
                    if i == 2:
                        script_hint = "run train.py"
                    if i == 3:
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

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_overview_md(self.model_version))

                if st.button("🚀 Explore Model Results"):
                    st.session_state.page = "results"
                    st.rerun()

            with col2:
                st.subheader("Selected Feature List")
                if self.training_results:
                    features_selected = self.training_results['feature_names']
                    st.info(f"Number of Features : {self.training_results['n_features']}")
                    st.json(features_selected)
                else:
                    st.info("Training Report not found. Run train.py to generate a training report.")



def main():
    app = DemoApp()

    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": "home",
        "📈 Data & Model" : "overview",
        "📊 Model Results": "results",
        "🔍 xAI - Explainability": "explainability",
        "🧩 Cluster Analysis": "clustering",
        "🔮 Test Your Data": "predictions",
        "🔬 Experiments": "experiments",
    }

    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("###### Note: These fields update when new page is selected.")

    st.sidebar.markdown("### Data Version (current)")
    st.sidebar.info(st.session_state.data_version)

    st.sidebar.markdown("### Model Version (current)")
    st.sidebar.info(st.session_state.model_version)

    st.sidebar.markdown("### Model Info")
    st.sidebar.caption(get_model_info(st.session_state.model_version))

    # Ensure artifacts exist for selected version
    if not app.load_results():
        st.error(f"Could not find results for **{st.session_state.model_version}** at `Results/{st.session_state.model_version}`.")
        # st.stop()

    # Route to selected page
    if st.session_state.page == "home":
        app.show_home_page()
    elif st.session_state.page == "overview":
        app.show_model_and_data_page()
    elif st.session_state.page == "results":
        app.show_results_page()
    elif st.session_state.page == "predictions":
        app.show_prediction_page()
    elif st.session_state.page == "explainability":
        app.show_explainability_page()

if __name__ == "__main__":
    main()

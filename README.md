================================================================
ML CLUSTERING STUDIO
End-to-End Unsupervised Clustering Pipeline (Streamlit)
================================================================

OVERVIEW
--------
ML Clustering Studio is a modular Streamlit application that walks a
user through a complete unsupervised machine learning workflow -
from raw CSV upload to a trained clustering model with evaluation
metrics and exportable results. It is built as a step-by-step
"wizard" with sidebar navigation, progress tracking, and recovery
handling if a step errors out.

Live demo: https://clusterml.streamlit.app/


PIPELINE STEPS
--------------
1. Load Data           - Upload a CSV (max 50 MB), preview schema
2. EDA                 - Distributions, boxplots, correlation heatmap,
                          categorical breakdowns
3. Cleaning            - Missing-value imputation, outlier removal
                          (Z-score based), low-variance / high-
                          correlation column detection
4. Feature Engineering - Scaling (StandardScaler etc.), encoding,
                          custom feature creation, dimensionality
                          reduction (PCA / other) for 2D visualisation
5. Clustering          - Choose and fit an algorithm; elbow chart and
                          silhouette sweep to help pick K; optional
                          AutoML-style comparison across algorithms
6. Results             - Cluster map, cluster sizes, radar/heatmap
                          profiles, feature importance, scatter
                          matrix, dendrogram; downloadable analysis
                          report
7. Learn ML            - In-app explanations of each algorithm
                          (what it does, when to use it, when not to)


SUPPORTED ALGORITHMS
---------------------
- KMeans          (Beginner)     - fast, spherical clusters
- DBSCAN          (Intermediate) - density-based, finds outliers
- Agglomerative   (Intermediate) - hierarchical, dendrogram support
- Spectral        (Advanced)     - graph-based, non-convex clusters
- Birch           (Intermediate) - memory-efficient, large datasets
- MeanShift       (Intermediate) - auto-detects number of clusters

Each algorithm's info card (best-fit and worst-fit scenarios) is
defined in config/settings.py under ALGO_INFO.


PROJECT STRUCTURE
------------------
ML_Cluster_APP/
|
|-- app.py                    Entry point (run this). Page config,
|                              CSS injection, session state init,
|                              sidebar navigation, and step router.
|
|-- config/
|   |-- settings.py           Colors, Plotly theme, pipeline step
|   |                         labels, algorithm metadata, session
|   |                         state defaults, step-reset key map
|   |-- theme.py              All custom CSS (dark theme, cards,
|                              buttons, tabs, layout)
|
|-- components/
|   |-- ui.py                 Reusable UI blocks: hero header,
|                              sidebar, stepper, metric strips,
|                              explanation boxes
|
|-- utils/
|   |-- data.py                CSV loading, column typing, imputation,
|   |                          outlier removal, scaling/encoding,
|   |                          PCA/variance helpers, correlation and
|   |                          low-variance feature detection
|   |-- metrics.py             Silhouette, Davies-Bouldin,
|   |                          Calinski-Harabasz scoring
|   |-- charts.py              All Plotly/Matplotlib figure builders
|                              (scatter, bar, pie, heatmap, elbow,
|                              silhouette sweep, radar, dendrogram,
|                              AutoML comparison, etc.)
|
|-- pipeline/
|   |-- steps.py              One function per wizard step
|                              (step_load -> step_learn)
|
|-- .streamlit/config.toml    Streamlit app configuration/theme
|-- .devcontainer/            Dev container config
|-- requirements.txt          Python dependencies
|-- runtime.txt               Python runtime version pin


REQUIREMENTS
------------
streamlit==1.42.0
pandas==2.2.3
numpy==2.2.2
scikit-learn==1.6.1
plotly==5.24.0
matplotlib==3.10.0
scipy==1.15.2
pillow==11.1.0
tabulate

Install with:
    pip install -r requirements.txt


RUNNING LOCALLY
----------------
1. Clone the repo and cd into it
2. (Optional) create a virtual environment
3. pip install -r requirements.txt
4. streamlit run app.py
5. Upload a CSV from the sidebar to begin


NOTES
-----
- The app is stateful across steps via st.session_state; restarting
  from an earlier step clears that step's downstream state (see
  STEP_RESET_KEYS in config/settings.py) so results stay consistent.
- An in-app error boundary in app.py catches exceptions during a
  step and offers recovery back to Step 1 rather than crashing the
  whole session.
- The Results step includes a downloadable analysis report
  summarising the run (data shape, preprocessing choices, algorithm,
  parameters, and evaluation metrics).

================================================================

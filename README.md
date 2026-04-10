# ClusterForge Pro — Modular ML Clustering Platform
## 📁 Project Structure

```
clusterforge/
│
├── app.py                  ← Entry point. Run this. Contains NO logic.
│
├── config/
│   ├── settings.py         ← Colors, pipeline steps, algorithm info, session defaults
│   └── theme.py            ← All CSS (dark theme, fonts, cards, buttons, tabs)
│
├── components/
│   └── ui.py               ← Reusable UI: hero, sidebar, stepper, metric strip, explain boxes
│
├── utils/
│   ├── data.py             ← Load, clean, impute, scale, encode, feature engineering
│   ├── metrics.py          ← Silhouette, Davies-Bouldin, Calinski-Harabasz
│   └── charts.py           ← Every Plotly/Matplotlib figure builder
│
└── pipeline/
    └── steps.py            ← One function per step (step_load → step_learn)
```

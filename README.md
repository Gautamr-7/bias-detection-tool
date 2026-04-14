# Bias Detection & Fairness Auditing Tool

A fairness auditing system for machine learning models that detects demographic bias, visualizes disparate impact across protected attributes, and applies mitigation techniques — aligned with **auditable and governable AI** principles.

Built as part of the **Hack2Skill Solution Challenge 2026** (Unbiased AI Decision track).

---

##  Problem Statement

Computer programs now make life-changing decisions — who gets a job, a bank loan, or medical care. When these models are trained on historical data that contains discrimination, they **learn and amplify those exact same biases**.

This tool exposes that hidden bias and fixes it.

---

##  What It Does

| Step | Description |
|------|-------------|
| **Detect** | Measures bias in ML models using 3 fairness metrics |
| **Visualize** | Charts showing bias levels across demographic groups |
| **Mitigate** | Applies Reweighing algorithm to reduce bias |
| **Compare** | Before vs after fairness scores side by side |

---

##  Fairness Metrics Used

- **Disparate Impact** — Ratio of positive outcome rates between groups (fair = ≥ 0.80)
- **Statistical Parity Difference** — Difference in prediction rates (fair = ≈ 0)
- **Equal Opportunity Difference** — Difference in true positive rates (fair = ≈ 0)

---

##  Tech Stack

- **Python** — Core logic
- **IBM AIF360** — Bias detection and mitigation library
- **Scikit-learn** — Logistic Regression model
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Visualizations
- **Streamlit** — Interactive web dashboard

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Gautamr-7/bias-detection-tool
cd bias-detection-tool
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

##  Project Structure

```
bias-detection-tool/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← All dependencies
└── README.md           ← This file
```

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
matplotlib
scikit-learn
aif360
```

---

##  Dataset

Uses the **Adult Income Dataset** (UCI ML Repository) — built into IBM AIF360.

- **Task**: Predict whether income exceeds $50K/year
- **Protected Attributes**: `sex` (Male/Female), `race` (White/Non-White)
- **Samples**: ~48,000 individuals
- **Classic bias case**: Historical data shows higher approval rates for males and white individuals

---

##  Sample Results

| Metric | Before Mitigation | After Mitigation |
|--------|-------------------|------------------|
| Disparate Impact | ~0.36  | ~0.85  |
| Statistical Parity Diff | ~-0.19  | ~-0.05  |
| Model Accuracy | ~85% | ~83% |

> Fairness improved significantly with only a **~2% drop in accuracy** — proving fairness and performance are not mutually exclusive.

---

##  Mitigation Technique: Reweighing

Reweighing is a **pre-processing** bias mitigation technique from IBM AIF360.

Instead of changing the data, it assigns **sample weights** to training examples — giving underrepresented groups more importance during model training. This helps the model learn fairer decision boundaries without altering the underlying dataset.

---

##  Relevance to AI Governance

This project directly addresses:
- **Auditable AI** — Every fairness decision is measurable and explainable
- **Governable AI** — Organizations can detect and fix bias before deployment
- **Responsible AI** — Prevents real-world harm from biased automated decisions


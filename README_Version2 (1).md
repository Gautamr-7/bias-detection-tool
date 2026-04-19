# AI Fairness Auditor

Detect algorithmic bias before it causes harm — with **counterfactual causal testing**, proxy risk scoring, and professional audit reporting.

## Why this project stands out

Most fairness tools stop at group-level disparity.  
This project adds **causal evidence** and **decision-stability diagnostics**:

- **Proxy Detector** (MI + Pearson/Cramér’s V + bootstrap CI)
- **Counterfactual Tester** (decision-flip testing)
- **Counterfactual Stability Index (CSI)** *(unique metric)*  
- **Mitigation Simulator** *(what-if feature suppression)*
- **Multi-metric Fairness Scorecard** (DP, DI, EqOdds, PPV, Calibration)
- **One-click HTML Compliance Report**

---

## Features

### 1) Proxy Variable Detection
Finds variables that may act as hidden stand-ins for protected attributes (e.g., zip code, first name, neighborhood).

### 2) Counterfactual Causal Testing
For rejected cases, flips one suspicious variable while holding everything else constant.  
If prediction flips to approval, that is evidence of proxy-driven bias.

### 3) Counterfactual Stability Index (CSI) — Unique Measure
`CSI = 1 - bias_flip_rate`  
Higher CSI indicates a model with more stable/fair behavior under controlled perturbation.

### 4) Fairness Metrics Suite
- Demographic Parity
- Disparate Impact (4/5ths rule)
- Equalized Odds
- Predictive Parity
- Calibration Gap

### 5) Mitigation Simulator
Runs a fast what-if scenario by removing top risky proxy features and retraining to estimate fairness/utility tradeoff.

### 6) Professional Report Export
Self-contained HTML report with:
- executive summary
- proxy findings
- counterfactual evidence
- fairness scorecard
- prioritized recommendations

---

## Project Structure

```bash
bias-detection-tool/
├── app.py
├── proxy_detector.py
├── counterfactual_tester.py
├── fairness_metrics.py
├── report_generator.py
├── generate_sample_data.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
python generate_sample_data.py
streamlit run app.py
```

Open: `http://localhost:8501`

---

## Input Data Format

Your CSV should contain:
- one row per decision
- feature columns (numeric/categorical)
- one binary outcome column (`1/0`)

Example:
```csv
applicant_id,age,income,credit_score,zip_code,approved
APP_001,34,65000,720,10025,0
APP_002,28,82000,750,10075,1
```

---

## Resume-Ready Talking Points

- Designed and implemented a modular fairness auditing pipeline for tabular ML.
- Added a novel **Counterfactual Stability Index (CSI)** to separate model instability from structural bias.
- Built mitigation simulation for practical model governance decisions.
- Delivered compliance-friendly reporting layer and operational Streamlit interface.
- Balanced fairness diagnostics with model performance (AUC/accuracy tradeoff visibility).

---

## Limitations

- This tool detects statistical/causal risk patterns; it is **not legal advice**.
- Fairness definitions are context-dependent and often incompatible simultaneously.
- Final governance decisions should involve legal/compliance and domain experts.

---

## License

MIT
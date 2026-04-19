import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from config import DOMAIN_CONFIGS, MODEL_CANDIDATES, DEFAULT_MODEL
from proxy_detector import ProxyDetector
from counterfactual_tester import CounterfactualTester
from fairness_metrics import FairnessMetrics
from report_generator import generate_html_report


st.set_page_config(page_title="AI Fairness Auditor", page_icon="⚖️", layout="wide")


def load_csv_with_fallback(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def encode_dataframe(df: pd.DataFrame):
    out = df.copy()
    encoders = {}
    for col in out.columns:
        if out[col].dtype == object:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            encoders[col] = le
    return out, encoders


def get_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=600)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_recommendations(proxy_findings, cf_results, fairness_score):
    recs = []

    high_risk = [f.feature for f in proxy_findings if f.is_high_risk]
    if high_risk:
        recs.append({
            "priority": "P0",
            "effort": "Medium",
            "action": f"Constrain or remove high-risk proxy variables: {', '.join(high_risk[:3])}",
            "rationale": "These variables show strong proxy behavior and likely contribute to discriminatory outcomes.",
        })

    if cf_results and any(r.is_significant for r in cf_results.values()):
        recs.append({
            "priority": "P0",
            "effort": "High",
            "action": "Run mitigation retraining with proxy suppression and threshold tuning",
            "rationale": "Counterfactual tests show causal decision flips, which is stronger evidence than correlation alone.",
        })

    if fairness_score < 70:
        recs.append({
            "priority": "P1",
            "effort": "Medium",
            "action": "Add fairness gates in CI/CD for every model release",
            "rationale": "Current fairness profile is below recommended threshold and should block deployment until improved.",
        })

    recs.append({
        "priority": "P2",
        "effort": "Low",
        "action": "Monitor fairness drift monthly",
        "rationale": "Data shifts can degrade fairness over time even when initial results are acceptable.",
    })

    return recs


def mitigation_simulation(df, y_col, risky_features):
    """
    What-if mitigation:
    - Train baseline model
    - Drop top risky feature
    - Retrain and compare AUC
    """
    if not risky_features:
        return {"status": "No high-risk features found for mitigation simulation."}

    X = df.drop(columns=[y_col]).copy()
    y = df[y_col].astype(int)

    X_enc, _ = encode_dataframe(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.25, random_state=42, stratify=y
    )

    base_model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])

    top_feature = risky_features[0]
    if top_feature not in X.columns:
        return {"status": f"Top risky feature '{top_feature}' not found in current dataframe."}

    X_mit = X.drop(columns=[top_feature])
    X_mit_enc, _ = encode_dataframe(X_mit)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_mit_enc, y, test_size=0.25, random_state=42, stratify=y
    )

    mit_model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")
    mit_model.fit(X_train2, y_train2)
    mit_auc = roc_auc_score(y_test2, mit_model.predict_proba(X_test2)[:, 1])

    return {
        "status": "ok",
        "dropped_feature": top_feature,
        "base_auc": round(float(base_auc), 4),
        "mitigated_auc": round(float(mit_auc), 4),
        "auc_delta": round(float(mit_auc - base_auc), 4),
    }


def validate_binary_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found."

    vals = pd.Series(df[target_col]).dropna().unique()
    if len(vals) != 2:
        return False, f"Selected outcome column '{target_col}' is not binary. Found {len(vals)} unique values."

    # strict 0/1 normalization check
    try:
        vals_int = set(pd.Series(df[target_col]).dropna().astype(int).unique())
    except Exception:
        return False, f"Outcome column '{target_col}' could not be cast to integers (0/1)."

    if not vals_int.issubset({0, 1}):
        return False, f"Outcome column '{target_col}' must contain only 0/1. Found values: {sorted(vals_int)}"

    return True, "ok"


# -----------------------
# UI
# -----------------------
st.title("⚖️ AI Fairness Auditor")
st.caption("Counterfactual bias detection + proxy analysis + fairness scorecards")

with st.sidebar:
    st.header("Setup")
    domain = st.selectbox(
        "Domain",
        list(DOMAIN_CONFIGS.keys()),
        format_func=lambda x: DOMAIN_CONFIGS[x]["label"],
    )
    model_choice = st.selectbox("Model", MODEL_CANDIDATES, index=MODEL_CANDIDATES.index(DEFAULT_MODEL))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    run_btn = st.button("Run Fairness Audit", width="stretch")

if uploaded is None:
    st.info("Upload a CSV to start. Optional: generate sample files with `python generate_sample_data.py`.")
    st.stop()

# load data
df = load_csv_with_fallback(uploaded)
st.subheader("Data Preview")
st.dataframe(df.head(20), width="stretch")

# target detection + selection
possible_targets = [
    c for c in df.columns
    if c.lower() in ["approved", "hired", "treatment_approved", "target", "label", "outcome"]
]
domain_default = DOMAIN_CONFIGS[domain]["outcome_col"]
if domain_default in df.columns:
    default_target = domain_default
elif possible_targets:
    default_target = possible_targets[0]
else:
    default_target = df.columns[-1]  # fallback, user can change manually

target_col = st.selectbox(
    "Outcome Column (binary 0/1)",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_target),
)

# validate before running
is_valid, msg = validate_binary_target(df, target_col)
if not is_valid:
    st.error(msg)
    st.stop()

if run_btn:
    work_df = df.copy()
    y = work_df[target_col].astype(int)

    id_col = DOMAIN_CONFIGS[domain]["id_col"]
    drop_cols = [target_col]
    if id_col and id_col in work_df.columns:
        drop_cols.append(id_col)

    features_df = work_df.drop(columns=drop_cols, errors="ignore")

    if features_df.shape[1] == 0:
        st.error("No feature columns left after removing target/id columns.")
        st.stop()

    # Layer 1: proxy detection
    detector = ProxyDetector()
    detector.fit(features_df, y)
    proxy_findings = detector.findings_
    suspicious = detector.suspicious_features()

    # Layer 2: model training + eval
    X_enc, _ = encode_dataframe(features_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.25, random_state=42, stratify=y
    )

    model = get_model(model_choice)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    auc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))

    # Layer 3: counterfactual testing
    cf_tester = CounterfactualTester(model=model)
    cf_tester.fit(features_df, y)
    cf_results = cf_tester.run(features_df, y, suspicious[:8])

    # Fairness metrics across top suspicious vars
    groups = {}
    for c in suspicious[:3]:
        if c in features_df.columns:
            groups[c] = features_df[c].reset_index(drop=True).iloc[:len(y_test)]

    fm = FairnessMetrics(
        y_true=y_test.reset_index(drop=True),
        y_pred=pd.Series(y_pred).reset_index(drop=True),
        y_proba=pd.Series(y_proba).reset_index(drop=True),
        groups=groups,
    ).compute_all()

    avg_csi = float(np.mean([r.csi for r in cf_results.values()])) if cf_results else 1.0
    overall_score = float(np.clip(0.75 * fm.overall_score_ + 25 * avg_csi, 0, 100))

    mitigation = mitigation_simulation(work_df, target_col, [f.feature for f in proxy_findings if f.is_high_risk])
    recommendations = build_recommendations(proxy_findings, cf_results, overall_score)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Fairness Score", f"{overall_score:.1f}/100")
    m2.metric("Model AUC", f"{auc:.3f}")
    m3.metric("Model Accuracy", f"{acc:.3f}")
    m4.metric("Avg CSI", f"{avg_csi:.3f}")

    # Tables
    st.subheader("Top Proxy Findings")
    st.dataframe(detector.summary_df(), width="stretch")

    st.subheader("Counterfactual Results")
    if cf_results:
        st.dataframe(pd.DataFrame([r.to_dict() for r in cf_results.values()]), width="stretch")
    else:
        st.info("No counterfactual results were generated.")

    st.subheader("Fairness Metrics Scorecard")
    st.dataframe(fm.summary_df(), width="stretch")

    st.subheader("Mitigation Simulator (What-If)")
    st.json(mitigation)

    # HTML report export
    report_html = generate_html_report(
        dataset_name=uploaded.name,
        overall_score=overall_score,
        proxy_findings=proxy_findings,
        cf_results=cf_results,
        fairness_results=fm.results_,
        recommendations=recommendations,
        model_accuracy=acc,
        model_auc=auc,
    )

    st.download_button(
        "Download HTML Report",
        data=report_html.encode("utf-8"),
        file_name=f"fairness_report_{uploaded.name.rsplit('.', 1)[0]}.html",
        mime="text/html",
    )
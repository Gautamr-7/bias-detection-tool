import io
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
        return LogisticRegression(max_iter=500)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    return RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1)


def build_recommendations(proxy_findings, cf_results, fairness_score):
    recs = []

    if any(f.is_high_risk for f in proxy_findings):
        top = [f.feature for f in proxy_findings if f.is_high_risk][:3]
        recs.append({
            "priority": "P0",
            "effort": "Medium",
            "action": f"Constrain or remove high-risk proxy variables: {', '.join(top)}",
            "rationale": "These features show strong proxy behavior and likely contribute to discriminatory outcomes.",
        })

    if cf_results and any(r.is_significant for r in cf_results.values()):
        recs.append({
            "priority": "P0",
            "effort": "High",
            "action": "Run mitigation retraining with proxy suppression and monotonic constraints",
            "rationale": "Counterfactual tests show causal decision flips, indicating potentially harmful model behavior.",
        })

    if fairness_score < 70:
        recs.append({
            "priority": "P1",
            "effort": "Medium",
            "action": "Establish pre-deployment fairness gates in CI/CD",
            "rationale": "Current fairness profile is below safe threshold and needs automated policy enforcement.",
        })

    recs.append({
        "priority": "P2",
        "effort": "Low",
        "action": "Schedule monthly fairness drift monitoring",
        "rationale": "Fairness can degrade over time due to data shift and feedback loops.",
    })

    return recs


def mitigation_simulation(df, y_col, risky_features):
    """
    Simple what-if mitigation:
    drop top risky feature(s), retrain RF, compare AUC and approval disparity.
    """
    if not risky_features:
        return {"status": "No high-risk features found."}

    base_X = df.drop(columns=[y_col])
    y = df[y_col].astype(int)

    base_X_enc, _ = encode_dataframe(base_X)
    X_train, X_test, y_train, y_test = train_test_split(base_X_enc, y, test_size=0.25, random_state=42, stratify=y)
    base_model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    base_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])

    # remove top 1 risky feature
    drop_feature = risky_features[0]
    if drop_feature not in base_X.columns:
        return {"status": f"Feature {drop_feature} unavailable for mitigation sim."}

    mit_X = base_X.drop(columns=[drop_feature])
    mit_X_enc, _ = encode_dataframe(mit_X)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(mit_X_enc, y, test_size=0.25, random_state=42, stratify=y)
    mit_model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")
    mit_model.fit(X_train2, y_train2)
    mit_auc = roc_auc_score(y_test2, mit_model.predict_proba(X_test2)[:, 1])

    return {
        "status": "ok",
        "dropped_feature": drop_feature,
        "base_auc": round(float(base_auc), 4),
        "mitigated_auc": round(float(mit_auc), 4),
        "auc_delta": round(float(mit_auc - base_auc), 4),
        "base_positive_rate": round(float(np.mean(base_pred)), 4),
    }


st.title("⚖️ AI Fairness Auditor")
st.caption("Counterfactual bias detection + proxy analysis + fairness scorecards")

with st.sidebar:
    st.header("Setup")
    domain = st.selectbox("Domain", list(DOMAIN_CONFIGS.keys()), format_func=lambda x: DOMAIN_CONFIGS[x]["label"])
    model_choice = st.selectbox("Model", MODEL_CANDIDATES, index=MODEL_CANDIDATES.index(DEFAULT_MODEL))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    run_btn = st.button("Run Fairness Audit", use_container_width=True)

if uploaded is None:
    st.info("Upload a CSV to start. Optional: generate sample files with `python generate_sample_data.py`.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

possible_targets = [c for c in df.columns if c.lower() in ["approved", "hired", "treatment_approved", "outcome", "label", "target"]]
default_target = DOMAIN_CONFIGS[domain]["outcome_col"] if DOMAIN_CONFIGS[domain]["outcome_col"] in df.columns else (possible_targets[0] if possible_targets else None)

if default_target is None:
    st.error("Could not auto-detect target column.")
    st.stop()

target_col = st.selectbox("Outcome Column (binary 0/1)", df.columns, index=list(df.columns).index(default_target))
id_col = DOMAIN_CONFIGS[domain]["id_col"] if DOMAIN_CONFIGS[domain]["id_col"] in df.columns else None

if run_btn:
    work_df = df.copy()
    if id_col and id_col in work_df.columns:
        features_df = work_df.drop(columns=[id_col, target_col])
    else:
        features_df = work_df.drop(columns=[target_col])

    y = work_df[target_col].astype(int)

    # 1) Proxy detection
    detector = ProxyDetector()
    detector.fit(features_df, y)
    proxy_findings = detector.findings_
    suspicious = detector.suspicious_features()

    # 2) Train model + predictions
    X_enc, _ = encode_dataframe(features_df)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=42, stratify=y)
    model = get_model(model_choice)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    auc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))

    # 3) Counterfactual testing
    cf = CounterfactualTester(model=model)
    cf.fit(features_df, y)  # fit own scaler + consistency
    cf_results = cf.run(features_df, y, suspicious[:8])

    # 4) Fairness metrics (use selected proxies as group vars where possible)
    group_vars = {}
    for c in suspicious[:3]:
        if c in features_df.columns:
            group_vars[c] = features_df[c]

    fm = FairnessMetrics(y_true=y_test.reset_index(drop=True), y_pred=pd.Series(y_pred), y_proba=pd.Series(y_proba), groups={k: group_vars[k].iloc[:len(y_test)].reset_index(drop=True) for k in group_vars})
    fm.compute_all()

    # 5) Overall score blend
    fairness_score = fm.overall_score_
    avg_csi = np.mean([r.csi for r in cf_results.values()]) if cf_results else 1.0
    overall_score = max(0.0, min(100.0, 0.75 * fairness_score + 25 * avg_csi))

    # 6) Mitigation simulation
    mitigation = mitigation_simulation(work_df, target_col, [f.feature for f in proxy_findings if f.is_high_risk])

    # UI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Fairness Score", f"{overall_score:.1f}/100")
    c2.metric("Model AUC", f"{auc:.3f}")
    c3.metric("Model Accuracy", f"{acc:.3f}")
    c4.metric("Avg Counterfactual Stability Index", f"{avg_csi:.3f}")

    st.subheader("Top Proxy Findings")
    st.dataframe(detector.summary_df(), use_container_width=True)

    st.subheader("Counterfactual Results")
    if cf_results:
        st.dataframe(pd.DataFrame([r.to_dict() for r in cf_results.values()]), use_container_width=True)
    else:
        st.info("No counterfactual results generated.")

    st.subheader("Fairness Metrics")
    st.dataframe(fm.summary_df(), use_container_width=True)

    st.subheader("Mitigation Simulator (What-If)")
    st.json(mitigation)

    recommendations = build_recommendations(proxy_findings, cf_results, overall_score)

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
        file_name=f"fairness_report_{uploaded.name.rsplit('.',1)[0]}.html",
        mime="text/html",
    )
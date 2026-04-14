import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bias Detection & Fairness Auditing Tool",
    page_icon="⚖️",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0fdff; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #0891b2;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .red-card { border-left: 5px solid #ef4444; }
    .green-card { border-left: 5px solid #22c55e; }
    .title-banner {
        background: linear-gradient(135deg, #0891b2, #06b6d4);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #0891b2;
        border-bottom: 2px solid #e0f7fa;
        padding-bottom: 6px;
        margin: 20px 0 14px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-banner">
    <h1>⚖️ Bias Detection & Fairness Auditing Tool</h1>
    <p style="font-size:16px; opacity:0.9;">
        Detect, visualize, and mitigate bias in machine learning models — 
        for auditable, fair, and governable AI
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=80)
    st.title("⚙️ Settings")
    st.markdown("---")
    protected = st.selectbox(
        "Protected Attribute",
        ["sex", "race"],
        help="The demographic attribute to audit for bias"
    )
    st.markdown("---")
    st.markdown("### 📖 What is Bias?")
    st.info(
        "AI bias occurs when a model makes unfair decisions based on "
        "sensitive attributes like gender or race — often learned from "
        "historical data. This tool detects and fixes that."
    )
    st.markdown("### 📊 Fairness Threshold")
    st.success("Disparate Impact ≥ 0.80 = Fair ✅")
    st.error("Disparate Impact < 0.80 = Biased ❌")

# ─── Load Dataset ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(protected_attr):
    priv = [['Male']] if protected_attr == 'sex' else [['White']]
    dataset = AdultDataset(
        protected_attribute_names=[protected_attr],
        privileged_classes=priv,
        features_to_drop=['fnlwgt']
    )
    return dataset

@st.cache_data
def get_metrics(protected_attr):
    priv = [{'sex': 1}] if protected_attr == 'sex' else [{'race': 1}]
    unpriv = [{'sex': 0}] if protected_attr == 'sex' else [{'race': 0}]
    dataset = load_data(protected_attr)
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=priv,
        unprivileged_groups=unpriv
    )
    return metric, dataset, priv, unpriv

dataset_load = st.empty()
with dataset_load:
    with st.spinner("Loading Adult Income Dataset..."):
        metric, dataset, priv_groups, unpriv_groups = get_metrics(protected)

dataset_load.empty()

# ─── Split dataset ────────────────────────────────────────────────────────────
train, test = dataset.split([0.7], shuffle=True, seed=42)

# ─── Train BIASED model ───────────────────────────────────────────────────────
@st.cache_resource
def train_biased(protected_attr):
    d = load_data(protected_attr)
    tr, te = d.split([0.7], shuffle=True, seed=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr.features)
    X_te = sc.transform(te.features)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, tr.labels.ravel())
    preds = clf.predict(X_te)
    pred_ds = te.copy()
    pred_ds.labels = preds.reshape(-1, 1)
    return pred_ds, te, clf, sc, accuracy_score(te.labels, preds)

@st.cache_resource
def train_fair(protected_attr):
    d = load_data(protected_attr)
    priv = [{'sex': 1}] if protected_attr == 'sex' else [{'race': 1}]
    unpriv = [{'sex': 0}] if protected_attr == 'sex' else [{'race': 0}]
    tr, te = d.split([0.7], shuffle=True, seed=42)
    rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
    tr_rw = rw.fit_transform(tr)
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr_rw.features)
    X_te = sc.transform(te.features)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, tr_rw.labels.ravel(), sample_weight=tr_rw.instance_weights)
    preds = clf.predict(X_te)
    pred_ds = te.copy()
    pred_ds.labels = preds.reshape(-1, 1)
    return pred_ds, te, accuracy_score(te.labels, preds)

with st.spinner("Training models..."):
    biased_preds, test_ds, clf_b, scaler, acc_biased = train_biased(protected)
    fair_preds, _, acc_fair = train_fair(protected)

# ─── Compute metrics ─────────────────────────────────────────────────────────
def get_clf_metric(pred_ds, test_d):
    return ClassificationMetric(
        test_d, pred_ds,
        privileged_groups=priv_groups,
        unprivileged_groups=unpriv_groups
    )

m_biased = get_clf_metric(biased_preds, test_ds)
m_fair   = get_clf_metric(fair_preds, test_ds)

di_before = m_biased.disparate_impact()
di_after  = m_fair.disparate_impact()
spd_before = m_biased.statistical_parity_difference()
spd_after  = m_fair.statistical_parity_difference()
eod_before = m_biased.equal_opportunity_difference()
eod_after  = m_fair.equal_opportunity_difference()

# ─── Section 1: Dataset Overview ─────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Dataset Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", f"{len(dataset.features):,}")
col2.metric("Features", f"{dataset.features.shape[1]}")
col3.metric("Protected Attribute", protected.capitalize())
col4.metric("Prediction Task", "Income > 50K")

priv_label = "Male" if protected == "sex" else "White"
unpriv_label = "Female" if protected == "sex" else "Non-White"

# Show group distribution
priv_idx = dataset.protected_attributes[:, 0] == 1
n_priv = priv_idx.sum()
n_unpriv = (~priv_idx).sum()

st.markdown(f"**Group Breakdown:** {priv_label}: `{n_priv:,}` samples &nbsp;|&nbsp; {unpriv_label}: `{n_unpriv:,}` samples")

# ─── Section 2: Bias Metrics ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Bias Detection Results</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

def metric_card(col, title, before, after, ideal, invert=False):
    bias_flag = "❌ Biased" if (before < ideal if not invert else before > ideal) else "✅ Fair"
    fix_flag  = "✅ Fixed"  if (after  >= ideal if not invert else after  <= ideal) else "⚠️ Improved"
    col.markdown(f"""
    <div class="metric-card {'red-card' if '❌' in bias_flag else 'green-card'}">
        <b>{title}</b><br>
        Before: <b>{before:.3f}</b> {bias_flag}<br>
        After:  <b>{after:.3f}</b> {fix_flag}<br>
        <small>Ideal: {ideal}</small>
    </div>""", unsafe_allow_html=True)

metric_card(col1, "Disparate Impact",            di_before,  di_after,  "≥ 0.80")
metric_card(col2, "Statistical Parity Diff",     spd_before, spd_after, "≈ 0.00", invert=True)
metric_card(col3, "Equal Opportunity Diff",      eod_before, eod_after, "≈ 0.00", invert=True)

# ─── Section 3: Charts ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visual Analysis</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#f0fdff')

cyan  = '#0891b2'
green = '#22c55e'
red   = '#ef4444'
gray  = '#e5e7eb'

# Chart 1 — Disparate Impact
bars = axes[0].bar(['Before\nMitigation', 'After\nMitigation'],
                   [di_before, di_after],
                   color=[red if di_before < 0.8 else green, green if di_after >= 0.8 else '#f97316'],
                   width=0.5, edgecolor='white', linewidth=2)
axes[0].axhline(0.8, color=cyan, linestyle='--', linewidth=2, label='Fair Threshold (0.80)')
axes[0].set_title('Disparate Impact', fontweight='bold', color='#0e7490')
axes[0].set_ylim(0, 1.2)
axes[0].legend(fontsize=8)
axes[0].set_facecolor('#f0fdff')
for bar, val in zip(bars, [di_before, di_after]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)

# Chart 2 — Model Accuracy
bars2 = axes[1].bar(['Biased\nModel', 'Fair\nModel'],
                    [acc_biased*100, acc_fair*100],
                    color=[red, green], width=0.5, edgecolor='white', linewidth=2)
axes[1].set_title('Model Accuracy (%)', fontweight='bold', color='#0e7490')
axes[1].set_ylim(0, 100)
axes[1].set_facecolor('#f0fdff')
for bar, val in zip(bars2, [acc_biased*100, acc_fair*100]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

# Chart 3 — All metrics comparison
metrics_names  = ['Disparate\nImpact', 'Stat Parity\nDiff', 'Equal Opp\nDiff']
before_vals = [di_before, abs(spd_before), abs(eod_before)]
after_vals  = [di_after,  abs(spd_after),  abs(eod_after)]
x = np.arange(len(metrics_names))
w = 0.3
axes[2].bar(x - w/2, before_vals, w, label='Before', color=red,  edgecolor='white', linewidth=1.5)
axes[2].bar(x + w/2, after_vals,  w, label='After',  color=green, edgecolor='white', linewidth=1.5)
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics_names, fontsize=9)
axes[2].set_title('All Fairness Metrics', fontweight='bold', color='#0e7490')
axes[2].legend()
axes[2].set_facecolor('#f0fdff')

plt.tight_layout()
st.pyplot(fig)

# ─── Section 4: What does it mean ────────────────────────────────────────────
st.markdown('<div class="section-header">💡 What This Means</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.error(f"""
    **❌ Before Mitigation**
    
    The model had a Disparate Impact of **{di_before:.2f}** — below the 0.80 fairness threshold.
    
    This means **{unpriv_label}** individuals were being approved at only **{di_before*100:.0f}%** 
    the rate of **{priv_label}** individuals, for the same qualifications.
    
    This is a textbook case of **algorithmic discrimination** — the model learned historical bias from the data.
    """)
with col2:
    st.success(f"""
    **✅ After Mitigation (Reweighing)**
    
    After applying the **Reweighing** algorithm, Disparate Impact improved to **{di_after:.2f}**.
    
    The model now treats both groups more equitably — and model accuracy remained at **{acc_fair*100:.1f}%**, 
    showing that **fairness and accuracy are not mutually exclusive**.
    
    Technique used: **IBM AIF360 Reweighing** — assigns weights to training samples to reduce bias.
    """)

# ─── Section 5: How It Works ─────────────────────────────────────────────────
with st.expander("📚 How Does This Tool Work?"):
    st.markdown(f"""
    ### Pipeline

    1. **Dataset** — Adult Income Dataset (UCI/AIF360): predicts if income > $50K  
       Contains sensitive attributes: `sex`, `race`

    2. **Bias Detection** — Three fairness metrics are computed:
       - **Disparate Impact**: ratio of positive outcome rates between groups (ideal ≥ 0.80)
       - **Statistical Parity Difference**: difference in positive prediction rates (ideal = 0)
       - **Equal Opportunity Difference**: difference in true positive rates (ideal = 0)

    3. **Model Training** — Logistic Regression baseline model trained on raw data

    4. **Mitigation** — **Reweighing** (IBM AIF360): adjusts sample weights in training data 
       so underrepresented groups get fair representation — without changing the data itself

    5. **Comparison** — Before vs After metrics visualized to show improvement

    ### Key Concept
    > If a loan approval model approves {priv_label}s 80% of the time but {unpriv_label}s only 50% of the time 
    > for similar profiles — that's bias. Disparate Impact = 50/80 = **0.625** → below 0.80 threshold → **biased**.
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built by <b>Gautam R</b> · Flink · Model Engineering College · "
    "Aligns with auditable & governable AI research · IBM AIF360</small></center>",
    unsafe_allow_html=True
)

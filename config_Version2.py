"""
Centralized configuration for the AI Fairness Auditor.
"""

from dataclasses import dataclass

# ----------------------------
# Global thresholds
# ----------------------------
PROXY_SUSPICION_THRESHOLD = 0.15
PROXY_HIGH_RISK_THRESHOLD = 0.30
TOP_K_PROXIES = 12

CF_MAX_SAMPLES = 700
CF_MIN_GROUP_SIZE = 8
CF_SEVERITY_MODERATE = 0.10
CF_SEVERITY_CRITICAL = 0.20

DEMOGRAPHIC_PARITY_TOLERANCE = 0.10
EQUALIZED_ODDS_TOLERANCE = 0.10
PREDICTIVE_PARITY_TOLERANCE = 0.10
CALIBRATION_TOLERANCE = 0.10
DISPARATE_IMPACT_MIN = 0.80

BOOTSTRAP_ITER = 300
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.25

# Counterfactual
N_COUNTERFACTUAL_SAMPLES = 300

# RF defaults
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12
RF_RANDOM_STATE = 42

# UI thresholds
SCORE_EXCELLENT = 90
SCORE_MODERATE = 70

PALETTE = {
    "danger":   "#EF4444",
    "warning":  "#F59E0B",
    "success":  "#10B981",
    "accent":   "#6366F1",
    "muted":    "#94A3B8",
    "bg_dark":  "#0B1220",
    "bg_card":  "#111827",
    "bg_hover": "#1F2937",
    "border":   "#334155",
    "text":     "#E2E8F0",
    "text_dim": "#94A3B8",
}

DEFAULT_MODEL = "random_forest"

MODEL_CANDIDATES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]

DOMAIN_CONFIGS = {
    "lending": {
        "label": "💳 Lending / Credit",
        "outcome_col": "approved",
        "id_col": "applicant_id",
        "known_proxies": ["zip_code", "application_hour", "neighborhood"],
        "legitimate": ["credit_score", "income", "debt_to_income", "loan_amount"],
        "regulation": "ECOA / Fair Housing Act",
    },
    "hiring": {
        "label": "💼 Hiring / Recruitment",
        "outcome_col": "hired",
        "id_col": "candidate_id",
        "known_proxies": ["first_name", "university"],
        "legitimate": ["years_experience", "skill_score", "gpa"],
        "regulation": "Title VII / NYC Local Law 144",
    },
    "healthcare": {
        "label": "🏥 Healthcare / Insurance",
        "outcome_col": "treatment_approved",
        "id_col": "patient_id",
        "known_proxies": ["insurance_type", "neighborhood"],
        "legitimate": ["severity_score", "age", "prior_visits"],
        "regulation": "ACA Section 1557 / ADA",
    },
    "custom": {
        "label": "⚙️ Custom Dataset",
        "outcome_col": None,
        "id_col": None,
        "known_proxies": [],
        "legitimate": [],
        "regulation": "Varies",
    },
}
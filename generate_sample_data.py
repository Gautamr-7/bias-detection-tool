import numpy as np
import pandas as pd


def generate_loan_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    n_a = int(n * 0.45)

    for i in range(n):
        in_a = i < n_a
        zip_code = int(np.random.randint(10000, 10050) if in_a else np.random.randint(10050, 10100))
        credit_score = int(np.clip(np.random.normal(685 if in_a else 715, 68), 300, 850))
        income = int(np.clip(np.random.lognormal(np.log(58000 if in_a else 74000), 0.42), 20000, 400000))
        age = int(np.clip(np.random.normal(38, 11), 22, 72))
        dti = round(float(np.clip(np.random.beta(2, 5) * 0.85, 0.02, 0.90)), 3)
        loan_amount = int(np.clip(np.random.lognormal(np.log(16000), 0.5), 2000, 150000))
        app_hour = int(np.random.choice(list(range(9, 24))))

        merit = (
            (credit_score - 300) / 550 * 0.45
            + min(income / 120000, 1.0) * 0.30
            + (1 - dti) * 0.25
        )
        prob = float(np.clip(merit - (0.15 if in_a else 0.0) + np.random.normal(0, 0.08), 0.03, 0.97))
        approved = int(np.random.random() < prob)

        rows.append({
            "applicant_id": f"APP_{i+1:05d}",
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "debt_to_income": dti,
            "loan_amount": loan_amount,
            "zip_code": zip_code,
            "application_hour": app_hour,
            "approved": approved,
        })

    df = pd.DataFrame(rows)
    df.to_csv("sample_loan_data.csv", index=False)
    return df


def generate_hiring_data(n: int = 900, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    group_a = ["James", "John", "Robert", "Michael", "William", "David", "Christopher", "Andrew"]
    group_b = ["DeShawn", "DeAndre", "Marquis", "Darnell", "Tyrone", "Jamal", "Leroy", "Reginald"]
    elite = ["State University", "Tech Institute", "National Business School"]
    regional = ["Community College", "Online University", "City College", "Regional State"]

    rows = []
    for i in range(n):
        in_a = np.random.random() < 0.55
        name = np.random.choice(group_a if in_a else group_b)
        school = np.random.choice(elite if np.random.random() < 0.48 else regional)

        yrs_exp = int(min(np.random.exponential(4) + 1, 22))
        gpa = round(float(np.clip(np.random.normal(3.2, 0.38), 2.0, 4.0)), 2)
        skill_score = int(np.clip(np.random.normal(72, 15), 30, 100))

        merit = (
            min(yrs_exp / 10, 1.0) * 0.35
            + (gpa - 2.0) / 2.0 * 0.30
            + (skill_score - 30) / 70 * 0.35
        )
        prob = float(np.clip(merit + (0.04 if school in elite else 0.0) - (0.0 if in_a else 0.12) + np.random.normal(0, 0.07), 0.03, 0.97))
        hired = int(np.random.random() < prob)

        rows.append({
            "candidate_id": f"CAND_{i+1:04d}",
            "first_name": name,
            "years_experience": yrs_exp,
            "gpa": gpa,
            "skill_score": skill_score,
            "university": school,
            "hired": hired,
        })

    df = pd.DataFrame(rows)
    df.to_csv("sample_hiring_data.csv", index=False)
    return df


def generate_healthcare_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []

    for i in range(n):
        insurance = np.random.choice(["Private", "Medicare", "Medicaid"], p=[0.44, 0.31, 0.25])
        neighborhood = np.random.choice(["Affluent", "Suburban", "Underserved", "Rural"], p=[0.24, 0.36, 0.24, 0.16])
        age = int(np.clip(np.random.normal(52, 18), 18, 90))
        severity = int(np.clip(np.random.normal(55, 20), 10, 100))
        prior_visits = int(min(np.random.exponential(3), 20))

        merit = severity / 100 * 0.60 + min(prior_visits / 10, 1.0) * 0.40
        ins_pen = {"Private": 0.00, "Medicare": 0.06, "Medicaid": 0.16}[insurance]
        nbhd_pen = {"Affluent": 0.00, "Suburban": 0.02, "Underserved": 0.13, "Rural": 0.09}[neighborhood]
        prob = float(np.clip(merit - ins_pen - nbhd_pen + np.random.normal(0, 0.07), 0.03, 0.97))
        approved = int(np.random.random() < prob)

        rows.append({
            "patient_id": f"PAT_{i+1:04d}",
            "age": age,
            "severity_score": severity,
            "prior_visits": prior_visits,
            "insurance_type": insurance,
            "neighborhood": neighborhood,
            "treatment_approved": approved,
        })

    df = pd.DataFrame(rows)
    df.to_csv("sample_healthcare_data.csv", index=False)
    return df


if __name__ == "__main__":
    generate_loan_data(1200)
    generate_hiring_data(900)
    generate_healthcare_data(1000)
    print("All sample datasets generated.")

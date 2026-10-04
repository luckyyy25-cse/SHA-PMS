import os
import pandas as pd
import numpy as np

def generate_hospital_dataset(n_samples=1500, output_path="data/hospital_appointments.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    patient_ids = [f"PAT-{1000 + i}" for i in range(n_samples)]
    ages = np.random.randint(1, 90, size=n_samples)
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.45, 0.55])
    
    departments = ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology", "Dermatology"]
    dept_weights = [0.30, 0.20, 0.18, 0.12, 0.10, 0.10]
    assigned_depts = np.random.choice(departments, size=n_samples, p=dept_weights)

    # Lead days (days between booking and appointment)
    lead_days = np.random.exponential(scale=10, size=n_samples).astype(int)
    lead_days = np.clip(lead_days, 0, 60)

    # Medical conditions dependent on age
    hypertension = ((ages > 45) & (np.random.rand(n_samples) < 0.45)).astype(int)
    diabetes = ((ages > 40) & (np.random.rand(n_samples) < 0.30)).astype(int)
    alcoholism = ((ages > 18) & (np.random.rand(n_samples) < 0.08)).astype(int)

    previous_no_shows = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.70, 0.15, 0.08, 0.04, 0.03])
    sms_reminder = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])

    # Realistic No-Show probability formula
    # Higher lead days, higher past no-shows increase risk. SMS reminder decreases risk.
    prob_score = (
        0.15 
        + 0.025 * lead_days 
        + 0.35 * previous_no_shows 
        - 0.25 * sms_reminder 
        - 0.002 * ages 
        + 0.10 * alcoholism
    )
    
    # Sigmoidal transformation for probability
    probs = 1 / (1 + np.exp(-prob_score))
    # Thresholding with noise
    no_show = (np.random.rand(n_samples) < probs).astype(int)

    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "Department": assigned_depts,
        "Lead_Days": lead_days,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Previous_No_Shows": previous_no_shows,
        "SMS_Reminder": sms_reminder,
        "No_Show": no_show
    })

    # Add a few intentional missing values to demonstrate data wrangling/preprocessing
    missing_idx_age = np.random.choice(df.index, size=15, replace=False)
    missing_idx_lead = np.random.choice(df.index, size=10, replace=False)
    df.loc[missing_idx_age, "Age"] = np.nan
    df.loc[missing_idx_lead, "Lead_Days"] = np.nan

    df.to_csv(output_path, index=False)
    print(f"Successfully generated dataset with {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_hospital_dataset()

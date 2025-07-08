"""
Data Generation Script for Clinical Study Churn Prediction
Generates synthetic clinical study data with realistic patterns
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

from config import *


def generate_synthetic_clinical_data(n_records=NUM_RECORDS, random_seed=RANDOM_SEED):
    """
    Generate synthetic clinical study data with realistic patterns

    Args:
        n_records (int): Number of records to generate
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Synthetic clinical study data
    """
    np.random.seed(random_seed)

    # Generate participant IDs
    participant_ids = [f"P{i:06d}" for i in range(1, n_records + 1)]

    # Generate basic demographics
    ages = np.random.normal(55, 15, n_records)
    ages = np.clip(ages, AGE_RANGE[0], AGE_RANGE[1])

    genders = np.random.choice(GENDERS, n_records, p=[0.45, 0.45, 0.1])

    # Income with some correlation to age and location
    base_income = np.random.normal(60000, 20000, n_records)
    age_factor = (ages - 40) * 500  # Older participants tend to have higher income
    income = base_income + age_factor
    income = np.clip(income, INCOME_RANGE[0], INCOME_RANGE[1])

    # Location with some correlation to income
    location_probs = [0.4, 0.35, 0.25]  # Urban, Suburban, Rural
    locations = np.random.choice(LOCATIONS, n_records, p=location_probs)

    # Study type and condition
    study_types = np.random.choice(STUDY_TYPES, n_records, p=[0.2, 0.5, 0.3])
    conditions = np.random.choice(
        CONDITIONS, n_records, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]
    )

    # Visit adherence with correlation to age, income, and location
    base_adherence = np.random.beta(3, 1, n_records)  # Beta distribution for adherence
    age_adherence_factor = (
        (85 - ages) / 67 * 0.2
    )  # Younger participants slightly less adherent
    income_adherence_factor = (
        (income - 20000) / 130000 * 0.1
    )  # Higher income slightly more adherent
    visit_adherence_rate = (
        base_adherence + age_adherence_factor + income_adherence_factor
    )
    visit_adherence_rate = np.clip(
        visit_adherence_rate, VISIT_ADHERENCE_RANGE[0], VISIT_ADHERENCE_RANGE[1]
    )

    # Tenure in months
    tenure_months = np.random.exponential(12, n_records)
    tenure_months = np.clip(tenure_months, TENURE_RANGE[0], TENURE_RANGE[1])

    # Last visit gap (correlated with adherence and tenure)
    base_gap = np.random.exponential(15, n_records)
    adherence_factor = (1 - visit_adherence_rate) * 30  # Lower adherence = longer gaps
    last_visit_gap_days = base_gap + adherence_factor
    last_visit_gap_days = np.clip(
        last_visit_gap_days, LAST_VISIT_GAP_RANGE[0], LAST_VISIT_GAP_RANGE[1]
    )

    # Number of medications
    num_medications = np.random.poisson(3, n_records)
    num_medications = np.clip(
        num_medications, NUM_MEDICATIONS_RANGE[0], NUM_MEDICATIONS_RANGE[1]
    )

    # Side effects (correlated with number of medications)
    side_effect_prob = (
        0.3 + (num_medications / 8) * 0.4
    )  # More medications = higher side effect risk
    has_side_effects = np.random.binomial(1, side_effect_prob, n_records)

    # Transport support (correlated with location and income)
    transport_prob = (
        0.2
        + (np.array([loc == "Rural" for loc in locations]) * 0.3)
        + (income < 40000) * 0.2
    )
    transport_support = np.random.binomial(1, transport_prob, n_records)

    # Monthly stipend (correlated with study type and condition)
    base_stipend = np.random.normal(400, 150, n_records)
    study_type_bonus = np.array([0, 100, 200])[
        np.array([st == "Phase I" for st in study_types]) * 0
        + np.array([st == "Phase II" for st in study_types]) * 1
        + np.array([st == "Phase III" for st in study_types]) * 2
    ]
    monthly_stipend = base_stipend + study_type_bonus
    monthly_stipend = np.clip(
        monthly_stipend, MONTHLY_STIPEND_RANGE[0], MONTHLY_STIPEND_RANGE[1]
    )

    # Contact frequency (correlated with adherence and tenure)
    base_contact = np.random.poisson(3, n_records)
    adherence_contact_factor = (
        1 - visit_adherence_rate
    ) * 2  # Lower adherence = more contact
    contact_frequency = base_contact + adherence_contact_factor
    contact_frequency = np.clip(
        contact_frequency, CONTACT_FREQUENCY_RANGE[0], CONTACT_FREQUENCY_RANGE[1]
    )

    # Support group membership (correlated with condition and age)
    support_prob = (
        0.15
        + (np.array([cond in ["Mental Health", "Cancer"] for cond in conditions]) * 0.2)
        + (ages > 60) * 0.1
    )
    support_group_member = np.random.binomial(1, support_prob, n_records)

    # Language barrier (correlated with location and income)
    language_prob = (
        0.1
        + (np.array([loc == "Urban" for loc in locations]) * 0.05)
        + (income < 30000) * 0.1
    )
    language_barrier = np.random.binomial(1, language_prob, n_records)

    # Device usage compliance (correlated with age and income)
    base_compliance = np.random.beta(2, 1, n_records)
    age_compliance_factor = (
        (85 - ages) / 67 * 0.3
    )  # Younger participants less compliant
    income_compliance_factor = (
        (income - 20000) / 130000 * 0.2
    )  # Higher income more compliant
    device_usage_compliance = (
        base_compliance + age_compliance_factor + income_compliance_factor
    )
    device_usage_compliance = np.clip(
        device_usage_compliance, DEVICE_COMPLIANCE_RANGE[0], DEVICE_COMPLIANCE_RANGE[1]
    )

    # Survey score (correlated with multiple factors)
    base_score = np.random.normal(7, 1.5, n_records)
    adherence_score_factor = (
        visit_adherence_rate * 2
    )  # Higher adherence = higher satisfaction
    stipend_score_factor = (
        (monthly_stipend - 100) / 900 * 1
    )  # Higher stipend = higher satisfaction
    side_effect_score_factor = has_side_effects * (
        -1
    )  # Side effects = lower satisfaction
    survey_score_avg = (
        base_score
        + adherence_score_factor
        + stipend_score_factor
        + side_effect_score_factor
    )
    survey_score_avg = np.clip(
        survey_score_avg, SURVEY_SCORE_RANGE[0], SURVEY_SCORE_RANGE[1]
    )

    # Generate churn target with realistic patterns
    churn_prob = generate_churn_probability(
        visit_adherence_rate,
        last_visit_gap_days,
        tenure_months,
        has_side_effects,
        survey_score_avg,
        contact_frequency,
        device_usage_compliance,
        monthly_stipend,
        income,
    )
    churned = np.random.binomial(1, churn_prob, n_records)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "age": ages,
            "gender": genders,
            "income": income,
            "location": locations,
            "study_type": study_types,
            "condition": conditions,
            "visit_adherence_rate": visit_adherence_rate,
            "tenure_months": tenure_months,
            "last_visit_gap_days": last_visit_gap_days,
            "num_medications": num_medications,
            "has_side_effects": has_side_effects,
            "transport_support": transport_support,
            "monthly_stipend": monthly_stipend,
            "contact_frequency": contact_frequency,
            "support_group_member": support_group_member,
            "language_barrier": language_barrier,
            "device_usage_compliance": device_usage_compliance,
            "survey_score_avg": survey_score_avg,
            "churned": churned,
        }
    )

    return data


def generate_churn_probability(
    visit_adherence,
    last_visit_gap,
    tenure,
    has_side_effects,
    survey_score,
    contact_freq,
    device_compliance,
    stipend,
    income,
):
    """
    Generate realistic churn probabilities based on feature interactions

    Args:
        Various feature arrays

    Returns:
        np.array: Churn probabilities
    """
    # Base churn probability
    base_prob = 0.15

    # Feature effects on churn
    adherence_effect = (1 - visit_adherence) * 0.4  # Lower adherence = higher churn
    gap_effect = (last_visit_gap / 90) * 0.3  # Longer gaps = higher churn
    tenure_effect = (tenure / 36) * (-0.2)  # Longer tenure = lower churn
    side_effect_impact = has_side_effects * 0.15  # Side effects = higher churn
    satisfaction_effect = (
        (10 - survey_score) / 10
    ) * 0.25  # Lower satisfaction = higher churn
    contact_effect = (
        contact_freq / 8
    ) * 0.1  # More contact = slightly higher churn (intervention)
    compliance_effect = (1 - device_compliance) * 0.2  # Lower compliance = higher churn
    stipend_effect = ((1000 - stipend) / 900) * 0.1  # Lower stipend = higher churn
    income_effect = (
        (150000 - income) / 130000
    ) * 0.05  # Lower income = slightly higher churn

    # Combine all effects
    churn_prob = (
        base_prob
        + adherence_effect
        + gap_effect
        + tenure_effect
        + side_effect_impact
        + satisfaction_effect
        + contact_effect
        + compliance_effect
        + stipend_effect
        + income_effect
    )

    # Ensure probabilities are between 0 and 1
    churn_prob = np.clip(churn_prob, 0.01, 0.95)

    return churn_prob


def save_data(data, filepath):
    """
    Save data to CSV file

    Args:
        data (pd.DataFrame): Data to save
        filepath (str): File path to save to
    """
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print(f"Dataset shape: {data.shape}")
    print(f"Churn rate: {data['churned'].mean():.3f}")


def main():
    """Main function to generate and save synthetic clinical data"""
    print("Generating synthetic clinical study data...")

    # Ensure directories exist
    ensure_directories()

    # Generate data
    data = generate_synthetic_clinical_data()

    # Save data
    save_data(data, RAW_DATA_FILE)

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(data)}")
    print(f"Churn rate: {data['churned'].mean():.3f}")
    print(f"Average age: {data['age'].mean():.1f}")
    print(f"Average income: ${data['income'].mean():,.0f}")
    print(f"Average visit adherence: {data['visit_adherence_rate'].mean():.3f}")

    print("\nFeature distributions:")
    print(f"Gender: {data['gender'].value_counts().to_dict()}")
    print(f"Location: {data['location'].value_counts().to_dict()}")
    print(f"Study Type: {data['study_type'].value_counts().to_dict()}")
    print(f"Condition: {data['condition'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

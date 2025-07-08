"""
Streamlit Dashboard for Clinical Study Churn & CLV Prediction
Provides interactive UI for patient prediction and SHAP explainability
Enhanced with business-focused visualizations and insights
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import aiohttp
from typing import Dict, Any, List
import time
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Clinical Study Churn & CLV Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = "http://localhost:8000"


# Cache configuration for Streamlit
@st.cache_data(ttl=3600)  # 1 hour cache
def get_cached_prediction(patient_data_hash: str):
    """Cache predictions to avoid repeated API calls"""
    return None  # Will be implemented with actual caching


@st.cache_data(ttl=1800)  # 30 minutes cache
def get_cached_explanation(patient_data_hash: str):
    """Cache explanations to avoid repeated API calls"""
    return None  # Will be implemented with actual caching


def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


async def get_prediction_async(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get prediction from FastAPI backend asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/predict", json=patient_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    st.error(f"API Error: {response.status}")
                    return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


async def get_explanation_async(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get SHAP explanation from FastAPI backend asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/explain", json=patient_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    st.error(f"API Error: {response.status}")
                    return None
    except Exception as e:
        st.error(f"Error getting explanation: {str(e)}")
        return None


async def get_batch_predictions_async(
    patients_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Get batch predictions from FastAPI backend asynchronously"""
    try:
        batch_request = {"patients": patients_data}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/predict/batch", json=batch_request
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    st.error(f"Batch API Error: {response.status}")
                    return None
    except Exception as e:
        st.error(f"Error getting batch predictions: {str(e)}")
        return None


def get_prediction(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get prediction from FastAPI backend (synchronous fallback)"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=patient_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def get_explanation(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get SHAP explanation from FastAPI backend (synchronous fallback)"""
    try:
        response = requests.post(f"{API_BASE_URL}/explain", json=patient_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting explanation: {str(e)}")
        return None


def create_patient_input_form():
    """Create the patient input form"""
    st.subheader("📋 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Demographics**")
        age = st.slider("Age", 18, 85, 55, help="Patient age in years")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        income = st.slider(
            "Monthly Income ($)",
            20000,
            150000,
            60000,
            step=5000,
            help="Monthly income in USD",
        )
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

    with col2:
        st.write("**Study Information**")
        study_type = st.selectbox("Study Type", ["Phase I", "Phase II", "Phase III"])
        condition = st.selectbox(
            "Condition",
            [
                "Diabetes",
                "Hypertension",
                "Cardiovascular Disease",
                "Obesity",
                "Respiratory Disease",
                "Mental Health",
                "Cancer",
                "Autoimmune Disease",
            ],
        )
        tenure_months = st.slider("Tenure (months)", 1, 36, 12, help="Months in study")
        monthly_stipend = st.slider(
            "Monthly Stipend ($)",
            100,
            1000,
            400,
            step=50,
            help="Monthly incentive amount",
        )

    with col3:
        st.write("**Engagement Metrics**")
        visit_adherence_rate = st.slider(
            "Visit Adherence Rate",
            0.3,
            1.0,
            0.7,
            0.05,
            help="Percentage of scheduled visits attended",
        )
        last_visit_gap_days = st.slider(
            "Days Since Last Visit", 0, 90, 15, help="Days since last clinic visit"
        )
        contact_frequency = st.slider(
            "Contact Frequency", 1.0, 8.0, 3.0, 0.5, help="Staff contacts per month"
        )
        device_usage_compliance = st.slider(
            "Device Compliance",
            0.2,
            1.0,
            0.6,
            0.05,
            help="Device/wearable compliance rate",
        )

    # Additional features
    st.write("**Clinical & Support Factors**")
    col4, col5, col6 = st.columns(3)

    with col4:
        num_medications = st.slider("Number of Medications", 0, 8, 3)
        has_side_effects = st.checkbox("Has Side Effects")
        survey_score_avg = st.slider(
            "Survey Score", 1.0, 10.0, 7.0, 0.5, help="Average satisfaction score"
        )

    with col5:
        transport_support = st.checkbox("Transport Support Provided")
        support_group_member = st.checkbox("Support Group Member")

    with col6:
        language_barrier = st.checkbox("Language Barrier")

    # Create patient data dictionary
    patient_data = {
        "age": age,
        "gender": gender,
        "income": income,
        "location": location,
        "study_type": study_type,
        "condition": condition,
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
    }

    return patient_data


def display_prediction_results(prediction_result: Dict[str, Any]):
    """Display prediction results"""
    st.subheader("🎯 Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        churn_prob = prediction_result["churn_probability"]
        st.metric(label="Churn Probability", value=f"{churn_prob:.1%}", delta=None)

    with col2:
        clv_estimate = prediction_result["clv_estimate"]
        st.metric(label="CLV Estimate", value=f"${clv_estimate:,.0f}", delta=None)

    with col3:
        risk_category = prediction_result["risk_category"]
        risk_color = {
            "Low Risk": "green",
            "Medium Risk": "orange",
            "High Risk": "red",
        }.get(risk_category, "gray")

        st.markdown(
            f"""
        <div style="text-align: center;">
            <h4>Risk Category</h4>
            <p style="color: {risk_color}; font-size: 24px; font-weight: bold;">
                {risk_category}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        retention_prob = 1 - churn_prob
        st.metric(
            label="Retention Probability", value=f"{retention_prob:.1%}", delta=None
        )


def plot_shap_waterfall(shap_values: Dict[str, float]):
    """Create SHAP waterfall plot"""
    # Sort features by absolute SHAP value
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color bars based on direction
    colors = ["red" if v > 0 else "blue" for v in values]

    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title("SHAP Feature Impact Analysis", fontsize=14, fontweight="bold")

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_width() + (0.01 if value > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


def plot_shap_force(shap_values: Dict[str, float], patient_data: Dict[str, Any]):
    """Create SHAP force plot"""
    # Create a simple force plot visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort features by absolute SHAP value
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_features[:10]]  # Top 10 features
    values = [x[1] for x in sorted_features[:10]]

    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    colors = ["red" if v > 0 else "blue" for v in values]

    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("SHAP Value")
    ax.set_title(
        "SHAP Force Plot - Feature Contributions", fontsize=14, fontweight="bold"
    )

    # Add feature values as text
    for i, (feature, value) in enumerate(zip(features, values)):
        feature_value = patient_data.get(feature, "N/A")
        ax.text(
            value + (0.01 if value > 0 else -0.01),
            i,
            f" = {feature_value}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def display_shap_explanation(
    explanation_result: Dict[str, Any], patient_data: Dict[str, Any]
):
    """Display SHAP explanation"""
    st.subheader("🔍 SHAP Explanation")

    shap_values = explanation_result["shap_values"]
    feature_importance = explanation_result["feature_importance"]

    # Display feature importance table
    st.write("**Feature Importance Ranking**")

    def color_direction(val):
        if val > 0:
            return "background-color: lightcoral"
        else:
            return "background-color: lightblue"

    # Create DataFrame for display
    df_importance = pd.DataFrame(feature_importance)
    df_importance["feature_value"] = df_importance["feature"].map(
        lambda x: patient_data.get(x, "N/A")
    )

    # Style the DataFrame
    styled_df = df_importance.style.applymap(
        lambda x: color_direction(x) if isinstance(x, (int, float)) else "",
        subset=["shap_value"],
    )

    st.dataframe(styled_df, use_container_width=True)

    # Display plots
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Waterfall Plot**")
        fig_waterfall = plot_shap_waterfall(shap_values)
        st.pyplot(fig_waterfall)

    with col2:
        st.write("**Force Plot**")
        fig_force = plot_shap_force(shap_values, patient_data)
        st.pyplot(fig_force)


def create_batch_input_form():
    """Create batch prediction input form"""
    st.subheader("📊 Batch Prediction")

    # File upload option
    uploaded_file = st.file_uploader(
        "Upload CSV file with patient data",
        type=["csv"],
        help="CSV should have columns matching the patient input fields",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} patients from file")
            st.dataframe(df.head(), use_container_width=True)
            return df.to_dict("records")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # Manual batch input
    st.write("**Or create batch manually:**")
    num_patients = st.slider("Number of patients", 2, 50, 5)

    if st.button("Generate Sample Batch"):
        # Generate sample data
        sample_patients = []
        for i in range(num_patients):
            patient = {
                "age": np.random.randint(18, 85),
                "gender": np.random.choice(["Male", "Female", "Other"]),
                "income": np.random.randint(20000, 150000),
                "location": np.random.choice(["Urban", "Suburban", "Rural"]),
                "study_type": np.random.choice(["Phase I", "Phase II", "Phase III"]),
                "condition": np.random.choice(
                    [
                        "Diabetes",
                        "Hypertension",
                        "Cardiovascular Disease",
                        "Obesity",
                        "Respiratory Disease",
                        "Mental Health",
                        "Cancer",
                        "Autoimmune Disease",
                    ]
                ),
                "visit_adherence_rate": np.random.uniform(0.3, 1.0),
                "tenure_months": np.random.randint(1, 36),
                "last_visit_gap_days": np.random.randint(0, 90),
                "num_medications": np.random.randint(0, 8),
                "has_side_effects": np.random.choice([True, False]),
                "transport_support": np.random.choice([True, False]),
                "monthly_stipend": np.random.randint(100, 1000),
                "contact_frequency": np.random.uniform(1.0, 8.0),
                "support_group_member": np.random.choice([True, False]),
                "language_barrier": np.random.choice([True, False]),
                "device_usage_compliance": np.random.uniform(0.2, 1.0),
                "survey_score_avg": np.random.uniform(1.0, 10.0),
            }
            sample_patients.append(patient)

        st.session_state.sample_patients = sample_patients
        st.success(f"Generated {num_patients} sample patients")

    if "sample_patients" in st.session_state:
        st.write("**Sample Batch Data:**")
        df_sample = pd.DataFrame(st.session_state.sample_patients)
        st.dataframe(df_sample, use_container_width=True)
        return st.session_state.sample_patients

    return None


def display_batch_results(batch_result: Dict[str, Any]):
    """Display batch prediction results"""
    st.subheader("📊 Batch Prediction Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", batch_result["total_patients"])

    with col2:
        st.metric("Successful Predictions", batch_result["successful_predictions"])

    with col3:
        st.metric("Failed Predictions", batch_result["failed_predictions"])

    with col4:
        success_rate = (
            batch_result["successful_predictions"]
            / batch_result["total_patients"]
            * 100
        )
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Display predictions
    if batch_result["predictions"]:
        st.write("**Predictions:**")
        df_predictions = pd.DataFrame(batch_result["predictions"])
        st.dataframe(df_predictions, use_container_width=True)

        # Create summary statistics
        if len(df_predictions) > 0:
            st.write("**Summary Statistics:**")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Churn Probability Distribution:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(
                    df_predictions["churn_probability"],
                    bins=20,
                    alpha=0.7,
                    color="skyblue",
                )
                ax.set_xlabel("Churn Probability")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Churn Probabilities")
                st.pyplot(fig)

            with col2:
                st.write("**CLV Distribution:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(
                    df_predictions["clv_estimate"],
                    bins=20,
                    alpha=0.7,
                    color="lightgreen",
                )
                ax.set_xlabel("CLV Estimate ($)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of CLV Estimates")
                st.pyplot(fig)

    # Display errors
    if batch_result["errors"]:
        st.write("**Errors:**")
        df_errors = pd.DataFrame(batch_result["errors"])
        st.dataframe(df_errors, use_container_width=True)


def generate_sample_data(n_patients: int = 100) -> pd.DataFrame:
    """Generate sample patient data for business visualizations"""
    np.random.seed(42)

    data = {
        "patient_id": range(1, n_patients + 1),
        "age": np.random.normal(55, 15, n_patients).clip(18, 85),
        "churn_probability": np.random.beta(2, 8, n_patients),
        "clv_estimate": np.random.lognormal(10, 0.5, n_patients),
        "tenure_months": np.random.exponential(12, n_patients).clip(1, 36),
        "condition": np.random.choice(
            ["Diabetes", "Hypertension", "Cardiovascular", "Obesity", "Mental Health"],
            n_patients,
        ),
        "study_type": np.random.choice(
            ["Phase I", "Phase II", "Phase III"], n_patients
        ),
        "location": np.random.choice(["Urban", "Suburban", "Rural"], n_patients),
        "income": np.random.normal(60000, 20000, n_patients).clip(20000, 150000),
    }

    df = pd.DataFrame(data)

    # Add risk categories
    df["risk_category"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    # Add CLV categories
    df["clv_category"] = pd.cut(
        df["clv_estimate"],
        bins=[0, 20000, 50000, float("inf")],
        labels=["Low CLV", "Medium CLV", "High CLV"],
    )

    return df


def display_business_overview():
    """Display business overview with key metrics and insights"""
    st.header("📊 Business Overview")
    st.markdown(
        "*Key insights to drive retention strategies and maximize patient value*"
    )

    # Generate sample data for demonstration
    df = generate_sample_data(200)

    # Key Metrics Row
    st.subheader("🎯 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_churn = df["churn_probability"].mean() * 100
        st.metric(
            label="Overall Churn Rate",
            value=f"{avg_churn:.1f}%",
            delta=(
                f"{avg_churn - 25:.1f}%" if avg_churn > 25 else f"{25 - avg_churn:.1f}%"
            ),
            delta_color="inverse",
        )
        st.caption("📈 Lower is better - target <25%")

    with col2:
        avg_clv = df["clv_estimate"].mean()
        st.metric(
            label="Average CLV",
            value=f"${avg_clv:,.0f}",
            delta=(
                f"${avg_clv - 30000:,.0f}"
                if avg_clv > 30000
                else f"${30000 - avg_clv:,.0f}"
            ),
            delta_color="normal",
        )
        st.caption("💰 Higher is better - target >$30K")

    with col3:
        high_risk_count = len(df[df["risk_category"] == "High Risk"])
        st.metric(
            label="High-Risk Patients",
            value=f"{high_risk_count}",
            delta=(
                f"{high_risk_count - 40}"
                if high_risk_count > 40
                else f"{40 - high_risk_count}"
            ),
            delta_color="inverse",
        )
        st.caption("⚠️ Immediate attention needed")

    with col4:
        total_value_at_risk = df[df["risk_category"] == "High Risk"][
            "clv_estimate"
        ].sum()
        st.metric(
            label="Value at Risk", value=f"${total_value_at_risk:,.0f}", delta=None
        )
        st.caption("💸 Potential revenue loss")

    # Risk Distribution Chart
    st.subheader("📈 Patient Risk Distribution")
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df["risk_category"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patient Risk Categories",
            color_discrete_map={
                "Low Risk": "#2E8B57",
                "Medium Risk": "#FFA500",
                "High Risk": "#DC143C",
            },
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🎯 Focus retention efforts on high-risk patients")

    with col2:
        fig = px.histogram(
            df,
            x="churn_probability",
            nbins=20,
            title="Churn Probability Distribution",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.add_vline(
            x=0.7,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Most patients have low churn risk")

    # CLV Analysis
    st.subheader("💰 Customer Lifetime Value Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df,
            x="risk_category",
            y="clv_estimate",
            title="CLV by Risk Category",
            color="risk_category",
            color_discrete_map={
                "Low Risk": "#2E8B57",
                "Medium Risk": "#FFA500",
                "High Risk": "#DC143C",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 High-risk patients often have high CLV - prioritize retention")

    with col2:
        fig = px.scatter(
            df,
            x="churn_probability",
            y="clv_estimate",
            color="risk_category",
            size="tenure_months",
            title="Churn Risk vs CLV",
            color_discrete_map={
                "Low Risk": "#2E8B57",
                "Medium Risk": "#FFA500",
                "High Risk": "#DC143C",
            },
        )
        fig.add_hline(
            y=50000,
            line_dash="dash",
            line_color="green",
            annotation_text="High CLV Threshold",
        )
        fig.add_vline(
            x=0.7,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🎯 Target high CLV, high-risk patients for retention programs")


def display_customer_analysis():
    """Display customer analysis with business insights"""
    st.header("👥 Customer Analysis")
    st.markdown("*Deep dive into patient segments and retention opportunities*")

    df = generate_sample_data(200)

    # Top High-Risk Customers
    st.subheader("🚨 Top 10 High-Risk Patients")
    high_risk_df = df[df["risk_category"] == "High Risk"].nlargest(10, "clv_estimate")

    if not high_risk_df.empty:
        display_df = high_risk_df[
            [
                "patient_id",
                "age",
                "condition",
                "churn_probability",
                "clv_estimate",
                "tenure_months",
            ]
        ].copy()
        display_df["churn_probability"] = display_df["churn_probability"].apply(
            lambda x: f"{x:.1%}"
        )
        display_df["clv_estimate"] = display_df["clv_estimate"].apply(
            lambda x: f"${x:,.0f}"
        )
        display_df.columns = [
            "Patient ID",
            "Age",
            "Condition",
            "Churn Risk",
            "CLV",
            "Tenure (months)",
        ]

        st.dataframe(display_df, use_container_width=True)
        st.caption(
            "🎯 Immediate action required - these patients represent high value at risk"
        )

    # Condition Analysis
    st.subheader("🏥 Condition-Based Analysis")
    col1, col2 = st.columns(2)

    with col1:
        condition_stats = (
            df.groupby("condition")
            .agg(
                {
                    "churn_probability": "mean",
                    "clv_estimate": "mean",
                    "patient_id": "count",
                }
            )
            .round(3)
        )
        condition_stats.columns = ["Avg Churn Risk", "Avg CLV", "Patient Count"]
        st.dataframe(condition_stats, use_container_width=True)
        st.caption("📊 Mental Health patients show highest churn risk")

    with col2:
        fig = px.bar(
            df.groupby("condition")["churn_probability"].mean().reset_index(),
            x="condition",
            y="churn_probability",
            title="Average Churn Risk by Condition",
            color="churn_probability",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 Mental Health and Diabetes patients need special attention")

    # Retention Opportunity Analysis
    st.subheader("💡 Retention Opportunity Analysis")

    # Calculate potential revenue saved
    high_risk_high_clv = df[
        (df["risk_category"] == "High Risk") & (df["clv_estimate"] > 50000)
    ]
    potential_revenue_saved = high_risk_high_clv["clv_estimate"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("High-Risk, High-CLV Patients", len(high_risk_high_clv))

    with col2:
        st.metric("Potential Revenue at Risk", f"${potential_revenue_saved:,.0f}")

    with col3:
        # Assuming 20% improvement with intervention
        potential_savings = potential_revenue_saved * 0.2
        st.metric("Potential Savings (20% improvement)", f"${potential_savings:,.0f}")

    st.info(
        "💡 **Business Recommendation:** Focus retention efforts on high-CLV patients "
        "with churn risk >70%. A 20% improvement in retention could save "
        "$100K+ in revenue."
    )


def display_technical_details():
    """Display technical details including SHAP plots"""
    st.header("🔬 Technical Details")
    st.markdown("*Model explanations and technical insights for data scientists*")

    # Create input form for technical analysis
    st.subheader("📋 Patient Information for Technical Analysis")
    patient_data = create_patient_input_form()

    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button("🔬 Analyze", type="primary")

    with col2:
        use_async = st.checkbox("Use Async Operations", value=True)

    if predict_button:
        with st.spinner("Running technical analysis..."):
            start_time = time.time()

            if use_async:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    prediction_result = loop.run_until_complete(
                        get_prediction_async(patient_data)
                    )
                    explanation_result = loop.run_until_complete(
                        get_explanation_async(patient_data)
                    )
                finally:
                    loop.close()
            else:
                prediction_result = get_prediction(patient_data)
                explanation_result = get_explanation(patient_data)

            end_time = time.time()

            if prediction_result and explanation_result:
                st.success(
                    f"✅ Analysis completed in {end_time - start_time:.2f} seconds"
                )

                # Display prediction results
                display_prediction_results(prediction_result)

                # Display SHAP explanation
                st.subheader("🔍 SHAP Model Explanation")
                display_shap_explanation(explanation_result, patient_data)
            else:
                st.error("❌ Analysis failed")


def main():
    """Main function"""
    st.title("🏥 Clinical Study Churn & CLV Prediction Dashboard")
    st.markdown("**Enhanced with Async Operations and Caching**")

    # Check API health
    if not check_api_health():
        st.error("❌ FastAPI backend is not running. Please start the backend first.")
        st.info("Run: `uvicorn api.main:app --reload`")
        return

    st.success("✅ FastAPI backend is running")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "Single Prediction",
            "Batch Prediction",
            "Cache Statistics",
            "Business Overview",
            "Customer Analysis",
            "Technical Details",
        ],
    )

    if page == "Single Prediction":
        st.header("🔮 Single Patient Prediction")

        # Create input form
        patient_data = create_patient_input_form()

        # Prediction button
        col1, col2 = st.columns([1, 3])
        with col1:
            predict_button = st.button("🚀 Predict", type="primary")

        with col2:
            use_async = st.checkbox(
                "Use Async Operations",
                value=True,
                help="Enable async operations for better performance",
            )

        if predict_button:
            with st.spinner("Making prediction..."):
                start_time = time.time()

                if use_async:
                    # Use async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        prediction_result = loop.run_until_complete(
                            get_prediction_async(patient_data)
                        )
                    finally:
                        loop.close()
                else:
                    # Use synchronous operations
                    prediction_result = get_prediction(patient_data)

                end_time = time.time()

                if prediction_result:
                    st.success(
                        f"✅ Prediction completed in {end_time - start_time:.2f} seconds"
                    )
                    display_prediction_results(prediction_result)

                    # SHAP explanation
                    with st.expander("🔍 View SHAP Explanation"):
                        with st.spinner("Generating SHAP explanation..."):
                            if use_async:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    explanation_result = loop.run_until_complete(
                                        get_explanation_async(patient_data)
                                    )
                                finally:
                                    loop.close()
                            else:
                                explanation_result = get_explanation(patient_data)

                            if explanation_result:
                                display_shap_explanation(
                                    explanation_result, patient_data
                                )
                            else:
                                st.error("Failed to generate SHAP explanation")
                else:
                    st.error("❌ Prediction failed")

    elif page == "Batch Prediction":
        st.header("📊 Batch Prediction")

        # Create batch input
        patients_data = create_batch_input_form()

        if patients_data:
            col1, col2 = st.columns([1, 3])
            with col1:
                batch_predict_button = st.button("🚀 Batch Predict", type="primary")

            with col2:
                use_async_batch = st.checkbox(
                    "Use Async Operations",
                    value=True,
                    help="Enable async operations for better performance",
                )

            if batch_predict_button:
                with st.spinner(f"Processing {len(patients_data)} patients..."):
                    start_time = time.time()

                    if use_async_batch:
                        # Use async operations
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            batch_result = loop.run_until_complete(
                                get_batch_predictions_async(patients_data)
                            )
                        finally:
                            loop.close()
                    else:
                        # Use synchronous operations (fallback)
                        st.warning("Batch predictions require async operations")
                        batch_result = None

                    end_time = time.time()

                    if batch_result:
                        st.success(
                            f"✅ Batch prediction completed in "
                            f"{end_time - start_time:.2f} seconds"
                        )
                        display_batch_results(batch_result)
                    else:
                        st.error("❌ Batch prediction failed")

    elif page == "Cache Statistics":
        st.header("📈 Cache Statistics")

        try:
            response = requests.get(f"{API_BASE_URL}/cache/stats")
            if response.status_code == 200:
                cache_stats = response.json()

                st.write("**In-Memory Cache:**")
                st.json(cache_stats["in_memory_cache"])

                st.write("**Redis Cache:**")
                st.json(cache_stats["redis_cache"])
            else:
                st.error("Failed to get cache statistics")
        except Exception as e:
            st.error(f"Error getting cache statistics: {e}")

    elif page == "Business Overview":
        display_business_overview()

    elif page == "Customer Analysis":
        display_customer_analysis()

    elif page == "Technical Details":
        display_technical_details()


if __name__ == "__main__":
    main()

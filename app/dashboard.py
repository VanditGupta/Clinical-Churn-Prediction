"""
Streamlit Dashboard for Clinical Study Churn & CLV Prediction
Provides interactive UI for patient prediction and SHAP explainability
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import json

# Page configuration
st.set_page_config(
    page_title="Clinical Study Churn & CLV Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_prediction(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get prediction from FastAPI backend"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=patient_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_explanation(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get SHAP explanation from FastAPI backend"""
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
        income = st.slider("Monthly Income ($)", 20000, 150000, 60000, step=5000, 
                          help="Monthly income in USD")
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    
    with col2:
        st.write("**Study Information**")
        study_type = st.selectbox("Study Type", ["Phase I", "Phase II", "Phase III"])
        condition = st.selectbox("Condition", [
            "Diabetes", "Hypertension", "Cardiovascular Disease", "Obesity",
            "Respiratory Disease", "Mental Health", "Cancer", "Autoimmune Disease"
        ])
        tenure_months = st.slider("Tenure (months)", 1, 36, 12, help="Months in study")
        monthly_stipend = st.slider("Monthly Stipend ($)", 100, 1000, 400, step=50,
                                   help="Monthly incentive amount")
    
    with col3:
        st.write("**Engagement Metrics**")
        visit_adherence_rate = st.slider("Visit Adherence Rate", 0.3, 1.0, 0.7, 0.05,
                                        help="Percentage of scheduled visits attended")
        last_visit_gap_days = st.slider("Days Since Last Visit", 0, 90, 15,
                                       help="Days since last clinic visit")
        contact_frequency = st.slider("Contact Frequency", 1.0, 8.0, 3.0, 0.5,
                                     help="Staff contacts per month")
        device_usage_compliance = st.slider("Device Compliance", 0.2, 1.0, 0.6, 0.05,
                                           help="Device/wearable compliance rate")
    
    # Additional features
    st.write("**Clinical & Support Factors**")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        num_medications = st.slider("Number of Medications", 0, 8, 3)
        has_side_effects = st.checkbox("Has Side Effects")
        survey_score_avg = st.slider("Survey Score", 1.0, 10.0, 7.0, 0.5,
                                    help="Average satisfaction score")
    
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
        "survey_score_avg": survey_score_avg
    }
    
    return patient_data

def display_prediction_results(prediction_result: Dict[str, Any]):
    """Display prediction results"""
    st.subheader("🎯 Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_prob = prediction_result["churn_probability"]
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob:.1%}",
            delta=None
        )
    
    with col2:
        clv_estimate = prediction_result["clv_estimate"]
        st.metric(
            label="CLV Estimate",
            value=f"${clv_estimate:,.0f}",
            delta=None
        )
    
    with col3:
        risk_category = prediction_result["risk_category"]
        risk_color = {
            "Low Risk": "green",
            "Medium Risk": "orange", 
            "High Risk": "red"
        }.get(risk_category, "gray")
        
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Risk Category</h4>
            <p style="color: {risk_color}; font-size: 24px; font-weight: bold;">
                {risk_category}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        retention_prob = 1 - churn_prob
        st.metric(
            label="Retention Probability",
            value=f"{retention_prob:.1%}",
            delta=None
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
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title('SHAP Feature Impact Analysis', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + (0.01 if value > 0 else -0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', 
                va='center', 
                ha='left' if value > 0 else 'right',
                fontsize=10)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_shap_force(shap_values: Dict[str, float], patient_data: Dict[str, Any]):
    """Create SHAP force plot"""
    # Prepare data for force plot
    feature_values = []
    feature_names = []
    shap_values_list = []
    
    for feature, shap_value in shap_values.items():
        if feature in patient_data:
            feature_values.append(patient_data[feature])
            feature_names.append(feature)
            shap_values_list.append(shap_value)
    
    # Create force plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute SHAP value
    sorted_indices = sorted(range(len(shap_values_list)), 
                           key=lambda i: abs(shap_values_list[i]), reverse=True)
    
    sorted_values = [shap_values_list[i] for i in sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_feature_values = [feature_values[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    colors = ['red' if v > 0 else 'blue' for v in sorted_values]
    bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors, alpha=0.7)
    
    # Add feature value annotations
    for i, (bar, value, feature_value) in enumerate(zip(bars, sorted_values, sorted_feature_values)):
        ax.text(bar.get_width() + (0.01 if value > 0 else -0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{feature_value}', 
                va='center', 
                ha='left' if value > 0 else 'right',
                fontsize=9, style='italic')
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Force Plot - Feature Values and Impact', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_shap_explanation(explanation_result: Dict[str, Any], patient_data: Dict[str, Any]):
    """Display SHAP explanation results"""
    st.subheader("🔍 SHAP Explanation")
    
    shap_values = explanation_result["shap_values"]
    feature_importance = explanation_result["feature_importance"]
    
    # Display top features
    st.write("**Top Features by Impact:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Waterfall plot
        fig_waterfall = plot_shap_waterfall(shap_values)
        st.pyplot(fig_waterfall)
        plt.close()
    
    with col2:
        # Force plot
        fig_force = plot_shap_force(shap_values, patient_data)
        st.pyplot(fig_force)
        plt.close()
    
    # Feature importance table
    st.write("**Detailed Feature Analysis:**")
    importance_df = pd.DataFrame(feature_importance)
    importance_df['shap_value'] = importance_df['shap_value'].round(4)
    importance_df['abs_value'] = importance_df['abs_value'].round(4)
    
    # Color code the direction
    def color_direction(val):
        if val == 'positive':
            return 'background-color: lightcoral'
        else:
            return 'background-color: lightblue'
    
    styled_df = importance_df.style.applymap(color_direction, subset=['direction'])
    st.dataframe(styled_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.title("🏥 Clinical Study Churn & CLV Prediction Dashboard")
    st.markdown("---")
    
    # Check API health
    if not check_api_health():
        st.error("""
        ⚠️ **FastAPI backend is not running!**
        
        Please start the backend server first:
        ```bash
        uvicorn api.main:app --reload
        ```
        
        Then refresh this page.
        """)
        return
    
    st.success("✅ Connected to FastAPI backend")
    
    # Create patient input form
    patient_data = create_patient_input_form()
    
    # Prediction button
    if st.button("🚀 Get Prediction & Explanation", type="primary"):
        with st.spinner("Getting prediction..."):
            # Get prediction
            prediction_result = get_prediction(patient_data)
            
            if prediction_result:
                # Display prediction results
                display_prediction_results(prediction_result)
                
                # Get SHAP explanation
                with st.spinner("Generating SHAP explanation..."):
                    explanation_result = get_explanation(patient_data)
                    
                    if explanation_result:
                        # Display SHAP explanation
                        display_shap_explanation(explanation_result, patient_data)
                    else:
                        st.error("Failed to get SHAP explanation")
            else:
                st.error("Failed to get prediction")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This dashboard provides:
        
        **🎯 Churn Prediction**
        - Predicts likelihood of patient dropping out
        - Calculates Customer Lifetime Value (CLV)
        - Categorizes risk levels
        
        **🔍 SHAP Explainability**
        - Shows which features drive predictions
        - Visualizes feature impact and values
        - Provides interpretable AI insights
        
        **📊 Risk Categories**
        - **Low Risk**: 0-30% churn probability
        - **Medium Risk**: 30-60% churn probability  
        - **High Risk**: 60-100% churn probability
        """)
        
        st.header("🔧 Technical Details")
        st.markdown("""
        - **Model**: LightGBM Gradient Boosting
        - **Features**: 18 clinical and demographic variables
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Backend**: FastAPI with async endpoints
        - **Frontend**: Streamlit interactive dashboard
        """)

if __name__ == "__main__":
    main() 
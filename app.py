import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.pkl')

try:
    model = load_model()
except Exception:
    st.warning("Model pipeline not found! Please run `python -m src.train` first.")

st.title("🛍️ Customer Churn Prediction Engine")
st.markdown("Enter customer metrics below to evaluate churn risk in real time.")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

    submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        # Match the exact schema and column names the model was trained on
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'AvgSpendPerMonth': [total_charges / (tenure + 1)]
        })
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.divider()
            if prediction == 1:
                st.error(f"⚠️ **High Risk of Churn!** (Probability: {probability:.2%})")
                st.markdown("*Action:* Send targeted retention discount or loyalty offer.")
            else:
                st.success(f"✅ **Low Risk of Churn.** (Probability: {probability:.2%})")
                st.markdown("*Action:* Customer is stable.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer is at risk of leaving.")

# Load saved model and training information
try:
    model = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    model_features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error(
        "Model files not found. Run churn_analysis.py first "
        "and make sure the saved files are in this folder."
    )
    st.stop()

st.subheader("Enter Customer Details")

tenure = st.number_input(
    "Tenure (months)", min_value=0, max_value=100, value=12
)
monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
total_charges = st.number_input(
    "Total Charges", min_value=0.0, value=840.0
)
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)

if st.button("Predict Churn"):
    # Start with every model feature set to zero
    input_data = pd.DataFrame(
        0, index=[0], columns=model_features
    )

    # Scale numeric fields using the saved training scaler
    numeric_values = pd.DataFrame([[
        tenure, monthly_charges, total_charges
    ]], columns=[
        "tenure", "MonthlyCharges", "TotalCharges"
    ])

    scaled_values = scaler.transform(numeric_values)

    for i, col in enumerate([
        "tenure", "MonthlyCharges", "TotalCharges"
    ]):
        input_data[col] = scaled_values[0][i]

    # Set categorical features using the training column names
    selections = {
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support
    }

    for column, value in selections.items():
        if column == "Contract":
            feature = "Contract_" + (
                "One year" if value == "One year"
                else "Two year" if value == "Two year"
                else ""
            )
        else:
            feature = column + "_" + value

        if feature in input_data.columns:
            input_data.loc[0, feature] = 1

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.1%}")

    if prediction == 1:
        st.warning("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted to stay.")

    st.caption(
        "This is a model estimate, not a guarantee of customer behavior."
    )

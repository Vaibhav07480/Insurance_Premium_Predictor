import streamlit as st

from predictor import predict_premium_category

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    [
        "retired",
        "freelancer",
        "student",
        "government_job",
        "business_owner",
        "unemployed",
        "private_job",
    ],
)

if st.button("Predict Premium Category"):
    try:
        prediction = predict_premium_category(
            age=age,
            weight=weight,
            height=height,
            income_lpa=income_lpa,
            smoker=smoker,
            city=city,
            occupation=occupation,
        )
        st.success(f"Predicted Insurance Premium Category: **{prediction}**")
    except Exception as exc:
        st.error("Could not generate a prediction. Check the inputs and try again.")
        st.exception(exc)

import streamlit as st
import pickle
import numpy as np

# ===== Load Model from Uploaded Path ===== #
@st.cache_resource
def load_model():
    with open(r"C:\Users\HP\Desktop\Project2\model_p2.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===== Streamlit UI ===== #
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Data Science Salary Prediction")

st.write("Enter the job and personal details to estimate salary in USD.")

# ------- Manually Enter Encoded Feature Values ------- #
st.markdown("### Enter Encoded Feature Values")

col1, col2 = st.columns(2)

with col1:
    work_year = st.number_input("Work Year", min_value=2010, max_value=2030, value=2024)
    experience_level = st.number_input("Experience Level (encoded)", min_value=0, max_value=5, value=2)
    employment_type = st.number_input("Employment Type (encoded)", min_value=0, max_value=5, value=1)
    job_title = st.number_input("Job Title (encoded)", min_value=0, max_value=200, value=10)

with col2:
    salary_currency = st.number_input("Salary Currency (encoded)", min_value=0, max_value=10, value=3)
    employee_residence = st.number_input("Employee Residence (encoded)", min_value=0, max_value=200, value=5)
    remote_ratio = st.slider("Remote Ratio", 0, 100, 50)
    company_location = st.number_input("Company Location (encoded)", min_value=0, max_value=200, value=8)

company_size = st.number_input("Company Size (encoded)", min_value=0, max_value=10, value=2)

salary = st.number_input("Current Salary Offered (Optional)", min_value=0, value=0)

# Final input array (adjust based on your model.feature_names)
input_data = np.array([[work_year, experience_level, employment_type, job_title,
                        salary_currency, employee_residence, remote_ratio,
                        company_location, company_size, salary]])

if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: **${int(prediction):,} / year**")
    except Exception as e:
        st.error(f"Prediction Failed: {e}")

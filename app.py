import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Customer Churn Prediction', page_icon='??', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load('model/churn_model.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f'Model load nahi hua! Error: {e}')
    st.stop()

st.title(' Customer Churn Prediction Dashboard')
st.markdown('Predict whether a telecom customer is at risk of churning.')
st.write('---')

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (Months)', 0, 72, 12)

with col2:
    st.subheader('Services')
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

with col3:
    st.subheader('Billing & Contract')
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.0, max_value=130.0, value=65.0)
    total_charges = st.number_input('Total Charges ($)', min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))

st.write('---')

if st.button('Predict Churn Risk', type='primary', use_container_width=True):
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
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
        'TotalCharges': [total_charges]
    })

    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.subheader('Prediction Result')
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        if pred == 1:
            st.error(' Result: Customer is LIKELY TO CHURN!')
        else:
            st.success('Result: Customer is LIKELY TO STAY!')
    with r_col2:
        st.metric('Churn Probability', f'{prob * 100:.1f}%')

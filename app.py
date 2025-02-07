import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = pickle.load(open("rfc_model.pkl", "rb"))

# Define categorical columns that need encoding
categorical_columns = [
    'Gender', 'Country', 'state', 'self_employed', 'family_history', 
    'work_interfere', 'remote_work', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]

# Encoding dictionary (to maintain consistency)
encoders = {col: LabelEncoder() for col in categorical_columns}

def encode_inputs(df):
    """Encodes categorical features dynamically."""
    for col in categorical_columns:
        df[col] = encoders[col].fit_transform(df[col])
    return df

# UI Title
st.title("Mental Health Prediction System")

# User Input Fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ['male', 'female', 'other'])
country = st.selectbox("Country", ['United States', 'France', 'United Kingdom', 'Canada', 'Portugal',
 'Switzerland', 'Poland', 'Australia', 'Germany', 'Mexico', 'Brazil',
 'Costa Rica', 'Austria', 'Ireland', 'India', 'South Africa', 'Russia', 'Italy',
 'Bulgaria', 'Netherlands', 'Sweden', 'Colombia', 'Latvia', 'Romania', 'Belgium',
 'New Zealand', 'Spain', 'Finland', 'Uruguay', 'Israel', 'Bosnia and Herzegovina',
 'Hungary', 'Singapore', 'Japan', 'Nigeria', 'Croatia', 'Norway', 'Thailand',
 'Denmark', 'Greece', 'Moldova', 'Georgia', 'China', 'Czech Republic', 'Philippines'])
state = st.selectbox("State (US Only)", ['MD', 'CA', 'NY', 'NC', 'MA', 'IA', 'TN', 'OH', 'PA', 'WA', 'WI', 'IN', 'TX', 'MI',
 'IL', 'UT', 'NM', 'OR', 'FL', 'MN', 'MO', 'AZ', 'CO', 'GA', 'DC', 'NE', 'WV', 'OK',
 'KS', 'VA', 'NH', 'KY', 'AL', 'NV', 'NJ', 'SC', 'VT', 'SD', 'ID', 'MS', 'RI', 'WY',
 'LA', 'CT', 'ME'])
self_employed = st.selectbox("Are you self-employed?", ['Yes', 'No'])
family_history = st.selectbox("Any family history of mental illness?", ['Yes', 'No'])
work_interfere = st.selectbox("Work interference due to mental health?", ['Sometimes', 'Never', 'Often', 'Rarely'])
no_employees = st.selectbox("Number of employees in your company", [1, 2, 4, 3, 6, 5])
remote_work = st.selectbox("Do you work remotely?", ['Yes', 'No'])
tech_company = st.selectbox("Do you work in a tech company?", ['Yes', 'No'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No', "Don't know"])
care_options = st.selectbox("Does your employer provide care options?", ['Not sure', 'No', 'Yes'])
wellness_program = st.selectbox("Does your employer offer a wellness program?", ['Yes', 'No', "Don't know"])
seek_help = st.selectbox("Does your employer support seeking help?", ["Don't know", 'No', 'Yes'])
anonymity = st.selectbox("Is seeking help anonymous?", ['Yes', 'No', "Don't know"])
leave = st.selectbox("Ease of taking mental health leave", ['Very easy', 'Somewhat easy', 'Somewhat difficult', "Don't know", 'Very difficult'])
mental_health_consequence = st.selectbox("Would discussing mental health have negative consequences?", ['No', 'Maybe', 'Yes'])
phys_health_consequence = st.selectbox("Would discussing physical health have negative consequences?", ['No', 'Maybe', 'Yes'])
coworkers = st.selectbox("Would you discuss mental health with coworkers?", ['Yes', 'Some of them', 'No'])
supervisor = st.selectbox("Would you discuss mental health with your supervisor?", ['Yes', 'Some of them', 'No'])
mental_health_interview = st.selectbox("Would mentioning mental health in an interview hurt you?", ['No', 'Maybe', 'Yes'])
phys_health_interview = st.selectbox("Would mentioning physical health in an interview hurt you?", ['Yes', 'Maybe', 'No'])
mental_vs_physical = st.selectbox("Does your employer value mental and physical health equally?", ['Yes', "Don't know", 'No'])
obs_consequence = st.selectbox("Have you observed negative mental health consequences at work?", ['Yes', 'No'])

# Prepare input data
data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Country': [country],
    'state': [state],
    'self_employed': [self_employed],
    'family_history': [family_history],
    'work_interfere': [work_interfere],
    'no_employees': [no_employees],
    'remote_work': [remote_work],
    'tech_company': [tech_company],
    'benefits': [benefits],
    'care_options': [care_options],
    'wellness_program': [wellness_program],
    'seek_help': [seek_help],
    'anonymity': [anonymity],
    'leave': [leave],
    'mental_health_consequence': [mental_health_consequence],
    'phys_health_consequence': [phys_health_consequence],
    'coworkers': [coworkers],
    'supervisor': [supervisor],
    'mental_health_interview': [mental_health_interview],
    'phys_health_interview': [phys_health_interview],
    'mental_vs_physical': [mental_vs_physical],
    'obs_consequence': [obs_consequence]
})

# Encode categorical variables
data = encode_inputs(data)

# Predict
if st.button("Predict Mental Health Risk"):
    prediction = model.predict(data)[0]
    # st.write("Prediction:", "You may need mental health support." if prediction == 1 else "You seem to be fine.")
    if prediction == 1:
        st.error("You may need mental health support.")
    else:
        st.success("You seem to be fine.")

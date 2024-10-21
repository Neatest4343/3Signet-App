import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('final_model.joblib')

# Title of the web app
st.title('Prediction App')

# Sidebar for user input
st.sidebar.header('Input Parameters')

# Function to take user input from sidebar
def user_input_features():
    # Example: replace with your actual feature columns
    application_mode = st.sidebar.selectbox('Application Mode', [1, 2, 3])
    application_order = st.sidebar.slider('Application Order', 1, 10, 1)
    course = st.sidebar.selectbox('Course', ['Course1', 'Course2'])
    nationality = st.sidebar.selectbox('Nationality', ['Country1', 'Country2'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    international = st.sidebar.selectbox('International', [0, 1])
    scholarship_holder = st.sidebar.selectbox('Scholarship Holder', [0, 1])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married'])

    # Dictionary of features
    data = {
        'Application mode': application_mode,
        'Application order': application_order,
        'Course': course,
        'Nacionality': nationality,
        'Gender': gender,
        'International': international,
        'Scholarship holder': scholarship_holder,
        'Marital status': marital_status
    }
    
    return pd.DataFrame(data, index=[0])

# Store user input features into dataframe
input_df = user_input_features()

# Display user input features
st.write('User Input Parameters')
st.write(input_df)

# Predict the result
prediction = model.predict(input_df)

# Display prediction
st.write('Prediction: ', prediction)

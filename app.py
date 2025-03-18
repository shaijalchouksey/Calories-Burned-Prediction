import streamlit as st
import pickle
import pandas as pd

# Set page config for a better UI
st.set_page_config(page_title='🔥 Calories Burned Prediction', page_icon='🔥', layout='centered')

# Custom CSS for background color and styling
st.markdown('''
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #e6f7ff;
            padding: 20px;
            border-radius: 15px;
        }
    </style>
''', unsafe_allow_html=True)

# Load the trained model
try:
    with open('calories_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error('Model file not found. Please train the model first.')
    exit()

# Stylish Title and Header
st.markdown('<h1 style="text-align: center; color: #FF5733; font-size: 3rem;">🔥 Calories Burned Prediction 🔥</h1>', unsafe_allow_html=True)
st.write("## Predict the number of calories burned based on your physical details.")

# Input fields for user data
age = st.number_input('**Age**', min_value=1, max_value=100, value=25)
gender = st.selectbox('**Gender**', ['Male', 'Female'])
height = st.number_input('**Height (cm)**', min_value=50, max_value=250, value=170)
weight = st.number_input('**Weight (kg)**', min_value=10, max_value=300, value=70)
duration = st.number_input('**Duration (mins)**', min_value=1, max_value=300, value=30)
heart_rate = st.number_input('**Heart Rate (bpm)**', min_value=40, max_value=200, value=100)
body_temp = st.number_input('**Body Temperature (°C)**', min_value=30.0, max_value=45.0, value=37.0)

# Encoding gender (same as during training)
gender_encoded = 0 if gender == 'Male' else 1

# Create a DataFrame with the input data
input_data = pd.DataFrame([[age, gender_encoded, height, weight, duration, heart_rate, body_temp]],
                          columns=['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

# Predict the calories burned
if st.button('🚀 Predict'):
    try:
        prediction = model.predict(input_data)
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #28a745; margin-top: 20px;">💪 Estimated Calories Burned: <b>{prediction[0]:.2f}</b> 🔥</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')

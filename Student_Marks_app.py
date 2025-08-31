# student_marks_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📚 Student Performance Predictor")
st.write("Enter student details below to predict the expected marks.")

# Sample Training Data
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Sleep_Hours": [9, 8, 7, 7, 6, 6, 5, 5],
    "Practice_Tests": [0, 1, 1, 2, 2, 3, 3, 4],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [30, 40, 50, 60, 70, 75, 85, 95]
}
df = pd.DataFrame(data)

# Train Model
X = df[["Hours_Studied", "Sleep_Hours", "Practice_Tests", "Attendance"]]
y = df["Marks"]
model = LinearRegression()
model.fit(X, y)

# User Inputs
st.subheader("🔧 Enter Student Details")
hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=12.0, step=0.5)
sleep = st.number_input("Sleep Hours per Day", min_value=4.0, max_value=10.0, step=0.5)
tests = st.number_input("Practice Tests Taken", min_value=0, max_value=10, step=1)
attendance = st.slider("Attendance (%)", 50, 100, 75)

# Predict
if st.button("Predict Marks"):
    input_data = np.array([[hours, sleep, tests, attendance]])
    prediction = model.predict(input_data)
    st.success(f"📈 Predicted Marks: {prediction[0]:.2f} / 100")

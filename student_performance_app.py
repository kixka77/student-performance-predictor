
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Set up the page
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Enter the details below to predict if a student is likely to pass or fail.")

# Input form
study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=40.0, value=10.0)
attendance_percent = st.slider("Attendance Percentage", 0, 100, 85)
engagement_level = st.slider("Engagement Level (1=Low, 5=High)", 1, 5, 3)
previous_gpa = st.slider("Previous GPA (1.0 - 4.0)", 1.0, 4.0, 3.0)

# Generate mock data for training
np.random.seed(42)
data = pd.DataFrame({
    'study_hours': np.random.uniform(0, 20, 200),
    'attendance_percent': np.random.uniform(50, 100, 200),
    'engagement_level': np.random.randint(1, 6, 200),
    'previous_gpa': np.round(np.random.uniform(1.0, 4.0, 200), 2),
    'passed': np.random.randint(0, 2, 200)
})

X = data.drop('passed', axis=1)
y = data['passed']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Gradient Boosting Classifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_scaled, y)

# Train Deep Neural Network
model_dnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_dnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_scaled, y, epochs=20, batch_size=16, verbose=0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'study_hours': study_hours,
        'attendance_percent': attendance_percent,
        'engagement_level': engagement_level,
        'previous_gpa': previous_gpa
    }])
    input_scaled = scaler.transform(input_data)

    gbdt_pred = gbdt.predict(input_scaled)[0]
    dnn_pred = model_dnn.predict(input_scaled)[0][0]

    st.subheader("Prediction Results")
    st.write(f"**Gradient Boosting Model:** {'✅ Pass' if gbdt_pred == 1 else '❌ Fail'}")
    st.write(f"**Deep Neural Network:** {'✅ Pass' if dnn_pred > 0.5 else '❌ Fail'}")

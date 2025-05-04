import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Student Performance Predictor")

# User inputs
study_hours = st.slider("Study Hours per Day", 0, 12, 4)
attendance = st.slider("Attendance (%)", 0, 100, 80)
assignments_done = st.slider("Assignments Done (%)", 0, 100, 85)
internet_usage = st.slider("Internet Usage for Study (%)", 0, 100, 60)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

# Simulate realistic data
def generate_data(n=300):
    data = []
    for _ in range(n):
        sh = np.random.randint(0, 13)
        at = np.random.randint(50, 101)
        ad = np.random.randint(50, 101)
        iu = np.random.randint(0, 101)
        sl = np.random.randint(4, 10)

        score = (sh * 0.4) + (at * 0.2) + (ad * 0.2) + (iu * 0.05) + (sl * 0.15)
        label = 1 if score > 60 else 0  # more realistic boundary
        data.append([sh, at, ad, iu, sl, label])

    df = pd.DataFrame(data, columns=[
        "study_hours", "attendance", "assignments_done", "internet_usage", "sleep_hours", "performance"
    ])
    return df

df = generate_data()

X = df.drop("performance", axis=1)
y = df["performance"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
model.fit(X_scaled, y)

# User input prediction
input_data = pd.DataFrame([[
    study_hours, attendance, assignments_done, internet_usage, sleep_hours
]], columns=X.columns)
input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)[0]
result = "Likely to Pass" if pred == 1 else "Needs Improvement"

if st.button("Predict"):
    st.success(f"Prediction: {result}")

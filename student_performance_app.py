import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Title
st.title("Student Performance Predictor")

# User inputs
study_hours = st.slider("Study Hours per Day", 0, 12, 4)
attendance = st.slider("Attendance (%)", 0, 100, 80)
assignments_done = st.slider("Assignments Done (%)", 0, 100, 85)
internet_usage = st.slider("Internet Usage for Study (%)", 0, 100, 60)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

# Mock dataset
np.random.seed(42)
data = {
    "study_hours": np.random.randint(0, 12, 200),
    "attendance": np.random.randint(60, 100, 200),
    "assignments_done": np.random.randint(50, 100, 200),
    "internet_usage": np.random.randint(0, 100, 200),
    "sleep_hours": np.random.randint(4, 10, 200)
}
df = pd.DataFrame(data)
df["performance"] = (
    (df["study_hours"] * 0.3 + df["attendance"] * 0.2 +
     df["assignments_done"] * 0.2 + df["internet_usage"] * 0.1 +
     df["sleep_hours"] * 0.2) > 55
).astype(int)

# Features and labels
X = df.drop("performance", axis=1)
y = df["performance"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300)
model.fit(X_scaled, y)

# Prediction
input_data = pd.DataFrame([[
    study_hours, attendance, assignments_done, internet_usage, sleep_hours
]], columns=X.columns)
input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)[0]
result = "Likely to Pass" if pred == 1 else "Needs Improvement"

# Show result
if st.button("Predict"):
    st.success(f"Prediction: {result}")

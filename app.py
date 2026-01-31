import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("log_reg_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction 🚢")

# User inputs
pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encode inputs
sex_map = {"male":0, "female":1}
embarked_map = {"S":0, "C":1, "Q":2}

input_data = pd.DataFrame({
    "Pclass":[pclass],
    "Sex":[sex_map[sex]],
    "Age":[age],
    "SibSp":[sibsp],
    "Parch":[parch],
    "Fare":[fare],
    "Embarked":[embarked_map[embarked]]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("💀 This passenger would NOT have survived.")

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details below:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Parents / Children Aboard", min_value=0, step=1)
fare = st.number_input("Fare Paid", min_value=0.0, step=1.0)
sex = st.radio("Sex", ["male", "female"])
embarked = st.radio("Embarked From", ["C", "Q", "S"])

# Convert inputs to model format
input_data = {
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': 1 if sex == "male" else 0,
    'Embarked_Q': 1 if embarked == "Q" else 0,
    'Embarked_S': 1 if embarked == "S" else 0
}

def predict(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                'Sex_male', 'Embarked_Q', 'Embarked_S']:
        if col not in df.columns:
            df[col] = 0

    df = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
             'Sex_male', 'Embarked_Q', 'Embarked_S']]
    
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    return "✅ Survived" if pred == 1 else "❌ Did not survive"

# Predict button
if st.button("Predict"):
    result = predict(input_data)
    st.subheader("Result:")
    st.success(result)

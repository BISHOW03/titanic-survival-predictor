import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_survival(data_dict):
    expected_cols = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_male', 'Embarked_Q', 'Embarked_S'
    ]
    
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)

    # Ensure all expected dummy columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    return "✅ Survived" if prediction == 1 else "❌ Did not survive"

if __name__ == "__main__":
    print("\n--- Titanic Survival Predictor ---\n")
    try:
        Pclass = int(input("Pclass (1-3): "))
        Age = float(input("Age: "))
        SibSp = int(input("Siblings/Spouses Aboard: "))
        Parch = int(input("Parents/Children Aboard: "))
        Fare = float(input("Fare: "))
        Sex = input("Sex (male/female): ").lower()
        Embarked = input("Embarked from (C/Q/S): ").upper()

        data = {
            'Pclass': Pclass,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Sex_male': 1 if Sex == "male" else 0,
            'Embarked_Q': 1 if Embarked == "Q" else 0,
            'Embarked_S': 1 if Embarked == "S" else 0
        }

        result = predict_survival(data)
        print(f"\nPrediction: {result}\n")

    except Exception as e:
        print(f"Error: {e}")

# 🚢 Titanic Survival Predictor

Predict whether a passenger would survive the Titanic disaster using machine learning.

This project uses the famous [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic) and includes data preprocessing, exploratory data analysis (EDA), model training, live terminal input, and a Streamlit web app.

---
## 🌐 Live Demo

👉 [Click to Open App](https://titanic-survival-predictor-bishow03.streamlit.app/)

---
## 📂 Project Structure

- titanic-survival-predictor/
- ├── data/
- │ └── train.csv # Titanic dataset
- ├── eda.py # Exploratory Data Analysis
- ├── preprocess.py # Data preprocessing and feature engineering
- ├── train_model.py # Trains and saves the ML model
- ├── predict.py # CLI-based live prediction
- ├── app.py # Streamlit web application
- ├── model.pkl # Trained ML model
- ├── scaler.pkl # Feature scaler used in training
- ├── requirements.txt
- └── README.md


---



# 🛠️ How to Run (Locally)
### 1. Clone the repo
```bash
git clone https://github.com/bishow03/titanic-survival-predictor.git
cd titanic-survival-predictor
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model
```bash
python train_model.py
```
### 4. Predict via Terminal (Live Input)
```bash
python predict.py
```
You'll be asked for:
- `Age` (int)
- `Sex` (M/F)
- Class
- Fare
- Siblings/Spouses
- Parents/Children
- Embark location

### 5. Run the Streamlit App
```bash
streamlit run app.py
```
    It opens in your browser at: http://localhost:8501

## 🧠 Model Info
- Algorithm: Logistic Regression
- Features: Pclass, Age, SibSp, Parch, Fare, Sex, Embarked
- Accuracy: ~81% (validation)

## 📌 Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- Target: Survived (1 = Yes, 0 = No)


## ⭐️ Support
If you found this project helpful:
- 👉 Give it a star
- 👉 Fork it and improve further
- 👉 Share it with others


## 🙋‍♂️ Author
[Bishow Ghimire](https://github.com/BISHOW03)

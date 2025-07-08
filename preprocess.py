import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

# Survival by Gender
sns.countplot(x='Survived', data=df, hue='Sex')
plt.title("Survival Count by Gender")
plt.savefig("survival_by_gender.png")
plt.clf()

# Age Distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title("Age Distribution by Survival")
plt.savefig("age_distribution.png")
plt.clf()

# Class vs Survival
sns.countplot(x='Pclass', data=df, hue='Survived')
plt.title("Survival by Passenger Class")
plt.savefig("survival_by_class.png")

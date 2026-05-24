import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Load Data
df = pd.read_csv('cardio_train.csv', sep=';')

# Data Preprocessing

df.drop('id', axis=1, inplace=True)

# Convert age from days to years
df['age'] = (df['age'] / 365.25).round().astype(int)

# Remove extreme outliers in blood pressure, height, and weight 
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
df = df[(df['ap_hi'] > df['ap_lo'])] 
df = df[(df['height'] >= 140) & (df['height'] <= 210)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 150)]

print(f"Data shape after removing outliers: {df.shape}")

# 3. Data Analysis and Visualization
# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

# Distribution of Age vs Cardio
plt.figure(figsize=(10, 6))
sns.countplot(x='age', hue='cardio', data=df, palette='Set2')
plt.title('Cardiovascular Disease Frequency by Age')
plt.show()

# 4. Model Building
X = df.drop(['cardio'], axis=1)
y = df['cardio']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

accuracies = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc*100:.2f}%")

# Plotting model accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), palette='viridis', legend=False)
plt.title('Machine Learning Models Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.6, 0.8)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.005, f"{v*100:.2f}%", ha='center', va='bottom', fontsize=12)
plt.show()

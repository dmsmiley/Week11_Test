import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def load_and_preprocess_data():
    df = sns.load_dataset('titanic')
    df.dropna(subset=['age'], inplace=True)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
    X = df[features]
    y = df['survived']
    return df, X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

def plot_accuracy_vs_k(k_values, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Streamlit App
# -----------------------------------------------
st.title("KNN on Titanic: Scaled vs. Unscaled Comparison")
st.markdown("""
This interactive app demonstrates how scaling affects the performance of K-Nearest Neighbors (KNN) on the Titanic dataset.
We will:
- Load and preprocess the data.
- Compare KNN performance on unscaled vs. scaled data (with k=5).
- Explore the effect of different odd k values on scaled data.
""")

# Load & preprocess data; split into training and testing sets
df, X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)

st.header("Data Preview")
st.subheader("Raw Data (first 5 rows)")
st.dataframe(df.head())

# ---------------------------
# KNN Without Scaling (k=5)
# ---------------------------
st.header("KNN Without Scaling (k=5)")
knn_unscaled = train_knn(X_train, y_train, n_neighbors=5)
y_pred_unscaled = knn_unscaled.predict(X_test)
accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
st.write(f"**Accuracy (Unscaled, k=5): {accuracy_unscaled:.2f}**")
cm_unscaled = confusion_matrix(y_test, y_pred_unscaled)
st.subheader("Confusion Matrix (Unscaled Data)")
plot_confusion_matrix(cm_unscaled, "KNN Confusion Matrix (Unscaled Data)")
st.subheader("Classification Report (Unscaled Data)")
st.text(classification_report(y_test, y_pred_unscaled))

# ---------------------------
# KNN With Scaling (k=5)
# ---------------------------
st.header("KNN With Scaling (k=5)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_scaled = train_knn(X_train_scaled, y_train, n_neighbors=5)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
st.write(f"**Accuracy (Scaled, k=5): {accuracy_scaled:.2f}**")
cm_scaled = confusion_matrix(y_test, y_pred_scaled)
st.subheader("Confusion Matrix (Scaled Data)")
plot_confusion_matrix(cm_scaled, "KNN Confusion Matrix (Scaled Data)")
st.subheader("Classification Report (Scaled Data)")
st.text(classification_report(y_test, y_pred_scaled))

# ---------------------------
# Interactive k Value Exploration
# ---------------------------
st.header("Explore Different k Values on Scaled Data")
k_slider = st.slider("Select Number of Neighbors (k)", min_value=1, max_value=21, step=2, value=5)
knn_temp = train_knn(X_train_scaled, y_train, n_neighbors=k_slider)
y_temp_pred = knn_temp.predict(X_test_scaled)
acc_temp = accuracy_score(y_test, y_temp_pred)
st.write(f"**Accuracy with k = {k_slider}: {acc_temp:.2f}**")

st.markdown("### Accuracy vs. Number of Neighbors (Odd k values)")
k_values = list(range(1, 21, 2))
accuracies_scaled = []
for k_val in k_values:
    knn_temp = train_knn(X_train_scaled, y_train, n_neighbors=k_val)
    y_temp_pred = knn_temp.predict(X_test_scaled)
    accuracies_scaled.append(accuracy_score(y_test, y_temp_pred))
plot_accuracy_vs_k(k_values, accuracies_scaled)

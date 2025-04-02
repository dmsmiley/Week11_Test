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
    # Load the Titanic dataset from seaborn
    df = sns.load_dataset('titanic')
    # Remove rows with missing 'age' values
    df.dropna(subset=['age'], inplace=True)
    # One-hot encode the 'sex' column (drop first category)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    # Define features and target
    features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
    X = df[features]
    y = df['survived']
    return df, X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, n_neighbors):
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

# -----------------------------------------------
# Streamlit App Layout
# -----------------------------------------------

st.title("KNN Performance: Scaled vs. Unscaled")

# Selection controls at the top
st.markdown("### Select Parameters")
k = st.slider("Select number of neighbors (k, odd values only)", min_value=1, max_value=21, step=2, value=5)
data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])

# Load and preprocess the data; split into training and testing sets
df, X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Depending on the toggle, optionally scale the data
if data_type == "Scaled":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Train KNN with the selected k value
knn_model = train_knn(X_train, y_train, n_neighbors=k)
if data_type == "Scaled":
    st.write(f"**Scaled Data: KNN (k = {k})**")
else:
    st.write(f"**Unscaled Data: KNN (k = {k})**")

# Predict and evaluate
y_pred = knn_model.predict(X_test)
accuracy_val = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy: {accuracy_val:.2f}**")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
plot_confusion_matrix(cm, f"KNN Confusion Matrix ({data_type} Data)")

# Display classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

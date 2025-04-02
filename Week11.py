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
# Helper functions
# -----------------------------------------------

def load_and_preprocess_data():
    # Load the Titanic dataset from seaborn
    df = sns.load_dataset('titanic')
    
    # Remove rows with missing 'age' values
    df.dropna(subset=['age'], inplace=True)
    
    # One-hot encode the 'sex' column (dropping the first category to avoid multicollinearity)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    
    # Define features and target
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
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

def plot_accuracy_vs_k(k_values, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs. Number of Neighbors (k) on Scaled Data')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Streamlit App
# -----------------------------------------------

st.title("K-Nearest Neighbors (KNN) Classification on the Titanic Dataset")
st.markdown(
    """
    In this interactive app, we will demonstrate how KNN performs on the Titanic dataset. We will:
    
    1. **View and inspect the data**
    2. **Run KNN without scaling**
    3. **Scale the data and see performance improvement**
    4. **Explore the effect of different (odd) k values on performance**
    """
)

# Sidebar for navigation
section = st.sidebar.radio("Choose Section", 
                           ("Data & Preprocessing", "KNN Without Scaling", "KNN With Scaling", "Explore k Values"))

# ----------------------------
# Section 1: Data & Preprocessing
# ----------------------------
if section == "Data & Preprocessing":
    st.header("Step 1 & 2: Load and Preprocess the Data")
    
    df, X, y, features = load_and_preprocess_data()
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Features Preview")
    st.dataframe(X.head())
    
    st.subheader("Target Preview")
    st.write(y.head())

    # Split the data for later sections
    X_train, X_test, y_train, y_test = split_data(X, y)
    st.success("Data has been loaded and preprocessed. The dataset has been split into training and testing sets.")

# ----------------------------
# Section 2: KNN Without Scaling
# ----------------------------
elif section == "KNN Without Scaling":
    st.header("Step 4: Evaluate KNN Without Scaling")
    
    # Load and preprocess the data, then split it
    _, X, y, features = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train KNN model on unscaled data (using k=5)
    knn_unscaled = train_knn(X_train, y_train, n_neighbors=5)
    
    # Predict on unscaled test data
    y_pred_unscaled = knn_unscaled.predict(X_test)
    accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
    
    st.write(f"**Accuracy without scaling (k=5): {accuracy_unscaled:.2f}**")
    
    cm_unscaled = confusion_matrix(y_test, y_pred_unscaled)
    st.subheader("Confusion Matrix (Unscaled Data)")
    plot_confusion_matrix(cm_unscaled, "KNN Confusion Matrix (Unscaled Data)")
    
    st.subheader("Classification Report (Unscaled Data)")
    st.text(classification_report(y_test, y_pred_unscaled))
    
# ----------------------------
# Section 3: KNN With Scaling
# ----------------------------
elif section == "KNN With Scaling":
    st.header("Step 5 & 6: Scale the Data and Evaluate KNN With Scaling")
    
    # Load and preprocess the data, then split it
    _, X, y, features = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.subheader("Scaled Training Features Preview")
    st.dataframe(pd.DataFrame(X_train_scaled, columns=features, index=X_train.index).head())
    
    # Train KNN model on scaled data (using k=5)
    knn_scaled = train_knn(X_train_scaled, y_train, n_neighbors=5)
    
    # Predict on scaled test data
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
    
    st.write(f"**Accuracy with scaling (k=5): {accuracy_scaled:.2f}**")
    
    cm_scaled = confusion_matrix(y_test, y_pred_scaled)
    st.subheader("Confusion Matrix (Scaled Data)")
    plot_confusion_matrix(cm_scaled, "KNN Confusion Matrix (Scaled Data)")
    
    st.subheader("Classification Report (Scaled Data)")
    st.text(classification_report(y_test, y_pred_scaled))
    
# ----------------------------
# Section 4: Explore Different k Values on Scaled Data
# ----------------------------
elif section == "Explore k Values":
    st.header("Step 7: Explore the Effect of Different k Values (Scaled Data)")
    
    # Load and preprocess the data, then split it
    _, X, y, features = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.markdown("Select an odd value for **k** using the slider below:")
    k = st.slider("Number of Neighbors (k)", min_value=1, max_value=21, step=2, value=5)
    
    # Train and evaluate for the selected k
    knn_temp = train_knn(X_train_scaled, y_train, n_neighbors=k)
    y_temp_pred = knn_temp.predict(X_test_scaled)
    acc_temp = accuracy_score(y_test, y_temp_pred)
    st.write(f"**Accuracy with k = {k}: {acc_temp:.2f}**")
    
    # Also, plot the accuracy for a range of odd k values
    st.markdown("### Accuracy vs. Number of Neighbors (Odd k values)")
    k_values = range(1, 21, 2)
    accuracies_scaled = []
    for k_val in k_values:
        knn_temp = train_knn(X_train_scaled, y_train, n_neighbors=k_val)
        y_temp_pred = knn_temp.predict(X_test_scaled)
        accuracies_scaled.append(accuracy_score(y_test, y_temp_pred))
        
    plot_accuracy_vs_k(list(k_values), accuracies_scaled)
    
    st.markdown("The plot above shows how the accuracy varies with different odd values of k on the scaled data.")

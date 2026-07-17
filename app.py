import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wafer Fault Detection",
    page_icon="🧇",
    layout="wide"
)

# Sidebar for Navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ("Prediction", "Training"))

st.title("🧇 Wafer Fault Detection")

# -----------------------------
# Utility functions
# -----------------------------
def load_objects():
    if os.path.exists("artifacts/model.pkl") and os.path.exists("artifacts/scalar.pkl"):
        with open("artifacts/model.pkl", "rb") as file:
            model = pickle.load(file)
        with open("artifacts/scalar.pkl", "rb") as file:
            scaler = pickle.load(file)
        return model, scaler
    return None, None

# -----------------------------
# Training Mode
# -----------------------------
if mode == "Training":
    st.header("Model Training")
    st.write("Upload a training dataset (CSV) to train the Wafer Fault Detection model.")
    
    uploaded_train_file = st.file_uploader("Upload Training CSV File", type=["csv"], key="train_uploader")
    
    if uploaded_train_file is not None:
        df_train = pd.read_csv(uploaded_train_file)
        
        st.subheader("Training Data Preview")
        st.dataframe(df_train.head())
        
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                # Preprocessing
                if "Unnamed: 0" in df_train.columns:
                    df_train.drop(columns=["Unnamed: 0"], inplace=True)
                
                if "Good/Bad" not in df_train.columns:
                    st.error("Target column 'Good/Bad' not found in the dataset.")
                else:
                    # Separate features and target
                    X = df_train.drop(['Good/Bad'], axis=1)
                    y = df_train[['Good/Bad']].copy()
                    
                    # Fill missing values
                    X.fillna(0, inplace=True)
                    
                    # Map target variables (-1 to 0, 1 to 1)
                    y['Good/Bad'] = y['Good/Bad'].map({-1: 0, 1: 1})
                    
                    # Train/Test Split
                    # Using test_size=0.20
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
                    
                    # Scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Ensure artifacts directory exists
                    os.makedirs("artifacts", exist_ok=True)
                    
                    # Save scaler
                    with open("artifacts/scalar.pkl", "wb") as file:
                        pickle.dump(scaler, file)
                        
                    # Model Training
                    model = XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    )
                    
                    # Fit model
                    model.fit(X_train_scaled, y_train)
                    
                    # Save model
                    with open("artifacts/model.pkl", "wb") as file:
                        pickle.dump(model, file)
                        
                    # Evaluation
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Model trained successfully! Accuracy on test split: {acc:.4f}")
                    
                    st.write("**Classification Report:**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# Prediction Mode
# -----------------------------
elif mode == "Prediction":
    st.header("Batch Prediction")
    st.write("Upload a CSV file for Batch Prediction using the trained model.")
    
    model, scaler = load_objects()
    
    if model is None or scaler is None:
        st.warning("Trained model or scaler not found. Please go to 'Training' mode to train the model first.")
    else:
        uploaded_file = st.file_uploader("Upload Prediction CSV File", type=["csv"], key="predict_uploader")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data")
            st.dataframe(df.head())
            
            # Remove unwanted column if present
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            
            # Remove target column if uploaded accidentally
            if "Good/Bad" in df.columns:
                df.drop(columns=["Good/Bad"], inplace=True)
                
            # Fill missing values
            df.fillna(0, inplace=True)
            
            # Scale
            X_scaled = scaler.transform(df)
            
            # Predict
            prediction = model.predict(X_scaled)
            
            # Convert back to original labels
            prediction = pd.Series(prediction).map({
                0: "bad wafer",
                1: "good wafer"
            })
            
            result = df.copy()
            result["Prediction"] = prediction
            
            st.subheader("Prediction Result")
            st.dataframe(result)
            
            csv = result.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                "📥 Download Prediction",
                data=csv,
                file_name="Wafer_Prediction.csv",
                mime="text/csv"
            )
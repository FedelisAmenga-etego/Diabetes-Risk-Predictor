import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models
log_model = joblib.load("logistic_regression.pkl")
rf_model = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost.pkl")
rf_tuned_model = joblib.load("rf_tuned.pkl")

models = {"Tuned Random Forest": rf_tuned_model,
          "Logistic Regression": log_model,
          "Random Forest": rf_model,
          "XGBoost": xgb_model}
# Page config
st.set_page_config(page_title="🩺 Diabetes Prediction", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f0f4f8, #d9e4ec);
        padding: 2rem;
    }
    h1, h2, h3 {
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-size: 16px;
        padding: 0.6rem;
    }
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
    }
    .non-diabetic {
        background-color: #d4edda;
        color: #155724;
    }
    .diabetic {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🩺 Diabetes Prediction")
st.subheader("Machine Learning powered tool for predicting likelihood of diabetes.")
st.markdown(
    """
    <style>
    .title {
        color: #2E86C1; /* Blue shade */
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Model Comparison Table ---
st.markdown("### 📊 Model Comparison")

comparison_data = {
    "Model": ["Tuned Random Forest", "Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy ℹ️": ["81%", "80%", "84%", "82%"],
    "Precision (Diabetes) ℹ️": ["89%", "74%", "77%", "77%"],
    "Recall (Diabetes) ℹ️": ["81%", "64%", "75%", "70%"],
    "F1-score ℹ️": ["74%", "69%", "76%", "73%"]
}

comparison_df = pd.DataFrame(comparison_data)



# Tooltips explanation
with st.expander("ℹ️ What do these metrics mean?"):
    st.markdown("""
    - **Accuracy**: Overall how many predictions were correct.  
    - **Precision (Diabetes)**: Of those predicted as diabetic, how many actually are.  
    - **Recall (Diabetes)**: Of all real diabetics, how many the model caught.  
    - **F1-score**: Balance between precision & recall, useful when classes are imbalanced.  
    """)

st.dataframe(comparison_df, use_container_width=True)

# --- Choose Model ---
st.markdown("### 🔍 Choose a Model")
model_choice = st.selectbox("Select a model for prediction:", list(models.keys()))

# explanation
st.markdown("""
#### 📊 Metrics Explanation
- **Accuracy**: Overall how many predictions were correct.  
- **Precision (Diabetes)**: Of those predicted as diabetic, how many actually are.  
- **Recall (Diabetes)**: Of all real diabetics, how many the model correctly identified.  
- **F1-score**: Balances precision and recall into one number.  
""")

# --- Patient Data Input (2x4 grid) ---
st.markdown("### 📝 Enter Patient Data")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Submit button
if st.button("🔮 Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction] * 100
    
    # Store in session history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "Patient #": len(st.session_state["history"]) + 1,
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Prediction": "Non-Diabetic" if prediction == 0 else "Diabetic",
        "Confidence": f"{confidence:.2f}%"
    })

    # --- Display result ---
    if prediction == 0:
        st.markdown(
            f"<div class='result-card non-diabetic'>🟢 The patient is NOT likely to have Diabetes (Non-Diabetic)<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card diabetic'>🔴 The patient is likely to have Diabetes (Diabetic)<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

# --- Prediction History ---
if "history" in st.session_state and len(st.session_state["history"]) > 0:
    st.markdown("### 📜 Prediction History")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df, use_container_width=True)

    # Download as CSV
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )



st.markdown(
    """
    <hr>
    <div style="text-align:center; color:grey; font-size:14px;">
    ⚡Built by: 
        <a href="https://github.com/FedelisAmenga-etego" target="_blank">Amenga-etego Fedelis</a>
    | Machine Learning for Healthcare
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")




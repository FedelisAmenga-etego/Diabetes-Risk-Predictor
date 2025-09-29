# Diabetes Risk Prediction Using Machine Learning

A comprehensive machine learning project that predicts diabetes risk in patients using diagnostic measurements. The project includes model development, hyperparameter tuning, and an interactive Streamlit web application for real-time predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops and compares multiple machine learning models to predict whether a patient has diabetes based on diagnostic measurements. The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases, focusing on female patients of Pima Indian heritage aged 21 years and older.

## 📊 Dataset

**Source:** National Institute of Diabetes and Digestive and Kidney Diseases

**Features:**
- `Pregnancies` - Number of times pregnant
- `Glucose` - Plasma glucose concentration
- `BloodPressure` - Diastolic blood pressure (mm Hg)
- `SkinThickness` - Triceps skin fold thickness (mm)
- `Insulin` - 2-Hour serum insulin (mu U/ml)
- `BMI` - Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction` - Diabetes pedigree function (genetic influence)
- `Age` - Age in years

**Target Variable:**
- `Outcome` - 0 (No diabetes) or 1 (Has diabetes)

## 🤖 Models

The project implements and compares four classification models:

1. **Logistic Regression** - Baseline linear model with StandardScaler
2. **Random Forest Classifier** - Ensemble method with class balancing
3. **XGBoost Classifier** - Gradient boosting with optimized hyperparameters
4. **Tuned Random Forest** - Hyperparameter-optimized Random Forest via GridSearchCV

## ✨ Features

- **Multiple Model Comparison** - Compare performance across 4 different models
- **Hyperparameter Optimization** - GridSearchCV implementation for Random Forest
- **Feature Importance Analysis** - Understand which features drive predictions
- **Interactive Web App** - User-friendly Streamlit interface for predictions
- **Model Persistence** - Saved models using joblib for deployment
- **Comprehensive Evaluation** - Precision, recall, F1-score, and accuracy metrics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FedelisAmenga-etego/diabetes-prediction.git
cd diabetes-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
matplotlib
seaborn
```

## 💻 Usage

### Running the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `diabetes.ipynb` and run all cells to train models

### Running the Streamlit App

1. Ensure models are trained and saved (`.pkl` files exist)

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

4. Use the app:
   - Select a model from the dropdown
   - Input patient diagnostic measurements
   - Click "Predict" to see results
   - View model comparison metrics

## 📈 Model Performance

### Overall Accuracy Comparison

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|---------------------|------------------|-------------------|
| Logistic Regression | 79.87% | 73.91% | 64.15% | 68.69% |
| Random Forest | 80.52% | 73.47% | 67.92% | 70.59% |
| XGBoost | **82.47%** | 77.08% | 69.81% | 73.27% |
| Tuned Random Forest | 80.65% | 68.00% | **81.00%** | 74.00% |

### Key Insights

- **Best Overall Accuracy:** XGBoost (82.47%)
- **Best Recall:** Tuned Random Forest (81.00%) - Identifies more diabetic patients
- **Best Precision:** XGBoost (77.08%) - Fewer false positives

### Feature Importance (Tuned Random Forest)

1. **Glucose** (34.44%) - Most important predictor
2. **BMI** (20.84%) - Second most significant
3. **Age** (14.22%) - Third most important
4. **Diabetes Pedigree Function** (8.58%)
5. Other features contribute <7% each

## 📁 Project Structure

```
diabetes-prediction/
│
├── diabetes.ipynb              # Main notebook with model development
├── app.py                      # Streamlit web application
├── diabetes.csv                # Dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Saved model files
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── rf_tuned.pkl
│
└── assets/                     # Images and visualizations
    └── screenshots/
```

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models and preprocessing
- **XGBoost** - Gradient boosting implementation
- **Streamlit** - Web application framework
- **Joblib** - Model serialization
- **Jupyter Notebook** - Development environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by the National Institute of Diabetes and Digestive and Kidney Diseases
- Pima Indian Heritage community for data contribution
- Scikit-learn and XGBoost development teams

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/FedelisAmenga-etego/diabetes-prediction](https://github.com/yourusername/diabetes-prediction)

---

⭐ If you found this project helpful, please consider giving it a star!

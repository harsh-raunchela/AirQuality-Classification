# 🌫️ Air Quality Prediction System

An end-to-end Machine Learning project that predicts **Air Quality Index (AQI) categories** based on pollutant concentration and meteorological data using Indian AQI standards.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Web Application Architecture](#-web-application-architecture)
- [AQI Categories](#aqi-categories)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Models & Results](#models--results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Demo](#demo)

---

## 📌 Project Overview

This project builds a complete machine learning pipeline to classify air quality into 6 categories based on real-world pollutant and weather sensor data. The system includes:

- Data cleaning & preprocessing
- AQI label creation using Indian standards
- Feature engineering & selection
- Training 3 classification models
- Hyperparameter tuning with GridSearchCV
- Flask REST API serving serialized ML inference
- Interactive frontend designed with **Material 3 Expressive** principles
- Full evaluation with metrics, confusion matrix & ROC curve
- Saved models for instant prediction (no retraining needed)

---


## 🌐 Web Application Architecture

The system features an interactive web application allowing users to submit environmental sensor metrics and receive instant model predictions.

### 🎨 Frontend (`web application.html`)
- **Design Language:** Implements **Google's Material 3 Expressive** design system (dynamic color tokens, 28px rounded surface containers, expressive typography, and pill buttons).
- **Step-by-Step Wizard:** Interactive 19-step wizard with real-time percentage progress indicators to guide input entry.
- **Dynamic Result Card:** Visual badge rendering color-coded health impact levels (*Good*, *Satisfactory*, *Moderate*, *Poor*, *Very Poor*, *Severe*).
- **Asynchronous Execution:** Built with vanilla JS (`fetch` API) to communicate with the REST API without reloading the page.

### ⚡ Backend (`app.py`)
- **API Framework:** Powered by **Flask** with `Flask-CORS` for cross-origin client communication.
- **Model Inference:** Loads saved `.pkl` model via `joblib` and evaluates input data in real time.
- **Feature Alignment:** Auto-detects model feature names (`model.feature_names_in_`) to match exact column order and casing.
- **Endpoint:** `POST /predict` — Accepts 19-feature JSON payloads and returns predicted AQI categories.

---


## 🟢 AQI Categories

| Category | PM2.5 Range (µg/m³) | Health Impact |
|----------|-------------------|---------------|
| 🟢 Good | 0 – 30 | Minimal impact |
| 🟡 Satisfactory | 31 – 60 | Minor breathing discomfort |
| 🟠 Moderate | 61 – 90 | Discomfort for sensitive people |
| 🔴 Poor | 91 – 120 | Breathing discomfort for most |
| 🟣 Very Poor | 121 – 250 | Respiratory illness on prolonged exposure |
| ⚫ Severe | > 250 | Serious health effects for all |

---

## 📁 Project Structure

```
air-quality-prediction/
│
├── 🌐 index.html                        # Material 3 Expressive Web Frontend
├── ⚡ app.py                            # Flask REST API Backend
├── 🤖 knn.pklux / svm_model.pkl         # Trained Model Binary
|
├── 📓 air_quality_preprocessing.ipynb   # Data cleaning & feature engineering
├── 📓 air_quality_model_training.ipynb  # Model training & evaluation
├── 📓 project_demo.ipynb                # Demo notebook for predictions
│
├── 📊 data.csv                          # Original raw dataset
├── 📊 air_quality_model_ready.csv       # Preprocessed dataset
├── 📊 final_data.csv                    # After feature selection
│
├── 🤖 lr_model.pkl                      # Saved Logistic Regression model
├── 🤖 knn_model.pkl                     # Saved KNN model
├── 🤖 svm_model.pkl                     # Saved SVM model (Best)
├── 🔧 scaler.pkl                        # Saved StandardScaler
├── 🔧 label_encoder.pkl                 # Saved LabelEncoder
│
└── 📄 README.md
```

---

## 📊 Dataset

- **Total Rows:** 8,784 hourly readings
- **Original Features:** 25 columns
- **Final Features after selection:** 20 columns
- **Target Variable:** AQI Category (6 classes)

**Key Pollutants:**
`PM2.5`, `NO2`, `NOx`, `NH3`, `SO2`, `CO`, `Ozone`, `Benzene`, `Toluene`, `Xylene`

**Meteorological Features:**
`Temperature (AT)`, `Relative Humidity (RH)`, `Wind Speed (WS)`, `Wind Direction (WD)`, `Barometric Pressure (BP)`

---

## ⚙️ ML Pipeline

```
Raw Data (8784 rows, 25 cols)
        ↓
Data Preprocessing
  • Dropped empty column (O Xylene)
  • Created AQI labels from PM2.5 (Indian Standard)
  • Extracted Month, Hour, Weekday from Timestamp
  • Label Encoding + Standard Scaling
        ↓
Feature Selection
  • Correlation Analysis → removed PM10, NO, TOT-RF
  • SelectKBest (f_classif) → removed RF, Hour, Weekday
  → Final: 20 features
        ↓
Train/Test Split (80% / 20%)
        ↓
Model Training + Hyperparameter Tuning (GridSearchCV, cv=5)
  • Logistic Regression
  • KNN
  • SVM
        ↓
Evaluation
  • Accuracy, Precision, Recall, F1-Score
  • Confusion Matrix
  • ROC Curve (AUC)
        ↓
Best Model → SVM (98.6% accuracy) 🏆
        ↓
Deployment
• Saved Best Model → Served via Flask API & Material 3 UI 
```

---

## 🏆 Models & Results

### After Hyperparameter Tuning:

| Model | Best Parameters | Train Accuracy | Test Accuracy | F1 Score | AUC |
|-------|----------------|----------------|---------------|----------|-----|
| Logistic Regression | C=10, solver=lbfgs | 97.0% | 97.0% | 0.97 | 1.00 |
| KNN | K=11, metric=manhattan, weights=uniform | 83.1% | 78.4% | 0.77 | 0.94 |
| **SVM** 🏆 | **C=10, kernel=linear, gamma=scale** | **98.9%** | **98.6%** | **0.98** | **1.00** |

### SVM — Per Class Performance:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Good | 0.99 | 0.96 | 0.97 |
| Moderate | 0.99 | 0.99 | 0.99 |
| Poor | 0.98 | 0.96 | 0.97 |
| Satisfactory | 0.98 | 0.99 | 0.99 |
| Severe | 0.98 | 0.99 | 0.99 |
| Very Poor | 0.99 | 0.99 | 0.99 |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Plotting graphs |
| `seaborn` | Heatmaps & visualizations |
| `joblib` | Saving & loading trained models |

--------------------------------

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6) | Material 3 Expressive web application |
| **Backend** | Python, Flask, Flask-CORS | REST API serving model inference |
| **Machine Learning** | scikit-learn, joblib | Model training, pipeline tuning, and serialization |
| **Data Processing** | pandas, numpy | Dataset cleaning, vector transformation, feature extraction |
| **Visualization** | matplotlib, seaborn | Exploratory data analysis & confusion matrices |
---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/air-quality-prediction.git
cd air-quality-prediction
```

### 2. Install dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn matplotlib seaborn joblib
```

### 3. Run notebooks in order
```
Step 1 → air_quality_preprocessing.ipynb
Step 2 → air_quality_model_training.ipynb
Step 3 → project_demo.ipynb  (for predictions)
```
### 4. Launch the Backend Server
```
Bash
python app.py
The API server will run at http://127.0.0.1:5000.
```
### 5. Launch the Web-app
```
Double-click web_application.html or open it in any modern browser to use the interactive application.
```

> ⚡ **Skip retraining:** The trained models are already saved as `.pkl` files.
> Just run `project_demo.ipynb` directly to see predictions!

---

## 🎯 Demo

### Single Prediction:
```python
import joblib
import pandas as pd

# Load best model
svm_model = joblib.load('svm_model.pkl')

label_map = {0: 'Good', 1: 'Moderate', 2: 'Poor',
             3: 'Satisfactory', 4: 'Severe', 5: 'Very Poor'}

# Sample input
sample = {'PM2.5 (µg/m³)': 171.80, 'NO2 (µg/m³)': 35.04, ...}
sample_df = pd.DataFrame([sample])

prediction = label_map[svm_model.predict(sample_df)[0]]
print(f"Predicted AQI Category: {prediction}")
```

### Output:
```
========================================
       AIR QUALITY PREDICTION DEMO
========================================
Logistic Regression → Severe
KNN                 → Severe
SVM (Best Model)    → Severe
========================================
```

---

## 👨‍💻 Author

**Harsh Raunchela **
- GitHub: https://github.com/harsh-raunchela

**Devansh Dangwal **
- Github: https://github.com/Nickyyayy
---

## 📄 License

This project is for educational purposes as part of a PBL (Project Based Learning) assignment.

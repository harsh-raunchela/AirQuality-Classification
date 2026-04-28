#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import numpy as np
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

df = pd.read_csv(os.path.join(BASE_DIR, "final_data.csv"))

X = df.drop(columns=["AQI_Label"])
# ==========================================================
# BANNER
# ==========================================================
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║               AIR QUALITY PREDICTION SYSTEM                  ║
║      Machine Learning Based AQI Classification Project       ║
╚══════════════════════════════════════════════════════════════╝
"""

# ==========================================================
# HELPERS
# ==========================================================
def line():
    print("═" * 62)

def predict_sample(sample):
    pred = model.predict(sample)
    return le.inverse_transform(pred)[0]

# ==========================================================
# MODE 1: RANDOM DATASET ROW
# ==========================================================
def random_prediction():
    row = random.randint(0, len(X)-1)

    sample = X.iloc[[row]]
    pred = predict_sample(sample)
    actual = le.inverse_transform([df["AQI_Label"].iloc[row]])[0]

    line()
    print(" RANDOM SAMPLE PREDICTION")
    line()

    print(f"\n Selected Row       : {row}")
    print(f" Predicted AQI      : {pred}")
    print(f" Actual AQI         : {actual}")

# ==========================================================
# MODE 2: ENTER CUSTOM VALUES
# ==========================================================
def custom_prediction():

    line()
    print(" CUSTOM AQI PREDICTION")
    line()

    print("\nEnter values between -2 to +3")
    print("(0 = average, +high, -low)\n")

    sample = X.iloc[[0]].copy()

    sample['PM2.5 (µg/m³)'] = float(input("PM2.5 Level        : "))
    sample['NO2 (µg/m³)'] = float(input("NO2 Level          : "))
    sample['CO (mg/m³)'] = float(input("CO Level           : "))
    sample['Ozone (µg/m³)'] = float(input("Ozone Level        : "))
    sample['Month'] = float(input("Month (1-12 scaled): "))

    pred = predict_sample(sample)

    print("\n╔══════════════════════════════════╗")
    print(f"║ Predicted AQI : {pred:<16} ║")
    print("╚══════════════════════════════════╝")

# ==========================================================
# MODE 3: SHOW MODEL PERFORMANCE
# ==========================================================
def model_info():

    line()
    print(" MODEL INFORMATION")
    line()

    print("""
 Best Model Used : Support Vector Machine (SVM)

 Accuracy        : 98.6%
 Logistic Reg.   : 97%
 KNN             : 78%

 Classes:
 Good
 Satisfactory
 Moderate
 Poor
 Very Poor
 Severe
""")

# ==========================================================
# MAIN MENU
# ==========================================================
def main():

    print(BANNER)

    while True:

        line()
        print(" MAIN MENU")
        line()

        print("""
 [1] Predict Using Random Dataset Row
 [2] Predict Using Custom Input
 [3] Show Model Performance
 [4] Exit
""")

        choice = input("Select Option: ")

        if choice == '1':
            random_prediction()

        elif choice == '2':
            custom_prediction()

        elif choice == '3':
            model_info()

        elif choice == '4':
            print("\nThank you for using AQI Prediction System ")
            sys.exit()

        else:
            print("\nInvalid Option!")

        input("\nPress Enter to continue...")

# ==========================================================
if __name__ == "__main__":
    main()
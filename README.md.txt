# Student Dropout Prediction (Low / Medium / High Risk)

This project predicts student dropout risk based on activity and performance data, and provides a tutor dashboard via Streamlit.

## Features
- Train ML model (`RandomForestClassifier`)
- Single student risk prediction form
- Bulk CSV upload with predictions
- Tutor dashboard with charts and high-risk alerts

## Setup
```bash
cd student_dropout_prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


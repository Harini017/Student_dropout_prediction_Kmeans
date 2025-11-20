
# 🎓 Student Dropout Prediction – Machine Learning Project

This project predicts whether a student is likely to **drop out**, **graduate**, or **continue studying**, based on academic and demographic features.

## 📁 Project Structure

```
student_dropout_prediction/
│
├── app/                         # Web application (Flask/FastAPI depending on your code)
│
├── model_pipeline.joblib        # Trained ML model (saved pipeline)
│
├── student_dropout_dataset.xlsx # Dataset used for training
│
├── train_model.py               # Script to train and save the model
│
├── README.md                    # Project documentation
│
├── requirements.txt             # Python dependencies
│
├── .gitignore                   # Git ignored files
│
└── venv/                        # Python virtual environment (optional)
```

## 🚀 Features

* **Data preprocessing** (handling missing values, encoding, scaling)
* **Model training** using ML algorithm - Random Forest
* **Prediction pipeline** stored as `model_pipeline.joblib`
* **Web interface** for user-friendly predictions
* **Modular project structure** for easy updates and maintenance

---

## 🧠 How the Model Works

The model learns patterns from student-related factors such as:

* Academic performance
* Attendance
* Demographic details
* Financial factors
* Enrollment information

Based on this, it predicts:

* **1 – Dropout**
* **0 – Non-dropOut**


---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student_dropout_prediction.git
cd student_dropout_prediction
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

To retrain or update the model:

```bash
python train_model.py
```

This will generate a fresh `model_pipeline.joblib`.

---

## 🌐 Running the Web App

Inside the project folder:

```bash
python app/app.py
```

## 📊 Dataset

The dataset used is:

```
student_dropout_dataset.xlsx
```

It contains student records with a target column indicating dropout/graduate/enrolled status.

---

## 🧪 Example Prediction Code

```python
import joblib
import pandas as pd

model = joblib.load("model_pipeline.joblib")

sample = pd.DataFrame([{
    "age": 20,
    "attendance": 84,
    "gpa": 7.5,
    "num_failures": 1
}])

prediction = model.predict(sample)
print("Predicted Status:", prediction[0])
```

---

## 🤝 Contributing

Pull requests and improvements are welcome!

---

This project is for **academic and learning purposes**.


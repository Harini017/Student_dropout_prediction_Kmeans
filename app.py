# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ---- 1️⃣ Page config MUST be first ----
st.set_page_config(layout="wide", page_title="Tutor Dashboard & Dropout Risk Predictor")

# ---- 2️⃣ Load model ----
MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

model = load_model()

st.title("Student Dropout Risk Predictor — Tutor Dashboard")
tab1, tab2 = st.tabs(["Predict Single Student", "Tutor Dashboard / Bulk Predict"])

# ---- Tab 1: Single student prediction ----
with tab1:
    st.header("Enter student features and predict risk")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("student_id", value="")
        course_id = st.text_input("course_id", value="")
        enroll_date = st.date_input("enroll_date", value=datetime.today().date())
        last_active_date = st.date_input("last_active_date", value=datetime.today().date())
        lecture_completion_pct = st.number_input("lecture_completion_pct", min_value=0.0, max_value=100.0, value=0.0)
        sessions_pct = st.number_input("sessions_pct", min_value=0.0, max_value=100.0, value=0.0)
        forum_activity_pct = st.number_input("forum_activity_pct", min_value=0.0, max_value=100.0, value=0.0)
        activity_recency_pct = st.number_input("activity_recency_pct", min_value=0.0, max_value=100.0, value=0.0)
    with col2:
        notifications_engaged_pct = st.number_input("notifications_engaged_pct", min_value=0.0, max_value=100.0, value=0.0)
        assignment_submission_pct = st.number_input("assignment_submission_pct", min_value=0.0, max_value=100.0, value=0.0)
        assignment_avg_score_pct = st.number_input("assignment_avg_score_pct", min_value=0.0, max_value=100.0, value=0.0)
        quiz_avg_score_pct = st.number_input("quiz_avg_score_pct", min_value=0.0, max_value=100.0, value=0.0)
        final_exam_score_pct = st.number_input("final_exam_score_pct", min_value=0.0, max_value=100.0, value=0.0)
        overall_score_pct = st.number_input("overall_score_pct", min_value=0.0, max_value=100.0, value=0.0)
        certificate_eligible = st.selectbox("certificate_eligible", options=["no", "yes"])

    if st.button("Predict"):
        today = pd.Timestamp.now().normalize()
        enroll_ts = pd.to_datetime(enroll_date)
        last_active_ts = pd.to_datetime(last_active_date)
        days_since_last_active = (today - last_active_ts).days
        enrolled_days = (today - enroll_ts).days

        row = {
            "student_id": student_id,
            "course_id": course_id,
            "enroll_date": enroll_ts,
            "last_active_date": last_active_ts,
            "lecture_completion_pct": lecture_completion_pct,
            "sessions_pct": sessions_pct,
            "forum_activity_pct": forum_activity_pct,
            "activity_recency_pct": activity_recency_pct,
            "notifications_engaged_pct": notifications_engaged_pct,
            "assignment_submission_pct": assignment_submission_pct,
            "assignment_avg_score_pct": assignment_avg_score_pct,
            "quiz_avg_score_pct": quiz_avg_score_pct,
            "final_exam_score_pct": final_exam_score_pct,
            "overall_score_pct": overall_score_pct,
            "certificate_eligible": 1 if certificate_eligible == "yes" else 0,
            "days_since_last_active": days_since_last_active,
            "enrolled_days": enrolled_days
        }

        X_single = pd.DataFrame([row])
        try:
            pred = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0]
            classes = model.classes_
            probs = {c: float(p) for c, p in zip(classes, proba)}
            st.success(f"Predicted risk category: **{pred}**")
            st.write("Probabilities:")
            st.json(probs)
        except Exception as e:
            st.error("Prediction failed — ensure feature names match training set.")
            st.write(e)

# ---- Tab 2: Bulk upload / Dashboard ----
with tab2:
    st.header("Bulk upload CSV and Tutor Dashboard")
    uploaded_file = st.file_uploader("Upload students CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['enroll_date','last_active_date'])
        today = pd.Timestamp.now().normalize()
        df['days_since_last_active'] = (today - df['last_active_date']).dt.days.clip(lower=0)
        df['enrolled_days'] = (today - df['enroll_date']).dt.days.clip(lower=0)

        try:
            preds = model.predict(df)
            df['predicted_risk'] = preds
            st.subheader("Predicted risk sample")
            st.dataframe(df.head(50))
        except Exception as e:
            st.error("Prediction failed. Make sure columns match training features.")
            st.write(e)
            st.stop()

        # Dashboard metrics
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        counts = df['predicted_risk'].value_counts()
        col1.metric("Total students", len(df))
        col2.metric("High risk", int(counts.get('high',0)))
        col3.metric("Medium risk", int(counts.get('medium',0)))

        st.markdown("### Risk distribution")
        st.bar_chart(counts)

        numeric_cols = ['overall_score_pct','final_exam_score_pct','quiz_avg_score_pct','assignment_avg_score_pct','lecture_completion_pct']
        present_numeric = [c for c in numeric_cols if c in df.columns]
        if present_numeric:
            agg = df.groupby('predicted_risk')[present_numeric].mean().round(2)
            st.dataframe(agg)
            st.line_chart(agg.T)
        else:
            st.info("No score columns available to summarize.")

        if 'high' in df['predicted_risk'].unique():
            st.markdown("### High-risk students (sample)")
            st.dataframe(df[df['predicted_risk']=='high'].head(100))

        st.download_button(
    "Download predictions CSV",
    df.to_csv(index=False).encode('utf-8'),
    file_name="predictions.csv",
    mime="text/csv"
)

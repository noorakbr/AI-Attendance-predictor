"""Predict attendance for one student: python predict.py 80 10 45 1 0 70"""
import sys
import joblib
import pandas as pd

cols = ["previous_attendance_pct", "study_hours_per_week", "commute_minutes",
        "part_time_job", "morning_class", "assignments_submitted_pct"]
if len(sys.argv) != 7:
    sys.exit("Usage: python predict.py " + " ".join(cols))
row = pd.DataFrame([[float(v) for v in sys.argv[1:]]], columns=cols)
model = joblib.load("model.joblib")
p = model.predict_proba(row)[0][1]
print(f"Probability of regular attendance: {p:.1%} -> {'REGULAR' if p >= 0.5 else 'AT RISK'}")

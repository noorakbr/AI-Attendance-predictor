"""Generate a synthetic student attendance dataset (no real student data is used)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

df = pd.DataFrame({
    "previous_attendance_pct": rng.normal(78, 14, N).clip(30, 100),
    "study_hours_per_week": rng.normal(12, 5, N).clip(0, 40),
    "commute_minutes": rng.integers(5, 120, N),
    "part_time_job": rng.integers(0, 2, N),
    "morning_class": rng.integers(0, 2, N),
    "assignments_submitted_pct": rng.normal(75, 18, N).clip(0, 100),
})

# Attendance is driven mostly by past attendance, with some noise from other factors.
score = (
    0.06 * df["previous_attendance_pct"]
    + 0.04 * df["assignments_submitted_pct"] / 10
    + 0.03 * df["study_hours_per_week"]
    - 0.012 * df["commute_minutes"]
    - 0.5 * df["part_time_job"]
    - 0.4 * df["morning_class"]
    + rng.normal(0, 0.9, N)
)
df["attends_regularly"] = (score > np.median(score) - 0.2).astype(int)  # 1 = 75%+ attendance next term
df.round(1).to_csv("attendance_data.csv", index=False)
print(f"Saved attendance_data.csv with {N} rows")

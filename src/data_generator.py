import numpy as np
import pandas as pd

def generate_synthetic(n=200, seed=42):
    """
    Generates a synthetic student performance dataset.
    """
    np.random.seed(seed)
    study_hours = np.random.randint(1, 11, n)
    attendance = np.random.randint(40, 101, n)
    previous_score = np.random.randint(30, 96, n)
    assignment_marks = np.random.randint(30, 101, n)

    # Create a normalized final score proxy in range ~0-100
    final_score = 0.3 * previous_score + 0.3 * assignment_marks + 0.4 * (attendance)
    # Use median threshold to ensure both classes exist (balanced-ish)
    median_threshold = np.median(final_score)
    result = ["Pass" if s >= median_threshold else "Fail" for s in final_score]

    df = pd.DataFrame({
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousScore": previous_score,
        "AssignmentMarks": assignment_marks,
        "FinalScoreProxy": final_score,
        "Result": result
    })
    return df

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred):
    """
    Computes classification metrics: Accuracy, Precision, Recall, F1-Score.
    """
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 3),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
        'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 3)
    }

def ensure_two_classes(df):
    """
    Ensure df['Result'] contains at least two classes (0 and 1).
    If only one class exists, compute a score proxy and reassign Result using median threshold.
    Returns modified df and a flag indicating whether modification occurred.
    """
    df = df.copy()
    # Map textual labels to numeric if needed
    if df['Result'].dtype == object:
        df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})
    # Fill missing numeric features
    df[['StudyHours', 'Attendance', 'PreviousScore', 'AssignmentMarks']] = df[['StudyHours', 'Attendance', 'PreviousScore', 'AssignmentMarks']].fillna(df.median())
    
    # If Result has only one unique value, recompute using a score proxy
    if df['Result'].nunique() < 2:
        # Compute a final score proxy (same formula used in generation)
        final_score = 0.3 * df['PreviousScore'] + 0.3 * df['AssignmentMarks'] + 0.4 * df['Attendance']
        median_threshold = final_score.median()
        df['Result'] = (final_score >= median_threshold).astype(int)
        df['FinalScoreProxy'] = final_score
        modified = True
    else:
        modified = False
    
    # Ensure integer type
    df['Result'] = df['Result'].astype(int)
    return df, modified

def train_and_evaluate(df):
    """
    Trains multiple models (Logistic, Linear (thresholded), Decision Tree, Random Forest)
    and returns metrics and predictions.
    """
    df_checked, modified = ensure_two_classes(df)
    X = df_checked[['StudyHours', 'Attendance', 'PreviousScore', 'AssignmentMarks']]
    y = df_checked['Result']

    # Use stratify only if there are at least 2 classes
    strat = y if y.nunique() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)

    # Linear Regression -> thresholded classification (use PreviousScore as proxy)
    lin_model = LinearRegression()
    lin_model.fit(X_train[['StudyHours', 'Attendance', 'AssignmentMarks']], X_train['PreviousScore'])
    lin_pred_scores = lin_model.predict(X_test[['StudyHours', 'Attendance', 'AssignmentMarks']])
    lin_pred = (lin_pred_scores >= 50).astype(int)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Metrics
    results = {
        'Logistic Regression': classification_metrics(y_test, log_pred),
        'Linear Regression (thresholded)': classification_metrics(y_test, lin_pred),
        'Decision Tree': classification_metrics(y_test, dt_pred),
        'Random Forest': classification_metrics(y_test, rf_pred)
    }

    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    # Determine best model by F1-Score
    metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']] = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].astype(float)
    best_row = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    best_model_f1 = best_row['F1-Score']

    pred_map = {
        'Logistic Regression': log_pred,
        'Linear Regression (thresholded)': lin_pred,
        'Decision Tree': dt_pred,
        'Random Forest': rf_pred
    }
    best_pred = pred_map[best_model_name]
    cm = confusion_matrix(y_test, best_pred)

    return {
        'metrics_df': metrics_df.round(3),
        'best_model_name': best_model_name,
        'best_model_f1': round(best_model_f1, 3),
        'confusion_matrix': cm,
        'y_test': y_test.reset_index(drop=True),
        'predictions': {
            'Logistic Regression': pd.Series(log_pred),
            'Linear Regression (thresholded)': pd.Series(lin_pred),
            'Decision Tree': pd.Series(dt_pred),
            'Random Forest': pd.Series(rf_pred)
        },
        'modified_labels': modified
    }

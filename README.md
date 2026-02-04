# Student Performance Prediction System

A professional machine learning system that predicts student performance based on study hours, attendance, previous scores, and assignment marks. This project is structured as a modular Python application with a Tkinter GUI.

## 📌 Project Overview

This project aims to help educators and students understand the key factors influencing academic success. It uses synthetic data to simulate student records and applies various machine learning algorithms to predict:
1.  **Result:** Whether a student will Pass or Fail (Classification).
2.  **Final Score:** The predicted final numeric score (Regression) - *implemented as a feature in training logic*.

## 🚀 Features

*   **Modular Architecture:** Clean separation of concerns (Data Generation, Model Training, UI).
*   **Synthetic Data Generation:** Generates realistic datasets with configurable parameters.
*   **Multiple Models:**
    *   **Logistic Regression:** For binary classification (Pass/Fail).
    *   **Linear Regression:** For estimating continuous final scores (thresholded for classification).
    *   **Decision Tree & Random Forest:** For robust classification.
*   **Interactive GUI:** A Tkinter-based desktop application to:
    *   Generate or load datasets.
    *   Train models and view comparison metrics (Accuracy, Precision, Recall, F1-Score).
    *   Visualise Confusion Matrices and Actual vs. Predicted plots.
*   **Robust Error Handling:** Handles edge cases like single-class datasets by automatically refining labels.

## 🛠️ Technologies Used

*   **Python:** Core programming language.
*   **Pandas & NumPy:** Data manipulation and numerical operations.
*   **Scikit-learn:** Machine learning model training and evaluation.
*   **Matplotlib:** Data visualization.
*   **Tkinter:** Graphical User Interface (GUI).

## 📂 Project Structure

```
student-performance-prediction-system/
├── src/
│   ├── app.py              # Main GUI application entry point
│   ├── data_generator.py   # Synthetic data generation logic
│   └── train_model.py      # ML pipeline and evaluation logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/student-performance-prediction-system.git
    cd student-performance-prediction-system
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Running the Application
Run the main script from the root directory:
```bash
python src/app.py
```

**GUI Workflow:**
1.  Click **"Generate Synthetic Data"** or **"Load CSV"**.
2.  Click **"Train & Evaluate"** to build models and see the results table.
3.  Use **"Show Pass/Fail Pie"** or **"Show Actual vs Pred Points"** for visualizations.

## 📊 Model Performance

The project evaluates models using:
*   **Accuracy**
*   **Precision**
*   **Recall**
*   **F1-Score**

*Note: Performance metrics vary based on the generated dataset's random seed.*

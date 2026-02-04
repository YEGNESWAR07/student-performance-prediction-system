import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import from our modularized files
# Assuming src is not in python path by default if run from root, we might need adjustments.
# But for now, standard import assuming PYTHONPATH is set or run as module.
# If running 'python src/app.py' directly, we can use relative imports or modify sys path.
# Let's adjust sys.path for simplicity if run from src/
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_synthetic
from src.train_model import train_and_evaluate

def load_csv(path):
    df = pd.read_csv(path)
    required = {'StudyHours', 'Attendance', 'PreviousScore', 'AssignmentMarks', 'Result'}
    # A loose check: columns must be present. Result might be missing if we were predicting new data, 
    # but for this specific app workflow (train & eval), we need it.
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    return df

class StudentMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction System")
        self.root.geometry("1080x700") # Slightly larger
        self.df = None
        self.results = None

        ctrl_frame = ttk.Frame(root, padding=8)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl_frame, text="Generate Synthetic Data", command=self.generate_data).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_frame, text="Load CSV", command=self.load_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_frame, text="Train & Evaluate", command=self.train_models).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_frame, text="Show Pass/Fail Pie", command=self.show_pie).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_frame, text="Show Actual vs Pred Points", command=self.show_points).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_frame, text="Exit", command=root.quit).pack(side=tk.RIGHT, padx=6)

        mid_frame = ttk.Frame(root, padding=8)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(mid_frame, text="Dataset Preview", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tree = ttk.Treeview(left, columns=("StudyHours", "Attendance", "PreviousScore", "AssignmentMarks", "Result"), show='headings', height=12)
        for col in ("StudyHours", "Attendance", "PreviousScore", "AssignmentMarks", "Result"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill='y')
        self.tree.configure(yscrollcommand=vsb.set)

        right = ttk.LabelFrame(mid_frame, text="Model Comparison (Metrics)", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, reexpand=True, padx=6, pady=6) # Changed expand to reexpand? No, just expand.

        self.metrics_tree = ttk.Treeview(right, columns=("Model", "Accuracy", "Precision", "Recall", "F1-Score"), show='headings', height=12)
        for col in ("Model", "Accuracy", "Precision", "Recall", "F1-Score"):
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=120, anchor='center')
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb2 = ttk.Scrollbar(right, orient="vertical", command=self.metrics_tree.yview)
        vsb2.pack(side=tk.RIGHT, fill='y')
        self.metrics_tree.configure(yscrollcommand=vsb2.set)

        bottom = ttk.LabelFrame(root, text="Plots", padding=6)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.fig = plt.Figure(figsize=(9, 3.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_data(self):
        self.df = generate_synthetic(n=200)
        self._populate_dataset_preview()
        messagebox.showinfo("Data Generated", "Synthetic dataset generated (200 rows).")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df = load_csv(path)
            self._populate_dataset_preview()
            messagebox.showinfo("Loaded", f"Loaded dataset from:\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def _populate_dataset_preview(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        if self.df is None:
            return
        preview = self.df.head(50)
        for _, row in preview.iterrows():
            vals = (row['StudyHours'], row['Attendance'], row['PreviousScore'], row['AssignmentMarks'], row['Result'])
            self.tree.insert('', tk.END, values=vals)

    def train_models(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a CSV or generate synthetic data first.")
            return
        try:
            # We import ensure_two_classes implicitly via train_and_evaluate handling it internally
            self.results = train_and_evaluate(self.df)
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training:\n{e}")
            return

        for i in self.metrics_tree.get_children():
            self.metrics_tree.delete(i)
        for _, row in self.results['metrics_df'].iterrows():
            self.metrics_tree.insert('', tk.END, values=(row['Model'], row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']))

        best = self.results['best_model_name']
        f1 = self.results['best_model_f1']
        modified = self.results.get('modified_labels', False)
        info_msg = f"Best model by F1-Score:\n{best} (F1 = {f1})"
        if modified:
            info_msg += "\n\nNote: The dataset contained a single class and labels were recomputed using a score proxy (median threshold) to allow training."
        messagebox.showinfo("Training Complete", info_msg)

        self._plot_confusion_matrix(self.results['confusion_matrix'], title=f"Confusion Matrix ({best})")

    def show_pie(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load or generate data first.")
            return
        # We need to use proper checking logic or just raw data. 
        # Ideally we should use the same logic as training if training happened, 
        # but for raw visualization let's just show raw data distribution.
        # Actually, let's use the helper from train_model just to be consistent if possible,
        # but since we can't easily import it without circular issues or restructure, let's allow raw show.
        # Wait, we imported train_and_evaluate. We can import ensure_two_classes too if we want.
        from src.train_model import ensure_two_classes
        df_checked, _ = ensure_two_classes(self.df)
        
        counts = df_checked['Result'].value_counts().sort_index()
        # Handle cases where 0 or 1 might be missing in raw data (though ensure_two_classes fixes it usually)
        labels = ['Fail (0)', 'Pass (1)']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        colors = ['salmon', 'lightgreen']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0.05))
        ax.set_title("Pass vs Fail Distribution")
        self.canvas.draw()

    def show_points(self):
        if self.results is None:
            messagebox.showwarning("No Results", "Train models first to view Actual vs Predicted points.")
            return
        y_test = self.results['y_test']
        preds = self.results['predictions']
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        idx = np.arange(len(y_test))
        ax.scatter(idx, y_test, label='Actual', marker='o', color='black', s=50)
        jitter = 0.08
        colors = {'Logistic Regression': 'tab:blue', 'Linear Regression (thresholded)': 'tab:orange', 'Decision Tree': 'tab:green', 'Random Forest': 'tab:red'}
        markers = {'Logistic Regression': 'x', 'Linear Regression (thresholded)': '^', 'Decision Tree': 's', 'Random Forest': 'D'}
        offsets = {'Logistic Regression': -3*jitter, 'Linear Regression (thresholded)': -jitter, 'Decision Tree': jitter, 'Random Forest': 3*jitter}
        
        for name, series in preds.items():
            ax.scatter(idx + offsets[name], series.values, label=name, marker=markers[name], color=colors[name], s=40, alpha=0.9)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Fail (0)', 'Pass (1)'])
        ax.set_xlabel("Test Sample Index")
        ax.set_title("Actual vs Predicted (Points) — Test Set")
        # Move legend outside if too crowded
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='small')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Adjust layout to make room for legend
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred_Fail', 'Pred_Pass'])
        ax.set_yticklabels(['Actual_Fail', 'Actual_Pass'])
        
        # Loop over data dimensions and create text annotations.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
        
        ax.set_title(title)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentMLApp(root)
    root.mainloop()

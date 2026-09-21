from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "report_images"
OUTPUT = ROOT / "Project_Report.docx"

doc = Document()

# --------------------------------------------------
# Page setup
# --------------------------------------------------
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.8)
section.right_margin = Inches(0.8)

# --------------------------------------------------
# Default font
# --------------------------------------------------
styles = doc.styles
styles["Normal"].font.name = "Arial"
styles["Normal"].font.size = Pt(10.5)

for style_name in ["Title", "Heading 1", "Heading 2"]:
    styles[style_name].font.name = "Arial"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(22)
    r.font.name = "Arial"
    return p


def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(13)
    r.bold = True
    return p


def add_centered(text, size=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


def add_heading(text, level=1):
    doc.add_heading(text, level=level)


def add_image(filename, caption):
    path = IMAGE_DIR / filename

    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(6.2))

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
    else:
        p = doc.add_paragraph(
            f"[Image not found: {filename}]"
        )
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def add_number(text):
    doc.add_paragraph(text, style="List Number")


# ==================================================
# TITLE PAGE
# ==================================================
add_title("Predictive Maintenance of Industrial Machinery")

add_subtitle("AICTE-2026 | Problem Statement #39")

add_centered(
    "Machine Learning Based Predictive Maintenance and Failure Classification",
    11,
)

doc.add_paragraph()
add_centered(
    "Student Project Report",
    12,
)

doc.add_paragraph()
add_centered(
    "Technology: Python, Scikit-learn, Flask, Streamlit",
    10,
)

add_centered(
    "Dataset: Kaggle Machine Predictive Maintenance Classification",
    10,
)

doc.add_page_break()


# ==================================================
# ABSTRACT
# ==================================================
add_heading("1. Abstract", 1)

doc.add_paragraph(
    "Predictive maintenance uses machine learning to identify patterns in "
    "industrial sensor measurements that may be associated with equipment "
    "failure. This project develops a machine learning based predictive "
    "maintenance system using the Machine Predictive Maintenance Classification "
    "dataset. The system analyzes machine type, air temperature, process "
    "temperature, rotational speed, torque, and tool wear."
)

doc.add_paragraph(
    "The project uses a binary classification model to predict whether a "
    "machine failure is likely to occur. A secondary classification model is "
    "used to classify the type of failure when a failure is predicted. Several "
    "machine learning algorithms are compared using stratified cross-validation, "
    "with Gradient Boosting selected for the final models based on macro F1-score."
)

doc.add_paragraph(
    "The final primary model achieved an F1-macro score of 0.8764 and an "
    "ROC-AUC of 0.9649 on the held-out test set. A Flask REST API and a "
    "Streamlit web interface were developed to demonstrate how the trained "
    "models can be used for prediction."
)


# ==================================================
# PROBLEM STATEMENT
# ==================================================
add_heading("2. Problem Statement", 1)

doc.add_paragraph(
    "Develop a predictive maintenance model for industrial machinery by "
    "analyzing sensor and operational data. The system should identify "
    "patterns that precede machine failure and classify the failure type "
    "where applicable. The objective is to support early identification of "
    "potential failures and help reduce unplanned downtime and maintenance costs."
)


# ==================================================
# OBJECTIVES
# ==================================================
add_heading("3. Objectives", 1)

objectives = [
    "Load and understand industrial machine sensor data.",
    "Perform data cleaning and exploratory data analysis.",
    "Identify relationships between sensor measurements and machine failure.",
    "Build and compare multiple machine learning classification models.",
    "Select a suitable model using stratified cross-validation.",
    "Evaluate the final model on an unseen test dataset.",
    "Classify the type of failure when a machine failure is predicted.",
    "Save trained models for reuse.",
    "Develop a Flask API for model inference.",
    "Develop a Streamlit user interface for interactive predictions.",
]


for item in objectives:
    add_bullet(item)


# ==================================================
# DATASET
# ==================================================
add_heading("4. Dataset Description", 1)

doc.add_paragraph(
    "The project uses the Machine Predictive Maintenance Classification "
    "dataset. The dataset contains 10,000 records and 10 columns representing "
    "machine identifiers, machine type, sensor measurements, target status, "
    "and failure information."
)

table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"

hdr = table.rows[0].cells
hdr[0].text = "Feature"
hdr[1].text = "Type"
hdr[2].text = "Description"

features = [
    ("Type", "Categorical", "Machine type"),
    ("Air temperature [K]", "Numeric", "Air temperature"),
    ("Process temperature [K]", "Numeric", "Process temperature"),
    ("Rotational speed [rpm]", "Numeric", "Machine rotational speed"),
    ("Torque [Nm]", "Numeric", "Machine torque"),
    ("Tool wear [min]", "Numeric", "Tool wear measurement"),
    ("Target", "Binary", "Machine failure indicator"),
    ("Failure Type", "Categorical", "Recorded failure category"),
]

for feature, typ, desc in features:
    cells = table.add_row().cells
    cells[0].text = feature
    cells[1].text = typ
    cells[2].text = desc


# ==================================================
# DATA PREPROCESSING
# ==================================================
add_heading("5. Data Preprocessing", 1)

doc.add_paragraph(
    "The dataset was inspected for missing values, duplicate rows, data "
    "types, and consistency between the Target and Failure Type columns."
)

add_bullet("No missing values were identified.")
add_bullet("Duplicate rows were checked.")
add_bullet("UDI and Product ID were excluded from model features because they are identifiers.")
add_bullet("The Type column was one-hot encoded.")
add_bullet("Numeric sensor features were standardized.")
add_bullet("A stratified 80/20 train-test split was used.")
add_bullet("The training set contained 8,000 records and the test set contained 2,000 records.")


# ==================================================
# EDA
# ==================================================
add_heading("6. Exploratory Data Analysis", 1)

doc.add_paragraph(
    "Exploratory data analysis was performed to understand the distribution "
    "of the target variable, failure categories, machine types, sensor "
    "measurements, correlations, and differences between failure and "
    "non-failure records."
)

add_image(
    "01_target_class_distribution.png",
    "Figure 1. Target class distribution.",
)

add_image(
    "02_failure_type_distribution.png",
    "Figure 2. Failure type distribution.",
)

add_image(
    "03_machine_type_distribution.png",
    "Figure 3. Machine type distribution and failure rate.",
)

add_image(
    "04_sensor_feature_distributions.png",
    "Figure 4. Numeric sensor feature distributions by failure status.",
)

add_image(
    "05_correlation_heatmap.png",
    "Figure 5. Correlation heatmap.",
)

add_image(
    "06_boxplots_by_failure_status.png",
    "Figure 6. Sensor feature box plots by failure status.",
)


# ==================================================
# MODEL DEVELOPMENT
# ==================================================
add_heading("7. Model Development", 1)

doc.add_paragraph(
    "Five classification algorithms were evaluated for the primary binary "
    "failure prediction task:"
)

models = [
    "Gradient Boosting",
    "Random Forest",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Logistic Regression",
]

for model_name in models:
    add_bullet(model_name)

doc.add_paragraph(
    "Five-fold stratified cross-validation was performed using the training "
    "data. Macro F1-score was used as the main model-selection metric because "
    "the target classes are imbalanced."
)

table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"

hdr = table.rows[0].cells
hdr[0].text = "Model"
hdr[1].text = "Mean F1-Macro"
hdr[2].text = "ROC-AUC"

comparison = [
    ("Gradient Boosting", "0.8587", "0.9689"),
    ("Random Forest", "0.8284", "0.9676"),
    ("Decision Tree", "0.8076", "0.7833"),
    ("K-Nearest Neighbors", "0.6906", "0.8574"),
    ("Logistic Regression", "0.5671", "0.8965"),
]

for name, f1, auc in comparison:
    cells = table.add_row().cells
    cells[0].text = name
    cells[1].text = f1
    cells[2].text = auc

doc.add_paragraph(
    "Gradient Boosting was selected for the final primary classifier based "
    "on the cross-validation macro F1-score."
)


# ==================================================
# PRIMARY MODEL RESULTS
# ==================================================
add_heading("8. Primary Failure Prediction Results", 1)

doc.add_paragraph(
    "The selected Gradient Boosting model was evaluated once on the held-out "
    "test set of 2,000 records."
)

table = doc.add_table(rows=1, cols=2)
table.style = "Table Grid"

hdr = table.rows[0].cells
hdr[0].text = "Metric"
hdr[1].text = "Test Result"

metrics = [
    ("Accuracy", "0.9900"),
    ("Precision (Macro)", "0.9283"),
    ("Recall (Macro)", "0.8364"),
    ("F1-Macro", "0.8764"),
    ("ROC-AUC", "0.9649"),
]

for metric, value in metrics:
    cells = table.add_row().cells
    cells[0].text = metric
    cells[1].text = value

add_image(
    "07_confusion_matrix.png",
    "Figure 7. Confusion matrix for the primary failure prediction model.",
)

add_image(
    "08_roc_curve.png",
    "Figure 8. ROC curve for the primary failure prediction model.",
)

add_image(
    "09_feature_importance.png",
    "Figure 9. Feature importance of the primary model.",
)


# ==================================================
# CONFUSION MATRIX INTERPRETATION
# ==================================================
add_heading("9. Confusion Matrix Analysis", 1)

doc.add_paragraph(
    "The held-out test set produced 1,925 true negatives, 7 false positives, "
    "22 false negatives, and 46 true positives. These values describe the "
    "classification behavior of the model on the test data."
)

add_bullet("True Negatives: 1,925")
add_bullet("False Positives: 7")
add_bullet("False Negatives: 22")
add_bullet("True Positives: 46")


# ==================================================
# SECONDARY FAILURE TYPE MODEL
# ==================================================
add_heading("10. Failure Type Classification", 1)

doc.add_paragraph(
    "A secondary classifier was trained using records representing actual "
    "failures with a known failure category. Four failure categories were "
    "available for this task: Heat Dissipation Failure, Power Failure, "
    "Overstrain Failure, and Tool Wear Failure."
)

doc.add_paragraph(
    "The failure-type dataset contained 330 usable failure records. An "
    "80/20 stratified split was used, resulting in 264 training records and "
    "66 test records."
)

table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"

hdr = table.rows[0].cells
hdr[0].text = "Failure Type"
hdr[1].text = "Precision"
hdr[2].text = "Recall"

failure_results = [
    ("Heat Dissipation Failure", "0.81", "1.00"),
    ("Overstrain Failure", "0.83", "0.62"),
    ("Power Failure", "0.95", "1.00"),
    ("Tool Wear Failure", "0.86", "0.67"),
]

for name, precision, recall in failure_results:
    cells = table.add_row().cells
    cells[0].text = name
    cells[1].text = precision
    cells[2].text = recall

doc.add_paragraph(
    "The secondary Gradient Boosting model achieved a macro F1-score of "
    "approximately 0.83 on the held-out failure-type test set. The number "
    "of Tool Wear test examples was small, so this result should be "
    "interpreted with caution."
)

doc.add_paragraph(
    "The dataset also contains records labelled as Random Failures that do "
    "not correspond to Target=1 failure records. These records were not "
    "treated as a separate failure class in the secondary classifier."
)


# ==================================================
# SYSTEM ARCHITECTURE
# ==================================================
add_heading("11. System Architecture", 1)

doc.add_paragraph(
    "The project is organized into separate components for data, model "
    "training, backend inference, frontend interaction, and documentation."
)

add_bullet("Jupyter Notebook: complete analysis and machine learning workflow.")
add_bullet("train_model.py: supporting model-training workflow.")
add_bullet("model/: saved primary and secondary model files.")
add_bullet("backend/: Flask REST API for predictions.")
add_bullet("frontend/: Streamlit interface.")
add_bullet("data/: predictive maintenance dataset.")
add_bullet("report_images/: figures used in this report.")


# ==================================================
# BACKEND
# ==================================================
add_heading("12. Flask Backend", 1)

doc.add_paragraph(
    "A Flask REST API was developed to load the trained models and provide "
    "prediction services. The backend exposes a health endpoint and a "
    "prediction endpoint."
)

add_bullet("GET /health — checks whether the API is running.")
add_bullet("POST /predict — receives machine sensor values and returns predictions.")
add_bullet("The primary model predicts whether failure is expected.")
add_bullet("The secondary model is used when a failure is predicted.")
add_bullet("CORS is enabled for frontend communication.")
add_bullet("The API runs locally on port 5000 during development.")


# ==================================================
# FRONTEND
# ==================================================
add_heading("13. Streamlit Frontend", 1)

doc.add_paragraph(
    "A Streamlit web interface was developed so that users can enter the "
    "six model input features through a graphical interface. The frontend "
    "communicates with the Flask backend and displays the prediction result "
    "and failure probability."
)

add_bullet("Machine Type")
add_bullet("Air Temperature")
add_bullet("Process Temperature")
add_bullet("Rotational Speed")
add_bullet("Torque")
add_bullet("Tool Wear")


# ==================================================
# TEST PREDICTION
# ==================================================
add_heading("14. Example System Test", 1)

doc.add_paragraph(
    "An end-to-end test was performed through the Streamlit interface using "
    "the following sample inputs:"
)

test_table = doc.add_table(rows=1, cols=2)
test_table.style = "Table Grid"

hdr = test_table.rows[0].cells
hdr[0].text = "Input"
hdr[1].text = "Value"

test_inputs = [
    ("Machine Type", "M"),
    ("Air Temperature", "300 K"),
    ("Process Temperature", "310 K"),
    ("Rotational Speed", "1500 rpm"),
    ("Torque", "40 Nm"),
    ("Tool Wear", "100 min"),
]

for name, value in test_inputs:
    cells = test_table.add_row().cells
    cells[0].text = name
    cells[1].text = value

doc.add_paragraph(
    "For this software test input, the system returned 'No failure predicted' "
    "with a displayed failure probability of approximately 0.07%. This is "
    "a demonstration of the software pipeline and should not be interpreted "
    "as a safety guarantee for real industrial machinery."
)


# ==================================================
# PROJECT STRUCTURE
# ==================================================
add_heading("15. Project Structure", 1)

structure = """Predictive_Maintenance_Project/
├── backend/
├── data/
├── frontend/
├── model/
├── report_images/
├── Predictive_Maintenance.ipynb
├── Project_Report.docx
├── README.md
├── requirements.txt
├── train_model.py
└── generate_report.py
"""

p = doc.add_paragraph()
r = p.add_run(structure)
r.font.name = "Courier New"
r.font.size = Pt(9)


# ==================================================
# CONCLUSION
# ==================================================
add_heading("16. Conclusion", 1)

doc.add_paragraph(
    "This project demonstrates an end-to-end predictive maintenance workflow "
    "for industrial machinery. The workflow includes data inspection, "
    "exploratory analysis, preprocessing, model comparison, model evaluation, "
    "failure-type classification, model persistence, API development, and "
    "a user-facing Streamlit application."
)

doc.add_paragraph(
    "The primary Gradient Boosting model achieved an F1-macro score of 0.8764 "
    "and an ROC-AUC of 0.9649 on the held-out test set. The results demonstrate "
    "that the available sensor measurements contain useful information for "
    "predicting the failure target in this dataset."
)


# ==================================================
# FUTURE SCOPE
# ==================================================
add_heading("17. Future Scope", 1)

future = [
    "Integrate continuous real-time sensor streams.",
    "Evaluate the system on additional industrial datasets.",
    "Perform time-series modelling when timestamped sensor data is available.",
    "Add model monitoring and drift detection.",
    "Experiment with additional ensemble and deep learning approaches.",
    "Add authenticated deployment for production environments.",
    "Integrate maintenance scheduling and alerting workflows.",
]

for item in future:
    add_bullet(item)


# ==================================================
# REFERENCES
# ==================================================
add_heading("18. References", 1)

references = [
    "Kaggle, Machine Predictive Maintenance Classification dataset.",
    "Scikit-learn documentation for machine learning preprocessing, model selection, and evaluation.",
    "Flask documentation for Python web APIs.",
    "Streamlit documentation for Python-based data applications.",
    "IBM SkillsBuild / AICTE-2026 Agentic AI problem statement material.",
]

for reference in references:
    add_bullet(reference)


# ==================================================
# Save
# ==================================================
doc.save(OUTPUT)

print()
print("==========================================")
print("REPORT CREATED SUCCESSFULLY")
print("==========================================")
print("File:", OUTPUT)
print("Size:", OUTPUT.stat().st_size, "bytes")
print()
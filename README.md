# Sepsis Early Detection - Clinical Assistant & Deep Learning Models

This repository contains a comprehensive suite of Machine Learning (ML) and Deep Learning (DL) models for the **early detection of sepsis** in Intensive Care Unit (ICU) patients. Sepsis is a life-threatening medical emergency, and early detection is crucial for clinical intervention and patient survival.

The models are trained on the standard **PhysioNet Computing in Cardiology Challenge 2019** dataset, which includes patient vital signs, laboratory values, and demographic information.

## 📂 Repository Structure

- [notebooks/](file:///d:/early_sepsis_prediction_model/notebooks/)
  - [CapstoneProject_SepsisPrediction.ipynb](file:///d:/early_sepsis_prediction_model/notebooks/CapstoneProject_SepsisPrediction.ipynb): Traditional ML models (Logistic Regression, Random Forest, XGBoost), **SHAP explainability**, and a clinical assistant **Gradio web interface**.
  - [sepsis.ipynb](file:///d:/early_sepsis_prediction_model/notebooks/sepsis.ipynb): Sequential deep learning models, including **LSTM** and **TCN (1D CNN)** architectures, evaluated on 3-hour and 6-hour time-series windows.
- [.gitignore](file:///d:/early_sepsis_prediction_model/.gitignore): Prevents dataset, python cache, and checkpoint directories from being uploaded.
- [requirements.txt](file:///d:/early_sepsis_prediction_model/requirements.txt): Python dependency configurations.

---

## 📊 Dataset Details

The models ingest 40+ clinical features, categorized into:
1. **Vital Signs**: Heart Rate (`HR`), Oxygen Saturation (`O2Sat`), Temperature (`Temp`), Systolic Blood Pressure (`SBP`), Mean Arterial Pressure (`MAP`), Diastolic Blood Pressure (`DBP`), Respiration Rate (`Resp`), and End-Tidal Carbon Dioxide (`EtCO2`).
2. **Laboratory Measurements**: Blood Gases (`pH`, `BaseExcess`, `HCO3`, `FiO2`, `PaCO2`, `SaO2`), Hematology (`WBC`, `Platelets`, `Hct`, `Hgb`, `PTT`, `Fibrinogen`), and Metabolic Panels (`BUN`, `Creatinine`, `Bilirubin`, `Glucose`, `Lactate`, `Electrolytes`).
3. **Demographics**: `Age`, `Gender`, `Unit1`/`Unit2` (ICU type), hospital admission time (`HospAdmTime`), and ICU length of stay (`ICULOS`).

*Note: Sepsis is highly imbalanced in the dataset (approx. **2.17%** positive rate), which is addressed using SMOTE (Synthetic Minority Over-sampling Technique) and class weighting during model training.*

---

## 📈 Model Performance & Evaluation

### 1. Traditional Machine Learning Models (Tabular Prediction)
*Trained on features with SMOTE and standard scaling:*

| Model | Accuracy | Test ROC-AUC | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **Random Forest** | ~99% | **0.9998** | High precision and recall; explainable via SHAP. |
| **XGBoost** | ~93% | **0.9811** | Robust gradient boosted tree classifier. |
| **Logistic Regression** | ~71% | **0.7643** | Linear baseline model. |

### 2. Time-Series Deep Learning Models (Sequence Prediction)
*Trained using sequential windows (3-hour vs. 6-hour history) to predict Sepsis onset:*

| Model | Window Size | Accuracy | Test ROC-AUC |
| :--- | :--- | :--- | :--- |
| **LSTM** | 6-Hour History | ~88% | **0.7735** |
| **Temporal CNN (TCN)** | 6-Hour History | ~84% | **0.7642** |
| **LSTM** | 3-Hour History | ~81% | **0.7307** |
| **Temporal CNN (TCN)** | 3-Hour History | ~73% | **0.7655** |

---

## 🧠 Explainable AI (SHAP)
To build clinical trust, the Random Forest model incorporates **SHAP (SHapley Additive exPlanations)**. The notebooks render global feature impact plots and local force/waterfall plots to show which clinical features (e.g., elevated heart rate, low oxygen saturation, or lab values like lactate) drive the positive predictions.

---

## 🩺 Clinical Assistant Gradio Interface
The project contains an interactive **Gradio Clinical Assistant Web Application**. 
- Clinicians can input real-time patient vitals (HR, O2Sat, Temp, SBP, MAP, Resp, Age, etc.).
- The system predicts the sepsis probability and labels the patient risk level (**Low**, **Moderate**, or **High**).
- It generates an automated clinical **Agent Insight** text and embeds SHAP visualization for diagnostic transparency.

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure Python 3.8+ is installed. Clone this repository and navigate to the directory:
```bash
git clone <your-repo-url>
cd early_sepsis_prediction_model
```

### 2. Install Dependencies
Install all required libraries listed in [requirements.txt](file:///d:/early_sepsis_prediction_model/requirements.txt):
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset
Place the raw dataset `Dataset.csv` (from PhysioNet Challenge 2019) inside the root directory:
```text
early_sepsis_prediction_model/
└── Dataset.csv
```
*(Note: `Dataset.csv` is excluded from Git commits to keep the repository lightweight.)*

### 4. Run the Code & Web App
- Run [CapstoneProject_SepsisPrediction.ipynb](file:///d:/early_sepsis_prediction_model/notebooks/CapstoneProject_SepsisPrediction.ipynb) using Jupyter Notebook or Google Colab to run the ML pipeline and start the **Gradio interface**.
- Run [sepsis.ipynb](file:///d:/early_sepsis_prediction_model/notebooks/sepsis.ipynb) to experiment with Deep Learning LSTM/TCN models.

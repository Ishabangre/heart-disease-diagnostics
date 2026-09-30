# ❤️ Heart Disease Diagnostics

An end-to-end machine-learning project that estimates the risk of heart disease from 13 clinical values. It includes model comparison and tuning, probability calibration, explainable AI (SHAP), a what-if simulator, personalised daily-life advice, and a Streamlit web app.

**Live demo:** _add your Streamlit link here_ (for example `https://your-app.streamlit.app`)

> ⚠️ **Educational project only.** This is not a medical device and does not give a medical diagnosis. Always consult a qualified doctor for real health decisions.

---

## Features

| Page | What it does |
|---|---|
| 🏠 **Home** | Front page with a single *Start Diagnosis* button |
| 🩺 **Prediction** | Enter 13 clinical values and get a calibrated risk percentage, plus a SHAP waterfall showing *why* |
| 🎛️ **What-If** | Move sliders (cholesterol, blood pressure, age...) and watch the risk change live, with a risk curve for that patient |
| 🌿 **Health Advice** | Personalised daily routine, food, exercise and monitoring advice, a daily checklist, and emergency warning signs |
| 🔍 **Explainability** | Global SHAP summary and partial-dependence curves |
| 📊 **Dashboard** | Model comparison table (cross-validation mean ± std), threshold and calibration results, confusion matrix, ROC curve |

## How the model is built

1. **Honest data handling:** duplicate rows are removed (the public dataset contains many repeats, which leak between train and test and inflate accuracy).
2. **Feature engineering:** `hr_ratio`, `bp_chol_index`, `stress_index`, `age_group`.
3. **Model comparison:** Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, XGBoost, LightGBM.
4. **Hyper-parameter tuning:** `RandomizedSearchCV` with stratified 5-fold cross-validation.
5. **Ensembles:** soft-voting and stacking of the top-3 tuned models.
6. **Model selection:** best model by cross-validated ROC-AUC. The test set is **not** used for selection.
7. **Probability calibration:** `CalibratedClassifierCV` (sigmoid), so "risk 70%" is a trustworthy probability.
8. **Threshold tuning:** decision threshold chosen on out-of-fold predictions using an F2 score (recall counts twice as much as precision, because missing a sick patient is worse than a false alarm).
9. **Explainability:** SHAP (model-agnostic, in probability space), partial dependence and ICE curves.

## Results

Example results from one run on the UCI Heart Disease data (302 unique rows: 241 train, 61 test). Update these numbers after your own final training run.

| Metric | Value |
|---|---|
| Best model | Voting ensemble (XGBoost + Gradient Boosting + KNN) |
| Cross-validated ROC-AUC | ~0.91 |
| Test ROC-AUC | 0.93 |
| Test recall at tuned threshold (0.25) | 0.94 |
| Test accuracy at tuned threshold | 0.82 |
| Brier score (before → after calibration) | 0.108 → 0.102 |

The test set is small (61 rows), so one prediction changes accuracy by about 1.6%. Trust the cross-validation mean ± std (see the Dashboard) more than a single test number.

## Project structure

```
HEART_DISEASE_PREDICTION/
│
├── dataset/
│   └── heart.csv                  # not committed, see "Dataset"
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   └── 04_Testing.ipynb
│
├── src/
│   ├── preprocessing.py           # load, clean (drop duplicates), split
│   ├── feature_engineering.py     # derived features
│   ├── model.py                   # model zoo, search spaces, ensembles
│   ├── train.py                   # full training pipeline
│   ├── evaluate.py                # metrics, confusion matrix, ROC, calibration
│   ├── explain.py                 # SHAP, partial dependence, what-if curves
│   ├── predict.py                 # load model and predict one patient
│   └── recommendations.py         # daily-life advice logic
│
├── models/
│   ├── heart_model.pkl            # calibrated best model
│   ├── feature_columns.pkl
│   ├── background.pkl             # sample of training rows for SHAP / what-if
│   └── metrics.json
│
├── outputs/
│   ├── graphs/                    # ROC, SHAP summary, feature importance, calibration, PDP
│   ├── confusion_matrix/
│   └── reports/                   # classification report, model_comparison.csv
│
├── app/
│   ├── main.py                    # Streamlit app (button navigation)
│   └── dashboard.py               # Dashboard page
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (the commonly used `heart.csv` version with 13 features and a binary target). Place the file at `dataset/heart.csv`. It is not committed to the repository.

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0 to 3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar above 120 mg/dl (1 = yes) |
| `restecg` | Resting ECG result (0 to 2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak-exercise ST segment |
| `ca` | Number of major vessels highlighted (0 to 4) |
| `thal` | Thalassemia test result code |
| `target` | 1 = heart disease, 0 = no heart disease |

## Installation

```bash
git clone https://github.com/<your-username>/heart-disease-diagnostics.git
cd heart-disease-diagnostics

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Then put `heart.csv` into the `dataset/` folder.

## Usage

Run everything from the project root.

```bash
python -m src.train              # full pipeline: compare, tune, ensemble, calibrate, save
python -m src.train --n-iter 5   # quicker run (fewer tuning trials)
python -m src.evaluate           # confusion matrix, ROC curve, report
python -m src.explain            # feature importance, SHAP summary, partial dependence
python -m src.predict            # sample prediction in the terminal
streamlit run app/main.py        # start the web app
```

Useful training options:

```bash
python -m src.train --keep-duplicates   # keep duplicate rows (inflates scores, not recommended)
python -m src.train --no-fe             # disable feature engineering
```

Use it from Python:

```python
from src.predict import predict

patient = dict(age=55, sex=1, cp=0, trestbps=130, chol=240, fbs=0, restecg=1,
               thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=2)
print(predict(patient))
# {'prediction': 1, 'probability': 0.63, 'threshold': 0.25, 'message': 'Person HAS heart disease'}
```

## Deployment (Streamlit Community Cloud)

1. Train the model locally and commit `models/` and `outputs/` (they are needed by the app).
2. Pin exact package versions in `requirements.txt` (the pickled model must be loaded with the same scikit-learn version).
3. Push to GitHub.
4. On [share.streamlit.io](https://share.streamlit.io) create an app with **Main file path** `app/main.py` and the same Python version as your local environment.

## Limitations

- **Small dataset:** only about 300 unique patients, so all numbers have wide uncertainty.
- **Old, single-source data:** the data comes from a few hospitals decades ago and may not generalise to other populations.
- **Correlation, not causation:** SHAP, partial-dependence and what-if results show what the *model* learned, not what would happen to a real person.
- **Recall-focused threshold:** catching more sick patients means more false alarms.
- **Not clinically validated.** Do not use it for real diagnosis or treatment decisions.

## Future work

- Patient history with trends (SQLite)
- PDF report download
- Hindi / English language toggle
- Training on multiple datasets (Cleveland, Hungarian, Switzerland, VA)
- FastAPI backend and Docker image
- Real-time heart-rate input from a wearable or sensor

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, Streamlit.

## Author

**Your Name**: add your GitHub / LinkedIn link here.


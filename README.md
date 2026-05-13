# 🚢 Titanic Survival Prediction

A machine learning pipeline that predicts passenger survival on the Titanic using a tuned Random Forest classifier. Built for the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

---

## 📌 Overview

This project goes beyond basic feature usage by engineering meaningful signals from raw data — extracting social titles, estimating family survival rates, and binning fares — before training and hyperparameter-tuning a Random Forest model to predict which passengers survived.

---

## 🗂️ Dataset

The dataset comes from the Kaggle Titanic competition. Train and test sets are combined upfront for consistent cleaning and feature engineering, then split back before modelling.

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender (encoded: female=1, male=0) |
| `Age` | Age in years (missing values imputed) |
| `Fare` | Ticket fare |
| `Ticket` | Ticket number |
| `Name` | Passenger name (used for title extraction) |
| `Survived` | Target: 1 = survived, 0 = did not survive |

---

## 🔧 Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, modelling, hyperparameter tuning
- `matplotlib`, `seaborn` — visualisation
- Kaggle environment

---

## 🚀 Pipeline

### 1. Data Loading
Train and test CSVs are loaded and concatenated into a single DataFrame for unified preprocessing.

### 2. Feature Engineering

**Title Extraction** — Titles are parsed from passenger names, rare titles (Dr, Rev, Sir, etc.) are grouped under `Rare`, and all titles are ordinally encoded.

**Family Survival** — A `Family_Group` key is created from last name and fare. For each passenger, the notebook looks at whether other members of their group survived, assigning `1` (someone survived), `0` (no one survived), or `0.5` (unknown/travelling alone).

**Missing Value Imputation** — Age is imputed using the median age for each Title × Pclass group. Fare missing values are filled with the global median.

**Fare Binning** — Fare is discretised into 5 quantile bins (`FareBin`) to reduce the effect of outliers.

**Ticket Frequency** — `Ticket_Freq` counts how many passengers shared the same ticket number, acting as a proxy for group size.

### 3. Feature Selection
The final feature set used for modelling:

```
Pclass, Sex, Age, Family_Survival, FareBin, Title, Ticket_Freq
```

### 4. Train / Validation Split
Stratified 80/20 split with `random_state=42`.

### 5. Baseline Random Forest
An initial Random Forest (100 trees, OOB scoring enabled) is trained and evaluated on the validation set.

### 6. Feature Importance Visualisation
A horizontal bar chart plots Mean Gini Decrease for each feature, with the most important features highlighted in a distinct colour.

### 7. Hyperparameter Tuning
`GridSearchCV` with 5-fold cross-validation searches over:

```python
{
    'n_estimators': [100, 300, 500],
    'max_depth': [6, 7, 8],
    'min_samples_leaf': [3, 5, 10],
    'criterion': ['gini', 'entropy']
}
```

The best estimator is extracted and re-evaluated. Training accuracy is compared against the CV score to flag potential overfitting.

### 8. Submission
Predictions from the tuned model are written to `submission.csv` in the format required by Kaggle.

---

## 📊 Evaluation

The model is assessed at two stages — baseline and post-tuning — using:

- Training and validation accuracy
- Out-of-bag (OOB) score (baseline)
- Cross-validation accuracy from GridSearch (tuned)
- Full classification report (precision, recall, F1 for `Survived` and `Not Survived`)
- Overfitting check: flags if training accuracy exceeds CV score by more than 5%

---

## 📁 Repository Structure

```
├── titanic-survival-prediction-random-forest.ipynb   # Main notebook
├── submission.csv                                     # Kaggle submission output
└── README.md
```

---

## ▶️ How to Run

### On Kaggle
1. Add the [Titanic competition dataset](https://www.kaggle.com/competitions/titanic/data) to your notebook
2. Open and run all cells in the notebook
3. `submission.csv` will be saved to the working directory

### Locally
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook titanic-survival-prediction-random-forest.ipynb
```

Update the file paths in the data loading cells from `/kaggle/input/...` to your local paths.

---

## 🔭 Potential Improvements

- Add cabin deck extraction from the `Cabin` feature
- Engineer an `IsAlone` flag from `SibSp` and `Parch`
- Try gradient boosting models (XGBoost, LightGBM)
- Use SHAP for deeper model explainability beyond Gini importance
- Explore threshold tuning to optimise F1 rather than accuracy

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

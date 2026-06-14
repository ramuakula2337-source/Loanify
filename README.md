Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval. Implemented Binary classification along with EDA, feature engineering &amp; model evaluation (Precision, Recall, F1).
# Loanify — Intelligent Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)

## Problem Statement

**SecureTrust Bank** processes hundreds of loan applications daily. Their existing manual verification process is time-consuming, biased, and inconsistent — causing good customers to get rejected and high-risk customers to get approved. This project builds an **intelligent ML system** that automatically analyses applicant details and predicts whether a loan should be Approved or Rejected before final human verification.

---

## Dataset

- **Source:** Kaggle
- **Features:** 19 (Applicant Income, Coapplicant Income, Employment Status, Age, Marital Status, Dependents, Credit Score, Existing Loans, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term, Loan Purpose, Property Area, Education Level, Gender, Employer Category)
- **Target:** `Loan_Approved` — 1 = Approved, 0 = Rejected

---

## Project Workflow

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Best Model
```

### Steps
1. **EDA** — Class balance check, income distributions, outlier detection (box plots), credit score histogram, correlation heatmap
2. **Preprocessing** — Mean/Mode imputation, Label Encoding, One-Hot Encoding, StandardScaler, dropped Applicant_ID
3. **Feature Engineering** — Added `DTI_Ratio²` and `Credit_Score²` to capture non-linear relationships
4. **Evaluation** — Precision selected as primary metric to minimise false approvals (approving risky applicants is costlier)

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **88.0%** | 78.5% | 83.6% | **81.0%** |
| KNN (k=5) | 78.5% | 67.3% | 57.4% | 62.0% |
| Naive Bayes | 86.0% | **81.1%** | 70.5% | 75.4% |

### Recommendation
- **Best overall:** Logistic Regression — highest accuracy (88%) and F1 (81%)
- **Best precision:** Naive Bayes (81.1%) — use when minimising false approvals is the top priority

---

## Key Insights

- **Precision chosen as primary metric** — approving a high-risk customer leads to financial loss; false negatives are less costly than false positives here
- **KNN underperformed** — financial tabular data does not cluster well geometrically at k=5
- **Feature engineering helped** — squaring DTI and Credit Score captured non-linear risk thresholds

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML Models | Scikit-learn |
| Algorithms | Logistic Regression, KNN, Naive Bayes |
| Encoding | LabelEncoder, OneHotEncoder |
| Scaling | StandardScaler |
| Imputation | SimpleImputer |
| Environment | Jupyter Notebook |

---

## Files

| File | Description |
|------|-------------|
| `Loanify.ipynb` | Main notebook — EDA, preprocessing, feature engineering, model training |
| `loan_approval_data.csv` | Loan applicant records (Kaggle) |
| `README.md` | Project documentation |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ramuakula2337-source/loanify-loan-approval-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Open the notebook
jupyter notebook Loanify.ipynb
```

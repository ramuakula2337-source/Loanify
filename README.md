#  Loanify — Intelligent Loan Approval Prediction System
Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval. Implemented Binary classification along with EDA, feature engineering &amp; model evaluation (Precision, Recall, F1).
##  Problem Statement

A mid-sized financial company **SecureTrust Bank** offers personal and home loans to customers across urban and rural regions of India. Their existing **manual verification process** is time-consuming, biased, and inconsistent — causing good customers to get rejected and high-risk customers to get approved.

This project builds an **intelligent ML system** that automatically analyses applicant details and **predicts whether a loan should be Approved or Rejected** before final human verification.

---

## Dataset Description

The dataset contains **loan applicant records** with 19 features covering personal, financial, and credit information.

| Feature | Description |
|---|---|
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| **Loan_Approved (Target)** | **1 = Approved, 0 = Rejected** |

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Best Model
```

### 1️⃣ Exploratory Data Analysis (EDA)
- Class balance check (pie chart — Approved vs Rejected)
- Distribution plots for Applicant Income & Co-applicant Income
- Box plots to detect outliers across Income, Credit Score, DTI Ratio, Savings
- Histogram of Credit Score split by Loan Approval status
- Correlation heatmap across all numerical features

### 2️⃣ Data Preprocessing
- **Missing Value Imputation** — Mean for numerical columns, Mode for categorical columns
- **Label Encoding** — Education Level, Loan Approved (target)
- **One-Hot Encoding** — Employment Status, Marital Status, Loan Purpose, Property Area, Gender, Employer Category
- **Feature Scaling** — StandardScaler applied before model training
- **Dropped** — Applicant_ID (not useful for prediction)

### 3️⃣ Feature Engineering
- Added `DTI_Ratio²` — squared Debt-to-Income ratio
- Added `Credit_Score²` — squared Credit Score
- Replaced original columns with engineered versions for better model signal

### 4️⃣ Models Built & Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors (KNN) | Distance-based classifier (k=5) |
| Naive Bayes (GaussianNB) | Probabilistic classifier — **Best Precision** ✅ |

### 5️⃣ Evaluation Metrics
- Accuracy
- Precision ← primary metric (avoid approving high-risk customers)
- Recall
- F1-Score
- Confusion Matrix

---
##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML Models | Scikit-learn |
| Algorithms | Logistic Regression, KNN, Naive Bayes |
| Encoding | LabelEncoder, OneHotEncoder |
| Scaling | StandardScaler |
| Imputation | SimpleImputer |
| Environment | Jupyter Notebook |

--
## 📁 Files

| File | Description |
|------|-------------|
| `Loanify.ipynb` | Main notebook — EDA, preprocessing, feature engineering, model training |
| `loan_approval_data.csv` | Dataset with loan applicant records |
| `README.md` | Project documentation |

##  Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 88.0% | 78.46% | 83.61% | 80.95% |
| KNN | 78.5% | 67.31% | 57.38% | 61.95% |
| Naive Bayes | 86.0% | 81.13% | 70.49% | 75.44% |

##  Result

- Logistic Regression achieved the best overall performance with 88% Accuracy and 80.95% F1 Score.
- Naive Bayes achieved the highest Precision (81.13%), making it suitable when minimizing false loan approvals is the primary objective.


 

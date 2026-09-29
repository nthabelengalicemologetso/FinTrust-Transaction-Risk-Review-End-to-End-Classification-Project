[FinTrust_Part_A_Data_Prep.py](https://github.com/user-attachments/files/32812991/FinTrust_Part_A_Data_Prep.py)# FinTrust-Transaction-Risk-Review-End-to-End-Classification-Project 
#<img width="1031" height="577" alt="Screenshot 2026-09-29 175922" src="https://github.com/user-attachments/assets/d8d58c8f-78d2-4436-a476-ca5f3566fb68" />



# 📌 Overview

**FinTrust** is a simulated Nigerian retail bank. This project uses customer and transaction data to investigate and predict whether a transaction should be flagged for **manual risk review**.

The project follows an end-to-end data science workflow, covering:

- Data preparation
- Exploratory Data Analysis (EDA)
- Statistical testing
- Feature engineering
- Classification modelling
- Model evaluation
- Model interpretation

The goal is to identify transaction patterns associated with higher review risk and build a classification model that can help support transaction-risk screening.

---

## 🎯 Project Objective

The primary objective is to predict the target variable:

```text
Risk_Review_Flag
```

This is a binary classification problem where transactions are classified according to whether they are flagged for manual risk review.

The analysis also investigates behavioural and transaction-level patterns that may help explain why certain transactions are more likely to be flagged.

---

## 📊 Dataset Overview

The project analyses:

| Metric | Value |
|---|---:|
| Transactions | **12,000** |
| Customers | **1,500** |
| Transactions flagged for review | **19.6%** |
| Engineered features | **12** |
| Best baseline ROC-AUC | **0.67** |

The dataset contains customer and transaction information that can be used to analyse transaction behaviour and identify potential risk patterns.

---

## 🔎 Exploratory Data Analysis

EDA was performed to understand the distribution of transaction behaviour and identify patterns associated with the risk-review flag.

Key areas explored included:

- Transaction types
- Cash withdrawals
- Transfers
- Deposit activity
- Night-time transactions
- Customer behaviour
- Risk-review rates across transaction categories

### Key observations

The analysis identified differences in review rates across transaction types and transaction timing.

For example, the dashboard highlights elevated review rates for:

- Transfers
- Cash withdrawals
- Night-time transactions

These patterns were investigated further through statistical analysis and feature engineering.

---

## 🧪 Statistical Testing

Statistical tests were used to determine whether observed relationships between transaction characteristics and the target variable were statistically meaningful.

The statistical analysis helped distinguish potentially meaningful patterns from differences that could occur due to random variation.

The results were used to inform feature engineering and modelling decisions.

---

## ⚙️ Feature Engineering

Feature engineering was performed to convert the raw customer and transaction information into variables suitable for classification models.

The final modelling workflow included **12 engineered features**.

Feature engineering focused on extracting useful information from areas such as:

- Transaction behaviour
- Transaction timing
- Customer activity
- Transaction categories
- Risk-related behavioural indicators

---

## 🤖 Modelling

The project evaluates baseline classification models for predicting `Risk_Review_Flag`.

The models include:

### Logistic Regression

Logistic Regression provides an interpretable baseline for binary classification and helps establish how well the engineered features can distinguish between reviewed and non-reviewed transactions.

### Random Forest

Random Forest was used as a tree-based classification model capable of capturing non-linear relationships and interactions between features.

---

## 📈 Model Evaluation

The models were evaluated using classification metrics including:

- Precision
- Recall
- F1-score
- ROC-AUC

The evaluation was performed on a held-out test set to assess model performance on unseen data.

### Baseline performance

The project dashboard reports a best baseline **ROC-AUC of approximately 0.67**.

This indicates that the baseline models provide useful predictive signal while leaving room for further model improvement.

> **Note:** Model performance should be interpreted in the context of the simulated dataset and the chosen train/test methodology.

---

## 📊 Evaluation Metrics

### Precision

Measures the proportion of transactions predicted as requiring review that were actually flagged.

### Recall

Measures the proportion of actual risk-review transactions that the model successfully identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two target classes across different classification thresholds.

---

## 💡 Business Interpretation

From a transaction-risk perspective, the analysis provides insight into customer and transaction characteristics associated with manual review.

The findings can support questions such as:

- Which transaction types receive more risk-review flags?
- Are transactions occurring at unusual times associated with higher review rates?
- Which customer behaviours provide useful predictive information?
- How effectively can transaction-level features identify potentially risky activity?

The project demonstrates how data science can be used to move from raw transaction data to a measurable classification workflow.

---

## 🛠️ Tools & Technologies

| Technology | Purpose |
|---|---|
| **Python** | Data analysis and modelling |
| **Pandas** | Data manipulation and preparation |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualisation |
| **SciPy** | Statistical testing |
| **Scikit-learn** | Feature engineering, modelling and evaluation |

---

## 🔄 Project Workflow

```text
Raw Customer & Transaction Data
              │
              ▼
       Data Preparation
              │
              ▼
             EDA
              │
              ▼
     Statistical Testing
              │
              ▼
      Feature Engineering
              │
              ▼
       Classification
        ┌─────┴─────┐
        ▼           ▼
Logistic Regression  Random Forest
        │           │
        └─────┬─────┘
              ▼
       Model Evaluation
              │
              ▼
       Interpretation
              │
              ▼
     Transaction Risk Insights

#data preparation
[Upload# ============================
# FinTrust Risk Review - Part A
# Data Preparation Workflow
# ============================

import pandas as pd
from pathlib import Path

# ---- 0. Config ----
DATA_DIR = Path(".")  # change to your data folder if needed
CUSTOMER_FILE = DATA_DIR / "FinTrust_Customer_Data.csv"
TRANSACTION_FILE = DATA_DIR / "FinTrust_Transaction_Data.csv"

# ---- 1. Load customer dataset ----
customers = pd.read_csv(
    CUSTOMER_FILE,
    sep=";", decimal=",", encoding="utf-8-sig"
)

# ---- 2. Load transaction dataset ----
transactions = pd.read_csv(
    TRANSACTION_FILE,
    sep=";", decimal=",", encoding="utf-8-sig"
)

# ---- 3. Data-quality checks BEFORE merging ----
# (duplicate IDs only mean something at their native grain)
print("Duplicate Customer_IDs in customer file:", customers["Customer_ID"].duplicated().sum())
print("Duplicate Transaction_IDs in transaction file:", transactions["Transaction_ID"].duplicated().sum())

# ---- 4. Parse the transaction timestamp properly ----
transactions["Transaction_DateTime"] = pd.to_datetime(
    transactions["Transaction_DateTime"], format="%m/%d/%Y %H:%M"
)

# ---- 5. Join datasets using Customer_ID ----
df = transactions.merge(customers, on="Customer_ID", how="left")

# ---- 6. Inspect missing values ----
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

# ---- 7. Check for true full-row duplicates ----
print("\nTotal fully duplicated rows:", df.duplicated().sum())

# ---- 8. Confirm every transaction matched a customer ----
unmatched = df["Customer_Name"].isna().sum()
print("Transactions with no matching customer record:", unmatched)

# ---- 9. Review data types ----
print("\nData types:")
print(df.dtypes)

# ---- 10. Summary statistics (numeric vs categorical, kept separate for readability) ----
print("\nNumeric summary:")
print(df.describe())
print("\nCategorical summary:")
print(df.describe(include="str"))

# ---- 11. Investigate unusual values ----
print("\nAge outliers (below 18 or above 100):")
print(df[(df["Age"] < 18) | (df["Age"] > 100)][["Customer_ID", "Age"]])

print("\nDigital Engagement Score outside 10-100:")
print(df[(df["Digital_Engagement_Score"] < 10) | (df["Digital_Engagement_Score"] > 100)]
      [["Customer_ID", "Digital_Engagement_Score"]])

print("\nNon-positive transaction amounts (Amount_NGN <= 0):")
print(df[df["Amount_NGN"] <= 0][["Transaction_ID", "Amount_NGN"]])

print("\nTenure outliers (negative, or above 100 years / 1200 months):")
print(df[(df["Tenure_Months"] < 0) | (df["Tenure_Months"] > 1200)][["Customer_ID", "Tenure_Months"]])

# ---- 12. Examine target distribution ----
if "Risk_Review_Flag" in df.columns:
    print("\nRisk Review Flag distribution:")
    print(df["Risk_Review_Flag"].value_counts(normalize=True))
else:
    print("\nRisk_Review_Flag not found in dataset yet - check transaction file schema.")

# ---- 13. Identify variables inappropriate for modelling ----
# IDs/names are identifiers, not predictive features - drop before modelling
inappropriate_vars = ["Customer_ID", "Customer_Name", "Transaction_ID"]
print("\nVariables to exclude from modelling:", inappropriate_vars)

model_ready_df = df.drop(columns=inappropriate_vars)

# ---- 14. Save outputs for Part B ----
df.to_csv(DATA_DIR / "FinTrust_Merged_Full.csv", index=False)
model_ready_df.to_csv(DATA_DIR / "FinTrust_Merged_ModelReady.csv", index=False)
print("\nSaved: FinTrust_Merged_Full.csv and FinTrust_Merged_ModelReady.csv")
ing FinTrust_Part_A_Data_Prep.py…]()

```

---feature engineering 
[FinTrust_Part_C_Feature_Engineering.py](https://github.com/user-attachments/files/32813541/FinTrust_Part_C_Feature_Engineering.py)
# ============================
# FinTrust Risk Review - Part C
# Feature Engineering
# ============================
#
# Builds predictive features on top of the Part A merged dataset, informed by
# the Part B EDA and significance testing (Transaction_Type, Amount_NGN,
# International_Transaction, Channel, Transaction_Status and Account_Type came
# back as the statistically significant drivers of Risk_Review_Flag).

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")
TARGET = "Risk_Review_Flag"

df = pd.read_csv(DATA_DIR / "FinTrust_Merged_Full.csv")
df["Transaction_DateTime"] = pd.to_datetime(df["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)

# =====================================================================
# Feature 1: Transaction_Amount_Band
# How: Amount_NGN split into quartile-based bands.
# Why: Part B/C found Amount_NGN significant (p < 0.001) with flagged
#      transactions having a higher median amount. A banded version turns
#      that continuous, skewed variable into stable, interpretable groups
#      that models handle more robustly than raw skewed values.
# =====================================================================
df["Transaction_Amount_Band"] = pd.qcut(
    df["Amount_NGN"], q=4, labels=["Low", "Medium", "High", "Very High"]
)

# =====================================================================
# Feature 2: Transaction_Hour
# How: Hour (0-23) extracted from Transaction_DateTime.
# Why: Fraud and risk-review patterns often cluster at unusual hours (e.g.
#      late night/early morning) when legitimate customer activity is low.
#      Raw hour lets a model learn any such time-of-day pattern.
# =====================================================================
df["Transaction_Hour"] = df["Transaction_DateTime"].dt.hour

# =====================================================================
# Feature 3: Is_Night_Transaction
# How: Binary flag = 1 if Transaction_Hour falls between 00:00 and 04:59.
# Why: Turns the hour feature into a specific, interpretable risk signal -
#      transactions in the small hours are disproportionately associated
#      with compromised accounts or automated fraud attempts, and a single
#      binary flag is easier for a model (and a reviewer) to act on than an
#      hour bucket alone.
# =====================================================================
df["Is_Night_Transaction"] = df["Transaction_Hour"].between(0, 4).astype(int)

# =====================================================================
# Feature 4: Transaction_Day_Of_Week
# How: Day name extracted from Transaction_DateTime (Monday ... Sunday).
# Why: Spending and withdrawal behaviour differs across the week (e.g.
#      payday-adjacent days, weekends), and unusual day-of-week activity
#      relative to a customer's norm can be a useful risk signal.
# =====================================================================
df["Transaction_Day_Of_Week"] = df["Transaction_DateTime"].dt.day_name()

# =====================================================================
# Feature 5: Is_Weekend
# How: Binary flag = 1 if Transaction_Day_Of_Week is Saturday or Sunday.
# Why: Weekend transactions happen outside normal banking-hours oversight
#      and staffing, and behave differently from weekday activity - a
#      simple binary is easier for a model to weight than 7 day categories.
# =====================================================================
df["Is_Weekend"] = df["Transaction_DateTime"].dt.dayofweek.isin([5, 6]).astype(int)

# =====================================================================
# Feature 6: Customer_Tenure_Band
# How: Tenure_Months grouped into New (<12), Established (12-59), and
#      Loyal (60+) bands.
# Why: Newer customers are statistically more likely to be involved in
#      account-takeover or first-party fraud than long-tenured customers,
#      who have an established, predictable transaction history. Banding
#      converts a continuous variable into a business-interpretable segment.
# =====================================================================
df["Customer_Tenure_Band"] = pd.cut(
    df["Tenure_Months"], bins=[-1, 11, 59, np.inf],
    labels=["New (<1yr)", "Established (1-5yr)", "Loyal (5yr+)"]
)

# =====================================================================
# Feature 7: Digital_Engagement_Category
# How: Digital_Engagement_Score split into Low (<40), Medium (40-70), and
#      High (70+) categories.
# Why: A sudden transaction from a customer with a Low digital-engagement
#      history (someone who rarely transacts digitally) is more anomalous,
#      and therefore potentially more risk-worthy, than the same
#      transaction from a High-engagement customer for whom it's routine.
# =====================================================================
df["Digital_Engagement_Category"] = pd.cut(
    df["Digital_Engagement_Score"], bins=[-1, 39, 70, 100],
    labels=["Low", "Medium", "High"]
)

# =====================================================================
# Feature 8: International_Flag
# How: International_Transaction ("Yes"/"No") encoded as a numeric 0/1.
# Why: Part B/C found this the second-strongest significant driver of
#      Risk_Review_Flag (37% flagged rate vs 19% for domestic). Encoding it
#      numerically makes it directly usable by most ML algorithms without
#      losing the strong signal already confirmed in the EDA.
# =====================================================================
df["International_Flag"] = (df["International_Transaction"] == "Yes").astype(int)

# =====================================================================
# Feature 9: Customer_Transaction_Count
# How: Total number of transactions on record for each Customer_ID.
# Why: Customers transacting far more (or less) often than typical may
#      represent unusual account usage patterns; raw activity volume is a
#      standard behavioural feature in transaction-risk models.
#      Caution: computed over the full dataset (not just prior history), so
#      this is a descriptive/behavioural feature rather than a point-in-time
#      "known at transaction time" feature - flag this if used for
#      time-ordered predictive modelling.
# =====================================================================
df["Customer_Transaction_Count"] = df.groupby("Customer_ID")["Transaction_ID"].transform("count")

# =====================================================================
# Feature 10: Customer_Avg_Amount & Amount_Deviation_From_Customer_Avg
# How: Customer_Avg_Amount = each customer's mean Amount_NGN across all
#      their transactions. Amount_Deviation_From_Customer_Avg = how many
#      standard deviations this transaction sits from that customer's own
#      average (a personalised z-score).
# Why: A ₦50,000 transaction is unremarkable for a customer who typically
#      moves that much, but highly unusual for one who typically moves
#      ₦5,000. Comparing each transaction to the customer's own baseline -
#      rather than to the population average - is a core technique in
#      transaction-monitoring/fraud systems and captures risk that a
#      population-level Amount_Band would miss.
# =====================================================================
customer_stats = df.groupby("Customer_ID")["Amount_NGN"].agg(["mean", "std"])
df["Customer_Avg_Amount"] = df["Customer_ID"].map(customer_stats["mean"])
customer_std = df["Customer_ID"].map(customer_stats["std"]).replace(0, np.nan)
df["Amount_Deviation_From_Customer_Avg"] = (
    (df["Amount_NGN"] - df["Customer_Avg_Amount"]) / customer_std
).fillna(0)

# =====================================================================
# Feature 11: Transaction_Frequency_Per_Day
# How: Customer_Transaction_Count divided by the number of days between a
#      customer's first and last transaction in the data (minimum 1 day to
#      avoid division by zero for single-transaction customers).
# Why: Two customers can have the same total transaction count but very
#      different activity intensity - one spread over three months, another
#      compressed into three days. A frequency (rate) feature captures
#      bursts of activity, a common early indicator of account compromise,
#      that a raw count alone does not.
# =====================================================================
span_days = df.groupby("Customer_ID")["Transaction_DateTime"].agg(lambda s: (s.max() - s.min()).days)
span_days = span_days.clip(lower=1)
df["Transaction_Frequency_Per_Day"] = df["Customer_Transaction_Count"] / df["Customer_ID"].map(span_days)

# =====================================================================
# Feature 12: High_Risk_Transaction_Type
# How: Binary flag = 1 if Transaction_Type is "Transfer" or "Cash
#      Withdrawal" - the two categories with the highest risk-review rates
#      (28.5% and 25.3% respectively) identified in Part B.
# Why: Condenses the strongest categorical driver (Transaction_Type,
#      Cramer's V = 0.173, the largest effect size of any variable tested)
#      into a single high-signal binary flag, which is often more useful to
#      a model than six sparsely-populated dummy categories.
# =====================================================================
df["High_Risk_Transaction_Type"] = df["Transaction_Type"].isin(
    ["Transfer", "Cash Withdrawal"]
).astype(int)

# ---------------------------------------------------------------
# Documentation table
# ---------------------------------------------------------------
feature_doc = pd.DataFrame([
    ("Transaction_Amount_Band", "Quartile bands (Low/Medium/High/Very High) of Amount_NGN.",
     "Amount_NGN was statistically significant (p<0.001); banding stabilises the skewed variable."),
    ("Transaction_Hour", "Hour (0-23) extracted from Transaction_DateTime.",
     "Lets a model learn time-of-day risk patterns."),
    ("Is_Night_Transaction", "1 if Transaction_Hour is between 00:00-04:59.",
     "Late-night activity is a classic fraud/risk indicator; a direct binary flag is model-friendly."),
    ("Transaction_Day_Of_Week", "Day name extracted from Transaction_DateTime.",
     "Captures weekly behavioural rhythms (e.g. payday effects)."),
    ("Is_Weekend", "1 if the transaction falls on Saturday or Sunday.",
     "Weekend transactions occur with less oversight/staffing and behave differently from weekday activity."),
    ("Customer_Tenure_Band", "Tenure_Months grouped into New / Established / Loyal.",
     "Newer customers carry higher account-takeover/fraud risk than long-tenured ones."),
    ("Digital_Engagement_Category", "Digital_Engagement_Score grouped into Low / Medium / High.",
     "A transaction is more anomalous for a normally low-engagement customer than a high-engagement one."),
    ("International_Flag", "International_Transaction ('Yes'/'No') encoded as 0/1.",
     "Confirmed 2nd-strongest significant driver in Part B/C (37% vs 19% flagged rate)."),
    ("Customer_Transaction_Count", "Total transactions on record per Customer_ID.",
     "Unusually high or low activity volume is a standard behavioural risk feature."),
    ("Customer_Avg_Amount / Amount_Deviation_From_Customer_Avg",
     "Each customer's mean transaction amount, and this transaction's z-score against it.",
     "Flags amounts that are unusual for THIS customer, not just unusual overall - core fraud-monitoring logic."),
    ("Transaction_Frequency_Per_Day", "Customer_Transaction_Count / active days (first-to-last transaction span).",
     "Captures bursts of activity intensity that a raw count can hide."),
    ("High_Risk_Transaction_Type", "1 if Transaction_Type is Transfer or Cash Withdrawal.",
     "Condenses the single strongest categorical driver from Part B (Cramer's V = 0.173) into one clean flag."),
], columns=["Feature", "How It Was Created", "Why It May Matter"])

pd.set_option("display.max_colwidth", None)
print(feature_doc.to_string(index=False))

feature_doc.to_csv(DATA_DIR / "FinTrust_Feature_Documentation.csv", index=False)
df.to_csv(DATA_DIR / "FinTrust_Features.csv", index=False)
print(f"\nFinal feature set shape: {df.shape}")
print("Saved: FinTrust_Feature_Documentation.csv and FinTrust_Features.csv")

#modelling 
[FinTrust_Part_E_Model_Evaluation.py](https://github.com/user-attachments/files/32813610/FinTrust_Part_E_Model_Evaluation.py)
# ============================
# FinTrust Risk Review - Part E
# Model Evaluation
# ============================
#
# Re-evaluates the Part D baseline models with the full required metric set,
# and explains why each metric is/isn't trustworthy on its own for this
# problem (Risk_Review_Flag is imbalanced: ~80% No / ~20% Yes).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
)

DATA_DIR = Path(".")
CHART_DIR = DATA_DIR / "evaluation_charts"
CHART_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ---- Rebuild the same train/test split and models as Part D ----
df = pd.read_csv(DATA_DIR / "FinTrust_Features.csv")
TARGET = "Risk_Review_Flag"
y = (df[TARGET] == "Yes").astype(int)

DROP_COLS = [
    "Transaction_ID", "Customer_ID", "Customer_Name", "Transaction_DateTime",
    "Transaction_Status", TARGET,
    "Customer_Transaction_Count", "Customer_Avg_Amount",
    "Amount_Deviation_From_Customer_Avg", "Transaction_Frequency_Per_Day",
    "Transaction_Amount_Band", "Digital_Engagement_Category", "Customer_Tenure_Band",
    "International_Transaction",
]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
NUMERIC_FEATURES = [
    "Amount_NGN", "Age", "Tenure_Months", "Digital_Engagement_Score",
    "Transaction_Hour", "Is_Weekend", "Is_Night_Transaction",
    "International_Flag", "High_Risk_Transaction_Type",
]
CATEGORICAL_FEATURES = [c for c in X.columns if c not in NUMERIC_FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]),
}

# =====================================================================
# Metric selection rationale (printed once, not per model)
# =====================================================================
print("""
Why these metrics, for this problem:

- Accuracy: reported for completeness, but treated with caution. Only ~20%
  of transactions are flagged, so a model that predicts "No" for everything
  would score ~80% accuracy while catching zero risky transactions. Accuracy
  alone would therefore be misleading here.
- Precision (Yes class): of the transactions the model flags, what share are
  genuinely risky? Matters because low precision means investigators waste
  time chasing false alarms - a real operational cost for a risk-review team.
- Recall (Yes class): of the genuinely risky transactions, what share does
  the model catch? Matters most for this use case - a missed risky
  transaction (false negative) is the costlier error for a bank, since it
  means fraud/risk goes undetected.
- F1-score: the harmonic mean of precision and recall, useful as a single
  number to compare models when both false positives and false negatives
  carry a cost and neither should be optimised in isolation.
- Confusion Matrix: shows the actual counts of true/false positives and
  negatives, which is what precision/recall/F1 are computed from - included
  so the trade-off is visible in absolute terms, not just as ratios.
- ROC-AUC: measures ranking quality (how well the model separates the two
  classes) independent of any specific decision threshold. Appropriate here
  because the 0.5 default threshold is unlikely to be the right operating
  point for a risk-review team that may want to tune sensitivity - ROC-AUC
  summarises performance across all possible thresholds.
""")

# =====================================================================
# Evaluate each model
# =====================================================================
summary_rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision (Yes): {prec:.3f}")
    print(f"Recall (Yes):    {rec:.3f}")
    print(f"F1-score (Yes):  {f1:.3f}")
    print(f"ROC-AUC:         {auc:.3f}")
    print(f"Confusion matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  -> Of {tp+fn} genuinely risky transactions, {tp} were caught and {fn} were missed.")
    print(f"  -> Of {tp+fp} transactions flagged, {tp} were genuinely risky ({prec:.1%} precision).")

    summary_rows.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1_Score": f1, "ROC_AUC": auc, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    })

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
pd.set_option("display.width", 120)
print("\n\nMetric comparison table:")
print(summary_df.round(3).to_string(index=False))
summary_df.to_csv(DATA_DIR / "FinTrust_Evaluation_Metrics.csv", index=False)

# =====================================================================
# Metric bar chart (Accuracy / Precision / Recall / F1 / ROC-AUC side by side)
# =====================================================================
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(metrics_to_plot))
width = 0.35
for i, (name, row) in enumerate(zip(summary_df["Model"], summary_df.itertuples())):
    vals = [getattr(row, m) for m in metrics_to_plot]
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Baseline model metric comparison (Yes / flagged class)")
ax.legend()
ax.axhline(0.196, color="gray", linestyle=":", linewidth=1, label="Base flagged rate")
fig.tight_layout()
fig.savefig(CHART_DIR / "metric_comparison_bar.png", dpi=150)
plt.close(fig)

print(f"\nAll charts saved to: {CHART_DIR.resolve()}")
print("Saved: FinTrust_Evaluation_Metrics.csv")

#visuals
#<img width="675" height="675" alt="confusion_matrix_random_forest" src="https://github.com/user-attachments/assets/1cf34814-ff69-427e-8bdd-23bffaad4121" />
#<img width="1200" height="750" alt="metric_comparison_bar" src="https://github.com/user-attachments/assets/fa8d8cc1-fd74-4ee4-8e0e-14c2d855d1d3" />
#<img width="900" height="825" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/1435b59f-dfdb-4e66-a335-83a6fe262e95" />
#<img width="900" height="825" alt="roc_curve_comparison" src="https://github.com/user-attachments/assets/b82b4451-5454-45ea-9b3c-295121877ef3" />
#<img width="675" height="675" alt="confusion_matrix_logistic_regression (1)" src="https://github.com/user-attachments/assets/ba8b19b5-033d-4151-b606-0f96ead27b2e" />
#<img width="1050" height="900" alt="logistic_regression_coefficients" src="https://github.com/user-attachments/assets/67e9ece8-82e9-4965-84d2-663430e143cf" />







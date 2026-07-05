import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / "dataset" / "Charlotin-hallucination_cases.csv"
IMAGES_DIR = BASE_DIR / ".." / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Data Loading
# The raw case data is loaded from CSV and inspected to understand its shape,
# column types, and basic statistics before any cleaning is done.

df = pd.read_csv(DATA_PATH)  # dataset
print("Shape:", df.shape)
print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nSummary Statistics")
print(df.describe(include="all").T)

# 2. Data Cleaning
# In this step, missing values, inconsistent text formatting, and messy
# multi-valued fields are handled to improve the quality of the dataset.
# Dates are converted to a proper datetime type, and the target column
# (Professional Sanction) is converted to a 0/1 label.

missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
print("\nMissing Values")
print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}))

df_clean = df.copy()

# Normalize text casing on key categorical columns
for col in ["Outcome", "AI Tool", "Party(ies)", "Legal Field Primary", "Legal Field Secondary",
            "Professional Sanction", "Alleged"]:
    df_clean[col] = df_clean[col].astype(str).str.strip()
    df_clean.loc[df_clean[col].isin(["nan", "None", ""]), col] = np.nan

df_clean["Outcome_norm"] = df_clean["Outcome"].str.lower()
df_clean["AI_Tool_norm"] = df_clean["AI Tool"].str.lower()

# Parse the Date column and pull out the year
df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
df_clean["Year"] = df_clean["Date"].dt.year

# Simplify multi-valued Party(ies) to the primary (first-listed) party
df_clean["Party_primary"] = df_clean["Party(ies)"].str.split(";").str[0].str.strip()

# Simplify AI tool into a small set of top categories, rest -> 'Other'
top_tools = df_clean["AI_Tool_norm"].value_counts().nlargest(6).index
df_clean["AI_Tool_grouped"] = np.where(
    df_clean["AI_Tool_norm"].isin(top_tools), df_clean["AI_Tool_norm"], "other/unidentified"
)

# Convert Yes/No columns to 0/1
df_clean["Professional_Sanction_bin"] = df_clean["Professional Sanction"].map({"Yes": 1, "No": 0})
df_clean["Alleged_bin"] = df_clean["Alleged"].map({"Yes": 1, "No": 0})

# Fill remaining categorical missingness with an explicit 'Unknown' label
for col in ["Legal Field Primary", "Party_primary"]:
    df_clean[col] = df_clean[col].fillna("Unknown")

# Drop rows with no target label (can't train/evaluate on them)
before = len(df_clean)
df_clean = df_clean.dropna(subset=["Professional_Sanction_bin"])
print(f"\nDropped {before - len(df_clean)} rows with missing target label.")
print("Remaining rows:", len(df_clean))

# 3. Feature Engineering (text-based counts)
# New numeric features are created from the free-text Hallucination Items
# column: how many hallucination items a filing contains, and how many of
# those were fabricated citations.

def count_items(text):
    if pd.isna(text):
        return 0
    return len(str(text).split("||"))


def count_type(text, keyword):
    if pd.isna(text):
        return 0
    return str(text).lower().count(keyword.lower())


df_clean["n_hallucination_items"] = df_clean["Hallucination Items"].apply(count_items)
df_clean["n_fabricated"] = df_clean["Hallucination Items"].apply(lambda t: count_type(t, "Fabricated"))
df_clean["has_monetary_penalty"] = df_clean["Monetary Penalty"].notna().astype(int)

numeric_cols = ["n_hallucination_items", "n_fabricated"]
print("\nNumeric Feature Summary")
print(df_clean[numeric_cols].describe())

# 4. Data Visualization
# One meaningful chart is created with each required library: a Matplotlib
# bar chart, a Seaborn countplot, and an interactive Plotly bar chart.

# Matplotlib: cases by year
yearly_counts = df_clean["Year"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(yearly_counts.index.astype(int).astype(str), yearly_counts.values, color="#4C72B0")
plt.title("AI Hallucination Cases by Year")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "fig_cases_by_year.png", dpi=110)
plt.show()

# Seaborn: outcome distribution, split by whether a sanction followed
top_outcomes = df_clean["Outcome_norm"].value_counts().nlargest(8).index
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df_clean[df_clean["Outcome_norm"].isin(top_outcomes)],
    y="Outcome_norm", order=top_outcomes, hue="Professional_Sanction_bin", palette="viridis"
)
plt.title("Top Outcomes, Split by Whether a Professional Sanction Followed")
plt.xlabel("Number of Cases")
plt.ylabel("Outcome")
plt.legend(title="Professional Sanction", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(IMAGES_DIR / "fig_outcome_by_sanction.png", dpi=110)
plt.show()

# Seaborn: which AI tools appear most often
tool_counts = df_clean["AI_Tool_grouped"].value_counts().nlargest(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=tool_counts.values, y=tool_counts.index)
plt.title("Most Frequently Used AI Tools")
plt.xlabel("Number of Cases")
plt.ylabel("AI Tool")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "fig_ai_tool_distribution.png", dpi=110)
plt.show()

# Seaborn: which legal fields see the most hallucination cases
top_fields = df_clean["Legal Field Primary"].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_fields.values, y=top_fields.index)
plt.title("Top Legal Fields with AI Hallucination Cases")
plt.xlabel("Number of Cases")
plt.ylabel("Legal Field")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "fig_legal_field_distribution.png", dpi=110)
plt.show()

# Plotly: top jurisdictions (interactive bar chart)
top_states = df_clean["State(s)"].value_counts().nlargest(12).reset_index()
top_states.columns = ["State", "Cases"]
fig = px.bar(
    top_states, x="Cases", y="State", orientation="h",
    title="Top 12 Jurisdictions by Number of AI-Hallucination Cases",
    color="Cases", color_continuous_scale="Blues"
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html(IMAGES_DIR / "fig_top_jurisdictions.html")
fig.show()

# 5. Feature Engineering for Modeling
# New features are created and categorical variables are encoded into
# numerical values (one-hot encoding) so that machine learning models can
# process them effectively.

feature_cols_numeric = ["Year", "n_hallucination_items", "n_fabricated",
                         "has_monetary_penalty", "Alleged_bin"]
feature_cols_categorical = ["Party_primary", "AI_Tool_grouped", "Legal Field Primary"]

model_df = df_clean.copy()
model_df["Year"] = model_df["Year"].fillna(model_df["Year"].median())
model_df[feature_cols_numeric] = model_df[feature_cols_numeric].fillna(0)

X = pd.get_dummies(model_df[feature_cols_numeric + feature_cols_categorical],
                    columns=feature_cols_categorical, drop_first=True)
y = model_df["Professional_Sanction_bin"].astype(int)

print("\nFeature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts(normalize=True).round(3))

# 6. Train/Test Split & Feature Scaling
# The dataset is split into training and testing sets. Logistic Regression
# needs scaled inputs, so a StandardScaler is fit on the training data only.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# 7. Model Training
# A Logistic Regression model and a Random Forest model are trained on the
# training data.

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling

print("\nModels trained.")

# 8. Model Evaluation
# The trained models are evaluated using Accuracy, Precision, Recall,
# F1-score, and a Confusion Matrix to assess their predictive performance.


def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_true, y_pred):.3f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["No Sanction", "Sanctioned"]))


y_pred_lr = log_reg.predict(X_test_scaled)
evaluate("Logistic Regression", y_test, y_pred_lr)

y_pred_rf = rf.predict(X_test)
evaluate("Random Forest", y_test, y_pred_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Sanction", "Sanctioned"],
                yticklabels=["No Sanction", "Sanctioned"])
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "fig_confusion_matrices.png", dpi=110)
plt.show()

print("\nProject completed successfully.")
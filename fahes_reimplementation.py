import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. Parameter Settings
# ============================================================
SEED = 42
N = 1000

# Injection rates for different types of DMVs
SYNTAX_RATE = 0.05       # 5%
NUMERICAL_RATE = 0.05    # 5%
RANDOMDMVD_RATE = 0.10   # 10%

rng = np.random.default_rng(SEED)


# ============================================================
# 2. Generate Normal Data
# ============================================================
first_names = [
    "Alice", "Bob", "Charlie", "David", "Emma",
    "Frank", "Grace", "Helen", "Ivan", "Julia",
    "Kevin", "Laura", "Mike", "Nancy", "Oscar",
    "Peter", "Rachel", "Steve", "Tom", "Wendy"
]

last_names = [
    "Smith", "Brown", "Wilson", "Taylor", "Miller",
    "Davis", "Clark", "Lewis", "Walker", "Hall"
]

names = [
    f"{rng.choice(first_names)} {rng.choice(last_names)}"
    for _ in range(N)
]

# Age: 18-70 years
ages = rng.integers(18, 71, size=N)

# Salary: Use a log-normal distribution to simulate
# a right-skewed salary distribution.
# Most salaries range from tens of thousands to over one hundred thousand.
salaries = rng.lognormal(
    mean=np.log(60000),
    sigma=0.25,
    size=N
).round(2)

df = pd.DataFrame({
    "Name": names,
    "Age": ages,
    "Salary": salaries
})

# Labels: Each row receives at most one type of DMV
df["DMV_Type"] = "Normal"


# ============================================================
# 3. Inject Three Types of Disguised Missing Values
# ============================================================
all_indices = rng.permutation(N)

n_syntax = int(N * SYNTAX_RATE)
n_numerical = int(N * NUMERICAL_RATE)
n_randomdmvd = int(N * RANDOMDMVD_RATE)

syntax_idx = all_indices[:n_syntax]
numerical_idx = all_indices[n_syntax:n_syntax + n_numerical]
random_idx = all_indices[
    n_syntax + n_numerical:
    n_syntax + n_numerical + n_randomdmvd
]

# 3.1 Syntactic DMVs
# Half are repeated characters; the others are invalid strings in a numeric column.
for i, idx in enumerate(syntax_idx):
    if i % 2 == 0:
        df.loc[idx, "Name"] = "5555555"
    else:
        df.loc[idx, "Salary"] = "unknown"

    df.loc[idx, "DMV_Type"] = "Syntax"

# 3.2 Numerical DMVs
# Create outliers using implausible ages and salaries.
for i, idx in enumerate(numerical_idx):
    if i % 2 == 0:
        df.loc[idx, "Age"] = 999
    else:
        df.loc[idx, "Salary"] = 0

    df.loc[idx, "DMV_Type"] = "Numerical"

# 3.3 RandomDMVD: High-Frequency Legitimate-Looking False Values
# 5000 is a valid number, but it is unusually low
# relative to the salary distribution in this experiment.
df.loc[random_idx, "Salary"] = 5000
df.loc[random_idx, "DMV_Type"] = "RandomDMVD"

# Save a labeled dataset for auditing and reproducibility.
df.to_csv("synthetic_dmv_dataset.csv", index=False)

print("Dataset generated successfully")
print(f"Total rows: {N}")
print(df["DMV_Type"].value_counts())
print()


# ============================================================
# 4. Detector 1: Syntax Detector
# ============================================================
def syntax_detector(data):
    """
    Detect:
    1. A single character repeated multiple times,
       such as 5555555 or aaaaaa.
    2. Invalid non-numeric values in Age and Salary.

    Returns a Boolean Series: True indicates a suspected DMV.
    """
    result = pd.Series(False, index=data.index)

    # Detect consecutive repetitions of a single character.
    repeated_pattern = re.compile(r"^(.)\1{4,}$")

    for col in ["Name", "Age", "Salary"]:
        text_values = data[col].astype("string").fillna("")
        repeated = text_values.str.match(repeated_pattern)
        result |= repeated

    # Detect invalid data types in numeric columns.
    for col in ["Age", "Salary"]:
        numeric_values = pd.to_numeric(data[col], errors="coerce")
        invalid = numeric_values.isna() & data[col].notna()
        result |= invalid

    return result


# ============================================================
# 5. Detector 2: IsolationForest Numerical Detector
# ============================================================
def numerical_detector(data, contamination=0.06):
    """
    Convert Age and Salary to numeric values and apply IsolationForest.
    Non-convertible strings are replaced with the column median
    to prevent invalid values from causing the model to fail.
    """
    features = data[["Age", "Salary"]].apply(
        pd.to_numeric, errors="coerce"
    )

    features = features.replace([np.inf, -np.inf], np.nan)

    features = features.fillna(features.median())

    # Scale features to normalize their numerical ranges.
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=SEED
    )

    predictions = model.fit_predict(X)

    # IsolationForest: -1 indicates an outlier.
    return pd.Series(predictions == -1, index=data.index)


# ============================================================
# 6. Detector 3: Approximate RandomDMVD
# ============================================================
def randomdmvd_detector(data):
    """
    Approximate the frequency and distribution comparison
    principles of RandomDMVD:

    1. Count the occurrences of each Salary value.
    2. Identify values with unusually high frequencies.
    3. Require candidate values to also fall outside
       the expected range of the normal salary distribution.

    Note: This is an approximation using the empirical distribution
    of the entire dataset, not the complete RandomDMVD algorithm
    described in the original paper.
    """
    result = pd.Series(False, index=data.index)

    salary = pd.to_numeric(data["Salary"], errors="coerce")
    valid = salary.dropna()

    if len(valid) == 0:
        return result

    # Count the exact frequency of each numerical value.
    counts = valid.value_counts()

    # Frequency threshold: at least 8 occurrences,
    # or 3% of the valid observations.
    frequency_threshold = max(8, int(np.ceil(len(valid) * 0.03)))

    # Use the lower and upper quantiles as distribution references.
    lower_bound = valid.quantile(0.01)
    upper_bound = valid.quantile(0.99)

    candidates = counts[counts >= frequency_threshold].index

    for value in candidates:
        # High frequency alone is insufficient.
        # The value must also be outside the expected distribution range.
        outside_distribution = (
            value < lower_bound or value > upper_bound
        )

        if outside_distribution:
            result |= salary.eq(value).fillna(False)

    return result


# ============================================================
# 7. Run All Three Detectors
# ============================================================
syntax_pred = syntax_detector(df)
numerical_pred = numerical_detector(df)
randomdmvd_pred = randomdmvd_detector(df)

predictions = {
    "Syntax Detector": syntax_pred,
    "IsolationForest": numerical_pred,
    "RandomDMVD": randomdmvd_pred
}

# Evaluate each detector only against its target DMV type.
# For example, Syntax Detector treats Syntax as the positive class.
# Other DMV types are considered negative for this evaluation.
target_types = {
    "Syntax Detector": "Syntax",
    "IsolationForest": "Numerical",
    "RandomDMVD": "RandomDMVD"
}

results = []

for detector_name, pred in predictions.items():
    target_type = target_types[detector_name]

    y_true = (df["DMV_Type"] == target_type).astype(int)
    y_pred = pred.astype(int)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    results.append({
        "Detector": detector_name,
        "Target DMV": target_type,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

results_df = pd.DataFrame(results)

print("========== Detection Results ==========")
print(results_df.to_string(
    index=False,
    formatters={
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format
    }
))
print()

results_df.to_csv("dmv_detection_results.csv", index=False)


# ============================================================
# 8. Plot the Precision / Recall Comparison
# ============================================================
labels = results_df["Detector"].tolist()
precision_values = results_df["Precision"].to_numpy()
recall_values = results_df["Recall"].to_numpy()

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    precision_values,
    width,
    label="Precision",
    color="#A78BFA"
)

bars2 = ax.bar(
    x + width / 2,
    recall_values,
    width,
    label="Recall",
    color="#34D399"
)

ax.set_xlabel("Detector")
ax.set_ylabel("Score")
ax.set_title("FAHES-inspired DMV Detection: Precision vs Recall")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)

plt.tight_layout()
plt.savefig("dmv_precision_recall.png", dpi=200)
plt.show()


# ============================================================
# 9. Optional: Inspect the Rows Flagged by Each Detector
# ============================================================
for detector_name, pred in predictions.items():
    print(f"\nFirst 10 rows flagged by {detector_name}:")

    cols = ["Name", "Age", "Salary", "DMV_Type"]
    print(df.loc[pred, cols].head(10).to_string())

print("\nOutput files:")
print("1. synthetic_dmv_dataset.csv")
print("2. dmv_detection_results.csv")
print("3. dmv_precision_recall.png")

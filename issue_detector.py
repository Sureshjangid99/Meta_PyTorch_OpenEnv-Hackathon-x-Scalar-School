"""
Issue detector: scans a DataFrame and returns a list of human-readable issue strings.
Used to populate Observation.issues at each step.
"""

import pandas as pd
import re
from typing import List


def detect_issues(df: pd.DataFrame) -> List[str]:
    """Return a list of detected data quality issues in the given DataFrame."""
    issues = []

    # 1. Missing values
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            issues.append(f"Missing values in '{col}': {count} nulls")

    # 2. Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate rows detected: {dup_count} duplicates")

    # 3. Type issues — columns that look numeric but are stored as strings
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            sample = df[col].dropna().head(20)
            numeric_count = 0
            for val in sample:
                try:
                    float(str(val).replace(",", ""))
                    numeric_count += 1
                except ValueError:
                    pass
            # If >60% are numeric-looking but dtype is object, flag it
            if len(sample) > 0 and numeric_count / len(sample) > 0.6:
                issues.append(f"Column '{col}' appears numeric but stored as string/object")

    # 4. All-null columns
    for col in df.columns:
        if df[col].isnull().all():
            issues.append(f"Column '{col}' is entirely null — candidate for removal")

    # 5. Constant columns (single unique non-null value)
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            issues.append(f"Column '{col}' has only one unique value — likely useless")

    # 6. Outlier detection for numeric columns
    for col in df.select_dtypes(include="number").columns:
        col_data = df[col].dropna()
        if len(col_data) < 3:
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        outlier_count = ((col_data < lower) | (col_data > upper)).sum()
        if outlier_count > 0:
            issues.append(f"Column '{col}' has {outlier_count} extreme outlier(s)")

    # 7. Negative values in likely-positive columns
    for col in df.select_dtypes(include="number").columns:
        if any(kw in col.lower() for kw in ["salary", "price", "quantity", "spend", "amount", "cost"]):
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"Column '{col}' has {neg_count} negative value(s) — likely invalid")

    # 8. Non-standard date formats
    for col in df.columns:
        if "date" in col.lower() and df[col].dtype == object:
            sample = df[col].dropna().head(10)
            formats_seen = set()
            for val in sample:
                val_str = str(val)
                if re.match(r"\d{4}-\d{2}-\d{2}", val_str):
                    formats_seen.add("ISO")
                elif re.match(r"\d{2}/\d{2}/\d{4}", val_str):
                    formats_seen.add("DD/MM/YYYY")
                elif re.match(r"\d{2}-\d{2}-\d{4}", val_str):
                    formats_seen.add("MM-DD-YYYY")
                elif re.match(r"\d{2}/\d{4}", val_str):
                    formats_seen.add("other")
            if len(formats_seen) > 1:
                issues.append(f"Column '{col}' has mixed date formats: {formats_seen}")

    # 9. Invalid emails
    for col in df.columns:
        if "email" in col.lower() and df[col].dtype == object:
            email_re = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
            bad = df[col].dropna().apply(lambda x: not email_re.match(str(x))).sum()
            if bad > 0:
                issues.append(f"Column '{col}' has {bad} invalid email address(es)")

    return issues

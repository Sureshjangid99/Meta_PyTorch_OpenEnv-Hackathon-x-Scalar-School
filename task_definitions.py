"""
Task definitions with programmatic graders for all 3 tasks.
Each grader returns a score in [0.0, 1.0].
"""

from __future__ import annotations
import pandas as pd
import re
from typing import Tuple, Dict, Any


# ──────────────────────────────────────────────
# TASK EASY: Fill Missing Values
# ──────────────────────────────────────────────

EASY_TASK_ID = "task_easy"
EASY_DESCRIPTION = (
    "You have a sales CSV with missing values across several columns. "
    "Your goal is to fill ALL missing values with appropriate strategies: "
    "use 'Unknown' for text columns and the column median/mean for numeric columns. "
    "Once all missing values are filled, call 'submit'."
)


def grade_easy(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Score 0.0–1.0 based on how completely missing values are filled.
    Full score only if zero nulls remain.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0, "Empty dataframe."

    null_count = df.isnull().sum().sum()
    filled_fraction = 1.0 - (null_count / total_cells)

    # Bonus: check that numeric columns are actually numeric
    type_bonus = 0.0
    numeric_cols = ["quantity", "unit_price"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors="raise")
                type_bonus += 0.05
            except Exception:
                pass

    score = min(1.0, filled_fraction * 0.9 + type_bonus)

    if null_count == 0:
        reason = f"All missing values filled. Score: {score:.2f}"
    else:
        reason = f"{null_count} null(s) remain across {df.shape[1]} columns. Score: {score:.2f}"

    return round(score, 3), reason


# ──────────────────────────────────────────────
# TASK MEDIUM: Deduplicate + Fix Types
# ──────────────────────────────────────────────

MEDIUM_TASK_ID = "task_medium"
MEDIUM_DESCRIPTION = (
    "You have a customer CSV with two problems: (1) duplicate rows and "
    "(2) columns stored as strings that should be numeric or boolean. "
    "Remove all duplicate rows, then fix data types: 'age' → int, "
    "'spend_total' → float, 'is_premium' → bool (True/False). "
    "Rows with unparseable values in those columns should be dropped. "
    "Submit when complete."
)


def grade_medium(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Score based on:
    - 40%: duplicate removal
    - 30%: correct age type
    - 30%: correct spend_total type
    """
    score = 0.0
    details = []

    # Deduplication (40%)
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        score += 0.40
        details.append("✓ No duplicates (0.40)")
    else:
        partial = max(0.0, 0.40 - dup_count * 0.05)
        score += partial
        details.append(f"✗ {dup_count} duplicate(s) remain (partial: {partial:.2f})")

    # Age is numeric (30%)
    if "age" in df.columns:
        try:
            numeric_ages = pd.to_numeric(df["age"], errors="coerce")
            valid_pct = numeric_ages.notna().mean()
            age_score = 0.30 * valid_pct
            score += age_score
            details.append(f"{'✓' if valid_pct == 1.0 else '~'} age numeric ({valid_pct*100:.0f}% valid) (+{age_score:.2f})")
        except Exception:
            details.append("✗ age column unreadable (+0.00)")

    # spend_total is numeric (30%)
    if "spend_total" in df.columns:
        try:
            numeric_spend = pd.to_numeric(df["spend_total"], errors="coerce")
            valid_pct = numeric_spend.notna().mean()
            spend_score = 0.30 * valid_pct
            score += spend_score
            details.append(f"{'✓' if valid_pct == 1.0 else '~'} spend_total numeric ({valid_pct*100:.0f}% valid) (+{spend_score:.2f})")
        except Exception:
            details.append("✗ spend_total unreadable (+0.00)")

    return round(min(1.0, score), 3), " | ".join(details)


# ──────────────────────────────────────────────
# TASK HARD: Full Pipeline Clean
# ──────────────────────────────────────────────

HARD_TASK_ID = "task_hard"
HARD_DESCRIPTION = (
    "You have a messy employee dataset requiring a FULL cleaning pipeline. "
    "You must: (1) drop columns that are entirely null or constant, "
    "(2) remove duplicate rows, "
    "(3) fill missing values (use 'Unknown' for strings, median for numerics), "
    "(4) remove rows with outlier/invalid salaries (negative or >500000), "
    "(5) fix invalid emails (set to null or drop), "
    "(6) standardize phone numbers to format 555-XXXX where possible. "
    "All six checks must pass for a perfect score. Submit when done."
)


def grade_hard(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Hard grader: 6 checks each worth ~1/6 of the score.
    """
    checks: Dict[str, float] = {}

    # 1. Useless columns dropped (useless_col, constant_col)
    useless = [c for c in ["useless_col", "constant_col"] if c in df.columns]
    if len(useless) == 0:
        checks["dropped_useless_cols"] = 1.0
    elif len(useless) == 1:
        checks["dropped_useless_cols"] = 0.5
    else:
        checks["dropped_useless_cols"] = 0.0

    # 2. No duplicates
    dup_count = df.duplicated().sum()
    checks["no_duplicates"] = max(0.0, 1.0 - dup_count * 0.2)

    # 3. Missing values filled
    null_count = df.isnull().sum().sum()
    total = df.shape[0] * df.shape[1]
    checks["nulls_filled"] = round(1.0 - (null_count / max(total, 1)), 3)

    # 4. Bad salaries removed
    if "salary" in df.columns:
        try:
            sal = pd.to_numeric(df["salary"], errors="coerce").dropna()
            bad = ((sal < 0) | (sal > 500000)).sum()
            checks["salary_outliers"] = max(0.0, 1.0 - bad * 0.3)
        except Exception:
            checks["salary_outliers"] = 0.0
    else:
        checks["salary_outliers"] = 1.0  # column removed = issue resolved

    # 5. Invalid emails fixed (no "MISSING", "N/A", "no_email", "not_valid" literals)
    if "email" in df.columns:
        bad_literals = ["MISSING", "N/A", "no_email", "not_valid", "bad-email"]
        bad_count = df["email"].dropna().apply(
            lambda x: any(b.lower() in str(x).lower() for b in bad_literals)
        ).sum()
        checks["emails_fixed"] = max(0.0, 1.0 - bad_count * 0.15)
    else:
        checks["emails_fixed"] = 1.0

    # 6. Phone standardization (at least 50% match XXX-XXXX pattern)
    if "phone" in df.columns:
        phone_re = re.compile(r"^\d{3}-\d{4}$")
        valid = df["phone"].dropna().apply(lambda x: bool(phone_re.match(str(x)))).mean()
        checks["phones_standardized"] = round(valid, 3)
    else:
        checks["phones_standardized"] = 1.0

    score = sum(checks.values()) / len(checks)
    details = " | ".join(f"{k}={v:.2f}" for k, v in checks.items())
    return round(min(1.0, score), 3), details


# ──────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    EASY_TASK_ID: {
        "id": EASY_TASK_ID,
        "description": EASY_DESCRIPTION,
        "difficulty": "easy",
        "grader": grade_easy,
    },
    MEDIUM_TASK_ID: {
        "id": MEDIUM_TASK_ID,
        "description": MEDIUM_DESCRIPTION,
        "difficulty": "medium",
        "grader": grade_medium,
    },
    HARD_TASK_ID: {
        "id": HARD_TASK_ID,
        "description": HARD_DESCRIPTION,
        "difficulty": "hard",
        "grader": grade_hard,
    },
}

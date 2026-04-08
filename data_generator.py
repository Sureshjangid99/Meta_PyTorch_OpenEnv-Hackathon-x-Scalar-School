"""
Data generators for all three tasks.
Each returns a messy DataFrame that the agent must clean.
"""

import pandas as pd
import numpy as np
import random


def make_easy_dataset() -> pd.DataFrame:
    """
    Easy task: Sales data with missing values only.
    Agent must fill all NaN values appropriately.
    """
    random.seed(42)
    np.random.seed(42)

    data = {
        "order_id": [f"ORD-{i:04d}" for i in range(1, 31)],
        "customer_name": [
            "Alice Smith", "Bob Jones", None, "Diana Prince", "Eve White",
            None, "George Hill", "Hannah Brown", "Ivan Green", None,
            "Karen Black", "Leo King", "Mia Lee", None, "Oscar Wild",
            "Paula Dean", None, "Rachel Ray", "Sam Gold", "Tina Turner",
            "Uma Fox", None, "Victor Hugo", "Wendy Bell", "Xena Cruz",
            None, "Yara Flint", "Zack Myers", "Amy Adams", "Brian May"
        ],
        "product": [
            "Laptop", "Mouse", "Keyboard", None, "Monitor",
            "Laptop", None, "Mouse", "Keyboard", "Monitor",
            None, "Laptop", "Mouse", "Keyboard", None,
            "Monitor", "Laptop", None, "Mouse", "Keyboard",
            "Monitor", "Laptop", None, "Mouse", "Keyboard",
            "Monitor", None, "Laptop", "Mouse", None
        ],
        "quantity": [
            2, 5, None, 1, 3, 2, 4, None, 1, 2,
            3, None, 2, 1, 4, 3, None, 2, 1, 3,
            2, 4, None, 1, 3, 2, 1, None, 2, 4
        ],
        "unit_price": [
            999.99, 25.50, 75.00, None, 349.99, 999.99, 25.50, 75.00, None, 349.99,
            999.99, 25.50, None, 75.00, 349.99, 999.99, 25.50, 75.00, None, 349.99,
            999.99, None, 25.50, 75.00, 349.99, 999.99, 25.50, None, 75.00, 349.99
        ],
        "region": [
            "North", "South", "East", "West", None,
            "North", "South", None, "East", "West",
            None, "North", "South", "East", "West",
            None, "North", "South", "East", None,
            "West", "North", None, "South", "East",
            "West", "North", None, "South", "East"
        ]
    }
    return pd.DataFrame(data)


def make_medium_dataset() -> pd.DataFrame:
    """
    Medium task: Customer data with duplicates AND type errors.
    Agent must remove duplicates and fix data types.
    """
    random.seed(7)
    np.random.seed(7)

    base_data = {
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                         111, 112, 113, 114, 115],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve",
                  "Frank", "Grace", "Hank", "Ivy", "Jack",
                  "Karen", "Leo", "Mia", "Ned", "Olive"],
        "age": ["28", "35", "not_a_number", "42", "29",
                 "31", "45", "38", "27", "fifty",
                 "33", "40", "26", "39", "44"],
        "signup_date": ["2021-01-15", "2021-03-22", "2021-02-10", "2020-11-05", "2022-06-17",
                        "2021-08-30", "2020-07-14", "2022-01-01", "2021-09-09", "2020-12-31",
                        "2021-04-04", "2022-03-15", "2021-07-07", "2020-10-10", "2022-05-20"],
        "spend_total": ["1200.50", "850", "NOT_AVAILABLE", "3300.00", "450.75",
                        "1100.00", "two thousand", "670.25", "2200.00", "390.00",
                        "875.50", "1560.00", "490.00", "3100.75", "620.00"],
        "is_premium": ["True", "False", "True", "1", "0",
                       "yes", "no", "True", "False", "True",
                       "1", "0", "True", "False", "yes"]
    }

    df = pd.DataFrame(base_data)

    # Inject duplicates (5 exact duplicate rows from existing rows)
    dup_indices = random.sample(range(len(df)), 5)
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


def make_hard_dataset() -> pd.DataFrame:
    """
    Hard task: Employee dataset with ALL types of issues:
    - Missing values
    - Duplicate rows
    - Wrong data types
    - Non-standard formats (phone, email, date)
    - Outlier/bad values
    - Useless columns (all NaN, constant value)
    """
    random.seed(99)
    np.random.seed(99)

    data = {
        "emp_id": [f"E{i:03d}" for i in range(1, 26)],
        "full_name": [
            "John Doe", "Jane Smith", None, "Alice Johnson", "Bob Brown",
            "Charlie Davis", None, "Eva Martinez", "Frank Wilson", "Grace Lee",
            "Henry Taylor", None, "Isabella Moore", "James Anderson", "Karen Thomas",
            "Liam Jackson", "Mia White", None, "Noah Harris", "Olivia Martin",
            "Paul Thompson", "Quinn Garcia", None, "Rachel Robinson", "Steve Clark"
        ],
        "email": [
            "john.doe@company.com", "jane.smith@company.com", "no_email", "alice@company.com",
            "bob.brown@company.com", "charlie@company.com", "MISSING", "eva@company.com",
            "frank.wilson@company.com", "grace@company.com", "henry@company.com",
            "bad-email", "isabella@company.com", "james@company.com", "karen@company.com",
            "liam@company.com", "mia.white@company.com", "N/A", "noah@company.com",
            "olivia@company.com", "paul@company.com", "quinn@company.com",
            "not_valid", "rachel@company.com", "steve@company.com"
        ],
        "phone": [
            "555-1234", "5552345", "(555) 3456", "555.4567", "5554567890",
            "555-5678", None, "555-6789", "555 7890", "(555)8901",
            "555-9012", "555-0123", None, "555-1234", "555-2345",
            "555-3456", None, "555-4567", "555-5678", "555-6789",
            "5557890", "555-8901", "555-9012", None, "555-0123"
        ],
        "department": [
            "Engineering", "HR", "Engineering", "Marketing", "HR",
            "Finance", "Engineering", None, "Marketing", "Finance",
            "HR", "Engineering", "Marketing", None, "Finance",
            "Engineering", "HR", "Marketing", "Finance", None,
            "Engineering", "HR", "Marketing", "Finance", "Engineering"
        ],
        "salary": [
            75000, 55000, 82000, 61000, 57000,
            90000, 78000, 63000, 67000, 95000,
            52000, 84000, 71000, 58000, 88000,
            -5000,  # outlier: negative salary
            54000, 66000, 91000, 73000,
            9999999,  # outlier: unrealistic salary
            85000, 69000, 77000, 60000
        ],
        "hire_date": [
            "2019-03-15", "15/04/2020", "2018-07-22", "2021-01-10", "2020-08-30",
            "2017-11-05", "2022-03-01", "2019-06-15", "03-12-2021", "2018-09-20",
            "2020-02-14", "2021-07-07", "2019-12-01", "2022-04-18", "2017-05-30",
            "2020-10-10", "2021-03-25", "04/15/2019", "2018-11-11", "2022-01-20",
            "2019-08-08", "2020-07-04", "2021-09-15", "05-20-2018", "2022-06-01"
        ],
        "performance_score": [
            4.2, 3.8, 4.7, 3.1, 4.0,
            None, 4.5, 3.9, None, 4.8,
            3.5, 4.1, None, 3.7, 4.6,
            4.0, None, 3.6, 4.9, 3.4,
            None, 4.3, 3.8, 4.5, None
        ],
        "useless_col": [None] * 25,  # all NaN — should be dropped
        "constant_col": ["ACTIVE"] * 25,  # constant — no info, should be dropped
    }

    df = pd.DataFrame(data)

    # Add 3 duplicate rows
    dup_rows = df.iloc[[0, 5, 12]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)

    return df

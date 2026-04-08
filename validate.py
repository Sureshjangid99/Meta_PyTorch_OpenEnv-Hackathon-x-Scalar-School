"""
validate.py — Pre-submission OpenEnv validation script
Run this before submitting to check all requirements pass.

Usage:
    python validate.py
    python validate.py --server http://localhost:7860  # validate against live server
"""

import sys
import os
import json
import argparse

# ── Colors ──────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = []
failed = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  {GREEN}✓{RESET} {name}" + (f" — {detail}" if detail else ""))
        passed.append(name)
    else:
        print(f"  {RED}✗{RESET} {name}" + (f" — {detail}" if detail else ""))
        failed.append(name)


def section(title: str):
    print(f"\n{BOLD}── {title} ──{RESET}")


# ──────────────────────────────────────────────────────────────
section("1. File Structure")
# ──────────────────────────────────────────────────────────────

required_files = [
    "openenv.yaml",
    "app.py",
    "inference.py",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "env/__init__.py",
    "env/environment.py",
    "env/models.py",
    "env/data_generator.py",
    "env/issue_detector.py",
    "tasks/__init__.py",
    "tasks/task_definitions.py",
]

for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))

# ──────────────────────────────────────────────────────────────
section("2. openenv.yaml Compliance")
# ──────────────────────────────────────────────────────────────

try:
    import yaml
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)
    check("openenv.yaml parseable", True)
    check("Has 'name' field", "name" in spec)
    check("Has 'version' field", "version" in spec)
    check("Has 'observation_space'", "observation_space" in spec)
    check("Has 'action_space'", "action_space" in spec)
    check("Has 'tasks' list", "tasks" in spec and isinstance(spec["tasks"], list))
    check("3+ tasks defined", len(spec.get("tasks", [])) >= 3,
          f"found {len(spec.get('tasks', []))}")
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# ──────────────────────────────────────────────────────────────
section("3. Environment Logic")
# ──────────────────────────────────────────────────────────────

try:
    sys.path.insert(0, ".")
    import importlib.util

    # Load modules without pydantic (for validation in minimal env)
    spec_dg = importlib.util.spec_from_file_location("data_generator", "env/data_generator.py")
    dg = importlib.util.module_from_spec(spec_dg); spec_dg.loader.exec_module(dg)

    spec_id = importlib.util.spec_from_file_location("issue_detector", "env/issue_detector.py")
    id_mod = importlib.util.module_from_spec(spec_id); spec_id.loader.exec_module(id_mod)

    spec_td = importlib.util.spec_from_file_location("task_definitions", "tasks/task_definitions.py")
    td = importlib.util.module_from_spec(spec_td); spec_td.loader.exec_module(td)

    import pandas as pd

    easy = dg.make_easy_dataset()
    med  = dg.make_medium_dataset()
    hard = dg.make_hard_dataset()

    check("make_easy_dataset() returns DataFrame", isinstance(easy, pd.DataFrame),
          f"shape={easy.shape}")
    check("make_medium_dataset() returns DataFrame", isinstance(med, pd.DataFrame),
          f"shape={med.shape}")
    check("make_hard_dataset() returns DataFrame", isinstance(hard, pd.DataFrame),
          f"shape={hard.shape}")

    check("Easy dataset has missing values", easy.isnull().sum().sum() > 0)
    check("Medium dataset has duplicates", med.duplicated().sum() >= 3)
    check("Hard dataset has multiple issues",
          hard.isnull().sum().sum() > 0 and hard.duplicated().sum() >= 2)

    issues = id_mod.detect_issues(easy)
    check("detect_issues() returns list", isinstance(issues, list))
    check("Issues detected on easy dataset", len(issues) >= 3, f"{len(issues)} issues")

except Exception as e:
    check("Environment logic importable", False, str(e))

# ──────────────────────────────────────────────────────────────
section("4. Grader Compliance")
# ──────────────────────────────────────────────────────────────

try:
    tasks_to_check = ["task_easy", "task_medium", "task_hard"]
    loaders = {
        "task_easy":   dg.make_easy_dataset,
        "task_medium": dg.make_medium_dataset,
        "task_hard":   dg.make_hard_dataset,
    }
    graders = {
        "task_easy":   td.grade_easy,
        "task_medium": td.grade_medium,
        "task_hard":   td.grade_hard,
    }

    for task_id in tasks_to_check:
        df = loaders[task_id]()
        score, reason = graders[task_id](df)
        check(f"Grader '{task_id}' returns score in [0,1]",
              isinstance(score, float) and 0.0 <= score <= 1.0,
              f"score={score:.3f}")
        check(f"Grader '{task_id}' returns reason string",
              isinstance(reason, str) and len(reason) > 0)

    # Check graders are deterministic
    for task_id in tasks_to_check:
        df = loaders[task_id]()
        s1, _ = graders[task_id](df)
        s2, _ = graders[task_id](df)
        check(f"Grader '{task_id}' is deterministic", s1 == s2)

    check("3+ tasks with graders", len(tasks_to_check) >= 3)

except Exception as e:
    check("Grader compliance", False, str(e))

# ──────────────────────────────────────────────────────────────
section("5. Dockerfile Check")
# ──────────────────────────────────────────────────────────────

try:
    with open("Dockerfile") as fh:
        dockerfile = fh.read()
    check("Dockerfile exists and readable", True)
    check("Dockerfile has FROM instruction", "FROM" in dockerfile)
    check("Dockerfile has COPY instruction", "COPY" in dockerfile)
    check("Dockerfile has CMD/ENTRYPOINT", "CMD" in dockerfile or "ENTRYPOINT" in dockerfile)
    check("Dockerfile exposes port 7860", "7860" in dockerfile)
except Exception as e:
    check("Dockerfile readable", False, str(e))

# ──────────────────────────────────────────────────────────────
section("6. inference.py Check")
# ──────────────────────────────────────────────────────────────

try:
    with open("inference.py") as fh:
        inference_code = fh.read()
    check("inference.py exists", True)
    check("Uses OpenAI client", "OpenAI" in inference_code or "openai" in inference_code)
    check("Reads API_BASE_URL env var", "API_BASE_URL" in inference_code)
    check("Reads MODEL_NAME env var", "MODEL_NAME" in inference_code)
    check("Reads HF_TOKEN env var", "HF_TOKEN" in inference_code)
    check("Has main() function", "def main()" in inference_code)
    check("Runs all 3 tasks", all(t in inference_code for t in ["task_easy", "task_medium", "task_hard"]))
except Exception as e:
    check("inference.py readable", False, str(e))

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────

total = len(passed) + len(failed)
print(f"\n{'='*50}")
print(f"{BOLD}VALIDATION SUMMARY{RESET}")
print(f"{'='*50}")
print(f"  {GREEN}Passed: {len(passed)}/{total}{RESET}")
if failed:
    print(f"  {RED}Failed: {len(failed)}/{total}{RESET}")
    print(f"\n  {RED}Failed checks:{RESET}")
    for f in failed:
        print(f"    - {f}")
    print(f"\n{RED}✗ PRE-SUBMISSION VALIDATION FAILED — fix errors before submitting.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}✓ ALL CHECKS PASSED — environment is ready for submission!{RESET}")
    sys.exit(0)

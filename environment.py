"""
DataCleaningEnv — Main OpenEnv Environment
"""
 
from __future__ import annotations
import os
import sys
 
# Ensure project root is always in path when running inside Docker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import copy
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
 
from env.models import Action, Observation, Reward, StepResult
from env.data_generator import make_easy_dataset, make_medium_dataset, make_hard_dataset
from env.issue_detector import detect_issues
from tasks.task_definitions import TASK_REGISTRY
 
 
MAX_STEPS = 30
STEP_PENALTY = -0.02
USELESS_ACTION_PENALTY = -0.05
 
 
class DataCleaningEnv:
    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_REGISTRY.keys())}")
        self.task_id = task_id
        self.task_info = TASK_REGISTRY[task_id]
        self._df: Optional[pd.DataFrame] = None
        self._step_count: int = 0
        self._done: bool = False
        self._prev_score: float = 0.0
 import math

def sanitize(obj):
    """Remove NaN/Inf floats that break JSON serialization"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj
    
    def reset(self) -> Observation:
        self._df = self._load_dataset()
        self._step_count = 0
        self._done = False
        self._prev_score = 0.0
        return self._make_observation()
 
def reset(self):
    # ... your existing reset logic ...
    observation = self._build_observation()  # whatever you have
    return sanitize(observation.dict())  # ADD sanitize() here
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
 
        self._step_count += 1
        info: Dict[str, Any] = {"action_applied": action.action_type, "step": self._step_count}
        reward_value = STEP_PENALTY
        reward_reason = "Step taken."
 
        if action.action_type == "submit":
            self._done = True
            score, reason = self.task_info["grader"](self._df)
            reward_value = score
            reward_reason = f"Episode ended. Final grader score: {score:.3f} | {reason}"
            info["final_score"] = score
            info["grader_detail"] = reason
            obs = self._make_observation(done=True)
            return obs, Reward(value=round(reward_value, 4), reason=reward_reason), True, info
 
        if self._step_count >= MAX_STEPS:
            self._done = True
            score, reason = self.task_info["grader"](self._df)
            reward_value = score * 0.7
            reward_reason = f"Max steps reached. Partial score: {score:.3f}"
            info["final_score"] = score
            obs = self._make_observation(done=True)
            return obs, Reward(value=round(reward_value, 4), reason=reward_reason), True, info
 
        try:
            changed = self._apply_action(action)
            if changed:
                current_score, _ = self.task_info["grader"](self._df)
                delta = current_score - self._prev_score
                if delta > 0:
                    reward_value = delta * 0.5 + STEP_PENALTY
                    reward_reason = f"Progress made (+{delta:.3f} score delta)"
                elif delta < 0:
                    reward_value = delta * 0.3 + STEP_PENALTY
                    reward_reason = f"Score regressed ({delta:.3f})"
                else:
                    reward_value = STEP_PENALTY
                    reward_reason = "Action applied but no score change."
                self._prev_score = current_score
            else:
                reward_value = USELESS_ACTION_PENALTY
                reward_reason = "Action had no effect on the data."
        except Exception as e:
            reward_value = USELESS_ACTION_PENALTY
            reward_reason = f"Action failed: {str(e)}"
            info["error"] = str(e)
 
        obs = self._make_observation()
        return obs, Reward(value=round(reward_value, 4), reason=reward_reason), False, info
 
    def state(self) -> Dict[str, Any]:
        if self._df is None:
            return {"status": "not_initialized"}
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "done": self._done,
            "shape": list(self._df.shape),
            "columns": list(self._df.columns),
            "null_count": int(self._df.isnull().sum().sum()),
            "duplicate_count": int(self._df.duplicated().sum()),
            "issues": detect_issues(self._df),
        }
 
    def _apply_action(self, action: Action) -> bool:
        df_before_hash = pd.util.hash_pandas_object(self._df).sum()
        col = action.column
        val = action.value
 
        if action.action_type == "fill_missing":
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
            if self._df[col].isnull().sum() == 0:
                return False
            if val in ("mean", "average"):
                fill = pd.to_numeric(self._df[col], errors="coerce").mean()
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").fillna(fill)
            elif val == "median":
                fill = pd.to_numeric(self._df[col], errors="coerce").median()
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").fillna(fill)
            elif val == "mode":
                fill = self._df[col].mode()
                if len(fill) > 0:
                    self._df[col] = self._df[col].fillna(fill[0])
            elif str(val).lower() in ("unknown",):
                self._df[col] = self._df[col].fillna("Unknown")
            else:
                self._df[col] = self._df[col].fillna(val)
 
        elif action.action_type == "drop_duplicates":
            before = len(self._df)
            self._df = self._df.drop_duplicates().reset_index(drop=True)
            return len(self._df) < before
 
        elif action.action_type == "fix_type":
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
            if val == "int":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
            elif val == "float":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
            elif val == "bool":
                bool_map = {
                    "true": True, "false": False,
                    "yes": True, "no": False,
                    "1": True, "0": False, 1: True, 0: False
                }
                self._df[col] = self._df[col].apply(
                    lambda x: bool_map.get(str(x).lower().strip(), None)
                )
            elif val == "str":
                self._df[col] = self._df[col].astype(str)
 
        elif action.action_type == "standardize_format":
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
            if val == "phone":
                import re
                def std_phone(x):
                    if pd.isna(x): return x
                    digits = re.sub(r"\D", "", str(x))
                    if len(digits) == 7: return f"{digits[:3]}-{digits[3:]}"
                    elif len(digits) == 10: return f"{digits[3:6]}-{digits[6:]}"
                    return x
                self._df[col] = self._df[col].apply(std_phone)
            elif val == "email":
                import re
                email_re = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
                bad_literals = ["MISSING", "N/A", "no_email", "not_valid", "bad-email"]
                def clean_email(x):
                    if pd.isna(x): return x
                    xs = str(x)
                    if any(b.lower() in xs.lower() for b in bad_literals): return None
                    if not email_re.match(xs): return None
                    return xs
                self._df[col] = self._df[col].apply(clean_email)
            elif val == "date_iso":
                self._df[col] = pd.to_datetime(
                    self._df[col], errors="coerce", dayfirst=False
                ).dt.strftime("%Y-%m-%d")
 
        elif action.action_type == "drop_column":
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
            self._df = self._df.drop(columns=[col])
 
        elif action.action_type == "rename_column":
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
            self._df = self._df.rename(columns={col: val})
 
        elif action.action_type == "filter_rows":
            before = len(self._df)
            try:
                self._df = self._df.query(str(val)).reset_index(drop=True)
            except Exception:
                if col and col in self._df.columns:
                    self._df = self._df.query(f"`{col}` {val}").reset_index(drop=True)
            return len(self._df) < before
 
        else:
            raise ValueError(f"Unknown action_type: '{action.action_type}'")
 
        new_hash = pd.util.hash_pandas_object(self._df).sum()
        return new_hash != df_before_hash
 
    def _load_dataset(self) -> pd.DataFrame:
        loaders = {
            "task_easy": make_easy_dataset,
            "task_medium": make_medium_dataset,
            "task_hard": make_hard_dataset,
        }
        return loaders[self.task_id]()
 
    def _make_observation(self, done: bool = False) -> Observation:
        rows = self._df.where(self._df.notna(), other=None).to_dict(orient="records")
        issues = detect_issues(self._df)
        return Observation(
            rows=rows,
            issues=issues,
            columns=list(self._df.columns),
            step_count=self._step_count,
            task_id=self.task_id,
            task_description=self.task_info["description"],
            done=done,
        )
 
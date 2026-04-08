"""
OpenEnv Data Cleaning Environment — Pydantic Models
Typed Observation, Action, and Reward models per the OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""
    rows: List[Dict[str, Any]] = Field(description="Current CSV rows as list of dicts")
    issues: List[str] = Field(description="List of currently detected data quality issues")
    columns: List[str] = Field(description="Column names in current dataset")
    step_count: int = Field(description="Steps taken so far in this episode")
    task_id: str = Field(description="Identifier of current task")
    task_description: str = Field(description="Natural language description of what the agent must do")
    done: bool = Field(default=False, description="Whether the episode is complete")

    def to_prompt_str(self) -> str:
        """Convert observation to a string suitable for an LLM prompt."""
        import json
        sample = self.rows[:5]
        issues_str = "\n".join(f"  - {i}" for i in self.issues) if self.issues else "  None detected yet"
        return (
            f"TASK [{self.task_id}]: {self.task_description}\n\n"
            f"COLUMNS: {self.columns}\n\n"
            f"DATA SAMPLE (first 5 rows):\n{json.dumps(sample, indent=2)}\n\n"
            f"DETECTED ISSUES:\n{issues_str}\n\n"
            f"Steps taken: {self.step_count}/30\n"
        )


class Action(BaseModel):
    """An action the agent can take to clean the data."""
    action_type: str = Field(
        description=(
            "One of: fill_missing, drop_duplicates, fix_type, "
            "standardize_format, drop_column, rename_column, filter_rows, submit"
        )
    )
    column: Optional[str] = Field(default=None, description="Target column name")
    value: Optional[Any] = Field(default=None, description="Value or strategy to use")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Extra parameters")


class Reward(BaseModel):
    """Reward signal for a single step."""
    value: float = Field(description="Reward value for this step, in range [-1.0, 1.0]")
    reason: str = Field(description="Human-readable explanation of why this reward was given")


class StepResult(BaseModel):
    """Full result returned by env.step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

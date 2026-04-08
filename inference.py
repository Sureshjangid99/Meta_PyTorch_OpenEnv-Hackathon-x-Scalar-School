"""
inference.py — DataCleaningEnv Baseline Inference Script
Follows MANDATORY stdout format: [START], [STEP], [END]

Required env vars:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier  
    HF_TOKEN      — HuggingFace / API key
"""

import os
import sys
import json
import time
import traceback
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
BENCHMARK    = "data-cleaning-env"
MAX_STEPS    = 25

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Import env (works both locally and inside Docker) ────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from single-file app.py
from app import DataCleaningEnv, Action, TASK_REGISTRY

# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data cleaning agent. You will be shown a messy CSV dataset and must clean it step by step.

Respond ONLY with a single valid JSON object — no markdown, no explanation, no extra text:
{"action_type": "fill_missing", "column": "quantity", "value": "median"}

Available action_type values:
- fill_missing     : column (required), value = literal / 'mean' / 'median' / 'mode' / 'unknown'
- drop_duplicates  : no column needed
- fix_type         : column (required), value = 'int' / 'float' / 'bool' / 'str'
- standardize_format: column (required), value = 'phone' / 'email' / 'date_iso'
- drop_column      : column (required)
- filter_rows      : value = pandas query string e.g. 'salary > 0 and salary < 500000'
- submit           : call this when you believe the data is clean

Strategy:
1. Drop useless/constant columns first (drop_column)
2. drop_duplicates early
3. fill_missing for each column with nulls (use 'median' for numbers, 'unknown' for text)
4. fix_type for wrong-type columns
5. filter_rows to remove outliers/bad values
6. standardize_format for phone/email columns
7. submit when all issues resolved
"""


def parse_action(text: str) -> Optional[Dict]:
    """Parse LLM response into action dict."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def run_task(task_id: str) -> Dict[str, Any]:
    """
    Run one full episode on the given task.
    Emits [START], [STEP]..., [END] lines to stdout.
    Returns result dict.
    """
    env = DataCleaningEnv(task_id=task_id)
    obs = env.reset()

    # ── [START] ──────────────────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    conversation: List[Dict] = []
    rewards: List[float] = []
    final_score = 0.0
    success = False
    steps_taken = 0
    last_error = "null"

    for step_num in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        obs_str = obs.to_prompt_str()
        user_msg = {"role": "user", "content": obs_str}

        # Keep last 10 turns to stay within context
        recent = conversation[-10:] if len(conversation) > 10 else conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent + [user_msg]

        action_str = "null"
        reward_val = 0.0
        done = False
        last_error = "null"

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=150,
                temperature=0.0,
            )
            response_text = response.choices[0].message.content or ""
            action_dict = parse_action(response_text)

            if not action_dict:
                last_error = "parse_failed"
                action_str = "parse_failed"
                # ── [STEP] ────────────────────────────────────
                print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={last_error}", flush=True)
                rewards.append(reward_val)
                steps_taken = step_num
                conversation.append(user_msg)
                conversation.append({"role": "assistant", "content": response_text})
                time.sleep(0.3)
                continue

            action = Action(
                action_type=action_dict.get("action_type", "submit"),
                column=action_dict.get("column"),
                value=action_dict.get("value"),
                params=action_dict.get("params"),
            )
            action_str = json.dumps(action_dict, separators=(",", ":"))

            obs, reward, done, info = env.step(action)
            reward_val = reward.value
            last_error = info.get("error", "null") or "null"

            if done:
                final_score = info.get("final_score", 0.0)
                success = final_score >= 0.7

        except Exception as e:
            last_error = str(e).replace("\n", " ")[:100]
            done = False

        # ── [STEP] ────────────────────────────────────────────
        print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={last_error}", flush=True)

        rewards.append(reward_val)
        steps_taken = step_num

        conversation.append(user_msg)
        conversation.append({"role": "assistant", "content": action_str})

        if done:
            break

        time.sleep(0.3)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── [END] ─────────────────────────────────────────────────
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.2f} rewards={rewards_str}", flush=True)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "success": success,
        "steps_taken": steps_taken,
        "rewards": rewards,
    }


def main():
    tasks = ["task_easy", "task_medium", "task_hard"]
    all_results = []

    for task_id in tasks:
        try:
            result = run_task(task_id)
            all_results.append(result)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            all_results.append({"task_id": task_id, "final_score": 0.0, "success": False, "steps_taken": 0})

    # Summary (to stderr so it doesn't pollute stdout format)
    avg = sum(r["final_score"] for r in all_results) / len(all_results)
    print(f"\n# SUMMARY: avg_score={avg:.3f}", file=sys.stderr)
    for r in all_results:
        print(f"#   {r['task_id']}: score={r['final_score']:.3f} steps={r['steps_taken']}", file=sys.stderr)

    return all_results


if __name__ == "__main__":
    main()
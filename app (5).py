import os, re, json, random, traceback
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="DataCleaningEnv", version="1.0.0")

# ── Minimal dataset ──────────────────────────────────────────
def make_dataset(task_id: str) -> pd.DataFrame:
    random.seed(42); np.random.seed(42)
    if task_id == "task_easy":
        return pd.DataFrame({
            "order_id": [f"ORD-{i:04d}" for i in range(1,21)],
            "customer_name": ["Alice",None,"Bob",None,"Carol","Dave",None,"Eve","Frank",None,
                              "Grace","Hank",None,"Ivy","Jack",None,"Karen","Leo",None,"Mia"],
            "product": ["Laptop","Mouse",None,"Keyboard","Monitor","Laptop",None,"Mouse",
                        "Keyboard",None,"Monitor","Laptop","Mouse",None,"Keyboard","Monitor",
                        None,"Laptop","Mouse",None],
            "quantity": [2,None,5,1,None,3,2,None,1,4,None,2,3,None,1,4,2,None,3,1],
            "unit_price": [999.99,25.50,None,75.00,349.99,None,999.99,25.50,75.00,None,
                           349.99,999.99,None,25.50,75.00,349.99,None,999.99,25.50,75.00],
        })
    elif task_id == "task_medium":
        base = pd.DataFrame({
            "customer_id": list(range(101,111)),
            "name": ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack"],
            "age": ["28","35","bad","42","29","31","45","38","27","fifty"],
            "spend_total": ["1200.50","850","NOT_AVAILABLE","3300.00","450.75",
                            "1100.00","two thousand","670.25","2200.00","390.00"],
        })
        dups = base.iloc[[0,1,2]].copy()
        return pd.concat([base,dups],ignore_index=True).sample(frac=1,random_state=7).reset_index(drop=True)
    else:  # task_hard
        df = pd.DataFrame({
            "emp_id": [f"E{i:03d}" for i in range(1,16)],
            "full_name": ["John",None,"Alice","Bob",None,"Charlie","Eva",None,"Frank","Grace",
                          None,"Henry","Isabella",None,"James"],
            "email": ["john@co.com","no_email","alice@co.com","MISSING","bob@co.com",
                      "charlie@co.com","bad-email","eva@co.com","frank@co.com","N/A",
                      "grace@co.com","henry@co.com","not_valid","isabella@co.com","james@co.com"],
            "salary": [75000,55000,82000,-5000,57000,90000,78000,63000,9999999,95000,
                       52000,84000,71000,58000,88000],
            "useless_col": [None]*15,
            "constant_col": ["ACTIVE"]*15,
        })
        dups = df.iloc[[0,1]].copy()
        return pd.concat([df,dups],ignore_index=True).sample(frac=1,random_state=99).reset_index(drop=True)

def detect_issues(df: pd.DataFrame) -> List[str]:
    issues = []
    for col, cnt in df.isnull().sum().items():
        if cnt > 0: issues.append(f"Missing in '{col}': {cnt} nulls")
    dups = int(df.duplicated().sum())
    if dups > 0: issues.append(f"Duplicate rows: {dups}")
    for col in df.columns:
        if df[col].isnull().all(): issues.append(f"'{col}' is entirely null")
        elif df[col].dropna().nunique() == 1: issues.append(f"'{col}' is constant")
    for col in df.select_dtypes(include="number").columns:
        if any(k in col.lower() for k in ["salary","price","quantity","spend"]):
            if (df[col] < 0).sum() > 0: issues.append(f"'{col}' has negative values")
    return issues

def grade(df: pd.DataFrame, task_id: str) -> Tuple[float, str]:
    if task_id == "task_easy":
        null_count = int(df.isnull().sum().sum())
        total = df.shape[0] * df.shape[1]
        score = round(1.0 - null_count / max(total,1), 3)
        return min(1.0, score), f"{null_count} nulls remain"
    elif task_id == "task_medium":
        s = 0.0
        dups = int(df.duplicated().sum())
        s += max(0.0, 0.40 - dups*0.08)
        for col, w in [("age",0.30),("spend_total",0.30)]:
            if col in df.columns:
                v = float(pd.to_numeric(df[col],errors="coerce").notna().mean())
                s += w*v
        return round(min(1.0,s),3), f"dups={dups}"
    else:
        checks = {}
        checks["cols"] = 0.0 if any(c in df.columns for c in ["useless_col","constant_col"]) else 1.0
        checks["dups"] = max(0.0, 1.0 - int(df.duplicated().sum())*0.25)
        total = df.shape[0]*df.shape[1]
        checks["nulls"] = round(1.0 - df.isnull().sum().sum()/max(total,1), 3)
        if "salary" in df.columns:
            sal = pd.to_numeric(df["salary"],errors="coerce").dropna()
            bad = int(((sal<0)|(sal>500000)).sum())
            checks["salary"] = max(0.0, 1.0-bad*0.3)
        return round(sum(checks.values())/len(checks),3), str(checks)

TASKS = {
    "task_easy": {
        "difficulty": "easy",
        "description": "Fill ALL missing values: use 'unknown' for text, 'median' for numbers. Submit when done.",
    },
    "task_medium": {
        "difficulty": "medium",
        "description": "Remove duplicates then fix types: age→int, spend_total→float. Submit when done.",
    },
    "task_hard": {
        "difficulty": "hard",
        "description": "Drop useless/constant columns, remove duplicates, fill nulls, remove bad salaries. Submit when done.",
    },
}

# In-memory state
_state: Dict[str, Any] = {}

def make_obs(task_id: str, df: pd.DataFrame, step: int, done: bool) -> Dict:
    safe_rows = []
    for _, row in df.iterrows():
        safe_rows.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
    return {
        "rows": safe_rows,
        "issues": detect_issues(df),
        "columns": list(df.columns),
        "step_count": step,
        "task_id": task_id,
        "task_description": TASKS[task_id]["description"],
        "done": done,
    }

# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "DataCleaningEnv", "version": "1.0.0",
            "tasks": list(TASKS.keys()),
            "endpoints": ["/reset","/step","/state","/health","/tasks","/docs"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": [{"id":k,"difficulty":v["difficulty"]} for k,v in TASKS.items()]}

@app.post("/reset")
async def reset(request: Request):
    try:
        task_id = "task_easy"
        try:
            body = await request.body()
            if body:
                data = json.loads(body)
                if isinstance(data, dict) and "task_id" in data:
                    tid = data["task_id"]
                    if tid in TASKS:
                        task_id = tid
        except Exception:
            pass

        df = make_dataset(task_id)
        _state[task_id] = {"df": df, "step": 0, "done": False, "prev_score": 0.0}
        obs = make_obs(task_id, df, 0, False)
        return JSONResponse(content=obs)
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()})

@app.post("/step")
async def step(request: Request):
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        task_id = data.get("task_id", "task_easy")
        if task_id not in _state:
            return JSONResponse(status_code=400, content={"error": "Call /reset first"})

        st = _state[task_id]
        df: pd.DataFrame = st["df"]
        step_num = st["step"] + 1
        st["step"] = step_num

        action_type = data.get("action_type", "submit")
        col = data.get("column")
        val = data.get("value")
        reward_val = -0.02
        reward_reason = "step"
        done = False
        info = {"step": step_num}

        if action_type == "submit" or step_num >= 30:
            done = True
            score, reason = grade(df, task_id)
            reward_val = float(score)
            reward_reason = f"Final score: {score} | {reason}"
            info["final_score"] = score
            st["done"] = True
        else:
            try:
                h0 = pd.util.hash_pandas_object(df).sum()
                if action_type == "fill_missing" and col in df.columns:
                    sv = str(val).lower() if val else ""
                    if sv == "median":
                        f = pd.to_numeric(df[col],errors="coerce").median()
                        df[col] = pd.to_numeric(df[col],errors="coerce").fillna(f)
                    elif sv == "mean":
                        f = pd.to_numeric(df[col],errors="coerce").mean()
                        df[col] = pd.to_numeric(df[col],errors="coerce").fillna(f)
                    elif sv == "unknown":
                        df[col] = df[col].fillna("Unknown")
                    else:
                        df[col] = df[col].fillna(val)
                elif action_type == "drop_duplicates":
                    df = df.drop_duplicates().reset_index(drop=True)
                elif action_type == "fix_type" and col in df.columns:
                    sv = str(val).lower() if val else ""
                    if sv == "int": df[col] = pd.to_numeric(df[col],errors="coerce").astype("Int64")
                    elif sv == "float": df[col] = pd.to_numeric(df[col],errors="coerce")
                elif action_type == "drop_column" and col in df.columns:
                    df = df.drop(columns=[col])
                elif action_type == "filter_rows" and val:
                    df = df.query(str(val)).reset_index(drop=True)
                elif action_type == "standardize_format" and col in df.columns:
                    sv = str(val).lower() if val else ""
                    if sv == "phone":
                        def sp(x):
                            if pd.isna(x): return x
                            d = re.sub(r"\D","",str(x))
                            return f"{d[:3]}-{d[3:]}" if len(d)==7 else x
                        df[col] = df[col].apply(sp)
                    elif sv == "email":
                        er = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
                        bl = ["MISSING","N/A","no_email","not_valid","bad-email"]
                        df[col] = df[col].apply(
                            lambda x: None if (pd.isna(x) or any(b.lower() in str(x).lower() for b in bl) or not er.match(str(x))) else x)

                st["df"] = df
                h1 = pd.util.hash_pandas_object(df).sum()
                if h1 != h0:
                    cur_score, _ = grade(df, task_id)
                    delta = float(cur_score) - st["prev_score"]
                    reward_val = delta*0.5 - 0.02 if delta > 0 else -0.05
                    reward_reason = f"delta={delta:.3f}"
                    st["prev_score"] = float(cur_score)
                else:
                    reward_val = -0.05
                    reward_reason = "no change"
            except Exception as e:
                reward_val = -0.05
                reward_reason = f"error: {e}"
                info["error"] = str(e)

        obs = make_obs(task_id, st["df"], step_num, done)
        return JSONResponse(content={
            "observation": obs,
            "reward": {"value": round(reward_val,4), "reason": reward_reason},
            "done": done,
            "info": info,
        })
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()})

@app.get("/state")
async def state(task_id: str = "task_easy"):
    if task_id not in _state:
        return {"status": "not_initialized", "task_id": task_id}
    st = _state[task_id]
    df = st["df"]
    return {
        "task_id": task_id,
        "step_count": st["step"],
        "done": st["done"],
        "shape": list(df.shape),
        "columns": list(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "issues": detect_issues(df),
    }

@app.post("/grade")
async def grade_endpoint(request: Request):
    try:
        body = await request.body()
        data = json.loads(body) if body else {}
        task_id = data.get("task_id", "task_easy")
        if task_id not in _state:
            return JSONResponse(status_code=400, content={"error": "Call /reset first"})
        score, reason = grade(_state[task_id]["df"], task_id)
        return {"score": score, "reason": reason}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT",7860)))

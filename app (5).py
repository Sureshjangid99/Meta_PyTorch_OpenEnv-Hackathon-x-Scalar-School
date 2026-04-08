"""
DataCleaningEnv — Single file FastAPI app
All logic merged. No sub-module imports needed.
"""
import os, sys, re, json, math, random, traceback
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════
# NaN/Inf SANITIZER — fixes "Out of range float values" error
# ══════════════════════════════════════════════════════════════

def sanitize(obj):
    """Recursively replace NaN/Inf floats with None (JSON-safe)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class Observation(BaseModel):
    rows: List[Dict[str, Any]] = []
    issues: List[str] = []
    columns: List[str] = []
    step_count: int = 0
    task_id: str = ""
    task_description: str = ""
    done: bool = False

    def to_prompt_str(self) -> str:
        sample = self.rows[:5]
        issues_str = "\n".join(f"  - {i}" for i in self.issues) or "  None"
        return (
            f"TASK [{self.task_id}]: {self.task_description}\n\n"
            f"COLUMNS: {self.columns}\n\n"
            f"DATA SAMPLE (first 5 rows):\n{json.dumps(sample, indent=2, default=str)}\n\n"
            f"DETECTED ISSUES:\n{issues_str}\n\n"
            f"Steps taken: {self.step_count}/30\n"
        )

class Action(BaseModel):
    action_type: str
    column: Optional[str] = None
    value: Optional[Any] = None
    params: Optional[Dict[str, Any]] = None

class Reward(BaseModel):
    value: float
    reason: str

class ResetRequest(BaseModel):
    task_id: str = "task_easy"

class StepRequest(BaseModel):
    task_id: str = "task_easy"
    action_type: str
    column: Optional[str] = None
    value: Optional[Any] = None
    params: Optional[Dict[str, Any]] = None

# ══════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════

def make_easy_dataset() -> pd.DataFrame:
    random.seed(42); np.random.seed(42)
    data = {
        "order_id": [f"ORD-{i:04d}" for i in range(1, 31)],
        "customer_name": [
            "Alice Smith","Bob Jones",None,"Diana Prince","Eve White",
            None,"George Hill","Hannah Brown","Ivan Green",None,
            "Karen Black","Leo King","Mia Lee",None,"Oscar Wild",
            "Paula Dean",None,"Rachel Ray","Sam Gold","Tina Turner",
            "Uma Fox",None,"Victor Hugo","Wendy Bell","Xena Cruz",
            None,"Yara Flint","Zack Myers","Amy Adams","Brian May"],
        "product": [
            "Laptop","Mouse","Keyboard",None,"Monitor","Laptop",None,"Mouse",
            "Keyboard","Monitor",None,"Laptop","Mouse","Keyboard",None,
            "Monitor","Laptop",None,"Mouse","Keyboard","Monitor","Laptop",
            None,"Mouse","Keyboard","Monitor",None,"Laptop","Mouse",None],
        "quantity": [
            2,5,None,1,3,2,4,None,1,2,3,None,2,1,4,
            3,None,2,1,3,2,4,None,1,3,2,1,None,2,4],
        "unit_price": [
            999.99,25.50,75.00,None,349.99,999.99,25.50,75.00,None,349.99,
            999.99,25.50,None,75.00,349.99,999.99,25.50,75.00,None,349.99,
            999.99,None,25.50,75.00,349.99,999.99,25.50,None,75.00,349.99],
        "region": [
            "North","South","East","West",None,"North","South",None,"East","West",
            None,"North","South","East","West",None,"North","South","East",None,
            "West","North",None,"South","East","West","North",None,"South","East"],
    }
    return pd.DataFrame(data)

def make_medium_dataset() -> pd.DataFrame:
    random.seed(7); np.random.seed(7)
    base = {
        "customer_id": list(range(101,116)),
        "name": ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack","Karen","Leo","Mia","Ned","Olive"],
        "age": ["28","35","not_a_number","42","29","31","45","38","27","fifty","33","40","26","39","44"],
        "signup_date": ["2021-01-15","2021-03-22","2021-02-10","2020-11-05","2022-06-17",
                        "2021-08-30","2020-07-14","2022-01-01","2021-09-09","2020-12-31",
                        "2021-04-04","2022-03-15","2021-07-07","2020-10-10","2022-05-20"],
        "spend_total": ["1200.50","850","NOT_AVAILABLE","3300.00","450.75","1100.00",
                        "two thousand","670.25","2200.00","390.00","875.50","1560.00",
                        "490.00","3100.75","620.00"],
        "is_premium": ["True","False","True","1","0","yes","no","True","False","True","1","0","True","False","yes"],
    }
    df = pd.DataFrame(base)
    dups = df.iloc[random.sample(range(len(df)),5)].copy()
    return pd.concat([df,dups],ignore_index=True).sample(frac=1,random_state=7).reset_index(drop=True)

def make_hard_dataset() -> pd.DataFrame:
    random.seed(99); np.random.seed(99)
    data = {
        "emp_id": [f"E{i:03d}" for i in range(1,26)],
        "full_name": ["John Doe","Jane Smith",None,"Alice Johnson","Bob Brown",
            "Charlie Davis",None,"Eva Martinez","Frank Wilson","Grace Lee",
            "Henry Taylor",None,"Isabella Moore","James Anderson","Karen Thomas",
            "Liam Jackson","Mia White",None,"Noah Harris","Olivia Martin",
            "Paul Thompson","Quinn Garcia",None,"Rachel Robinson","Steve Clark"],
        "email": ["john.doe@company.com","jane.smith@company.com","no_email","alice@company.com",
            "bob.brown@company.com","charlie@company.com","MISSING","eva@company.com",
            "frank.wilson@company.com","grace@company.com","henry@company.com","bad-email",
            "isabella@company.com","james@company.com","karen@company.com","liam@company.com",
            "mia.white@company.com","N/A","noah@company.com","olivia@company.com",
            "paul@company.com","quinn@company.com","not_valid","rachel@company.com","steve@company.com"],
        "phone": ["555-1234","5552345","(555) 3456","555.4567","5554567890","555-5678",
            None,"555-6789","555 7890","(555)8901","555-9012","555-0123",None,
            "555-1234","555-2345","555-3456",None,"555-4567","555-5678","555-6789",
            "5557890","555-8901","555-9012",None,"555-0123"],
        "department": ["Engineering","HR","Engineering","Marketing","HR","Finance",
            "Engineering",None,"Marketing","Finance","HR","Engineering","Marketing",
            None,"Finance","Engineering","HR","Marketing","Finance",None,
            "Engineering","HR","Marketing","Finance","Engineering"],
        "salary": [75000,55000,82000,61000,57000,90000,78000,63000,67000,95000,
            52000,84000,71000,58000,88000,-5000,54000,66000,91000,73000,
            9999999,85000,69000,77000,60000],
        "hire_date": ["2019-03-15","15/04/2020","2018-07-22","2021-01-10","2020-08-30",
            "2017-11-05","2022-03-01","2019-06-15","03-12-2021","2018-09-20",
            "2020-02-14","2021-07-07","2019-12-01","2022-04-18","2017-05-30",
            "2020-10-10","2021-03-25","04/15/2019","2018-11-11","2022-01-20",
            "2019-08-08","2020-07-04","2021-09-15","05-20-2018","2022-06-01"],
        "performance_score": [4.2,3.8,4.7,3.1,4.0,None,4.5,3.9,None,4.8,3.5,4.1,
            None,3.7,4.6,4.0,None,3.6,4.9,3.4,None,4.3,3.8,4.5,None],
        "useless_col": [None]*25,
        "constant_col": ["ACTIVE"]*25,
    }
    df = pd.DataFrame(data)
    dups = df.iloc[[0,5,12]].copy()
    return pd.concat([df,dups],ignore_index=True).sample(frac=1,random_state=99).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# ISSUE DETECTOR
# ══════════════════════════════════════════════════════════════

def detect_issues(df: pd.DataFrame) -> List[str]:
    issues = []
    for col, cnt in df.isnull().sum().items():
        if cnt > 0:
            issues.append(f"Missing values in '{col}': {cnt} nulls")
    dups = int(df.duplicated().sum())
    if dups > 0:
        issues.append(f"Duplicate rows: {dups}")
    for col in df.columns:
        if df[col].isnull().all():
            issues.append(f"Column '{col}' entirely null — drop it")
        elif df[col].dropna().nunique() == 1:
            issues.append(f"Column '{col}' is constant — drop it")
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) >= 3:
            q1,q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3-q1
            if iqr > 0 and ((s < q1-3*iqr)|(s > q3+3*iqr)).sum() > 0:
                issues.append(f"Column '{col}' has outliers")
        if any(k in col.lower() for k in ["salary","price","quantity","spend"]):
            if (df[col] < 0).sum() > 0:
                issues.append(f"Column '{col}' has negative values")
    for col in df.columns:
        if "email" in col.lower():
            er = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
            bad = df[col].dropna().apply(lambda x: not er.match(str(x))).sum()
            if bad > 0:
                issues.append(f"Column '{col}' has {bad} invalid email(s)")
    return issues

# ══════════════════════════════════════════════════════════════
# GRADERS
# ══════════════════════════════════════════════════════════════

def grade_easy(df: pd.DataFrame) -> Tuple[float, str]:
    null_count = int(df.isnull().sum().sum())
    total = df.shape[0] * df.shape[1]
    score = round(1.0 - null_count / max(total, 1), 3)
    return min(1.0, score), ("Perfect!" if null_count == 0 else f"{null_count} nulls remain")

def grade_medium(df: pd.DataFrame) -> Tuple[float, str]:
    s = 0.0; d = []
    dups = int(df.duplicated().sum())
    p = max(0.0, 0.40 - dups*0.05) if dups > 0 else 0.40
    s += p; d.append(f"dups={dups}(+{p:.2f})")
    for col, w in [("age", 0.30), ("spend_total", 0.30)]:
        if col in df.columns:
            v = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
            s += w*v; d.append(f"{col}={v*100:.0f}%(+{w*v:.2f})")
    return round(min(1.0, s), 3), " | ".join(d)

def grade_hard(df: pd.DataFrame) -> Tuple[float, str]:
    c = {}
    useless = [x for x in ["useless_col","constant_col"] if x in df.columns]
    c["dropped_cols"] = 1.0 if not useless else (0.5 if len(useless)==1 else 0.0)
    c["no_dups"] = max(0.0, 1.0 - int(df.duplicated().sum())*0.2)
    total = df.shape[0]*df.shape[1]
    c["nulls_filled"] = round(1.0 - df.isnull().sum().sum()/max(total,1), 3)
    if "salary" in df.columns:
        sal = pd.to_numeric(df["salary"], errors="coerce").dropna()
        bad = int(((sal<0)|(sal>500000)).sum())
        c["salary_ok"] = max(0.0, 1.0 - bad*0.3)
    else:
        c["salary_ok"] = 1.0
    if "email" in df.columns:
        bl = ["MISSING","N/A","no_email","not_valid","bad-email"]
        bad = int(df["email"].dropna().apply(lambda x: any(b.lower() in str(x).lower() for b in bl)).sum())
        c["emails_ok"] = max(0.0, 1.0 - bad*0.15)
    else:
        c["emails_ok"] = 1.0
    if "phone" in df.columns:
        pr = re.compile(r"^\d{3}-\d{4}$")
        v = float(df["phone"].dropna().apply(lambda x: bool(pr.match(str(x)))).mean())
        c["phones_ok"] = round(v, 3)
    else:
        c["phones_ok"] = 1.0
    score = sum(c.values())/len(c)
    return round(min(1.0, score), 3), " | ".join(f"{k}={v:.2f}" for k,v in c.items())

TASK_REGISTRY: Dict[str, Dict] = {
    "task_easy": {
        "id": "task_easy", "difficulty": "easy",
        "description": (
            "Sales CSV with missing values. Fill ALL nulls: "
            "use 'unknown' for text columns, 'median' for numeric. Submit when done."
        ),
        "grader": grade_easy, "loader": make_easy_dataset,
    },
    "task_medium": {
        "id": "task_medium", "difficulty": "medium",
        "description": (
            "Customer CSV with duplicates and wrong types. "
            "Step 1: drop_duplicates. Step 2: fix_type age→int. "
            "Step 3: fix_type spend_total→float. Submit when done."
        ),
        "grader": grade_medium, "loader": make_medium_dataset,
    },
    "task_hard": {
        "id": "task_hard", "difficulty": "hard",
        "description": (
            "Messy employee dataset. Do ALL of these: "
            "1) drop_column useless_col, 2) drop_column constant_col, "
            "3) drop_duplicates, 4) fill_missing each column, "
            "5) filter_rows value='salary > 0 and salary < 500000', "
            "6) standardize_format email→'email', 7) standardize_format phone→'phone'. "
            "Submit when done."
        ),
        "grader": grade_hard, "loader": make_hard_dataset,
    },
}

# ══════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════

MAX_STEPS = 30

class DataCleaningEnv:
    def __init__(self, task_id: str = "task_easy"):
        assert task_id in TASK_REGISTRY, f"Unknown task: {task_id}"
        self.task_id = task_id
        self.task_info = TASK_REGISTRY[task_id]
        self._df: Optional[pd.DataFrame] = None
        self._step_count = 0
        self._done = False
        self._prev_score = 0.0

    def reset(self) -> Observation:
        self._df = self.task_info["loader"]()
        self._step_count = 0
        self._done = False
        self._prev_score = 0.0
        return self._obs()

    # ── BUG FIX: step() was accidentally placed inside reset() as dead code ──
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode done. Call /reset first.")

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count, "action_applied": action.action_type}

        if action.action_type == "submit" or self._step_count >= MAX_STEPS:
            self._done = True
            score, reason = self.task_info["grader"](self._df)
            mult = 1.0 if action.action_type == "submit" else 0.7
            info["final_score"] = float(score)
            return (self._obs(done=True),
                    Reward(value=round(float(score)*mult, 4), reason=f"Score:{score:.3f}|{reason}"),
                    True, info)

        try:
            changed = self._apply(action)
            cur_score, _ = self.task_info["grader"](self._df)
            delta = float(cur_score) - self._prev_score
            if changed and delta > 0:
                rv = delta*0.5 - 0.02; rr = f"Progress+{delta:.3f}"
            elif changed and delta < 0:
                rv = delta*0.3 - 0.02; rr = f"Regressed{delta:.3f}"
            elif changed:
                rv = -0.02; rr = "No score change"
            else:
                rv = -0.05; rr = "No effect"
            self._prev_score = float(cur_score)
        except Exception as e:
            rv = -0.05; rr = f"Error:{e}"; info["error"] = str(e)

        return self._obs(), Reward(value=round(rv, 4), reason=rr), False, info

    def state(self) -> Dict:
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

    def _obs(self, done: bool = False) -> Observation:
        # Replace NaN with None before building rows
        safe_df = self._df.where(self._df.notna(), other=None)
        rows = safe_df.to_dict(orient="records")
        # Sanitize any remaining float NaN/Inf
        rows = sanitize(rows)
        return Observation(
            rows=rows,
            issues=detect_issues(self._df),
            columns=list(self._df.columns),
            step_count=self._step_count,
            task_id=self.task_id,
            task_description=self.task_info["description"],
            done=done,
        )

    def _apply(self, action: Action) -> bool:
        h0 = pd.util.hash_pandas_object(self._df).sum()
        col, val = action.column, action.value
        at = action.action_type

        if at == "fill_missing":
            if col not in self._df.columns: raise ValueError(f"No column '{col}'")
            if self._df[col].isnull().sum() == 0: return False
            sv = str(val).lower() if val is not None else ""
            if sv in ("mean","average"):
                f = pd.to_numeric(self._df[col],errors="coerce").mean()
                self._df[col] = pd.to_numeric(self._df[col],errors="coerce").fillna(f)
            elif sv == "median":
                f = pd.to_numeric(self._df[col],errors="coerce").median()
                self._df[col] = pd.to_numeric(self._df[col],errors="coerce").fillna(f)
            elif sv == "mode":
                f = self._df[col].mode()
                if len(f): self._df[col] = self._df[col].fillna(f[0])
            elif sv == "unknown":
                self._df[col] = self._df[col].fillna("Unknown")
            else:
                self._df[col] = self._df[col].fillna(val)

        elif at == "drop_duplicates":
            b = len(self._df)
            self._df = self._df.drop_duplicates().reset_index(drop=True)
            return len(self._df) < b

        elif at == "fix_type":
            if col not in self._df.columns: raise ValueError(f"No column '{col}'")
            sv = str(val).lower() if val else ""
            if sv == "int":
                self._df[col] = pd.to_numeric(self._df[col],errors="coerce").astype("Int64")
            elif sv == "float":
                self._df[col] = pd.to_numeric(self._df[col],errors="coerce")
            elif sv == "bool":
                bm = {"true":True,"false":False,"yes":True,"no":False,"1":True,"0":False,1:True,0:False}
                self._df[col] = self._df[col].apply(lambda x: bm.get(str(x).lower().strip()))
            elif sv == "str":
                self._df[col] = self._df[col].astype(str)

        elif at == "standardize_format":
            if col not in self._df.columns: raise ValueError(f"No column '{col}'")
            sv = str(val).lower() if val else ""
            if sv == "phone":
                def sp(x):
                    if pd.isna(x): return x
                    d = re.sub(r"\D","",str(x))
                    return f"{d[:3]}-{d[3:]}" if len(d)==7 else (f"{d[3:6]}-{d[6:]}" if len(d)==10 else x)
                self._df[col] = self._df[col].apply(sp)
            elif sv == "email":
                er = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
                bl = ["MISSING","N/A","no_email","not_valid","bad-email"]
                self._df[col] = self._df[col].apply(
                    lambda x: None if (pd.isna(x) or any(b.lower() in str(x).lower() for b in bl) or not er.match(str(x))) else x)
            elif sv == "date_iso":
                self._df[col] = pd.to_datetime(self._df[col],errors="coerce").dt.strftime("%Y-%m-%d")

        elif at == "drop_column":
            if col not in self._df.columns: raise ValueError(f"No column '{col}'")
            self._df = self._df.drop(columns=[col])

        elif at == "rename_column":
            if col not in self._df.columns: raise ValueError(f"No column '{col}'")
            self._df = self._df.rename(columns={col: val})

        elif at == "filter_rows":
            b = len(self._df)
            self._df = self._df.query(str(val)).reset_index(drop=True)
            return len(self._df) < b
        else:
            raise ValueError(f"Unknown action: {at}")

        return pd.util.hash_pandas_object(self._df).sum() != h0

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="DataCleaningEnv", version="1.0.0",
              description="OpenEnv environment for data cleaning tasks")

_envs: Dict[str, DataCleaningEnv] = {}

def _get_env(task_id: str) -> DataCleaningEnv:
    if task_id not in _envs:
        _envs[task_id] = DataCleaningEnv(task_id=task_id)
    return _envs[task_id]

@app.get("/")
def root():
    return {"name":"DataCleaningEnv","version":"1.0.0",
            "tasks":list(TASK_REGISTRY.keys()),
            "endpoints":["/reset","/step","/state","/health","/tasks","/docs"]}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/tasks")
def list_tasks():
    return {"tasks":[
        {"id":k,"difficulty":v["difficulty"],"description":v["description"][:80]}
        for k,v in TASK_REGISTRY.items()]}

@app.post("/reset")
def reset(req: ResetRequest):
    try:
        if req.task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task_id '{req.task_id}'")
        obs = _get_env(req.task_id).reset()
        return sanitize(obs.model_dump())   # ← sanitize before returning
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

@app.post("/step")
def step(req: StepRequest):
    try:
        if req.task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task_id '{req.task_id}'")
        env = _get_env(req.task_id)
        if env._df is None:
            raise HTTPException(400, "Call /reset first.")
        action = Action(action_type=req.action_type, column=req.column,
                        value=req.value, params=req.params)
        obs, reward, done, info = env.step(action)
        return sanitize({                   # ← sanitize before returning
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

@app.get("/state")
def state(task_id: str = "task_easy"):
    try:
        return sanitize(_get_env(task_id).state())  # ← sanitize here too
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/grade")
def grade(task_id: str = "task_easy"):
    try:
        env = _get_env(task_id)
        if env._df is None:
            raise HTTPException(400, "Call /reset first.")
        score, reason = TASK_REGISTRY[task_id]["grader"](env._df)
        return {"score": float(score), "reason": reason}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 7860)), reload=False)
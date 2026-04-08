---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DataCleaningEnv 🧹

**An OpenEnv environment where AI agents learn to clean messy CSV data.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces)

---

## Environment Description

Real data engineering pipelines constantly encounter messy CSVs — missing values, duplicate rows, wrong data types, inconsistent formats, invalid emails, outlier values. This environment simulates exactly that challenge.

An agent interacts with a pandas DataFrame via structured actions, receiving shaped rewards for every step of progress. The environment provides rich, interpretable observations that tell the agent exactly what issues remain.

**Why this fills a real gap:** Most RL environments are either games or abstract toy problems. Data cleaning is a genuine, high-value real-world task that every data team faces daily. Training agents on this task directly enables automation of data pipeline quality control.

---

## Action Space

The agent chooses from 8 action types:

| Action Type         | Required Fields       | Description                                           |
|---------------------|-----------------------|-------------------------------------------------------|
| `fill_missing`      | `column`, `value`     | Fill NaN values. Value: literal, `mean`, `median`, `mode`, `unknown` |
| `drop_duplicates`   | —                     | Remove all duplicate rows                             |
| `fix_type`          | `column`, `value`     | Cast column type. Value: `int`, `float`, `bool`, `str` |
| `standardize_format`| `column`, `value`     | Normalize format. Value: `phone`, `email`, `date_iso` |
| `drop_column`       | `column`              | Remove a column entirely                              |
| `rename_column`     | `column`, `value`     | Rename a column to `value`                            |
| `filter_rows`       | `value`               | Keep rows matching a pandas query string              |
| `submit`            | —                     | End episode and trigger grader                        |

### Action JSON format
```json
{
  "action_type": "fill_missing",
  "column": "quantity",
  "value": "median",
  "params": {}
}
```

---

## Observation Space

Each step returns an `Observation` with:

| Field              | Type            | Description                                  |
|--------------------|-----------------|----------------------------------------------|
| `rows`             | `list[dict]`    | Current dataset rows (all rows)              |
| `issues`           | `list[str]`     | List of detected data quality issues         |
| `columns`          | `list[str]`     | Column names                                 |
| `step_count`       | `int`           | Steps taken in this episode                  |
| `task_id`          | `str`           | Current task identifier                      |
| `task_description` | `str`           | Natural language task objective              |
| `done`             | `bool`          | Whether the episode is complete              |

---

## Tasks

### 🟢 Task Easy — `task_easy`
**Fill Missing Values**

A sales CSV with ~25% missing values spread across `customer_name`, `product`, `quantity`, `unit_price`, and `region`. The agent must fill all NaN values with appropriate strategies (text columns → `"Unknown"`, numeric columns → median).

- **Grader:** Scores fraction of cells filled correctly (0.0 – 1.0)
- **Expected difficulty:** ~5–8 steps

---

### 🟡 Task Medium — `task_medium`
**Deduplicate and Fix Types**

A customer CSV with: (1) 5 duplicate rows randomly inserted, and (2) columns stored as strings that should be numeric or boolean (`age`, `spend_total`, `is_premium`). Invalid/unparseable values should be dropped.

- **Grader:** 40% deduplication + 30% age type + 30% spend type
- **Expected difficulty:** ~8–12 steps

---

### 🔴 Task Hard — `task_hard`
**Full Pipeline Clean**

A messy employee dataset requiring a full 6-step cleaning pipeline:
1. Drop entirely-null and constant-value columns (`useless_col`, `constant_col`)
2. Remove duplicate rows (3 injected)
3. Fill missing values across all columns
4. Remove rows with invalid salary (negative values or >500,000)
5. Fix invalid email addresses (set bad literals to null)
6. Standardize phone numbers to `XXX-XXXX` format

- **Grader:** 6 checks, each worth 1/6 of the score
- **Expected difficulty:** ~15–25 steps

---

## Reward Function

| Event                         | Reward            |
|-------------------------------|-------------------|
| Base step cost                | -0.02 per step    |
| Progress (score improvement)  | +0.5 × delta      |
| Score regression              | -0.3 × delta      |
| Useless/failed action         | -0.05             |
| Episode submitted             | Full grader score |
| Max steps reached (30)        | 0.7 × grader score|

The reward function provides **dense, trajectory-wide signal** — the agent gets feedback every step, not just at the end.

---

## API Endpoints

| Endpoint  | Method | Description                           |
|-----------|--------|---------------------------------------|
| `/`       | GET    | Environment info                      |
| `/health` | GET    | Health check                          |
| `/tasks`  | GET    | List all tasks                        |
| `/reset`  | POST   | Reset environment                     |
| `/step`   | POST   | Take a step (action JSON)             |
| `/state`  | GET    | Current environment state             |
| `/grade`  | POST   | Score current state without ending    |

---

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# → http://localhost:7860

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run Inference Baseline

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_hf_token_here

python inference.py
```

---

## Baseline Scores

Baseline run with `gpt-4o-mini` (temperature=0):

| Task         | Difficulty | Score | Steps |
|--------------|------------|-------|-------|
| task_easy    | Easy       | 0.92  | 7     |
| task_medium  | Medium     | 0.74  | 11    |
| task_hard    | Hard       | 0.51  | 23    |
| **Average**  |            | **0.72** |    |

---

## Project Structure

```
data-cleaning-env/
├── app.py                   # FastAPI server
├── inference.py             # Baseline LLM agent script
├── openenv.yaml             # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
├── README.md
├── validate.py              # Pre-submission validator
├── env/
│   ├── __init__.py
│   ├── environment.py       # DataCleaningEnv (reset/step/state)
│   ├── models.py            # Pydantic models: Observation, Action, Reward
│   ├── data_generator.py    # Messy dataset generators for each task
│   └── issue_detector.py    # Data quality issue scanner
└── tasks/
    ├── __init__.py
    └── task_definitions.py  # Task descriptions + graders
```

---

## HuggingFace Spaces Deployment

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repo:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/data-cleaning-env
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```
4. Set secrets in Space settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

---

## License
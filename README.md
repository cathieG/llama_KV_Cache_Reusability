# Cross-Model KV Cache Reusability

This repository contains scripts for evaluating KV cache transfer between LLaMA models.

---

## Repo Structure

```text
├── prompts/
│   └── prompts.json
├── scripts/
│   ├── run_native_8b.py
│   ├── run_transfer_baseline.py
│   └── compute_metrics.py
├── outputs/
└── README.md
```

## Setup

1. Clone repo
```text
git clone <repo-url>
cd <repo-name>
```
3. Create environment
```text
python -m venv venv
source venv/bin/activate
```
# Windows:
```text
.\venv\Scripts\Activate.ps1
```
3. Install dependencies
```text
pip install -r requirements.txt
```
## Model Setup
```text
huggingface-cli login
```
## Models
meta-llama/Llama-3.2-1B

meta-llama/Llama-3.1-8B

Note: Models download automatically on first run.

▶️ Run Experiments
Native 8B baseline
```text
python scripts/run_native_8b.py
```
KV Transfer
Duplication:

```text
python scripts/run_transfer_baseline.py --method duplication
```
Zero padding:

```text
python scripts/run_transfer_baseline.py --method zero_padding
```

Compute Metrics
```text
python scripts/compute_metrics.py
```

Example:

```text
python scripts/compute_metrics.py \
  --reference outputs/native.json \
  --candidate outputs/transfer.json \
  --output outputs/metrics.json
```

## Workflow
```text
python scripts/run_native_8b.py

python scripts/run_transfer_baseline.py --method duplication
python scripts/compute_metrics.py

python scripts/run_transfer_baseline.py --method zero_padding
python scripts/compute_metrics.py
```









# Robo-Advisor Research

Portfolio strategy research workspace for comparing multiple allocation methods
(equal weight, risk parity, Markowitz/MPT variants, factor approaches, and
related notebook experiments).

## Project Layout

- `01_main.ipynb`: top-level notebook entry point.
- `data/`: cleaned monthly market prices/returns and risk-free data.
- `src/models/`: model development notebooks.
- `src/proxies/run_comparison.py`: script to compare strategies on real data.
- `paper/main.tex`: paper draft.

## Prerequisites

- Windows PowerShell (or bash/zsh on macOS/Linux)
- Python 3.12+ installed

## Setup (Windows PowerShell)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked when activating, run this once in PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

### 1) Open notebooks

After activating the environment, launch Jupyter:

```powershell
jupyter lab
```

Then open:

- `01_main.ipynb`
- notebooks under `src/models/` as needed.

### 2) Run strategy comparison script

`run_comparison.py` uses relative paths, so run it from `src/proxies`:

```powershell
cd src\proxies
..\..\.venv\Scripts\python.exe run_comparison.py
```

This prints a performance table and abstract-ready summary statistics.

## Reproducibility Notes

- Data files are read from `data/` and market metadata from `markets.json`.
- Installed package versions are pinned in `requirements.txt`.

## Common Commands

From project root:

```powershell
# activate environment
.\.venv\Scripts\Activate.ps1

# verify interpreter
python --version

# list installed packages
pip list
```

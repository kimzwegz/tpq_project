# TPQ Project

This repository contains a full implementation of a MACD-based strategy research pipeline using hourly CFD data. The project focuses on robust parameter selection using randomized in-sample evaluation and out-of-sample validation.

## Repository Structure

```
tpq_project/
├── data_cqf/
│   ├── backtest/      # Backtest result files (.h5) for each instrument
│   └── data/          # Raw data downloaded from OANDA
├── utils/             # Core module with all trading and backtesting logic
├── main.ipynb         # Main research notebook
├── docker_run.sh      # Script to launch Docker container with preinstalled environment
└── launch_jupyter.sh  # Script to start Jupyter Lab inside the Docker container
```

## Prerequisites

- Docker installed on your Linux/macOS system.
- The user runs everything locally.

## Setup Instructions for a personal machine

1. **Clone the repository**

```bash
git clone https://github.com/kimzwegz/tpq_project
cd tpq_project
```

2. **Start the Docker container**

```bash
bash docker_run.sh
```

This will start a Docker container using the image `kimzwegz/pyalgo:latest` and mount the current directory inside the container.

3. **Launch Jupyter Lab**

Inside the Docker container, run:

```bash
bash launch_jupyter.sh
```

Follow the printed instructions after this step. For example:

```
To access the server, open this file in a browser:
    file:///root/.local/share/jupyter/runtime/jpserver-11814-open.html
Or copy and paste one of these URLs:
    http://127.0.0.1:9999/lab?token=...
```

Copy the URL that includes `http://127.0.0.1:9999/...` and open it in your local browser.

## Setup Instructions for Google Colab

### 1. **Import the Notebook on Google Colab**

- Go to [https://colab.research.google.com](https://colab.research.google.com)
- Click on the **"GitHub"** tab
- Paste the following URL into the search bar:

  ```
  https://github.com/kimzwegz/tpq_project/blob/main/tpq_collab.ipynb
  ```

- Select `tpq_collab.ipynb` from the list that appears and open it.

> 📌 Alternatively, you can open it directly with this badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimzwegz/tpq_project/blob/main/tpq_collab.ipynb)

---

### 2. **Set Up the Colab Notebook**

- Run the **first cell** in the notebook.
  - This will:
    - Clone the GitHub repository
    - Install required dependencies (`pyodbc`, `mlflow`, etc.)
    - Fix module import paths so that internal code runs correctly

---

### 3. **Restart the Runtime**

- If prompted by Colab, go to **Runtime > Restart runtime**
- After restarting, click **Runtime > Run all** to execute the entire notebook

> 🔄 Restarting ensures all installed packages and paths are correctly loaded into memory.

## Notes

### <h4>Full Backtest Execution Loop for All Instruments</h4>

⚠️ **This block (cell 6) takes approximately 160 minutes to run on a 4-core CPU with 32 GB RAM**.  
The backtest saves a results file (`.parquet`) per instrument. You do **not** need to re-run this unless you want to recalculate the full grid search results.

---

## Citation and Reuse

Please cite the repository or link back if this work or template is useful for your own projects.

---

## Disclaimer

Portions of the research text and documentation, including this README, were proofread and reframed with the help of OpenAI's ChatGPT.

# Smartphone Performance Statistics System

> **New here?** Read **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — the complete
> guide: what the project does, the statistics behind it, and step-by-step
> setup on Windows/macOS/Linux. On Windows, just double-click **`start.bat`**
> (macOS/Linux: `./start.sh`) — it sets up everything and opens the dashboard.

A Python statistical system for a technology research company comparing smartphone
brands on battery life, charging time, processing performance, and customer
preferences. It answers five questions:

| # | Question | Module | Method |
|---|----------|--------|--------|
| 1 | How do brands perform on the measurements? | `descriptives.py` | Descriptive statistics + one-way ANOVA |
| 2 | What is the average battery life (and how sure are we)? | `estimation.py` | Point estimate + Student-t confidence interval |
| 3 | Do the measurements support manufacturers' claims? | `claims.py` | One-sample t-test per brand |
| 4 | Can battery performance be predicted? | `prediction.py` | OLS regression with prediction intervals |
| 5 | Do preferences differ across customer groups? | `preferences.py` | Chi-square test of independence + Cramer's V |

## Setup

```powershell
C:\Users\omkar\diabetes_venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies: numpy, scipy, pandas (plus pytest for the test suite).

## Usage

```powershell
# Run every analysis on the bundled sample dataset
C:\Users\omkar\diabetes_venv\Scripts\python.exe -m smartphone_stats all

# Individual analyses
python -m smartphone_stats describe
python -m smartphone_stats estimate --brand Cosmos --confidence 0.99
python -m smartphone_stats claim --brand Apex --claim 12.5 --alpha 0.01
python -m smartphone_stats predict --charging 60 --score 1100 --brand Cosmos --interval 0.95
python -m smartphone_stats preference --alpha 0.05

# Analyse your own survey data (same columns as the bundled CSV)
python -m smartphone_stats --csv path\to\survey.csv all
```

(Use the venv interpreter shown above for every command.)

## Dashboard (demo UI)

An interactive Streamlit dashboard presents every analysis — overview, battery-life
estimation, claim tests, prediction, and preference analysis — with sliders,
dropdowns and charts:

```powershell
C:\Users\omkar\diabetes_venv\Scripts\python.exe -m streamlit run app.py
```

It opens in the browser at `http://localhost:8501`. The sidebar controls the
confidence level, significance level, and lets you upload your own survey CSV
instead of the bundled sample. Close it with Ctrl+C in the terminal.

## Dataset

Each row is one surveyed unit: a tested phone plus the responding customer.

| Column | Meaning |
|--------|---------|
| `brand` | Manufacturer of the tested unit (Apex, Bolt, Cosmos, Delta) |
| `battery_life_hours` | Measured battery life |
| `charging_time_minutes` | Measured charging time |
| `processing_score` | Benchmark-style performance score |
| `customer_group` | Segment of the respondent (Student, Professional, Senior) |
| `preference` | Brand the respondent says they prefer |

The bundled file `smartphone_stats/data/sample_smartphones.csv` (240 rows) is
synthetic, generated deterministically by
`scripts/generate_sample_data.py` (seed 42). Replace it with real survey data
via `--csv`; all analyses validate the schema first.

## Statistical notes

- **Estimation** uses a two-sided Student-t interval on the sample mean; the
  level and tail shape follow the chosen confidence (or alpha) parameter.
- **Claim tests** are one-sample t-tests (`two-sided`/`greater`/`less`). The
  default per-brand advertised claims live in `claims.DEFAULT_CLAIMS`; a claim
  is "rejected" when its p-value falls below alpha.
- **Prediction** is OLS of battery life on charging time and processing score,
  with brand fixed effects. Charging time and processing score both vary
  systematically between brands, so without brand controls their slopes are
  confounded; the dummies separate within-brand from between-brand variation.
  Features are mean-centred for numerical conditioning, so the intercept is the
  adjusted mean battery life, and predictions carry a proper prediction
  interval that widens with leverage away from the data centre. Supply
  `--brand` to include a brand's fixed effect in the prediction.
- **Preference test** is a chi-square test of independence with Cramer's V as
  effect size and standardised residuals showing which cells drive the
  association. The CLI warns when any expected cell count is below 5.

## Tests

```powershell
C:\Users\omkar\diabetes_venv\Scripts\python.exe -m pytest
```

Configuration is in `pytest.ini` (`pythonpath = .`, `testpaths = tests`).

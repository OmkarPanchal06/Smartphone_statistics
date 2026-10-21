# PROJECT GUIDE — Smartphone Performance Statistics System

A complete, self-contained guide: what this project is, how it works, how to run
it on any laptop, and how to present it. Written so someone who has never seen
the code can download the folder and be running in five minutes.

---

## 1. What this project is (the story)

A technology research company wants to compare smartphone brands on four things:

1. **Battery life** — how long a phone actually lasts
2. **Charging time** — how quickly it refills
3. **Processing performance** — how fast it is (benchmark score)
4. **Customer preference** — which brand customers say they would buy

This project is a **Python statistical system** that takes survey measurements
(a tested phone + the customer who rated it) and answers five questions a
research company would actually ask:

| # | Business question | Statistical method |
|---|-------------------|--------------------|
| 1 | How do the brands compare on the raw measurements? | Descriptive statistics + one-way ANOVA |
| 2 | What is the average battery life, and how confident are we? | Point estimate + Student-t confidence interval |
| 3 | Do the measurements support what the manufacturers advertise? | One-sample t-test (per brand) |
| 4 | Can we **predict** battery life from charging time and performance? | Multiple linear regression (OLS) with brand fixed effects |
| 5 | Do customer preferences differ between customer groups (students vs professionals vs seniors)? | Chi-square test of independence + Cramér's V |

Everything is reproducible: the sample data is generated with a fixed random
seed (42), so every run on every laptop gives identical numbers.

---

## 2. What you get (deliverables)

- **`smartphone_stats/`** — the analysis library (8 modules), one per task
- **`app.py`** — an interactive **Streamlit dashboard** (the demo UI, 5 tabs with charts and sliders)
- **CLI** — `python -m smartphone_stats ...` with 6 subcommands
- **`tests/`** — 65 automated tests (all passing)
- **`smartphone_stats/data/sample_smartphones.csv`** — 240-row sample dataset (synthetic, seeded)
- **`requirements.txt`** — exact dependency list
- **`start.bat` / `start.sh`** — one-click launch (creates its own environment)

---

## 3. The dataset

Each row = **one surveyed unit**: a tested smartphone plus the responding customer.

| Column | Type | Meaning | Example |
|--------|------|---------|---------|
| `brand` | text | Manufacturer of the tested unit | `Apex`, `Bolt`, `Cosmos`, `Delta` |
| `battery_life_hours` | number | Measured battery life | `12.4` |
| `charging_time_minutes` | number | Measured charging time | `62.0` |
| `processing_score` | number | Benchmark-style performance score | `1120` |
| `customer_group` | text | Segment of the respondent | `Student`, `Professional`, `Senior` |
| `preference` | text | Brand the respondent prefers | `Cosmos` |

The bundled CSV has **240 rows** (60 per brand). It is *synthetic* — generated
by `scripts/generate_sample_data.py` with real statistical structure baked in:
battery life genuinely responds to charging time and processing score, and
customer preferences genuinely vary by group, so every statistical test has a
real effect to detect. **You can replace it with real survey data** using the
`--csv` flag or the dashboard's upload box (same columns required, validated).

---

## 4. How the analysis works (module by module)

### 4.1 `models.py` + `data.py` — data model and validation
- `Measurement` is a frozen dataclass for a single measurement; it rejects
  negative numbers, NaNs, and empty brand names at construction time.
- `validate_dataset()` checks the whole DataFrame: required columns present,
  numeric columns numeric and non-negative, text columns non-empty. Every
  analysis validates before computing, so garbage data fails loudly with a
  clear message instead of producing nonsense statistics.

### 4.2 `descriptives.py` — Question 1 (compare the brands)
- **Descriptive statistics**: per-brand count, mean, standard deviation, min,
  max for all three measurements.
- **One-way ANOVA**: tests H₀ — "all brand mean battery lives are equal".
  F = between-group variance / within-group variance. A large F with a tiny
  p-value means the brands really do differ, not just by sampling luck.

### 4.3 `estimation.py` — Question 2 (average battery life)
- Point estimate = sample mean `x̄`.
- Confidence interval (Student-t, because the population σ is unknown):

  ```
  CI = x̄ ± t(α/2, n−1) × s/√n
  ```

  where `s` is the sample standard deviation, `n` the sample size, and
  `t(α/2, n−1)` the critical value from the t-distribution.
- Works pooled ("All brands") and per brand. Higher confidence → wider interval.

### 4.4 `claims.py` — Question 3 (manufacturer claims)
- **One-sample t-test**: H₀ — "the true mean battery life equals the advertised
  claim".  `t = (x̄ − claim) / (s/√n)`.
- Supports `two-sided`, `greater`, and `less` alternatives; the reported
  confidence interval matches the chosen alternative.
- A claim is **rejected** when p < α. Default advertised claims per brand live
  in `claims.DEFAULT_CLAIMS` (Apex 12.0 h, Bolt 11.5 h, Cosmos 13.0 h, Delta 11.0 h).
- Handles the degenerate zero-variance case exactly (scipy returns NaN there).

### 4.5 `prediction.py` — Question 4 (predict battery life)
- **Multiple linear regression (OLS)** of battery life on charging time and
  processing score, **plus brand fixed effects** (drop-first dummy encoding).
- Why brand effects? Charging time and processing score both vary
  systematically *between* brands (fast-charging brands also score higher), so
  a naive pooled regression **confounds** the two effects — we caught exactly
  this bug in testing: the charging coefficient came out positive (wrong sign).
  The brand dummies separate within-brand effects from between-brand
  differences, giving the correct slopes: roughly **−0.05 h per extra charging
  minute** and **+0.008 h per performance point** on the sample data (R² ≈ 0.88).
- Features are mean-centred for numerical conditioning (the intercept becomes
  the adjusted mean battery life).
- Reports classical standard errors, t-tests on every coefficient, R², adjusted
  R², RMSE, and the overall F-test.
- **Prediction intervals** account for leverage: the interval widens for
  inputs far from the observed data — statistically proper, not just
  `mean ± 2·RMSE`.

### 4.6 `preferences.py` — Question 5 (preferences across groups)
- **Chi-square test of independence** on the customer-group × preferred-brand
  contingency table. H₀ — "preference and customer group are independent".
- Reports χ², p-value, degrees of freedom, and **Cramér's V** (effect size,
  0 = no association, 1 = perfect).
- **Standardized residuals** per cell show *which* groups drive the
  association (|z| > 2 is notable) — e.g. in the sample data, Students→Bolt has
  z = +4.7.
- Warns when any expected cell count is below 5 (chi-square rule of thumb).

---

## 5. Three ways to use the project

### 5.1 The dashboard (demo UI) — best for showing people

```powershell
streamlit run app.py        # or the one-click scripts, see below
```

Opens `http://localhost:8501` in the browser with **five tabs**: Overview,
Battery Life, Claim Tests, Prediction (interactive sliders!), Preferences.
The sidebar controls confidence level, α, and CSV upload.

### 5.2 The command line

```powershell
python -m smartphone_stats all                       # every analysis
python -m smartphone_stats describe                  # descriptives + ANOVA
python -m smartphone_stats estimate --brand Cosmos --confidence 0.99
python -m smartphone_stats claim --brand Apex --claim 12.5 --alpha 0.01
python -m smartphone_stats predict --charging 60 --score 1100 --brand Cosmos
python -m smartphone_stats preference --alpha 0.05
python -m smartphone_stats --csv my_data.csv all     # your own data
```

### 5.3 As a Python library

```python
from smartphone_stats import (
    load_dataset, mean_battery_life, test_manufacturer_claim,
    fit_battery_model, test_preference_association,
)

df = load_dataset()                                  # or your own CSV
print(mean_battery_life(df["battery_life_hours"]))   # estimate + 95% CI
model = fit_battery_model(df)
print(model.predict(60, 1100, brand="Cosmos"))       # prediction + interval
```

---

## 6. Running it on your laptop (step by step)

### 6.0 What you need

- **Python 3.10, 3.11, 3.12 or 3.13** (3.10+ required — the libraries need it).
  Check with `python --version`.
- ~500 MB of disk for Python packages.
- Internet once, to download the packages.

### 6.1 Windows — easiest (one click)

1. Put the project folder anywhere (e.g. Desktop).
2. Double-click **`start.bat`**.
3. The first run creates a private virtual environment (`.venv`) and installs
   everything (takes a few minutes, once). Then the dashboard opens in your
   browser at `http://localhost:8501`.
4. Later runs are instant. Close the dashboard by closing the black window (or
   pressing `Ctrl+C` inside it).

If double-clicking does nothing, open PowerShell **in the project folder**
(right-click the folder → "Open in Terminal" / "Open PowerShell window here")
and run:

```powershell
.\start.bat
```

If you see "execution of scripts is disabled", run once:
`Set-ExecutionPolicy -Scope Process Bypass` then `.\start.bat`.

### 6.2 macOS / Linux — one click

Open Terminal, `cd` into the folder, then:

```bash
chmod +x start.sh     # only the first time
./start.sh
```

### 6.3 Manual setup (any OS, shows what the script does)

```powershell
# 1. Go to the project folder
cd path/to/smartphone_stats_project

# 2. Create an isolated environment (once)
python -m venv .venv          # macOS/Linux: python3 -m venv .venv

# 3. Activate it
#    Windows PowerShell:  .\.venv\Scripts\Activate.ps1
#    Windows CMD:         .venv\Scripts\activate.bat
#    macOS / Linux:       source .venv/bin/activate

# 4. Install the dependencies (once)
pip install -r requirements.txt

# 5. Verify everything works (optional, ~15 seconds)
pytest

# 6. Run the dashboard
streamlit run app.py
```

Dependencies installed: `numpy`, `scipy`, `pandas` (statistics), `streamlit`
(dashboard), `pytest` (tests).

### 6.4 Verify the installation

```powershell
python -m smartphone_stats all    # prints the full analysis
pytest                            # should end with "65 passed"
```

If `all` prints four brands and pytest says 65 passed, everything works.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `TypeError: ... got an unexpected keyword argument 'alt'` | You are running an **old global Streamlit** instead of the project environment. Double-click `start.bat` (it uses its own environment), or run `pip install --upgrade streamlit` and re-run `streamlit run app.py`. The app also auto-detects old versions and disables the fancy UI bits. |
| `'python' is not recognized` | Install Python from python.org and tick **"Add Python to PATH"** during setup; or use `py -3` instead of `python` |
| `No module named smartphone_stats` | You are not in the project folder — `cd` into it first (the folder that contains `app.py`) |
| `No module named streamlit` / `numpy` | The environment isn't activated, or you skipped step 4 — rerun `pip install -r requirements.txt` |
| `Permission denied` / script disabled (Windows) | `Set-ExecutionPolicy -Scope Process Bypass` then `.\start.bat` |
| Dashboard opens but is blank | Hard-refresh with Ctrl+F5; check the terminal window for red error text |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `pytest` not found | Use `python -m pytest` instead, or activate the environment |
| Wrong Python picked (Mac) | Use `python3` / `py -3` everywhere instead of `python` |
| Uploaded CSV rejected | It must have exactly the six columns from section 3, non-negative numbers |

---

## 8. How the project is tested

`pytest` runs **65 tests** (~15 s) covering: data validation, reproducibility of
the synthetic dataset, correctness of every statistical computation against
hand calculations and SciPy reference values, edge cases (zero variance,
too-few samples, unknown brands), the CLI end-to-end, and the Streamlit
dashboard (via Streamlit's `AppTest` harness, which executes the whole UI
headlessly).

```powershell
python -m pytest        # 65 passed = everything healthy
```

Notable bugs these tests caught during development: a sign-flipped regression
coefficient (confounding, fixed with brand fixed effects), a pandas
misuse in the summary table, a numpy shape error in prediction, and NaN
p-values on zero-variance data.

---

## 9. Project structure

```
smartphone_stats_project/
├── app.py                          # Streamlit dashboard (demo UI)
├── start.bat                       # one-click launcher (Windows)
├── start.sh                        # one-click launcher (macOS/Linux)
├── requirements.txt                # dependencies
├── pytest.ini                      # test config
├── README.md                       # short overview
├── PROJECT_GUIDE.md                # this file
├── smartphone_stats/               # the analysis library
│   ├── __init__.py                 # public API
│   ├── __main__.py                 # enables `python -m smartphone_stats`
│   ├── models.py                   # Measurement dataclass
│   ├── data.py                     # load / validate / generate data
│   ├── descriptives.py             # Q1: stats + ANOVA
│   ├── estimation.py               # Q2: mean + confidence interval
│   ├── claims.py                   # Q3: manufacturer claim tests
│   ├── prediction.py               # Q4: regression + prediction
│   ├── preferences.py              # Q5: chi-square test
│   ├── cli.py                      # command-line interface
│   └── data/
│       └── sample_smartphones.csv  # 240-row sample dataset
├── scripts/
│   └── generate_sample_data.py     # regenerates the CSV (seed 42)
└── tests/                          # 65 tests
```

---

## 10. Results on the sample data (for your presentation)

- **Brand comparison**: Cosmos has the highest battery life (13.55 h), Delta the
  lowest (10.02 h). ANOVA: F = 165.8, p ≈ 8×10⁻⁵⁸ — brands differ decisively.
- **Average battery life** (pooled): 11.72 h, 95% CI [11.51, 11.93].
- **Claims**: Cosmos and Apex advertise *conservative* claims (they beat them);
  Bolt (claim 11.5 h, measured 10.98) and Delta (claim 11.0, measured 10.02)
  fall **significantly short** — the kind of finding a research company sells.
- **Prediction**: charging time −0.051 h/min, processing score +0.0079 h/point,
  R² = 0.884 — the model explains ~88% of battery-life variance.
- **Preferences**: preferences differ sharply by group, χ² = 45.3,
  p ≈ 4×10⁻⁸, Cramér's V = 0.31 (medium effect) — Students strongly favor
  Bolt, Professionals and Seniors favor Apex/Cosmos.

---

## 11. Academic honesty note

The bundled data is **synthetic** (clearly documented everywhere). The system is
designed to run on real survey data — drop in a CSV with the same six columns
via `--csv` or the dashboard upload, and every statistic, test and chart updates
automatically.

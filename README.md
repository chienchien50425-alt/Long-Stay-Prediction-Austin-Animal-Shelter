# Predicting Long-Stay Shelter Animals — Austin Animal Center (2013–2025)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange.svg">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg">
  <img src="https://img.shields.io/github/repo-size/chienchien50425-alt/austin-animal-center">
</p>

> Flagging the dogs and cats most likely to get stuck in the shelter, **on the day they
> arrive**, so staff can step in early instead of reacting weeks too late - the model catches roughly 7 to 8 out of every 10 animals that would go on to get stuck.

---

## 1. The problem

Austin Animal Center is a large open-intake municipal shelter. It has been facing constant overcapacity and prolonged kennel stays since 2021, due to a pandemic-related decline in adoptions and spay/neuter procedures. Animals that stay a long time consume kennel space, staff hours, and medical cost.

The question this project answers: using only what is known the day an animal walks in, can we predict whether it will become a long-stay (>30 days) case? A reliable early flag lets staff prioritise foster placement, targeted marketing, or behavioural support before an animal lingers.

This is framed as a binary classification task with target is_long_stay (1 = stay > 30 days).

---

## 2. The data

Everything here is built on Austin Animal Center's own public records, published on the
[City of Austin Open Data Portal](https://data.austintexas.gov/). The shelter logs two things:
every animal that comes **in** (an *intake* record) and every animal that leaves, for any
reason (an *outcome* record). This project uses the full history from **October 2013 to May
2025** — roughly 174,000 intake events and 174,000 outcome events.

After matching each departure back to the arrival record and narrowing to dogs and
cats, the analysis runs on **162,932 animal stays**.

> **Included in this repo — no download needed.** Both raw exports are committed under
> [`data/raw_dataset/`](data/raw_dataset/) (Intakes ≈ 29 MB, Outcomes ≈ 24 MB), so the
> full pipeline runs without touching the portal. **Data license: Public Domain** (City of
> Austin Open Data Portal).

---

## 3. What the project delivers

The result is an **early-warning flag**: on intake day, each dog or cat is scored for how
likely it is to become a long-stay case, using only what's actually known at that moment.

In plain terms, **the flag catches roughly 7 to 8 out of every 10 animals that would go on to
get stuck** — while they still have their best shot at a fast placement. That's the number
that matters operationally: the earlier a future long-stay animal is identified, the more
levers (foster, marketing, behavioural help) the shelter still has to pull.

The operating threshold is chosen to **maximise F1**, the point that best balances catching as many future long-stay animals as possible (recall) against the operational load of chasing false alarms (precision). At that balance point the flag happens to lean toward recall (~0.71–0.75) over precision (~0.40), which suits a triage tool: an unnecessary early foster nudge costs lower than a missed long-stay animal costs real kennel-weeks. The flag is built to rank and prioritise.

For the technically inclined, here are the underlying numbers, back-tested on a held-out
future year (2024) the models never saw during development:

| Species | Model | AUC | Precision | Recall | F1 |
|---------|-------|-----|-----------|--------|-----|
| Dog | XGBoost | 0.711 | 0.424 | 0.716 | 0.532 |
| Dog | Logistic Reg. | 0.654 | 0.395 | 0.712 | 0.508 |
| Cat | XGBoost | 0.747 | 0.419 | 0.748 | 0.537 |
| Cat | Logistic Reg. | 0.711 | 0.462 | 0.665 | 0.545 |

*Scored on the 2024 test year at the chosen threshold (to maximise F1 on the earlier 2022–2023 data).*
*Baseline was established using LR approach, while XGBoost was employed as the principal predictive model.*

| <img width="400" alt="Untitled design" src="https://github.com/user-attachments/assets/2dd803e2-433c-4653-9a99-79350efccdbb" /> | <img width="750" alt="image" src="https://github.com/user-attachments/assets/55b5bf1f-0727-4e78-9b37-1e6921526f20" /> |
| :---: | :---: |
| Figure 1. Test AUC by fold (2020–2024), per species and model. | Figure 2. Confusion matrices on the 2024 test fold, per species and model (at each model's operating threshold). |

---

## 4. How it works — data, modelling, and the judgment calls

<img width="1108" height="576" alt="Screenshot 2026-07-02 at 21 58 44" src="https://github.com/user-attachments/assets/92af5f6e-83bf-4021-8195-3bb90413878d" />

Figure 3. The five-stage data-flow pipeline, from raw Austin Open Data exports to per-species modeling

### The project runs as a five-stage data-flow pipeline

Stage 1 · Raw Data Sources — Two CSV exports from the City of Austin Open Data Portal (2013-10-01 → 2025-05-05): an Intakes file (173,812 rows, one row per intake event) and an Outcomes file (173,775 rows, one row per outcome event).  

Stage 2 · Clean & Merge (01_cleaning) — Python/pandas notebook that aligns both exports to a shared schema and backward-merges intakes with outcomes into one row per outcome event.  

Stage 3 · Processed Dataset — Writes df_full_merged.csv to disk (162,932 rows × 26 columns, filtered to dogs & cats). This single file is the source every downstream notebook reads from.  

Stage 4 · EDA (02_eda) — Exploratory analysis; its findings inform the feature choices used in modeling.  

Stage 5 · Modeling (03_modeling) — Predicts is_long_stay separately per species using Logistic Regression and XGBoost. Outputs predictions only.  

### Key judgment calls   

- Modelling dogs and cats separately.
- Each outcome is matched to its arrival with a **backward `merge_asof`** on animal ID. Every departure is joined to its *most recent prior* intake. Drop outcomes with no matching intake (~0.5%) and outcomes re-claiming the same intake.
- Long stay is derived from outcome, so *every* outcome-derived column is a leakage -> remove from features. 
- Random split would cause data leakage in time-series data, so the model is trained on data up to year $Y-1$ and tested on year $Y$, walking forward from 2020 to 2024.
  | fold | train years | test year | role |
  |---|---|---|---|
  | 1 | 2013–2019 | 2020 | test |
  | 2 | 2013–2020 | 2021 | test |
  | 3 | 2013–2021 | 2022 | **validation** (threshold decisions) |
  | 4 | 2013–2022 | 2023 | **validation** (threshold decisions + feature selection) |
  | 5 | 2013–2023 | 2024 | **reference** (test + interpretability) |

- The 2020–2022 test folds drive no decisions; they trace test AUC over time. They uncover the dog model's 2023 drop as a structural result and show the cat model stays flat across the same years.
- In 2023, the dog long_stay rate jumps structurally (Dog +0.096 from 2022) for reasons the recorded fields don't explain. Pooling 2022 (0.219) and 2023 (0.315) places the cut-point between the pre- and post-jump regimes, making it more robust for 2024 deployment. Cats show no jump but use the same rule for parity.
- Feature selection of breed and spay/neuter is decided by ablation on single 2023 fold. This holds on one year because an ablation compares AUC with vs. without a feature, and AUC is a ranking measure largely unaffected by a base-rate shift and the 2023 jump is broad-based (a uniform lift, not a structural reshuffle).
  - Keep *Cat breed*:  Mutual information against the target was **0.001**, which argues for dropping it, but mutual information misses interaction effects. Validation-fold ablation moved XGBoost AUC by **+0.0013** with breed included, effectively neutral.
  - Keep *Spay/neuter vs. age*:  These two are correlated (r ≈ 0.41 dog, 0.59 cat), raising a
  redundancy worry. Dropping the spay/neuter slightly lowered validation AUC for both species (Δ ≈ −0.003) in LR, drop will slightly hurt performance.
- Handling imbalance  
Long-stay cases are the minority class (dog ≈ 0.17-0.33, cat ≈ 0.27-0.34). Re-weighting the positive class (`class_weight='balanced'` for LR and `scale_pos_weight` for XGBoost). 

### Features — how each input is selected, encoded, or cleaned before modeling
 
| Feature handling | Dog | Cat | Where decided |
|---|---|---|---|
| Selected feature set | intake reason, breed, health condition, sex, age, `is_sn`, `is_mix` | same seven | MI / Pearson screening |
| Dropped features | colour (primary/secondary/pattern), intake month | same | Near-floor MI (≤ 0.008) for both species |
| Breed encoding | top-20 + Other | top-4 + Other | Cat breed MI ≈ 0.001, but MI misses interactions; ablation was neutral (+0.0013 AUC). Kept for feature set parity |
| `is_sn` (spay/neuter) | kept | kept | Collinearity ablation (Δ AUC ≈ −0.003) |
| Age (XGBoost) | raw `age_at_intake_days` | same | Tree model, scale-invariant |
| Age (Logistic Reg.) | 8 buckets: <2mo … 15yr+ | same | EDA |
| Health condition | keep Normal / Injured / Sick / Nursing / Neonatal, rest → Other | same | Keep the well-populated levels, merge <1000 categories into 'Other' |
| Encoder fitting | one-hot + top-N breeds, re-fit per fold on train only | same | Prevents fold-to-fold leakage |
 
### Model & validation — training, tuning, and back-test setup
 
| Setting | Dog | Cat | Where decided |
|---|---|---|---|
| Long-stay threshold (target) | > 30 days | > 30 days |  From shelter's announcement |
| Operating threshold (XGBoost) | 0.417 | 0.455 | Max-F1 on pooled 2022–2023 |
| Operating threshold (Logistic Reg.) | 0.446 | 0.517 | Same |
| Class imbalance | per-fold `scale_pos_weight` (XGB) / balanced weights (LR) | same | Recomputed per fold as base rate drifts |
| XGBoost tuning grid | depth ∈ {3, 5}, lr ∈ {0.03, 0.1}, n_estimators ≤ 800, early stop 30 | same | Nested CV (inner = last train year) |
| Logistic Reg. | L2, C = 1.0, max_iter = 2000 | same | Fixed regularised baseline |
| Back-test folds | test years 2020–2024, expanding window | same | Rolling-origin |
| Validation / test fold | feature decisions on 2023; headline on 2024 | same | 2024 never used for selection/tuning |

---

## 5. What actually drives a long stay

Drivers are read from **SHAP values on the reference-fold model** (train ≤ 2023), not impurity
importance (biased toward high-cardinality features like breed):

<img align="left" width="550" src="https://github.com/user-attachments/assets/bfd2dadc-2373-4142-9594-6994ba69f4f2" />


**Figure 4. SHAP summary for the dog XGBoost model**  

Each dot is one dog; x-position is that feature's push on the prediction (right = toward long-stay, left = toward faster exit). Color encodes the feature's value: for age, red = older; for 0/1 features (breed, owner-surrender, spay/neuter), red = positive.  

Age ranks first, and its red/blue spread on both sides is the non-monotonic effect. Pit Bull, owner-surrender and injured push toward long-stay; small popular breeds (Chihuahua, Dachshund, Miniature Schnauzer, Miniature Poodle) push the other way.

<br clear="left"/>
<br><br>

<img align="left" width="550" src="https://github.com/user-attachments/assets/87844d10-a50a-4459-9b43-c678e1f233cb" />

**Figure 5. SHAP summary for the cat XGBoost model**  

Each dot is one cat; x-position is the feature's push (right = toward long-stay, left = toward faster exit); color is the feature's value (for age, red = older; for 0/1 features, red = present).  

Age ranks first with a non-monotonic red/blue spread. Sex_Unknown: present (red) pushes strongly left; These intakes are overwhelmingly newborn kittens (median age 22 days) transferred out on day 0 (~86% transfer, ~0% adoption), likely too young to be sexed at intake, and routed straight to foster/rescue rather than entering the shelter pipeline. In contrast, Owner-surrender, nursing and injured push toward long-stay.


<br clear="left"/>
<br><br>

<img align="left" width="550" src="https://github.com/user-attachments/assets/8d738622-18d3-49cf-989c-1805516bfe98" />



**Figure 6. Long-stay rate by age at intake, dogs vs cats**  

In EDA we can see the age effect is *non-monotonic* for both dog and cat: the youngest puppies (under ~2 months) carry the highest long-stay risk (~27%), risk collapses in adolescence (~5% at 2–6 months), climbs again through prime adulthood, then falls for seniors. Very young kittens are highest-risk (~40%), and risk *rises* again into the senior years rather than falling.

<br clear="left"/>

---

## 6. Limitations

### Model & methodology

- **Precision at the operating threshold is ≈0.42 (XGBoost).**
  The thresholds maximize
  F1 on pooled 2022–2023 out-of-sample predictions, but the resulting operating point is
  recall-heavy. On the 2024 test fold, XGBoost reaches recall 0.716 (dog) / 0.748 (cat)
  at precision 0.424 / 0.419. In practice, roughly 2 of every 5 flagged animals actually
  become long-stays, and for a shelter with limited staff the false positives are a real
  operational cost. Since shelter capacity was unknown when the threshold was set, *recall
  and precision can be re-balanced along the PR curve to match actual operational needs.*

- **Predicted probabilities systematically overstate long-stay risk.**
  On the pooled
  reliability curves, every point for both species lies below the diagonal: e.g., dogs
  scored around 0.8 by XGBoost actually long-stay at only ~52%, and cats scored around
  0.8 at ~67%. This is a known consequence of per-fold class weighting, which inflates
  minority-class probabilities. The curves remain monotonic, so ranking (AUC) and
  thresholded flags are unaffected, but the raw scores must not be read as literal
  probabilities.

- **The 2023 dog regime shift is unexplained, and the base rate keeps drifting.**  
  The dog long-stay rate jumped structurally in 2023 and dog XGBoost AUC dropped from 0.766
  to 0.694; none of the recorded intake fields explain the shift, so the dog model is
  inherently less trustworthy than the cat model. More broadly, the long-stay base rate
  drifts for both species (24.6% in 2022, 30.7% in 2023, 29.7% in 2024). The pipeline has
  no drift detection, so a future shift of the same kind would silently degrade performance.

- **Only intake-day information is used (7 features per species).** No behavioral
  assessments, photos, or any post-intake signal. This is by design (the model must
  score an animal on day one), but it likely sets the AUC ceiling at ~0.70–0.75.

- **Interpretability is correlational, from a single fit.** SHAP values (and LR
  coefficients) describe association, not causation, e.g., owner surrender predicting
  long stays does not mean intervening on surrenders would shorten them.

### Data & scope

- **Record linkage is assumption-based.** Each outcome is matched to the nearest prior
  intake via `merge_asof`; unmatched outcomes (924 rows, 0.53%), outcomes re-claiming
  an already-used intake (274 rows), duplicates, and impossible dates were dropped.
  Total losses are under 1%, but mismatches cannot be fully ruled out.

- **Intake status does not guarantee adoption eligibility.** Per the Austin Animal
  Center metadata, intake records represent the status of animals as they arrive at
  the shelter; the data does not indicate whether an animal was ever eligible for
  adoption. Some long stays may therefore reflect legal holds or ineligibility rather
  than low adoption appeal, and the model cannot distinguish the two.

- **Scope: one shelter, two species.** All data come from Austin Animal Center and only
  dogs and cats are modeled. Nothing here has been tested for transfer to other shelters,
  regions, or species.
---

## 7. Future work

- **Calibrated probabilities** — *addresses "predicted probabilities systematically overstate long-stay risk".*  
  The per-fold class weighting that fixes imbalance
  also inflates minority-class scores, so raw outputs must not be read as literal
  probabilities. If actual probability scores are ever needed, fitting an isotonic or Platt calibrator on a held-out fold would let
  the scores be interpreted as true probabilities.

- **Drift monitoring and threshold re-tuning** — *addresses "no drift detection, and the base rate keeps drifting".*  
  The long-stay base rate moves year to year
  (24.6% → 30.7% → 29.7%), and the pipeline currently has no way to notice. Adding a
  base-rate / PSI drift check on incoming data, plus a scheduled re-tune of the
  operating threshold, would stop a future regime shift from silently degrading the
  flag.

- **Post-intake signals** — *addresses "only intake-day information is used".*
  The model deliberately scores an animal on day one, which likely caps AUC at
  ~0.70–0.75. If later signals (behavioral assessments, medical updates, photos)
  were incorporated as they arrive, a second-stage model could refine the day-one
  flag for animals still in the shelter.

## 8. Lessons learned

- **I didn't fully understand the business problem in the beginning**  
My initial instinct was to predict whether an animal would be adopted. Only when writing the report did I realize this missed the shelter's real pain point: limited space and capacity, where the true strain comes from animals that stay stuck for a long time. I therefore redefined the target from is_adopted to is_long_stay, shifting the focus from "Will this animal be adopted?" to "Will it occupy space long-term, so staff can intervene early?


- **Changing the business problem forces the data-cleaning logic to be re-audited.**  
When I switched the target from is_adopted to is_long_stay, I kept the original data-cleaning pipeline unchanged, and that inherited logic silently biased the dataset. Because the merge was keyed on outcomes, any animal still in the shelter, which has no outcome yet — was dropped entirely, and those are exactly the longest-staying animals the new target was meant to find. I recognized the problem and added back 467 still-in-shelter animals as confirmed long stays and restored the population the inherited merge had removed.

---

## 9. Pipeline & repository

```
01_cleaning.ipynb  →  02_eda.ipynb  →  03_modeling.ipynb
```

| Notebook | What it does |
|----------|--------------|
| **01_cleaning** | Aligns both raw exports to a shared snake_case schema; normalises mixed-timezone dates; backward `merge_asof` join of each outcome to its most recent prior intake; drops orphans/duplicates; parses sex/breed/colour; engineers the `is_long_stay` target and intake-time features; filters to dogs & cats → `df_full_merged.csv` (162,932 rows). |
| **02_eda** | Per-species univariate, bivariate (long-stay rate by feature), and temporal analysis; correlation heatmaps and mutual-information screening against the target. EDA only — no modelling. |
| **03_modeling** | Time-aware folds; per-species Logistic Regression + XGBoost; feature ablations on the validation fold; back-test on 2024; threshold selection; confusion matrices; SHAP interpretation. |

```
.
├── data/
│   ├── raw_dataset/        # two raw Austin exports, committed
│   │                       #   (filenames carry a _20260523 date stamp)
│   └── processed/
│       └── df_full_merged.csv   # cleaned, merged, dog/cat-only (162,932 rows)
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── LICENSE                 # MIT License (covers the code)
├── README.md
└── requirements.txt        # pinned dependencies (Python 3.13)
```

**Reproducing:**

No data download is required — both raw CSVs are already committed under
[`data/raw_dataset/`](data/raw_dataset/) (**license: Public Domain**, City of Austin Open
Data Portal). Install the dependencies and run the notebooks in order:

```bash
pip install -r requirements.txt   # or: pandas numpy scikit-learn xgboost shap matplotlib seaborn
# run notebooks in order: 01 → 02 → 03
# 01 writes data/processed/df_full_merged.csv, which 02 and 03 both read.
```

> ⚠️ **Filename convention — don't rename these.** `01_cleaning.ipynb` reads the two raw
> files by their **exact paths** via `FULL_INTAKES_PATH` / `FULL_OUTCOMES_PATH`. Each filename carries a
> **date-stamp suffix** `_20260523`. If you re-export fresh data from the portal, that date
> stamp will differ and the notebook will fail to find the file — either rename the new
> export to match, or edit those two path variables in `01_cleaning.ipynb`.

### Reproducibility

The scope here is deliberately honest: results reproduce in **ranking and metrics** on the
pinned environment, not guaranteed bit-for-bit across machines.

- **Seed = 42.** `03_modeling.ipynb` fixes `RANDOM_STATE = 42` (and `np.random.seed(42)`),
  and passes it to XGBoost — both the inner-CV models and the final refit — to the SHAP
  background sampling, and to the mutual-information screening in `02_eda.ipynb`
  (`mutual_info_classif(..., random_state=42)`). Those steps are reproducible run-to-run.
- **LogisticRegression is *not* explicitly seeded.** It is built as
  `LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')` with no `random_state`.
  Its default `lbfgs` solver is deterministic, so results are stable.
- **Numerical reproducibility depends on the pinned environment** — Python 3.13,
  `pandas==3.0.0`, `numpy==2.4.1` (`03_modeling` was run locally on this stack). Cleaning was
  done on **pandas 3.0**, and pandas changed `merge_asof` and timezone-parsing behaviour
  across major versions, so running the cleaning step on **pandas 2.x may yield a different
  post-merge row count**, which then propagates downstream.
- **Multi-threaded XGBoost.** XGBoost runs with `n_jobs=4` and `tree_method='hist'`.
  Multi-threaded floating-point summation is not guaranteed bit-for-bit identical across
  hardware, so metrics and rankings reproduce but the exact digits may differ machine to
  machine.

---

## License & citation

- **Code:** MIT License — see [LICENSE](LICENSE). Original work by Wei Ling Chien.
- **Data:** City of Austin Open Data Portal — Animal Center *Intakes* & *Outcomes*,
  **Public Domain**. Retrieved 2026-05-23. The data license is separate from the code
  license: the code is MIT, the underlying data is public domain and not covered by MIT.

**How to cite:**

```
Wei Ling Chien (2026). Predicting Long-Stay Shelter Animals — Austin Animal Center.
GitHub: https://github.com/chienchien50425-alt/austin-animal-center
```

**Acknowledgements:** City of Austin and Austin Animal Center for publishing the open
intake and outcome data that makes this project possible.

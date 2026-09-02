# Predicting Long-Stay Shelter Animals — Austin Animal Center (2013–2025)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange.svg">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg">
  <img src="https://img.shields.io/github/repo-size/chienchien50425-alt/austin-animal-center">
</p>

> Flagging the dogs and cats most likely to get stuck in the shelter, **on the day they
> arrive**, so staff can step in early instead of reacting weeks too late - flag the arrivals whose risk score lands in the top 30% of the last 90 days, and you catch about **half** the animals that go on to get stuck, with about half the flagged list turning out to be real.

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

After matching each arrival and departure record and narrowing to dogs and
cats, the analysis runs on **162,932 animal stays**.

> **Included in this repo, no download needed.** Both raw exports are committed under
> [`data/raw_dataset/`](data/raw_dataset/) (Intakes ≈ 29 MB, Outcomes ≈ 24 MB), so the
> full pipeline runs without touching the portal. **Data license: Public Domain** (City of
> Austin Open Data Portal).

---

## 3. What the project delivers

The result is an **early-warning flag**: on intake day, each dog or cat is scored for how
likely it is to become a long-stay case, using only what's actually known at that moment.

**How the System Works in Practice:**

- Built for Real-World Capacity: Shelters have limited resource. Instead of setting an unrealistic goal to capture every long-stay animal, the system flags the top 30% most at-risk animals.

- Why 30%? By keeping the alert list to this size, it remains manageable for staff. At this setting, the system successfully catches about half of the pets who will genuinely end up needing long-term help, and about half the pets on the list will truly be long-stays.

- Always Adapting: The system compares new arrivals to the shelter's population from the past 90 days, rather than using a rigid cutoff from the previous year. 

- An Adjustable Dial: The 30% threshold isn't permanent. If a shelter has the capacity to intervene more often, they can widen the net or vice versa.

For the technically inclined, here are the underlying numbers, tested on a held-out future year (2024):

| Species | Model | AUC | Flagged | Precision | Recall | F1 |
|---------|-------|-----|---------|-----------|--------|-----|
| Dog | XGBoost | 0.747 | 29.2% | 0.517 | 0.511 | 0.514 |
| Dog | Logistic Reg. | 0.681 | 26.6% | 0.436 | 0.394 | 0.414 |
| Cat | XGBoost | 0.758 | 32.5% | 0.513 | 0.560 | 0.536 |
| Cat | Logistic Reg. | 0.712 | 32.8% | 0.481 | 0.530 | 0.504 |

 
*LR is the baseline; XGBoost is the principal model and wins on every metric for both species.*  

Across the full five-fold back-test — each year predicted by a model trained only on the years before it:

| Species / model | 2020 | 2021 | 2022 | 2023 | 2024 | 
|---|---|---|---|---|---|
| Dog · XGBoost | 0.791 | 0.796 | 0.786 | 0.727 | 0.747 |
| Dog · Logistic Reg. | 0.774 | 0.779 | 0.756 | 0.672 | 0.681 |
| Cat · XGBoost | 0.744 | 0.740 | 0.741 | 0.749 | 0.758 |
| Cat · Logistic Reg. | 0.695 | 0.690 | 0.698 | 0.711 | 0.712 |

| <img width="250" alt="Test AUC by fold" src="reports/figures/fig1_auc_by_fold.png" /> | <img width="600" alt="Confusion matrices, 2024 test fold" src="reports/figures/fig2_confusion_2024.png" /> |
| :---: | :---: |
| Figure 1. Test AUC by fold (2020–2024), per species and model. | Figure 2. Confusion matrices on the 2024 test fold, at the rolling 90-day operating point. |

---

## 4. How it works — data, modelling, and the judgment calls

<img width="1108" height="576" alt="Screenshot 2026-07-02 at 21 58 44" src="https://github.com/user-attachments/assets/92af5f6e-83bf-4021-8195-3bb90413878d" />

Figure 3. The five-stage data-flow pipeline, from raw Austin Open Data exports to per-species modeling

### The project runs as a five-stage data-flow pipeline

Stage 1 · Raw Data Sources — Two CSV exports from the City of Austin Open Data Portal (2013-10-01 → 2025-05-05): an Intakes file (173,812 rows, one row per intake event) and an Outcomes file (173,775 rows, one row per outcome event).  

Stage 2 · Clean & Merge (01_cleaning) — Python/pandas notebook that aligns both exports to a shared schema and backward-merges each outcome to its most recent prior intake, producing one row per completed animal stay. It then adds 467 still-in-shelter animals that have no outcome yet but whose elapsed stay already exceeds 30 days, labeled positive.

Stage 3 · Processed Dataset — Writes df_full_merged.csv to disk (162,932 rows × 27 columns, filtered to dogs & cats). This single file is the source every downstream notebook reads from.  

Stage 4 · EDA (02_eda) — Exploratory analysis; its findings inform the feature choices used in modeling.  

Stage 5 · Modeling (03_modeling) — Predicts is_long_stay separately per species using Logistic Regression and XGBoost, and exports the headline XGBoost models to `models/`.  

### Key judgment calls   

- Modelling dogs and cats separately.
- Each outcome is matched to its arrival with a **backward `merge_asof`** on animal ID. Every departure is joined to its *most recent prior* intake. Drop outcomes with no matching intake (~0.5%) and outcomes re-claiming the same intake.
- Long stay is derived from outcome, so *every* outcome-derived column is a leakage -> remove from features. 
- Random split would cause data leakage in time-series data, so the model is trained on data up to year $Y-1$ and tested on year $Y$, walking forward.
  | fold | train years | test year | role |
  |---|---|---|---|
  | 1 | 2013–2019 | 2020 | trace only |
  | 2 | 2013–2020 | 2021 | trace only |
  | 3 | 2013–2021 | 2022 | **decision** (feature ablations) |
  | 4 | 2013–2022 | 2023 | **decision** (feature ablations) |
  | 5 | 2013–2023 | 2024 | **test** |

- Handling imbalance  
Long-stay cases are the minority class (by intake year over the full years 2014–2024, dog 0.11–0.32, cat 0.17–0.34). Re-weighting the positive class (`class_weight='balanced'` for LR and `scale_pos_weight` for XGBoost). 

### Features — how each input is selected, encoded, or cleaned before modeling
 
| Feature handling | Dog | Cat | Where decided |
|---|---|---|---|
| Selected feature set | intake reason, breed, **`breed_size`**, health condition, sex, age, `is_sn`, `is_mix`, intake month, intake year (10) | same minus `breed_size` (9) | MI screening, then back-test ablation |
| `breed_size` (small <25 lbs / big) | **added**, −0.002 / +0.007 / +0.009 AUC on the 2022 / 2023 / 2024 folds | **not used** | External domain knowledge |
| Breed encoding | top-**60** + Other | top-4 + Other | cat top 4 cover ~90% population |
| Age (XGBoost) | raw `age_at_intake_days` | same |  |
| Age (Logistic Reg.) | 8 buckets: <2mo … 15yr+ | same | EDA |
| Health condition | keep Normal / Injured / Sick / Nursing / Neonatal, rest → Other | same | Keep the well-populated levels, merge <1000 categories into 'Other' |
| Encoder fitting | one-hot + top-N breeds, re-fit per fold on train only | same |  |
 
### Model & validation — training, tuning, and back-test setup
 
| Setting | Dog | Cat |
|---|---|---|
| Long-stay threshold (target) | > 30 days | > 30 days |
| Operating point | top **30%** of the trailing **90 days** | same |
| Class imbalance | per-fold `scale_pos_weight` (XGB) / balanced weights (LR) | same |
| XGBoost tuning grid | depth ∈ {3, 5}, lr ∈ {0.03, 0.1}, n_estimators ≤ 800, early stop 30 | same |
| Logistic Reg. | L2, C = 1.0, max_iter = 2000 | same |
| Back-test folds | test years 2020–2024 | same | Rolling-origin |
| Validation / test fold | feature decisions on 2022 **and** 2023; the operating point is a policy rule (top 30% / 90 days); headline on 2024 | same | 2024 never used for selection/tuning |

---

## 5. What actually drives a long stay

Drivers are read from **SHAP values on the reference-fold model** (train ≤ 2023), not impurity
importance (biased toward high-cardinality features like breed):

<img align="left" width="550" src="reports/figures/fig4_shap_dog.png" />


**Figure 4. SHAP summary for the dog XGBoost model**  

Each dot is one dog; x-position is that feature's push on the prediction (right = toward long-stay, left = toward faster exit). Color encodes the feature's value: for age, red = older; for 0/1 features (breed, owner-surrender, spay/neuter), red = positive.  

Age ranks first, and its red/blue spread on both sides is the non-monotonic effect. `breed_size_small` is the clearest single band in the plot, red (small) sits entirely on the left, pushing toward a fast exit. Owner-surrender, injured and Pit Bull push toward long-stay.

<br clear="left"/>
<br><br>

<img align="left" width="550" src="reports/figures/fig5_shap_cat.png" />

**Figure 5. SHAP summary for the cat XGBoost model**  

Each dot is one cat; x-position is the feature's push (right = toward long-stay, left = toward faster exit); color is the feature's value (for age, red = older; for 0/1 features, red = present).  

Age ranks first with a non-monotonic red/blue spread. Sex_Unknown is the longest tail in the plot: present (red) pushes strongly left. These intakes are overwhelmingly newborn kittens (median age 22 days) transferred out on day 0 (~86% transfer, ~0% adoption), likely too young to be sexed at intake and routed straight to foster/rescue rather than entering the shelter pipeline. In contrast, owner-surrender, nursing and injured push toward long-stay.


<br clear="left"/>
<br><br>


## 6. 2023 AUC Drop Deep Dive

  1. **Is the model outdated?** → 76% of the decline is because the underlying shelter data actually became harder to predict (real signal loss). Proved this by comparing two different testing methods:
  - The Standard Test (Walk-Forward): When we train a model on past data to predict the future, we saw a total performance drop of 0.059 between 2022 and 2023.
  - The Up-to-Date Test (In-Year Oracle): Even with the perfectly up-to-date "In-Year Oracle" model, performance still dropped by 0.045 (from 0.818 in 2022 to 0.773 in 2023).
  2. **What got worse?  The dogs the model called safe stopped being safe.** Scoring both years with one model (train ≤ 2021), the mean score for long-stay dogs fell 0.630 → 0.582 while fast dogs rose 0.380 → 0.409, narrowing the gap AUC measures +0.250 → +0.173. In the lowest-risk quartile the actual long-stay rate x4 (3.0% → 12.5%), while the top quartile barely moved (48.4% → 51.0%).
  3. **Which way out got slower? → Adoption** (table below). Adoption went from 14 days to 27, but Transfer got faster (7 → 4) and Return to Owner barely moved.
  5. **Is it a big-dog problem?** → Adoption wait times doubled equally for all dogs: 16 to 30 days for large dogs and 7 to 13 days for small dogs. This universal slowdown pushed far more dogs to the 30-day boundary, degrading the model's accuracy because the difference between a fast exit and a long stay now often comes down to unpredictable daily luck.
  6. **What is still unexplained?** → **Why adoption slowed.** Not answerable from this export: no adopter counts, foot traffic, listing dates or policy records.

  **Median days to exit, dogs, by outcome**:

  | outcome | 2021 | 2022 | **2023** | 2024 | Δ 2022→2023 | 2023 share of exits |
  |---|---|---|---|---|---|---|
  | **Adoption** | 11 | 14 | **27** | 18 | **+13 d** | 61.1% |
  | **Transfer** | 6 | 7 | **4** | 4 | **−3 d** | 21.1% |
  | Return to Owner | 1 | 1 | 2 | 2 | +1 d | 13.6% |
  | Rto-Adopt | 7 | 10 | 12 | 8 | +2 d | 1.6% |
  | Euthanasia | 4 | 4 | 7.5 | 8 | +3.5 d | 1.5% |

## 7. Limitations

### Model & methodology

- **The 2023 dog break is diagnosed but not fully explained.** Appendix 1 narrows it to an
  adoption-pathway slowdown and rules out a general shelter jam, but *why* adoption slowed is not answerable from this export. The dog model remains less stable than the cat model.

- **Only intake-day information is used (10 features for dogs, 9 for cats).** No behavioral
  assessments, photos, or any post-intake signal. This is by design (the model must
  score an animal on day one), but it likely sets the AUC ceiling at ~0.75–0.80.


### Data & scope

- **System migration limits scope to pre-May 2025 legacy data**
This project utilizes historical CSV extracts from the shelter's legacy system, with records concluding on May 5, 2025. The shelter has since migrated to a new platform that employs a different identification schema for animals, intakes, and outcomes. Because the internal mapping logic required to reconcile the legacy IDs with the new structure is unavailable, recent records cannot be ingested. Consequently, this analysis is restricted to the historical dataset.

- **Intake status does not guarantee adoption eligibility.**  
Per the Austin Animal
  Center metadata, intake records represent the status of animals as they arrive at
  the shelter; the data does not indicate whether an animal was ever eligible for
  adoption. Some long stays may therefore reflect legal holds or ineligibility rather
  than low adoption appeal, and the model cannot distinguish the two.


## 8. Future work

- **Integrating post-migration data**  
Building a data bridge between the legacy and modern systems. By developing a crosswalk logic to reconcile the differing ID schemas, integrate post-migration records, update our ETL pipeline, and transition this historical analysis into a real-time predictive tool.

- **Post-intake signals** — *to addresses "only intake-day information is used".*  
  The model deliberately scores an animal on day one, which likely caps AUC at
  ~0.75–0.80. If later signals (behavioral assessments, medical updates, photos)
  were incorporated as they arrive, a second-stage model could refine the day-one
  flag for animals still in the shelter.

- **Adoption-side data** — *addresses the one gap Appendix 1 could not close.*  
  The 2023 break traces to the adoption pathway specifically, but this export holds no adopter
  counts, foot traffic, time-to-listing or adoption-event calendars. Those fields are what would
  turn "adoption slowed" into an explanation, and they would likely absorb the drift the model
  currently cannot see.

## 9. Lessons learned

- **I didn't fully understand the business problem in the beginning**  
My initial instinct was to predict whether an animal would be adopted. Only when writing the report did I realize this missed the shelter's real pain point: limited space and capacity, where the true strain comes from animals that stay stuck for a long time. I therefore redefined the target from is_adopted to is_long_stay, shifting the focus from "Will this animal be adopted?" to "Will it occupy space long-term, so staff can intervene early?"


- **Changing the business problem forces the data-cleaning logic to be re-audited.**  
When I switched the target from is_adopted to is_long_stay, I kept the original data-cleaning pipeline unchanged, and that inherited logic silently biased the dataset. Because the merge was keyed on outcomes, any animal still in the shelter, which has no outcome yet, was dropped entirely, and those are exactly the longest-staying animals the new target was meant to find. I recognized the problem and added back 467 still-in-shelter animals as confirmed long stays and restored the population the inherited merge had removed.

---

## 10. Pipeline & repository

```
01_cleaning.ipynb  →  02_eda.ipynb  →  03_modeling.ipynb
```

| Notebook | What it does |
|----------|--------------|
| **01_cleaning** | Audits every raw column as drop-down vs free text (which is how `Breed` was caught as free text carrying 236 colour values); aligns both raw exports to a shared snake_case schema; normalises mixed-timezone dates; backward `merge_asof` join of each outcome to its most recent prior intake; drops orphans/duplicates; parses sex/breed/colour; engineers the `is_long_stay` target and intake-time features; filters to dogs & cats → `df_full_merged.csv` (162,932 rows). |
| **02_eda** | Per-species univariate, bivariate (long-stay rate by feature), and temporal analysis; correlation heatmaps and mutual-information screening against the target. EDA only — no modelling. |
| **03_modeling** | Time-aware folds; per-species Logistic Regression + XGBoost; feature ablations on both decision folds (2022, 2023); five-fold back-test with the headline on 2024; `FLAG_RATE` operating point; confusion matrices; SHAP interpretation; Appendix 1's diagnosis of the 2023 dog break; exports the headline models to `models/`. |


```
.
├── data/
│   ├── raw_dataset/        # two raw Austin exports, committed
│   │                       #   (filenames carry a _20260523 date stamp)
│   └── processed/
│       └── df_full_merged.csv   # generated by 01_cleaning — gitignored, not committed
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── LICENSE                 
├── README.md
└── requirements.txt        # pinned dependencies (Python 3.13)
```

---

**Acknowledgements:** City of Austin and Austin Animal Center for publishing the open
intake and outcome data that makes this project possible.

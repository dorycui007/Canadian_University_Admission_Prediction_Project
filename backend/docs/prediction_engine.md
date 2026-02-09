# How the Prediction Engine Works

A complete technical walkthrough of how raw student application data becomes a calibrated admission probability. This document traces every step of the pipeline, from CSV ingestion to the final API response.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Ingestion and Normalization](#2-data-ingestion-and-normalization)
3. [Feature Engineering](#3-feature-engineering)
4. [The Design Matrix](#4-the-design-matrix)
5. [Math Foundation](#5-math-foundation)
6. [Model Layer](#6-model-layer)
7. [Evaluation and Calibration](#7-evaluation-and-calibration)
8. [API Serving](#8-api-serving)
9. [End-to-End Example](#9-end-to-end-example)

---

## 1. System Overview

The prediction engine is a six-layer pipeline that transforms a student's application into a calibrated admission probability:

```
Raw CSV Data
    |
    v
[1. NORMALIZE]  Fuzzy-match university/program names to canonical forms
    |
    v
[2. ENCODE]     Transform categorical + numeric fields into a numeric matrix X
    |
    v
[3. TRAIN]      Fit models using from-scratch linear algebra (QR, Ridge, IRLS)
    |
    v
[4. EVALUATE]   Measure calibration (Brier, ECE) and discrimination (AUC, PR)
    |
    v
[5. CALIBRATE]  Apply Platt scaling so predicted probabilities are trustworthy
    |
    v
[6. SERVE]      FastAPI endpoints return probability + explanation + uncertainty
```

### Source Files Map

| Layer | Files | Purpose |
|-------|-------|---------|
| Normalize | `src/utils/normalize.py` | RapidFuzz fuzzy matching for names |
| Encode | `src/features/encoders.py` | 8 domain-specific encoder classes |
| Design Matrix | `src/features/design_matrix.py` | Generic transformers + builder pattern |
| Math | `src/math/vectors.py`, `matrices.py`, `projections.py`, `qr.py`, `svd.py`, `ridge.py` | From-scratch linear algebra |
| Models | `src/models/baseline.py`, `logistic.py`, `hazard.py`, `embeddings.py`, `attention.py` | 4 model types + attention |
| Evaluation | `src/evaluation/calibration.py`, `discrimination.py`, `validation.py` | Scoring, calibration, cross-validation |
| API | `src/api/predictor.py` | FastAPI with prediction pipeline |

---

## 2. Data Ingestion and Normalization

### 2.1 Raw Data

The system ingests CSV files containing self-reported Canadian university admission results:

```
data/raw/
  2022_2023_Canadian_University_Results.csv
  2023_2024_Canadian_University_Results.csv
  2024_2025_Canadian_University_Results.csv
```

Each row is one application with fields like:
- Student's top-6 average, grade 11 average, grade 12 average
- University name (free-text, often with typos)
- Program name (free-text, many formats)
- Decision outcome (Accepted / Rejected / Waitlisted / Deferred)
- Province, application year

### 2.2 University Normalization

University names come in many forms: "UofT", "U of T", "University of Toronto", "toronto". The normalization module (`normalize.py`) resolves these to canonical names using a two-stage process:

**Stage 1: Exact Match**
Check the input against a dictionary of known aliases.

**Stage 2: Fuzzy Match (RapidFuzz)**
If no exact match, compute similarity scores against all known university names using RapidFuzz's `token_sort_ratio` scorer. Accept matches above a threshold of 85.

```
"UOFt" --> fuzzy match --> score 91 --> "University of Toronto"  (accepted)
"Ontario" --> fuzzy match --> score 62 --> no match               (rejected)
```

Key design decisions:
- **UofT Campuses**: Kept as 3 separate entries (St. George, Mississauga, Scarborough) because they have separate OUAC admission categories and different requirements.
- **Toronto Metropolitan University**: All "Ryerson" references map to the new name.
- **Threshold 85**: Tested to catch typos while avoiding false positives.

### 2.3 Program Normalization

Programs are more complex. A raw entry like `"BSc Honours: Computer Science (Co-op)"` is parsed into structured components:

```
Input:  "BSc Honours: Computer Science (Co-op)"
          |
          v
   Extract Components:
     degree:         BSc
     honours:        True
     coop:           True
     base_program:   Computer Science
          |
          v
   Fuzzy-match base_program against data/mappings/base_programs.yaml
          |
          v
Output: "Computer Science | BSc Honours | Co-op"
```

The mapping files in `data/mappings/` provide canonical forms:
- `universities.yaml` -- canonical university names
- `base_programs.yaml` -- canonical program names
- `decisions.yaml` -- decision outcome mappings
- `degree_abbreviations.yaml` -- degree format standardization

---

## 3. Feature Engineering

Once names are normalized, the feature engineering layer transforms raw application fields into numeric representations suitable for models.

### 3.1 Domain-Specific Encoders

The `encoders.py` module provides 8 specialized encoders (`src/features/encoders.py`):

| Encoder | Input | Output | Method |
|---------|-------|--------|--------|
| **GPAEncoder** | Top-6 average (e.g. 92.5) | Normalized score, GPA bucket flags | Z-score normalization, percentile binning |
| **UniversityEncoder** | "University of Toronto" | Binary indicators per university | One-hot or dummy encoding |
| **ProgramEncoder** | "Computer Science" | Binary indicators per program cluster | Clustered encoding to reduce dimensionality |
| **TermEncoder** | "Fall 2023" | Cyclical term features | Sin/cos encoding for cyclical nature |
| **DateEncoder** | Application/decision dates | Days elapsed, month indicators | Temporal feature extraction |
| **FrequencyEncoder** | Category name | Frequency of that category in training data | Target-independent frequency |
| **WOEEncoder** | Category name | Weight of Evidence score | `ln(% events / % non-events)` per category |
| **CompositeEncoder** | Multiple fields | Combined feature vector | Chains multiple encoders |

**Example: GPAEncoder**

```
Input: top_6_average = 92.5

Step 1 - Z-score: (92.5 - 87.3) / 4.2 = 1.24  (above average by 1.24 std devs)
Step 2 - Bucket:  92.5 falls in [90, 95) bucket -> is_90_95 = 1
Step 3 - Output:  [1.24, 0, 0, 0, 1, 0]
                   zscore  <80  80-85  85-90  90-95  95+
```

### 3.2 Convenience Function

The `create_admission_encoders()` factory function creates a standard set of all encoders configured for admission prediction. The `encode_admission_features()` function applies them in batch to a list of application dictionaries.

---

## 4. The Design Matrix

The design matrix module (`src/features/design_matrix.py`) is the bridge between raw features and the numeric matrix `X` that models consume. It uses a **builder pattern** to orchestrate transformations.

### 4.1 Architecture

```
  Raw application dicts
          |
          v
  DesignMatrixBuilder.fit(data)     <-- learns statistics from training data
          |
          v
  DesignMatrixBuilder.transform(data)  <-- applies learned transforms
          |
          v
  Numeric matrix X (n_samples x n_features) + feature name list
```

### 4.2 Core Transformers

The module provides 4 generic transformer classes, each implementing `fit()`, `transform()`, and `get_feature_names()`:

#### NumericScaler
Standardizes continuous features using z-score or min-max scaling.

```
Z-score:   x_scaled = (x - mean) / std
Min-max:   x_scaled = (x - min) / (max - min)
```

Handles edge cases: if `std = 0` (constant feature), outputs 0.0 to avoid division by zero.

#### OneHotEncoder
Converts categorical variables into binary indicator columns.

```
Input:  university = "UofT"
        categories learned: ["McMaster", "McGill", "UofT", "Waterloo"]

Output: [0, 0, 1, 0]
         McM  McG  UofT  Wat
```

Handles unknown categories with three strategies: `error`, `ignore`, or `encode` (adds an `<unknown>` column).

#### DummyEncoder
Like OneHotEncoder but drops the first (or specified) category to avoid multicollinearity. Essential for linear models where full one-hot creates a rank-deficient matrix.

```
Input:  university = "UofT"
        categories: ["McMaster", "McGill", "UofT", "Waterloo"]
        dropped: "McMaster" (reference category)

Output: [0, 1, 0]    <-- only 3 columns, McMaster is the reference
         McG  UofT  Wat
```

Interpretation: coefficients are relative to the dropped category.

#### OrdinalEncoder
Encodes ordered categories as integers preserving the ordering.

```
Input:  average_bucket = "90-95"
        order: ["<80", "80-85", "85-90", "90-95", "95+"]

Output: 3    (0-indexed position)
```

### 4.3 InteractionBuilder

Creates multiplicative interactions and polynomial features:

```
Multiplicative:  x_avg * x_is_competitive_program
                 (high average matters more for competitive programs)

Polynomial:      x_avg, x_avg^2, x_avg^3
                 (captures non-linear GPA effects)
```

### 4.4 DesignMatrixBuilder (Orchestrator)

The main builder class coordinates everything:

```python
config = DesignMatrixConfig(
    numeric_features=["top_6_average", "grade_11_average"],
    categorical_features={"university": "dummy", "program": "onehot"},
    interaction_specs=[InteractionSpec("top_6_average", "university", "multiply")],
    add_intercept=True
)

builder = DesignMatrixBuilder(config)
builder.fit(training_data)
X_train = builder.transform(training_data)    # shape: (n, p)
X_test  = builder.transform(test_data)        # same p columns

feature_names = builder.get_feature_names()
# ["intercept", "top_6_average", "grade_11_average", "university_McGill", ...]
```

**fit()** learns:
- Mean/std for numeric features
- Category sets for categorical features
- Interaction structures

**transform()** applies:
1. Extract each feature from input dicts
2. Handle missing values (median for numeric, mode for categorical)
3. Apply fitted transformers
4. Build interaction columns
5. Optionally prepend intercept column
6. Return stacked numpy matrix

### 4.5 Validation Utilities

After building the design matrix, validation functions check for problems:

| Function | What it checks |
|----------|---------------|
| `validate_design_matrix()` | NaN/Inf values, constant columns, rank, condition number, outliers |
| `check_column_rank()` | Uses SVD to detect rank deficiency (tolerance-based) |
| `identify_collinear_features()` | Finds pairs of features with correlation above threshold |

The condition number `kappa(X) = sigma_max / sigma_min` measures numerical stability. Note that solving the normal equations squares this: `kappa(X^T X) = kappa(X)^2`. If `kappa(X) > 10^3` (i.e., `kappa(X^T X) > 10^6`), ridge regularization is essential.

---

## 5. Math Foundation

The math modules implement core algorithms with from-scratch implementations where pedagogically valuable (e.g., Householder QR), and use numpy.linalg routines for production stability elsewhere. The modules build on each other in a conceptual dependency chain:

```
vectors.py  -->  matrices.py  -->  projections.py  -->  qr.py  -->  svd.py
                                                          |
                                                          v
                                                       ridge.py
                                                          |
                                                          v
                                                      logistic.py (IRLS)
```

> **Note**: This chain represents conceptual mathematical dependencies, not Python import dependencies. Each module imports numpy directly. The Householder QR factorization in `qr.py` is implemented from scratch; other modules leverage `numpy.linalg` routines (solve, svd, qr) for numerical stability and performance.

### 5.1 Vectors (`src/math/vectors.py`)

Foundation operations on 1D arrays:

| Function | Formula | Use |
|----------|---------|-----|
| `dot(u, v)` | `sum(u_i * v_i)` | Core of all predictions: `z = x^T * beta` |
| `norm(v)` | `sqrt(sum(v_i^2))` | Distance computation, normalization |
| `scale(v, c)` | `c * v` | Scaling vectors |
| `add(u, v)` | `u + v` | Vector addition |
| `subtract(u, v)` | `u - v` | Residuals: `r = y - X*beta` |
| `normalize(v)` | `v / norm(v)` | Unit vectors for projections |
| `cosine_similarity(u, v)` | `dot(u,v) / (norm(u)*norm(v))` | Embedding similarity |
| `angle(u, v)` | `arccos(cosine_similarity(u,v))` | Geometric interpretation |

The dot product is *the* core operation. Every prediction ultimately computes `z = x^T * beta`, which is a dot product between a feature vector and the coefficient vector.

### 5.2 Matrices (`src/math/matrices.py`)

Operations on 2D arrays:

| Function | Purpose |
|----------|---------|
| `multiply(A, B)` | Matrix multiplication -- computes `X @ beta` for all samples at once |
| `transpose(A)` | `A^T` -- needed for `X^T X` (normal equations) |
| `identity(n)` | `I_n` -- used in ridge regularization: `X^T X + lambda * I` |
| `trace(A)` | `sum(diag(A))` -- effective degrees of freedom |
| `rank(A)` | Number of linearly independent columns |
| `condition_number(A)` | `sigma_max / sigma_min` -- numerical stability indicator |

The design matrix `X` has shape `(n_samples x n_features)`. With all encoders active, this could be `(4900 x 250)`, though the current API serving path uses a simplified 8-feature representation (see Section 8).

### 5.3 Projections (`src/math/projections.py`)

Projections are the geometric heart of least squares:

```
The projection of y onto the column space of X:

    y_hat = X @ (X^T X)^{-1} @ X^T @ y = H @ y

where H = X(X^T X)^{-1}X^T is the "hat matrix"
```

The hat matrix `H` puts a "hat" on `y` to get `y_hat`. Its diagonal elements `h_ii` are the **leverage** of each observation -- how much influence that point has on its own fitted value.

| Function | Formula | Use |
|----------|---------|-----|
| `project_onto_subspace(v, basis)` | `sum(proj(v, b_i))` | Project vector onto span of basis |
| `compute_hat_matrix(X)` | `X(X^T X)^{-1}X^T` | Leverage for diagnostics |

### 5.4 QR Factorization (`src/math/qr.py`)

QR factorization decomposes `X = QR` where `Q` is orthogonal and `R` is upper triangular. This is the numerically stable way to solve least squares (compared to the normal equations `X^T X beta = X^T y`, which squares the condition number).

#### Householder Reflections

The from-scratch implementation uses Householder reflections:

```
For each column j of X:
  1. Extract the column below the diagonal: x = X[j:, j]
  2. Compute reflection vector: v = x + sign(x_1) * ||x|| * e_1
  3. Normalize: v = v / ||v||
  4. Apply reflection: X[j:, :] = X[j:, :] - 2 * v * (v^T @ X[j:, :])
```

**Why Householder?**
- Each reflection zeros out one column below the diagonal
- After n reflections, we have R (upper triangular)
- Q is the product of all reflection matrices
- Numerically stable: condition number preserved (not squared)

#### Step-by-step trace

Starting with a 3x3 example:

```
X = [[1.0,  2.0,  3.0],
     [4.0,  5.0,  6.0],
     [7.0,  8.0, 10.0]]

Column 0: x = [1, 4, 7]
  ||x|| = 8.124
  v = x + sign(1) * 8.124 * e_1 = [9.124, 4, 7]
  v_normalized = v / ||v||
  Apply H_0: zeros out below diagonal in column 0, updates columns 1-2

Column 1: x = X[1:, 1] (sub-diagonal part)
  Apply H_1: zeros out below diagonal in column 1

Result: Q (3x3 orthogonal) and R (3x3 upper triangular)
  X = Q @ R
```

#### Solving Least Squares via QR

```
Given X @ beta = y (overdetermined):
  1. Factor X = QR
  2. Compute Q^T @ y    (cheap: just matrix-vector multiply)
  3. Solve R @ beta = Q^T @ y by back substitution (R is upper triangular)
```

Back substitution starts from the last row and works upward:
```
beta_n = (Q^T y)_n / R_nn
beta_{n-1} = ((Q^T y)_{n-1} - R_{n-1,n} * beta_n) / R_{n-1,n-1}
...
```

### 5.5 SVD (`src/math/svd.py`)

Singular Value Decomposition factors any matrix `A = U * Sigma * V^T`:

- `U` (m x m): left singular vectors
- `Sigma` (m x n): diagonal with singular values `sigma_1 >= sigma_2 >= ... >= 0`
- `V` (n x n): right singular vectors

Key uses in this project:
- **Condition number**: `kappa(A) = sigma_1 / sigma_k` -- if large, matrix is ill-conditioned
- **Rank detection**: count singular values above tolerance
- **Low-rank approximation**: keep top-k singular values for embeddings
- **Ridge analysis**: ridge regression shrinks singular values by `sigma_i^2 / (sigma_i^2 + lambda)`

### 5.6 Ridge Regression (`src/math/ridge.py`)

Ridge regression adds L2 regularization to prevent overfitting and handle ill-conditioned matrices:

```
Standard least squares: beta = (X^T X)^{-1} X^T y
Ridge regression:       beta = (X^T X + lambda * I)^{-1} X^T y
```

Adding `lambda * I` to `X^T X` ensures the matrix is invertible even when `X` is rank-deficient.

#### Key Functions

| Function | Method | When Used |
|----------|--------|-----------|
| `ridge_solve(X, y, lambda)` | Solves `(X^T X + lambda*I) beta = X^T y` via `np.linalg.solve` | Standard ridge |
| `ridge_solve_qr(X, y, lambda)` | Augmented QR (stack `sqrt(lambda)*I` below X) | More numerically stable |
| `weighted_ridge_solve(X, y, W, lambda)` | Solves `(X^T W X + lambda*I) beta = X^T W z` via normal equations | **IRLS workhorse** |
| `ridge_loocv(X, y, lambda)` | Efficient LOO-CV via leverage formula | Lambda selection |
| `ridge_cv(X, y, lambdas, k)` | K-fold CV across lambda grid | Lambda selection |
| `ridge_path(X, y, lambdas)` | Coefficient paths across lambdas | Visualization |
| `ridge_gcv(X, y, lambda)` | Generalized cross-validation | Fast lambda selection |
| `ridge_effective_df(X, lambda)` | `trace(X(X^T X + lambda*I)^{-1}X^T)` | Model complexity |

#### The Weighted Ridge Solve

`weighted_ridge_solve()` is the single most important function in the entire math stack. It is called by every iteration of IRLS:

```
Solve: (X^T W X + lambda * I) @ beta = X^T W @ z

where:
  W = diagonal weight matrix
  z = working response vector

Steps:
  1. Compute X^T W X efficiently as X^T @ (w * X) (avoid forming full W)
  2. Add regularization: (X^T W X + lambda * I)
  3. Compute X^T W z as X^T @ (w * z)
  4. Solve the linear system via np.linalg.solve
```

---

## 6. Model Layer

The system implements four model types, each suited to different situations:

### 6.1 Beta-Binomial Baseline (`src/models/baseline.py`)

**Question answered**: What is the admission rate for this specific program?

**Approach**: Bayesian conjugate prior model. For each (university, program) pair, maintain a Beta distribution over the admission probability.

```
Prior:     P(admit) ~ Beta(alpha_0, beta_0)
Data:      k admissions out of n applications
Posterior: P(admit) ~ Beta(alpha_0 + k, beta_0 + n - k)
```

**Why Bayesian?**
The naive estimate `P = k/n` is unreliable for programs with few observations. If Queens Computing has 2 applications (1 admit, 1 reject), MLE says 50% -- but that's based on almost no data.

The Bayesian approach **shrinks** estimates toward a prior mean (the overall admission rate), with shrinkage inversely proportional to sample size:

```
Posterior mean = (prior_strength / (prior_strength + n)) * prior_mean
               + (n / (prior_strength + n)) * data_mean

n = 2 applications:  heavy shrinkage toward prior (unreliable data)
n = 500 applications: posterior ≈ data mean (reliable data)
```

**Shrinkage structure**:
- Global prior: overall admission rate (used as the prior for all programs)
- Per-program posterior: updated with each program's admit/reject counts

Programs with little data shrink toward the global rate. A full hierarchical structure (Global -> University -> Program) where programs shrink toward their university rate is planned but not yet implemented.

**Uncertainty quantification**: Unlike point estimates, this model provides **credible intervals**:
```
"65% admission probability [52%, 78%]"
```
Wide intervals flag uncertain predictions (need more data).

### 6.2 IRLS Logistic Regression (`src/models/logistic.py`)

**Question answered**: Given a student's full feature profile, what is their probability of admission?

This is the **core model**. It uses Iteratively Reweighted Least Squares (IRLS) to fit logistic regression, connecting linear algebra directly to classification.

#### The Logistic Model

```
P(admit | x) = sigmoid(x^T @ beta) = 1 / (1 + exp(-x^T @ beta))
```

Where:
- `x` = feature vector (average, university indicators, program indicators, ...)
- `beta` = learned coefficients
- `sigmoid` maps any real number to (0, 1)

#### Why Not Linear Regression?

Linear regression could predict `P = x^T @ beta`, but this can produce values outside [0, 1] (invalid probabilities). The sigmoid wrapper ensures valid probabilities.

#### The IRLS Algorithm

IRLS iteratively converts logistic regression into a sequence of weighted least squares problems:

```
Initialize: beta = zeros

Repeat until convergence:
  1. Compute predictions:  p_i = sigmoid(x_i^T @ beta)
  2. Compute weights:      w_i = p_i * (1 - p_i)
  3. Compute working response: z_i = x_i^T @ beta + (y_i - p_i) / w_i
  4. Solve weighted ridge:  beta_new = (X^T W X + lambda*I)^{-1} X^T W z
  5. Check convergence:     ||beta_new - beta|| < tolerance?

Return beta
```

**Why this works**: At each iteration, we approximate the logistic loss with a weighted quadratic loss, which is a standard weighted least squares problem. As beta converges, the approximation becomes exact at the optimum.

#### Step-by-step trace of one IRLS iteration

```
Given: X (4900 x 250), y (4900,), current beta (250,)

Step 1 - Predictions:
  z = X @ beta                     # (4900,) linear scores
  p = 1 / (1 + exp(-z))           # (4900,) probabilities
  # Numerically stable: split formula for positive/negative z

Step 2 - Weights:
  w = p * (1 - p)                  # (4900,) diagonal weights
  # Peaked at p=0.5 (most uncertain), near 0 at p near 0 or 1
  # Clipped: max(w, 1e-10) to avoid division by zero

Step 3 - Working response:
  z_working = X @ beta + (y - p) / w    # (4900,)
  # Linearization of the logistic loss around current beta

Step 4 - Weighted ridge solve:
  This calls weighted_ridge_solve(X, z_working, w, lambda)
  Internally:
    X_w = diag(sqrt(w)) @ X               # (4900 x 250) weighted X
    z_w = sqrt(w) * z_working              # (4900,) weighted z
    Solve via augmented QR:
      [X_w; sqrt(lambda)*I] @ beta = [z_w; 0]
    beta_new = back_substitution(R, Q^T @ z_augmented)

Step 5 - Convergence:
  ||beta_new - beta||_2 < 1e-6?
  Typical: converges in 5-10 iterations
```

#### Key Implementation Details

**Numerically stable sigmoid** (`logistic.py:284`):
```python
def sigmoid(z):
    # Split to avoid overflow in exp()
    pos = z >= 0
    result[pos] = 1 / (1 + exp(-z[pos]))           # for z >= 0
    result[~pos] = exp(z[~pos]) / (1 + exp(z[~pos])) # for z < 0
```

**Gradient and Hessian**:
```
Gradient:  g = X^T @ (p - y) + lambda * beta
Hessian:   H = X^T @ W @ X + lambda * I

where W = diag(p * (1 - p))
```

The Hessian is computed efficiently as `(W^{1/2} X)^T (W^{1/2} X) + lambda*I` to avoid forming the full n x n weight matrix.

**Feature importance**: After training, each coefficient `beta_j` has a direct interpretation:
- `exp(beta_j)` = odds ratio for a one-unit increase in feature j
- Sign: positive means higher values increase admission probability
- Magnitude: larger absolute value means stronger effect

### 6.3 Discrete-Time Hazard Model (`src/models/hazard.py`)

**Question answered**: *When* will the student receive a decision?

This model predicts the **timing** of admission decisions using discrete-time survival analysis.

#### The Hazard Function

```
h(t | x) = sigmoid(alpha_t + x^T @ beta)
```

Where:
- `h(t | x)` = probability of receiving a decision at time t, given no decision before t
- `alpha_t` = baseline hazard at time period t (time-specific intercepts)
- `beta` = covariate effects (same across all time periods)

#### Person-Period Expansion

The key trick: transform each application into multiple rows (one per time period at risk):

```
Student A gets decision at week 5:

Original: 1 row
Expanded: 5 rows

  Week  Features   Decision?
   1    [92.5, ...]    0
   2    [92.5, ...]    0
   3    [92.5, ...]    0
   4    [92.5, ...]    0
   5    [92.5, ...]    1     <-- decision here
```

After expansion, this becomes a standard logistic regression problem -- solved with IRLS.

#### Survival and CDF

From hazard rates, we compute:
```
Survival:  S(t) = product(1 - h(k)) for k = 1 to t
                = probability of NO decision by time t

CDF:       F(t) = 1 - S(t)
                = probability of decision BY time t
```

This enables predictions like: "You'll likely hear by mid-March" or "80% chance of decision by week 12."

### 6.4 Embeddings Model (`src/models/embeddings.py`) [Planned]

> **Status**: The embeddings module defines the architecture but is not yet integrated into the API serving path. The API currently returns placeholder similarity results.

**Question answered**: Can we represent universities and programs as dense vectors that capture similarity?

Traditional one-hot encoding creates sparse, high-dimensional vectors (50+ dims for universities, 200+ for programs). The embeddings model replaces these with learned dense vectors:

```
Traditional:  "UofT" --> [1, 0, 0, ..., 0]  (50 dimensions, sparse)
Embedding:    "UofT" --> [0.2, -0.5, ..., 0.8]  (16 dimensions, dense)
```

#### How Embeddings Are Learned

```
1. Initialize random embedding matrices:
   E_uni  (n_universities x 16)
   E_prog (n_programs x 32)

2. For each training sample:
   a. Look up embeddings: e_uni = E_uni[uni_id], e_prog = E_prog[prog_id]
   b. Concatenate: h = [e_uni, e_prog, numeric_features]
   c. Predict: p = sigmoid(h^T @ w)
   d. Compute loss: L = -y*log(p) - (1-y)*log(1-p)
   e. Backpropagate: update E_uni, E_prog, w via SGD

3. After training:
   - Similar programs have similar embeddings
   - Embeddings capture latent structure (competitiveness, field, etc.)
```

#### Dimension Sizing Rule of Thumb (fast.ai)

```
embedding_dim = min(50, (n_categories + 1) // 2)

Universities: min(50, (50+1)//2) = 25 --> use 16
Programs:     min(50, (200+1)//2) = 50 --> use 32
```

#### Benefits
- Fewer parameters than one-hot (48 vs 250 dimensions)
- Similar programs get similar embeddings automatically
- Embeddings are exportable to Weaviate for similarity search
- Natural input for the attention mechanism

### 6.5 Attention Mechanism (`src/models/attention.py`) [Planned]

> **Status**: The attention module defines the architecture but is not yet integrated into the API serving path.

**Question answered**: Which similar programs did the model consider, and how much weight did each get?

The attention model provides **interpretable** predictions by showing which historical programs influenced the prediction.

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

Where:
- **Q (Query)**: "What am I looking for?" -- the student's application
- **K (Keys)**: "What's available?" -- all known program embeddings
- **V (Values)**: "What to retrieve?" -- associated admission rates/features
- **sqrt(d)**: scaling to prevent dot products from growing too large

#### Step-by-step trace

```
Input: query embedding q (1 x 64), program embeddings K (200 x 64), values V (200 x 1)

Step 1 - Similarity scores:
  scores = q @ K^T / sqrt(64)     # (1 x 200) -- raw similarity to each program
  # scores[i] = how similar this application is to program i

Step 2 - Softmax normalization:
  weights = softmax(scores)        # (1 x 200) -- sum to 1.0
  # Numerically stable: subtract max before exp
  # weights[i] = attention weight for program i

Step 3 - Weighted aggregation:
  output = weights @ V             # (1 x 1) -- weighted sum of values

Example output:
  Waterloo CS:  0.35  (most similar)
  UofT CS:      0.25
  McGill CS:    0.15
  Other:        0.25 (spread across remaining programs)
```

#### Multi-Head Attention

Instead of one attention computation, use `h` parallel "heads" with different learned projections:

```
For each head i:
  Q_i = Q @ W_q_i    (project to different subspace)
  K_i = K @ W_k_i
  V_i = V @ W_v_i
  head_i = Attention(Q_i, K_i, V_i)

Concat all heads and project:
  output = Concat(head_1, ..., head_h) @ W_o
```

Each head can learn to attend to different aspects (e.g., one head focuses on program competitiveness, another on geographic similarity).

#### Self-Attention vs Cross-Attention

The module implements both:
- **SelfAttentionLayer**: programs attend to each other (learn inter-program relationships)
- **CrossAttentionLayer**: a student query attends to all program embeddings (prediction)

#### Full AttentionModel Pipeline

```
1. Embed application features (via EmbeddingsModel)
2. Cross-attention: query attends to all program embeddings
3. Concatenate attention output with original features
4. Sigmoid: p = sigmoid(concat @ w)
5. Return probability + attention weights (for explanation)
```

---

## 7. Evaluation and Calibration

### 7.1 Calibration (`src/evaluation/calibration.py`)

Calibration measures whether predicted probabilities match actual outcomes. If we predict 70% for 100 students, roughly 70 should be admitted.

#### Brier Score

```
Brier Score = (1/n) * sum((p_i - y_i)^2)
```

Lower is better. Ranges from 0 (perfect) to 1 (worst). This is the mean squared error for probabilities.

#### Brier Score Decomposition

The Brier score decomposes into three interpretable components:

```
Brier = Uncertainty - Resolution + Reliability

Uncertainty:  var(y) = base_rate * (1 - base_rate)
              How inherently unpredictable is the outcome?
              (Fixed for a dataset -- cannot be improved by any model)

Resolution:   sum(n_k * (bar_y_k - bar_y)^2) / n
              How much do predicted groups differ in actual rates?
              (Higher is better -- model separates groups well)

Reliability:  sum(n_k * (bar_p_k - bar_y_k)^2) / n
              How close are predicted probs to actual rates within bins?
              (Lower is better -- model is well-calibrated)
```

#### Expected Calibration Error (ECE)

```
ECE = sum(n_k / n * |bar_p_k - bar_y_k|)  over K bins
```

Bins predictions into groups (e.g., 0-10%, 10-20%, ...), then measures the average gap between predicted and actual rates in each bin. An ECE of 0.03 means predictions are off by 3% on average.

#### Maximum Calibration Error (MCE)

```
MCE = max(|bar_p_k - bar_y_k|)  over all bins
```

Worst-case bin error. Useful for detecting specific probability ranges where the model is miscalibrated.

#### Platt Scaling

Post-hoc calibration that fits a sigmoid to map raw model scores to calibrated probabilities:

```
p_calibrated = sigmoid(A * logit(p_raw) + B)

where logit(p) = log(p / (1-p))
```

Parameters `A` and `B` are fitted on a **held-out validation set** by minimizing cross-entropy loss via gradient descent:

```
Gradient descent on (A, B):
  For each iteration:
    logits = A * logit(p_raw) + B
    p_cal = sigmoid(logits)
    loss = -mean(y * log(p_cal) + (1-y) * log(1 - p_cal))
    A -= lr * d_loss/d_A
    B -= lr * d_loss/d_B
```

Typical fitted values:
- `A ≈ 1.0`: raw predictions were already well-calibrated
- `A > 1.0`: raw predictions were too uncertain (under-confident)
- `A < 1.0`: raw predictions were too confident (over-confident)
- `B > 0`: shifts predictions upward (higher calibrated probabilities)
- `B < 0`: shifts predictions downward (lower calibrated probabilities)

### 7.2 Discrimination (`src/evaluation/discrimination.py`)

Discrimination measures how well the model separates admits from rejects, regardless of calibration.

#### ROC-AUC

The ROC curve plots True Positive Rate vs False Positive Rate at every threshold:

```
For threshold t from 1.0 down to 0.0:
  Predict "admit" if p >= t
  TPR = true positives / actual positives
  FPR = false positives / actual negatives
  Plot (FPR, TPR)

AUC = area under this curve
```

AUC interpretation:
- 0.5 = random guessing (diagonal line)
- 0.8 = good discrimination
- 0.9 = excellent discrimination
- 1.0 = perfect separation

AUC has a probabilistic interpretation: the probability that a randomly chosen admitted student has a higher predicted probability than a randomly chosen rejected student.

#### Precision-Recall Curves

More informative than ROC when classes are imbalanced:

```
For threshold t from 1.0 down to 0.0:
  Precision = true positives / predicted positives
  Recall = true positives / actual positives
  Plot (Recall, Precision)

Average Precision (AP) = area under PR curve
```

#### Lift Analysis

Measures how much better the model is than random selection:

```
Lift at top k% = (actual positive rate in top k%) / (overall positive rate)

If overall admission rate is 40%:
  Lift = 2.0 means top k% has 80% admission rate (2x better than random)
```

#### DeLong Test

Statistical test to compare AUC values between two models:

```
H0: AUC_model_A = AUC_model_B
H1: AUC_model_A != AUC_model_B

Uses asymptotic normal approximation of AUC variance
p-value < 0.05 means statistically significant difference
```

### 7.3 Validation Strategy (`src/evaluation/validation.py`)

#### Why Temporal Splitting

Admission data is time-series -- using random train/test splits would leak future information:

```
WRONG (random split):
  Train on 2024 data, test on 2022 data
  Model "sees the future" -- artificially inflated metrics

RIGHT (temporal split):
  Train: 2022-2023
  Validation: 2023-2024  (tune hyperparameters)
  Test: 2024-2025        (final evaluation)
```

#### Validation Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Temporal Split** | Train on past, test on future | Default for this project |
| **Expanding Window** | Train window grows over time | When more data always helps |
| **Sliding Window** | Fixed-size training window slides forward | When recent data matters most |
| **Stratified K-Fold** | Random splits preserving class balance | Only if data is truly i.i.d. |

#### Expanding Window Cross-Validation

```
Fold 1: Train [2022-2023]           Validate [2023-2024]
Fold 2: Train [2022-2023, 2023-2024] Validate [2024-2025]
```

Each fold has more training data, simulating real deployment where the model is periodically retrained.

---

## 8. API Serving

The FastAPI application (`src/api/predictor.py`) exposes the prediction pipeline as REST endpoints.

### 8.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Single application prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1000) |
| `POST` | `/explain` | Detailed feature importance |
| `GET` | `/health` | Service health check |
| `GET` | `/model/info` | Model metadata and metrics |
| `GET` | `/universities` | List supported universities |
| `GET` | `/programs` | List programs (filterable by university) |

### 8.2 Request/Response Schema

**Request** (`POST /predict`):
```json
{
  "top_6_average": 92.5,
  "grade_11_average": 88.0,
  "grade_12_average": 91.0,
  "university": "University of Toronto",
  "program": "Computer Science",
  "province": "Ontario",
  "country": "Canada"
}
```

**Response**:
```json
{
  "probability": 0.73,
  "confidence_interval": {
    "lower": 0.65,
    "upper": 0.80,
    "method": "asymptotic"
  },
  "prediction": "LIKELY_ADMIT",
  "feature_importance": [
    {"feature_name": "top_6_average", "contribution": 0.8, "direction": "+"},
    {"feature_name": "is_ontario", "contribution": 0.1, "direction": "+"}
  ],
  "similar_programs": [
    {"university": "Waterloo", "program": "Computer Science", "similarity": 0.85}
  ],
  "model_version": "v1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 8.3 Prediction Pipeline (Single Request)

```
1. VALIDATE
   - Check required fields (top_6_average, university, program)
   - Validate ranges: 0 <= average <= 100
   - Validate province code against known set

2. BUILD FEATURES
   - Normalize university/program names
   - Construct feature vector x (currently 8 features):
     [bias, avg/100, g11/100, g12/100, is_ON, is_BC, is_AB, is_QC]
   - Missing grade_11/grade_12 values default to top_6_average
   - Future: university/program embeddings, interaction features

3. RAW PREDICTION
   - z = x^T @ beta       (linear combination)
   - p_raw = sigmoid(z)    (raw probability)

4. CALIBRATION (Platt Scaling)
   - logit = log(p_raw / (1 - p_raw))
   - p_cal = sigmoid(A * logit + B)
   - Clip to [0.001, 0.999]

5. CONFIDENCE INTERVAL (approximate)
   - SE = sqrt(p * (1-p) / n_training)   (binomial proportion SE)
   - CI = [p - 1.96*SE, p + 1.96*SE]
   - Note: This is a simplified lower bound. Proper individual prediction CIs
     require the Fisher information matrix via the delta method.

6. EXPLANATION
   - For each feature j: contribution_j = x_j * beta_j
   - Sort by |contribution|, return top 5
   - Find similar programs via embedding cosine similarity

7. LABEL
   - p >= 0.70: "LIKELY_ADMIT"
   - 0.40 <= p < 0.70: "UNCERTAIN"
   - p < 0.40: "UNLIKELY_ADMIT"

8. RETURN PredictionResponse
```

### 8.4 Prediction Labels

```
|<--- UNLIKELY_ADMIT --->|<--- UNCERTAIN --->|<--- LIKELY_ADMIT --->|
0                       0.40                0.70                    1.0
```

Thresholds should be tuned based on historical admission rates and business requirements.

---

## 9. End-to-End Example

Let's trace a complete prediction for a real application:

### Input
```
Student: Ontario resident, 92.5% top-6 average
Target:  University of Toronto, Computer Science
```

### Step 1: Normalization
```
University: "University of Toronto" --> exact match --> canonical
Program:    "Computer Science"      --> exact match --> canonical
```

### Step 2: Feature Engineering
```
Encoders produce:
  GPA features:    [1.24, 0, 0, 0, 1, 0]  (z-score + bucket flags)
  University:      [0, 0, 1, 0, ...]       (one-hot: UofT = 1)
  Program:         [1, 0, 0, ...]           (one-hot: CS = 1)
  Province:        [1, 0, 0, 0]             (Ontario = 1)
```

### Step 3: Design Matrix Row
```
DesignMatrixBuilder.transform() produces:
  x = [1.0, 0.925, 0.88, 0.91, 1.0, 0.0, 0.0, 0.0]
       bias  avg    g11   g12   ON    BC    AB    QC
```

### Step 4: Model Prediction (Logistic)
```
beta = [-0.5, 0.08, 0.02, 0.03, 0.1, 0.05, 0.05, 0.08]  (learned from IRLS)

z = x^T @ beta
  = 1.0*(-0.5) + 0.925*0.08 + 0.88*0.02 + 0.91*0.03 + 1.0*0.1 + 0*0.05 + 0*0.05 + 0*0.08
  = -0.5 + 0.074 + 0.0176 + 0.0273 + 0.1
  = -0.2811

p_raw = sigmoid(-0.2811) = 1 / (1 + exp(0.2811)) = 0.430
```

### Step 5: Calibration
```
Platt scaling (log-odds space):
  logit(p_raw) = log(0.430 / 0.570) = log(0.754) = -0.282
  p_cal = sigmoid(A * logit + B) = sigmoid(1.0 * (-0.282) + 0.0)
        = sigmoid(-0.282) = 0.430

With A=1.0, B=0.0 (identity calibration), p_cal = p_raw.
A trained calibrator would have A and B fitted on validation data.
```

### Step 6: Confidence Interval
```
Approximate (binomial proportion SE -- simplified lower bound):
SE = sqrt(0.430 * 0.570 / 10000) = 0.00495
CI = [0.430 - 1.96*0.00495, 0.430 + 1.96*0.00495]
   = [0.420, 0.440]

Note: Proper individual prediction CIs using the delta method would be wider.
```

### Step 7: Feature Importance
```
Contributions (sorted by magnitude):
  1. bias:          1.0 * (-0.5)  = -0.500  (-)
  2. is_ontario:    1.0 * 0.1     = +0.100  (+)
  3. top_6_average: 0.925 * 0.08  = +0.074  (+)
  4. grade_12:      0.91 * 0.03   = +0.027  (+)
  5. grade_11:      0.88 * 0.02   = +0.018  (+)
```

### Step 8: Label
```
p = 0.430 >= 0.40 --> "UNCERTAIN"
```

### Final Response
```json
{
  "probability": 0.430,
  "confidence_interval": {"lower": 0.420, "upper": 0.440},
  "prediction": "UNCERTAIN",
  "feature_importance": [
    {"feature_name": "top_6_average", "contribution": 0.074, "direction": "+"},
    {"feature_name": "is_ontario", "contribution": 0.1, "direction": "+"}
  ],
  "similar_programs": [
    {"university": "University of Waterloo", "program": "Computer Science", "similarity": 0.85},
    {"university": "McGill University", "program": "Computer Science", "similarity": 0.78}
  ],
  "model_version": "v1.0.0",
  "calibration_note": "Confidence interval uses simplified binomial approximation"
}
```

**Note**: This example uses simplified placeholder coefficients. With `A=1.0, B=0.0`, Platt scaling is an identity transform (no calibration effect). With a properly trained model on real data, the coefficients would be learned from thousands of historical applications via IRLS, calibration parameters would be fitted on a held-out validation set, and the predictions would be more accurate. The current API serving path uses 8 features; the full encoder pipeline (250+ features) is defined in `encoders.py` and `design_matrix.py` but not yet wired into the API.

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| **IRLS** | Iteratively Reweighted Least Squares -- converts logistic regression into a sequence of weighted least squares problems |
| **Ridge Regression** | Least squares with L2 penalty (lambda * \|\|beta\|\|^2) to prevent overfitting |
| **QR Factorization** | Decomposes X = QR where Q is orthogonal, R is upper triangular |
| **Householder Reflection** | Orthogonal transformation that zeros out vector elements below the diagonal |
| **SVD** | Singular Value Decomposition: A = U * Sigma * V^T |
| **Hat Matrix** | H = X(X^TX)^{-1}X^T -- projects y onto column space of X |
| **Platt Scaling** | Post-hoc calibration using a sigmoid fitted on validation data |
| **Brier Score** | Mean squared error of probability predictions |
| **ECE** | Expected Calibration Error -- average gap between predicted and actual rates across bins |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic curve |
| **Beta-Binomial** | Bayesian conjugate model for estimating binomial probabilities |
| **Hazard Function** | h(t) = P(event at time t \| no event before t) |
| **Embeddings** | Learned dense vector representations of categorical variables |
| **Attention** | Mechanism that computes weighted sum based on query-key similarity |

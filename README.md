# University Admissions Prediction Project Plan

## Project Overview

### Problem Statement

Canadian high school students face uncertainty about admission chances. They must decide where to apply, how many "safety" vs "reach" schools to include, and when to expect decisions—all without reliable data. This project builds a prediction system to answer: **"Given my average and target program, what are my chances?"**

### Project Goals

Build a machine learning prediction model that estimates a student's chance of admission to a Canadian university program based on:
- **High school average** (Top 6 average - the six highest U/M level courses)
- **University name** (e.g., University of Toronto, McMaster, Waterloo)
- **Program name** (e.g., Computer Science, Engineering, Life Sciences)

### Data Sources

Historical self-reported admission data from ~4,900 applications across 3 admission cycles:
- **2022-23 cycle:** 981 records
- **2023-24 cycle:** 1,845 records
- **2024-25 cycle:** 2,074 records

Data includes: admission outcome (accepted/rejected/waitlisted/deferred), reported average, application date, decision date, university, program, and student-provided context.

### Model Outputs

1. **P(admit):** Probability of admission (0-1) with calibrated uncertainty interval (e.g., 73% ± 8%)
2. **Decision timing:** Predicted weeks until decision based on historical patterns for that program
3. **Similar cases:** Retrieved examples of students with similar profiles and their outcomes

### Technical Approach

- **Embedding-based model:** Learn dense representations for universities and programs to capture similarity (UofT CS is "closer" to Waterloo CS than to UofT Art History)
- **Attention mechanism:** Let the model learn which features matter most for each prediction
- **Calibration:** Ensure predicted probabilities match actual admission rates (if we say 70%, ~70% should get in)
- **Uncertainty quantification:** Provide confidence intervals, not just point estimates

### Success Criteria

1. **Calibration:** Predictions match reality (calibration error < 5%)
2. **Discrimination:** Model correctly ranks applicants (AUC-ROC > 0.75)
3. **Usefulness:** Uncertainty intervals are narrow enough to be actionable
4. **Generalization:** Performs well on held-out 2024-25 data (no future leakage)

### Scope Boundaries

**In Scope:**
- Ontario universities and programs with sufficient data (>20 applications)
- Undergraduate admissions based on Grade 12 averages
- Binary outcome prediction (admitted vs not admitted)

**Out of Scope (for now):**
- Supplementary application scoring (essays, portfolios, interviews)
- Graduate admissions
- International student-specific predictions
- Real-time data collection/scraping

### Learning Goals (Prioritized)

This project serves as a vehicle for learning key technical concepts:

1. **PyTorch & Deep Learning (Primary):** tensors, autograd, nn.Module, optimizers, embeddings, attention mechanisms
2. **Machine Learning:** logistic regression fundamentals, probability calibration, model evaluation, neural network architectures
3. **MAT223 Linear Algebra:** Learn concepts as they appear in ML—gradients as vectors, matrix operations for batched computation, embeddings as learned basis vectors
4. **CSC148 OOP/Data Structures:** Apply through PyTorch patterns—nn.Module inheritance, custom Dataset classes, exception handling for convergence failures
5. **Databases:** MongoDB for structured storage, Weaviate for vector similarity search and retrieval

### Why PyTorch-First?

**Priority:** Learn modern deep learning tools first, then understand the math underneath.

Traditional ML courses teach: math → numpy → sklearn → (maybe) deep learning. This project inverts that:
- Start with PyTorch's high-level abstractions (nn.Module, optimizers, DataLoader)
- Build intuition through experimentation and visualization
- Dive into math (linear algebra, calculus) when it explains observed behavior
- Connect back to fundamentals (logistic regression is a 1-layer neural network)

---

## Validated MAT223 Concepts for This Project

Based on the MAT223 lecture notes table of contents, here are the exact sections you need:

### Must Learn (Core to Project)

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| 4.1 | Vectors | Feature vectors, predictions | Day 1 |
| 4.2 | Dot Product and Projections | Predictions x^T β, similarity | Day 1 |
| 4.2.2 | Projections onto Lines | Simplest least squares case | Day 1 |
| 4.4 | Subspaces | Column space col(X) | Day 1 |
| 4.4.1 | Span | Understanding model expressiveness | Day 1 |
| 4.5 | Linear Independence | Dropping reference categories | Day 2 |
| 4.6 | Rank of a Matrix | Checking full column rank | Day 2 |
| 4.7 | Orthonormal Bases | QR factorization foundation | Day 2 |
| 4.7.1 | Projections Onto Subspaces | Normal equations derivation | Day 2 |
| 4.7.2 | Gram-Schmidt (conceptual) | Understanding QR | Day 2 |
| 4.8 | Approximating Solutions | Least squares, IRLS | Day 2 |
| 1.3.2 | Rank of a Matrix | Singular matrix diagnosis | Day 2 |
| 1.4.2 | Column Vectors | Design matrix columns | Day 1 |
| 1.4.4 | Matrix Multiplication | Xβ predictions | Day 1 |

### Should Learn (Improves Understanding)

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| 3.4 | Eigenvalues/Eigenvectors | SVD understanding, conditioning | Day 6-7 |
| 3.6 | Applications | Ridge shrinkage geometry | Day 6-7 |
| 1.6 | Matrix Inversion | Why NOT to use it (prefer QR) | Day 2 |
| 1.3.1 | Gaussian Elimination | Row reduction understanding | Optional |

### Can Skip for This Project

| Section | Topic | Why Skip |
|---------|-------|----------|
| 2.x | Complex Numbers | Not used in real-valued ML |
| 3.1-3.3 | Determinants | Not directly needed for IRLS/ridge |
| 3.5 | Diagonalization | Nice to know but not essential |
| 4.3 | Cross Product and Planes | 3D geometry, not ML relevant |
| 1.5 | Linear Transformations | Conceptually useful but not core |

---

## Validated CSC148 Concepts for This Project

Based on the CSC148 course notes, here are the sections to focus on:

### Must Implement

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| 1.3-1.5 | Function Design Recipe | All functions with docstrings, types | Day 1+ |
| 3.1-3.4 | OOP + Representation Invariants | Model classes, DesignMatrix | Day 3+ |
| 3.5-3.8 | Inheritance | BaseModel → LogisticModel hierarchy | Day 5+ |
| 5.1-5.4 | Exceptions | ConvergenceError, RankDeficiencyError | Day 6+ |
| 2.1-2.4 | Testing | Unit tests for math functions | Throughout |

### Should Implement

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| 8.1-8.3 | Trees | Program hierarchy for partial pooling | Day 4 |
| 7.1-7.6 | Recursion | Tree traversal, aggregation | Day 4 |
| 4.1-4.2 | ADTs | Model interface/protocol | Day 10 |
| 4.4 | Running Time Analysis | Documenting O(np²) for IRLS | Day 6 |

### Can Reference but Not Implement from Scratch

| Section | Topic | Why Reference Only |
|---------|-------|-------------------|
| 6.x | Linked Lists | Use Python lists/numpy instead |
| 9.x | Recursive Sorting | Use numpy's built-in sorting |
| 4.2-4.3 | Stacks/Queues | Not core to this project |

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    UNIVERSITY ADMISSIONS PREDICTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   CSV Data   │───▶│   MongoDB    │───▶│   PyTorch    │───▶│   Weaviate   │  │
│  │  (Raw Input) │    │  (Storage)   │    │   (Models)   │    │  (Vectors)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         ▼                   ▼                   ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                              FastAPI                                      │  │
│  │                         (Prediction API)                                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│                          ┌──────────────────┐                                   │
│                          │   User Query:    │                                   │
│                          │  avg=92, UofT CS │                                   │
│                          └────────┬─────────┘                                   │
│                                   │                                             │
│                                   ▼                                             │
│                          ┌──────────────────┐                                   │
│                          │    Response:     │                                   │
│                          │  P(admit)=0.73   │                                   │
│                          │  CI: [0.65,0.81] │                                   │
│                          │  Decision: ~12wk │                                   │
│                          └──────────────────┘                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

     CSV FILES                         ETL                           MONGODB
  ┌─────────────┐                 ┌──────────┐                  ┌─────────────┐
  │ 2022-23.csv │─────┐           │          │                  │ universities│
  │  (981 rows) │     │           │  Clean   │                  ├─────────────┤
  ├─────────────┤     ├──────────▶│  Parse   │─────────────────▶│  programs   │
  │ 2023-24.csv │     │           │ Validate │                  ├─────────────┤
  │ (1845 rows) │─────┤           │          │                  │applications │
  ├─────────────┤     │           └──────────┘                  └──────┬──────┘
  │ 2024-25.csv │─────┘                                                │
  │ (2074 rows) │                                                      │
  └─────────────┘                                                      │
                                                                       ▼
                                                              ┌─────────────────┐
     TRAINING                        PYTORCH                  │  DataLoader     │
  ┌─────────────┐              ┌──────────────┐               │  - batch_size   │
  │ Train Set   │◀─────────────│   Dataset    │◀──────────────│  - shuffle      │
  │ (2022-2024) │              │    Class     │               │  - transforms   │
  ├─────────────┤              └──────────────┘               └─────────────────┘
  │ Test Set    │
  │ (2024-2025) │              Temporal Split: No Future Leakage!
  └─────────────┘
```

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL ARCHITECTURE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────────────────────┐
                         │            INPUT FEATURES           │
                         ├──────────┬──────────┬───────────────┤
                         │   avg    │ uni_id   │   prog_id     │
                         │  (float) │  (int)   │    (int)      │
                         └────┬─────┴────┬─────┴───────┬───────┘
                              │          │             │
                              ▼          ▼             ▼
                    ┌─────────────┐ ┌─────────┐ ┌─────────────┐
                    │  nn.Linear  │ │Embedding│ │  Embedding  │
                    │  (1 → 32)   │ │(50 → 32)│ │ (200 → 32)  │
                    └──────┬──────┘ └────┬────┘ └──────┬──────┘
                           │             │             │
                           │      ┌──────┴──────┐      │
                           │      │             │      │
                           │      ▼             ▼      │
                           │  ┌───────────────────┐    │
                           │  │    ATTENTION      │    │
                           │  │  ┌─────────────┐  │    │
                           │  │  │ Q = W_q(x)  │  │    │
                           │  │  │ K = W_k(x)  │  │    │
                           │  │  │ V = W_v(x)  │  │    │
                           │  │  │             │  │    │
                           │  │  │softmax(QK^T)│  │    │
                           │  │  │   ───────   │  │    │
                           │  │  │    √d_k     │  │    │
                           │  │  └─────────────┘  │    │
                           │  └─────────┬─────────┘    │
                           │            │              │
                           └────────────┼──────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   CONCATENATE   │
                              │   (32*3 = 96)   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   MLP HEAD      │
                              │  96 → 32 → 1    │
                              │  ReLU, Dropout  │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    SIGMOID      │
                              │   P(admit)      │
                              └─────────────────┘
```

---

## Attention Mechanism Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ATTENTION: "Which programs are similar?"                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Query Program: "UofT Computer Science"
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                    PROGRAM EMBEDDING SPACE                        │
    │                                                                   │
    │     ★ Query: UofT CS                                              │
    │                    ╲                                              │
    │                     ╲ high similarity                             │
    │                      ╲                                            │
    │    ● UW CS ──────────●─── UofT Engineering                        │
    │         ╲           ╱                                             │
    │          ╲         ╱                                              │
    │           ● McGill CS                                             │
    │                                                                   │
    │                          ○ Western Business (low similarity)      │
    │                                                                   │
    │    ○ = low attention weight                                       │
    │    ● = high attention weight                                      │
    │    ★ = query program                                              │
    └───────────────────────────────────────────────────────────────────┘

    Attention Weights (example):
    ┌────────────────────┬────────────┐
    │ Program            │ Weight     │
    ├────────────────────┼────────────┤
    │ UWaterloo CS       │ 0.35  ████ │
    │ UofT Engineering   │ 0.25  ███  │
    │ McGill CS          │ 0.20  ██   │
    │ Queen's Computing  │ 0.12  █    │
    │ Western Business   │ 0.08  ▌    │
    └────────────────────┴────────────┘
```

---

## Training Loop Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LOOP                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  for epoch in range(num_epochs):                                         │
    │      for batch in dataloader:                                            │
    │                                                                          │
    │          ┌─────────────┐                                                 │
    │          │ 1. FORWARD  │                                                 │
    │          │   probs =   │                                                 │
    │          │  model(X)   │                                                 │
    │          └──────┬──────┘                                                 │
    │                 │                                                        │
    │                 ▼                                                        │
    │          ┌─────────────┐                                                 │
    │          │ 2. LOSS     │     loss = BCE(probs, y)                        │
    │          │   compute   │     + λ * ||weights||²                          │
    │          └──────┬──────┘                                                 │
    │                 │                                                        │
    │                 ▼                                                        │
    │          ┌─────────────┐                                                 │
    │          │ 3. BACKWARD │     loss.backward()                             │
    │          │   autograd  │     ∂L/∂w for all weights                       │
    │          └──────┬──────┘                                                 │
    │                 │            MAT223: Gradients!                          │
    │                 ▼                                                        │
    │          ┌─────────────┐                                                 │
    │          │ 4. UPDATE   │     optimizer.step()                            │
    │          │   weights   │     w = w - lr * ∂L/∂w                          │
    │          └──────┬──────┘                                                 │
    │                 │                                                        │
    │                 ▼                                                        │
    │          ┌─────────────┐                                                 │
    │          │ 5. ZERO     │     optimizer.zero_grad()                       │
    │          │   gradients │     Reset for next batch                        │
    │          └─────────────┘                                                 │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
```

---

## Project Workflow (14-Day Timeline)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         14-DAY PROJECT TIMELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

PHASE 1: FOUNDATIONS (Days 1-3)
══════════════════════════════════════════════════════════════════════════════════
Day 1       Day 2       Day 3
  │           │           │
  ▼           ▼           ▼
┌─────┐     ┌─────┐     ┌─────┐
│Tensor│────▶│Auto-│────▶│nn.  │
│ ops │     │grad │     │Module│
└─────┘     └─────┘     └─────┘
  │           │           │
  │  MAT223:  │  MAT223:  │  CSC148:
  │  Vectors  │  Gradients│  Inheritance
  │  Matrices │           │
  ▼           ▼           ▼
[MongoDB Setup] [First Model] [Training Loop]


PHASE 2: CORE MODELS (Days 4-7)
══════════════════════════════════════════════════════════════════════════════════
Day 4       Day 5       Day 6       Day 7
  │           │           │           │
  ▼           ▼           ▼           ▼
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│Base-│────▶│Feat-│────▶│Embed│────▶│Regu-│
│line │     │ures │     │dings│     │lariz│
└─────┘     └─────┘     └─────┘     └─────┘
  │           │           │           │
  │  Stats:   │  CSC148:  │  MAT223:  │  MAT223:
  │  Beta     │  Dataset  │  Low-rank │  Ridge
  │  Binomial │  ADTs     │  UV^T     │
  ▼           ▼           ▼           ▼
[Calibration] [DataLoader] [Program vectors] [Hyperparams]


PHASE 3: ADVANCED (Days 8-11)
══════════════════════════════════════════════════════════════════════════════════
Day 8       Day 9       Day 10      Day 11
  │           │           │           │
  ▼           ▼           ▼           ▼
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│Attn │────▶│Attn │────▶│Multi│────▶│Timing│
│Basics│    │Model│     │Head │     │Model │
└─────┘     └─────┘     └─────┘     └─────┘
  │           │           │           │
  │  MAT223:  │           │  MAT223:  │
  │  QK^T     │  Visualize│  Subspaces│  Survival
  │  softmax  │  weights  │           │  analysis
  ▼           ▼           ▼           ▼
[Q,K,V impl] [Compare models] [Parallel attn] [Decision week]


PHASE 4: PRODUCTION (Days 12-14)
══════════════════════════════════════════════════════════════════════════════════
Day 12      Day 13      Day 14
  │           │           │
  ▼           ▼           ▼
┌─────┐     ┌─────┐     ┌─────┐
│Weavi│────▶│Analy│────▶│ API │
│ate  │     │sis  │     │Docs │
└─────┘     └─────┘     └─────┘
  │           │           │
  │  Vector   │  Error    │  FastAPI
  │  search   │  analysis │  Demo
  ▼           ▼           ▼
[Semantic search] [Model comparison] [Ship it!]
```

---

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTION DIAGRAM                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │           USER REQUEST               │
                    │  "What are my chances for UofT CS    │
                    │   with a 92% average?"               │
                    └─────────────────┬────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI SERVER                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  @app.post("/predict")                                                   │   │
│  │  def predict(avg: float, uni: str, prog: str):                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    MONGODB      │       │    PYTORCH      │       │    WEAVIATE     │
│                 │       │                 │       │                 │
│  Look up IDs:   │       │  Forward pass:  │       │  Find similar:  │
│  uni_id = 5     │──────▶│  model(X)       │◀──────│  programs for   │
│  prog_id = 42   │       │  → P(admit)     │       │  context        │
│                 │       │                 │       │                 │
└─────────────────┘       └────────┬────────┘       └─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        CALIBRATION           │
                    │  ┌────────────────────────┐  │
                    │  │ Adjust raw probability │  │
                    │  │ using isotonic         │  │
                    │  │ regression             │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │          RESPONSE            │
                    │  {                           │
                    │    "probability": 0.73,      │
                    │    "ci_lower": 0.65,         │
                    │    "ci_upper": 0.81,         │
                    │    "decision_week": 12,      │
                    │    "similar_programs": [     │
                    │      "UWaterloo CS",         │
                    │      "McGill CS"             │
                    │    ]                         │
                    │  }                           │
                    └──────────────────────────────┘
```

---

## Embedding Space Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LEARNED EMBEDDING SPACE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

    Universities (2D projection of 32-dim embeddings)
    ════════════════════════════════════════════════

              Competitive ▲
                         │
                         │    ★ UofT
                         │         ★ McGill
                         │    ★ UBC
                         │              ★ UWaterloo
                         │
    Large ◀──────────────┼──────────────────────▶ Small
                         │
                         │         ● Queen's
                         │    ● Western
                         │              ● Ottawa
                         │
                         │
                         ▼ Less Competitive


    Programs (clustered by field)
    ═══════════════════════════════

                              STEM
                         ┌─────────────┐
                         │ ● CS        │
                         │ ● Eng       │
                         │ ● Math      │
                         └─────────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────────────┐ ┌──────────┐ ┌──────────────┐
        │   HEALTH     │ │  SCIENCE │ │   BUSINESS   │
        │ ● Nursing    │ │ ● Biology│ │ ● Commerce   │
        │ ● Kinesiology│ │ ● Physics│ │ ● Economics  │
        └──────────────┘ └──────────┘ └──────────────┘
```

---

## Learning Pathway: Math-to-Code Bridge

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MATH ←→ CODE TRANSLATION                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

MAT223 CONCEPT              PYTORCH CODE                    WHAT IT DOES
═══════════════════════════════════════════════════════════════════════════════════

Vector x ∈ ℝⁿ        →      x = torch.tensor([...])        Feature vector

Dot product x·y      →      torch.dot(x, y)                 Similarity, prediction

Matrix mult Xβ       →      X @ beta                        Linear model output

||x||₂ (norm)        →      torch.norm(x)                   Vector length

Projection           →      Q @ (Q.T @ y)                   Least squares

Gradient ∇L          →      loss.backward()                 Autograd magic!
                            param.grad

Low-rank UV^T        →      nn.Embedding(n, k)              Learned vectors

Softmax              →      F.softmax(x, dim=-1)            Attention weights

═══════════════════════════════════════════════════════════════════════════════════

CSC148 CONCEPT              PYTORCH CODE                    WHAT IT DOES
═══════════════════════════════════════════════════════════════════════════════════

Inheritance          →      class Model(nn.Module):         Extend PyTorch base
                               def __init__(self):
                                   super().__init__()

Abstract methods     →      def forward(self, x):           Must implement
                               raise NotImplementedError

Representation       →      assert self.weight.shape ==     Verify state
invariants                       (out, in)

Iterators            →      for batch in DataLoader:        Batch processing

Exceptions           →      if cond > 1e10:                 Error handling
                               raise ConvergenceError()
```

---

## PyTorch Learning Path

### Core PyTorch Concepts (in order of introduction)

| Day | PyTorch Concept | What It Is | MAT223 Connection | CSC148 Connection |
|-----|-----------------|------------|-------------------|-------------------|
| 1 | **Tensors** | N-dimensional arrays with GPU support | Vectors, matrices (§4.1, §1.4) | Arrays, memory model (§1.1) |
| 1 | **Tensor Operations** | Matrix multiply, broadcasting, indexing | Matrix multiplication (§1.4.4) | - |
| 2 | **Autograd** | Automatic differentiation | Gradients, chain rule | - |
| 2 | **requires_grad** | Tracking computations for backprop | Partial derivatives | - |
| 3 | **nn.Module** | Base class for all neural networks | - | Inheritance, OOP (§3.5-3.8) |
| 3 | **nn.Linear** | Fully connected layer: y = Wx + b | Linear transformations (§1.5) | - |
| 4 | **Loss Functions** | BCEWithLogitsLoss, CrossEntropyLoss | - | - |
| 4 | **Optimizers** | SGD, Adam - update weights | Gradient descent | - |
| 5 | **DataLoader** | Batching, shuffling data | - | Iterators (§4.1) |
| 5 | **Custom Dataset** | torch.utils.data.Dataset | - | ADTs, protocols (§4.1-4.2) |
| 6 | **nn.Embedding** | Learnable lookup table for categories | Low-rank factorization (§3.6) | - |
| 7 | **Regularization** | weight_decay (L2), Dropout | Ridge regression | - |
| 8-9 | **Attention** | Query-Key-Value mechanism | Dot products, projections (§4.2) | - |
| 10 | **Multi-head Attention** | Parallel attention heads | Subspaces (§4.4) | - |

---

### PyTorch Fundamentals Code Examples

#### Day 1: Tensors (MAT223 §4.1 - Vectors)

```python
import torch

# Vectors are 1D tensors
x = torch.tensor([1.0, 2.0, 3.0])  # Shape: (3,)
y = torch.tensor([4.0, 5.0, 6.0])

# MAT223: Vector operations
dot_product = torch.dot(x, y)           # x^T y = 32
norm = torch.norm(x)                     # ||x|| = sqrt(14)
cosine_sim = torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

# Matrices are 2D tensors
X = torch.randn(100, 5)  # 100 samples, 5 features (design matrix)
beta = torch.randn(5)     # coefficient vector

# MAT223 §1.4.4: Matrix multiplication
predictions = X @ beta    # Shape: (100,) - this is Xβ!

print(f"Dot product: {dot_product}")
print(f"Norm: {norm}")
print(f"Predictions shape: {predictions.shape}")
```

**MAT223 Connection:** A tensor IS a vector/matrix. `X @ beta` is exactly the matrix-vector multiplication Xβ from linear algebra.

---

#### Day 2: Autograd (Gradients - MAT223 implicitly)

```python
import torch

# Create tensors with gradient tracking
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,)).float()  # Binary labels
beta = torch.randn(5, requires_grad=True)  # We want gradients for beta!

# Forward pass: logistic regression
logits = X @ beta                          # Linear: Xβ
probs = torch.sigmoid(logits)              # Nonlinear: σ(Xβ)
loss = torch.nn.functional.binary_cross_entropy(probs, y)

# Backward pass: compute gradients automatically!
loss.backward()

# beta.grad now contains ∂loss/∂beta
print(f"Loss: {loss.item():.4f}")
print(f"Gradient shape: {beta.grad.shape}")  # (5,)
print(f"Gradient: {beta.grad}")

# MAT223 connection: This gradient is X^T (p - y) / n
# PyTorch computed it automatically via chain rule!
```

**MAT223 Connection:** The gradient `beta.grad` is exactly X^T(p - y) from the logistic regression derivation. Autograd computes this for you!

---

#### Day 3: nn.Module (CSC148 §3.5-3.8 - Inheritance)

```python
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    Logistic regression as a PyTorch module.

    CSC148 Connection: This is INHERITANCE!
    - LogisticRegression inherits from nn.Module
    - Must implement __init__ and forward (like abstract methods)

    Representation Invariants:
        - self.linear.weight has shape (1, in_features)
        - self.linear.bias has shape (1,)
    """

    def __init__(self, in_features: int):
        super().__init__()  # CSC148: Call parent's __init__
        self.linear = nn.Linear(in_features, 1)  # y = Wx + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute P(y=1|x)

        MAT223 Connection:
            self.linear(x) computes Wx + b (matrix multiplication!)
        """
        logits = self.linear(x)  # Shape: (batch_size, 1)
        return torch.sigmoid(logits).squeeze()  # Shape: (batch_size,)


# Usage
model = LogisticRegression(in_features=5)
X = torch.randn(32, 5)  # Batch of 32 samples
probs = model(X)        # Calls forward() automatically
print(f"Probabilities: {probs.shape}")  # (32,)
```

---

#### Day 6: nn.Embedding (MAT223 §3.6 - Low-rank)

```python
import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    """
    Learn dense embeddings for universities and programs.

    MAT223 Connection:
        Instead of one-hot (high-dimensional, sparse):
            uni_onehot = [0, 0, 1, 0, ..., 0]  # 50 dimensions

        We learn dense embeddings (low-dimensional):
            uni_embedding = [0.3, -0.1, 0.8, ...]  # 16 dimensions

        This is LOW-RANK FACTORIZATION!
        The interaction term u_uni^T v_prog is a dot product (§4.2)
    """

    def __init__(self, n_universities: int, n_programs: int, embed_dim: int = 16):
        super().__init__()
        self.uni_embedding = nn.Embedding(n_universities, embed_dim)
        self.prog_embedding = nn.Embedding(n_programs, embed_dim)
        self.avg_linear = nn.Linear(1, embed_dim)
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, uni_ids, prog_ids, averages):
        """
        uni_ids: (batch,) - integer IDs for universities
        prog_ids: (batch,) - integer IDs for programs
        averages: (batch, 1) - high school averages
        """
        # Look up embeddings
        uni_emb = self.uni_embedding(uni_ids)    # (batch, embed_dim)
        prog_emb = self.prog_embedding(prog_ids)  # (batch, embed_dim)
        avg_emb = self.avg_linear(averages)       # (batch, embed_dim)

        # Combine: element-wise product captures interaction
        # MAT223: This is related to bilinear forms u^T v
        combined = uni_emb * prog_emb + avg_emb   # (batch, embed_dim)

        logits = self.output(combined)  # (batch, 1)
        return torch.sigmoid(logits).squeeze()

# Usage
model = EmbeddingModel(n_universities=50, n_programs=200, embed_dim=16)
```

---

#### Days 8-9: Attention Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProgramAttention(nn.Module):
    """
    Attention mechanism for program embeddings.

    The idea: "Which other programs are most similar/relevant to this one?"

    MAT223 Connections:
        - Query, Key, Value are LINEAR TRANSFORMATIONS (§1.5)
        - Attention scores use DOT PRODUCTS (§4.2)
        - Softmax creates a CONVEX COMBINATION (related to projections)
        - The output is a weighted sum of values

    Attention formula:
        Attention(Q, K, V) = softmax(QK^T / √d) V
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        # MAT223: These are matrix multiplications!
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_key = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query: (batch, embed_dim) - the program we're predicting for
        context: (n_programs, embed_dim) - all program embeddings

        Returns: (batch, embed_dim) - attention-weighted representation
        """
        # Project to Q, K, V spaces
        Q = self.W_query(query)      # (batch, embed_dim)
        K = self.W_key(context)      # (n_programs, embed_dim)
        V = self.W_value(context)    # (n_programs, embed_dim)

        # Compute attention scores: QK^T / sqrt(d)
        # MAT223: This is a MATRIX of dot products!
        # scores[i,j] = query_i · key_j (similarity)
        scores = torch.matmul(Q, K.T) / math.sqrt(self.embed_dim)  # (batch, n_programs)

        # Softmax to get attention weights (sum to 1)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_programs)

        # Weighted sum of values
        # MAT223: This is a LINEAR COMBINATION of the value vectors!
        output = torch.matmul(attention_weights, V)  # (batch, embed_dim)

        return output, attention_weights


class AdmissionModelWithAttention(nn.Module):
    """
    Full model with attention over similar programs.

    Intuition: "To predict admission to UofT CS, look at similar programs
    (UWaterloo CS, UofT Engineering) and use their patterns."
    """

    def __init__(self, n_universities: int, n_programs: int, embed_dim: int = 32):
        super().__init__()
        self.uni_embedding = nn.Embedding(n_universities, embed_dim)
        self.prog_embedding = nn.Embedding(n_programs, embed_dim)
        self.attention = ProgramAttention(embed_dim)

        self.avg_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, uni_ids, prog_ids, averages, all_prog_embeddings):
        """
        all_prog_embeddings: (n_programs, embed_dim) - embeddings of ALL programs
        """
        # Get embeddings for this sample
        uni_emb = self.uni_embedding(uni_ids)      # (batch, embed_dim)
        prog_emb = self.prog_embedding(prog_ids)   # (batch, embed_dim)
        avg_emb = self.avg_encoder(averages)       # (batch, embed_dim)

        # Attend over all programs to find similar ones
        attended, attn_weights = self.attention(prog_emb, all_prog_embeddings)

        # Combine everything
        combined = torch.cat([uni_emb, prog_emb + attended, avg_emb], dim=-1)

        logits = self.classifier(combined)
        return torch.sigmoid(logits).squeeze(), attn_weights
```

---

### Updated 14-Day Schedule (PyTorch-First)

#### Phase 1: PyTorch Foundations (Days 1-3)

**Day 1: Tensors + Data Loading**
- PyTorch tensors, operations, GPU basics
- Load CSV data into PyTorch Dataset/DataLoader
- MAT223 as it arises: vectors, matrix multiplication
- MongoDB setup for raw storage

**Day 2: Autograd + First Model**
- Understand automatic differentiation
- Implement logistic regression from scratch in PyTorch
- Train with manual gradient descent
- MAT223: gradients, chain rule intuition

**Day 3: nn.Module + Training Loop**
- Convert to nn.Module class structure
- Use nn.BCEWithLogitsLoss, torch.optim.Adam
- Proper train/val/test split (temporal)
- CSC148: inheritance, class design

---

#### Phase 2: Core ML Models (Days 4-7)

**Day 4: Baseline + Evaluation**
- Beta-binomial baseline (in numpy for comparison)
- Implement calibration metrics: Brier, ECE
- Reliability diagrams with matplotlib

**Day 5: Feature Engineering**
- One-hot encoding vs embeddings comparison
- Spline basis for average (can use PyTorch)
- Custom Dataset class
- CSC148: ADTs, iterators

**Day 6: Embedding Models**
- nn.Embedding for universities and programs
- Compare to one-hot: fewer parameters, better generalization
- MAT223: low-rank factorization intuition

**Day 7: Regularization + Hyperparameters**
- Weight decay (L2 regularization)
- Dropout
- Learning rate scheduling
- Hyperparameter tuning with validation set

---

#### Phase 3: Advanced Models (Days 8-11)

**Day 8: Attention Basics**
- Understand Query-Key-Value
- Implement single-head attention
- MAT223: dot products as similarity, linear combinations

**Day 9: Attention Model for Admissions**
- Attention over similar programs
- Visualize attention weights
- Compare to embedding-only model

**Day 10: Multi-head Attention**
- Multiple attention heads
- Concatenate and project
- MAT223: subspaces, each head learns different "view"

**Day 11: Decision Timing Model**
- Discrete hazard model in PyTorch
- Or: regression model for decision week
- Survival curves

---

#### Phase 4: Production + Polish (Days 12-14)

**Day 12: Weaviate + Vector Search**
- Store learned embeddings in Weaviate
- Semantic search: "find similar programs"
- Compare to attention-based similarity

**Day 13: Model Comparison + Analysis**
- Compare all models: baseline, logistic, embedding, attention
- Error analysis by program/university
- Feature importance / attention visualization

**Day 14: API + Documentation**
- FastAPI endpoint
- Demo notebook
- Write-up connecting PyTorch to math concepts

---

## Data Overview

**Source:** 3 CSV files with 4,900 total rows
- `2022_2023_Canadian_University_Results.csv` (981 rows)
- `2023_2024_Canadian_University_Results.csv` (1,845 rows)
- `2024_2025_Canadian_University_Results.csv` (2,074 rows)

**Key Fields:**
- University, Program name, OUAC Code
- Decision (Accepted/Rejected/Waitlisted/Deferred)
- Grade averages (Grade 11, Grade 12, Top 6)
- Application date, Decision date
- Citizenship, Province

---

## Project Structure

```
Grade_Prediction_Project/
├── data/
│   ├── raw/                    # Original CSVs
│   └── processed/              # Cleaned data
├── src/
│   ├── math/                   # From-scratch linear algebra implementations
│   │   ├── vectors.py          # Vector operations, dot product, norms
│   │   ├── matrices.py         # Matrix operations
│   │   ├── projections.py      # Projections, least squares
│   │   ├── qr.py               # QR factorization
│   │   ├── svd.py              # SVD and low-rank approximation
│   │   └── ridge.py            # Ridge regression solver
│   ├── models/
│   │   ├── baseline.py         # Beta-binomial smoothing baseline
│   │   ├── logistic.py         # Logistic regression with IRLS
│   │   ├── hazard.py           # Discrete-time hazard for timing
│   │   └── embeddings.py       # Low-rank uni/program embeddings
│   ├── features/
│   │   ├── design_matrix.py    # Build X with splines, one-hots
│   │   └── encoders.py         # Categorical encoding utilities
│   ├── evaluation/
│   │   ├── calibration.py      # Brier, ECE, reliability plots
│   │   ├── discrimination.py   # ROC-AUC, PR-AUC
│   │   └── validation.py       # Temporal split logic
│   ├── db/
│   │   ├── mongo.py            # MongoDB connection and operations
│   │   ├── weaviate_client.py  # Weaviate vector DB operations
│   │   └── etl.py              # CSV to database loading
│   ├── visualization/
│   │   ├── math_viz.py         # Visualize projections, SVD, etc.
│   │   └── eval_viz.py         # Calibration curves, reliability plots
│   └── api/
│       └── predictor.py        # Main prediction interface
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_linear_algebra_foundations.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_calibration_evaluation.ipynb
│   ├── 06_decision_timing.ipynb
│   └── 07_embeddings.ipynb
├── tests/
│   ├── test_math/              # Unit tests for math implementations
│   └── test_models/            # Model correctness tests
├── docs/
│   └── math_notes.md           # Your proofs and derivations
├── config.yaml                 # Configuration (DB connections, hyperparams)
├── requirements.txt
└── README.md
```

---

## 14-Day Implementation Schedule

### Phase 1: Foundations (Days 1-3)

#### Day 1: Setup + Vectors + Dot Products
**Math (2hr):** Vectors, norms, inner products, orthogonality
**Code (4hr):**
- Project setup, virtual environment, dependencies
- Implement `vectors.py` from scratch
- Load and explore CSV data with pandas
**Verify (2hr):**
- Visualize vectors in 2D/3D
- Numerical verification of Cauchy-Schwarz
- Compare your implementations to numpy

#### Day 2: Projections + Least Squares Geometry
**Math (2hr):** Projections onto lines/subspaces, column space, least squares as projection
**Code (4hr):**
- Implement `projections.py`
- Start `design_matrix.py` (one-hot encoding basics)
- Setup MongoDB, create schema, load CSV data
**Verify (2hr):**
- Visualize projection of y onto col(X)
- Show residuals are orthogonal to columns

#### Day 3: Rank, Conditioning, QR Factorization
**Math (2hr):** Rank, nullspace, ill-conditioning, QR decomposition
**Code (4hr):**
- Implement `qr.py` (Householder reflections)
- Add rank/condition checking to design matrix
- Complete MongoDB ETL pipeline
**Verify (2hr):**
- Compare QR solution stability vs normal equations
- Test on near-collinear matrices

---

### Phase 2: Core Model (Days 4-7)

#### Day 4: Baseline Model + Beta Smoothing
**Math (2hr):** Beta-binomial posterior, credible intervals
**Code (4hr):**
- Implement `baseline.py` with isotonic smoothing
- Create average bins, compute admit rates
- First reliability plot
**Verify (2hr):**
- Check monotonicity across bins
- Compare to raw empirical rates

#### Day 5: Logistic Regression Theory
**Math (2hr):** Sigmoid, log-odds, gradient/Hessian derivation, convexity
**Code (4hr):**
- Implement `logistic.py` gradient and Hessian functions
- Start IRLS loop (weighted least squares each iteration)
**Verify (2hr):**
- Gradient check (finite differences)
- Show Hessian is PSD

#### Day 6: Ridge Regularization + IRLS Training
**Math (2hr):** Ridge as Tikhonov, why λI makes SPD, shrinkage geometry
**Code (4hr):**
- Implement `ridge.py` solver using QR
- Complete IRLS training loop with ridge
- Unit test against sklearn LogisticRegression
**Verify (2hr):**
- Plot loss curve decreasing each iteration
- Compare coefficients to sklearn

#### Day 7: SVD + Model Diagnostics
**Math (2hr):** SVD decomposition, singular values, ridge shrinkage view
**Code (4hr):**
- Implement basic SVD understanding code
- Add SVD-based diagnostics to model
- Visualize eigenvalue spectrum of X^T W X
**Verify (2hr):**
- Show ridge shrinks along small singular directions
- Experiment with λ selection

---

### Phase 3: Evaluation + Timing (Days 8-10)

#### Day 8: Temporal Validation + Calibration
**Math (2hr):** Brier score, ECE, proper scoring rules
**Code (4hr):**
- Implement temporal split (train 22-24, test 24-25)
- Implement `calibration.py` (Brier, ECE)
- Create reliability plots by average bin and program
**Verify (2hr):**
- Compute metrics on holdout
- Optional isotonic recalibration

#### Day 9: Decision Timing Model
**Math (2hr):** Discrete hazard, survival function, median/IQR
**Code (4hr):**
- Implement `hazard.py` with weekly buckets
- Logistic hazard with same features
- Predict median decision week
**Verify (2hr):**
- Validate timing predictions on 24-25 holdout
- Plot survival curves

#### Day 10: API + Integration
**Code (6hr):**
- Implement `predictor.py` main interface
- CLI or simple web endpoint
- Input validation, error handling
**Verify (2hr):**
- End-to-end testing
- Document usage

---

### Phase 4: Embeddings + Polish (Days 11-14)

#### Day 11: Weaviate Setup + Vector Concepts
**Math (2hr):** Low-rank factorization, bilinear terms u^T v
**Code (4hr):**
- Setup Weaviate, create schema for programs/universities
- Implement initial embedding vectors from SVD
**Verify (2hr):**
- Query similar programs by vector similarity
- Visualize embedding space (PCA/t-SNE)

#### Day 12: Embedding-based Model
**Math (2hr):** Alternating ridge least squares for U, V
**Code (4hr):**
- Implement `embeddings.py` with alternating optimization
- Replace one-hot with learned embeddings in model
**Verify (2hr):**
- Compare Brier/ECE: one-hot vs embeddings
- Analyze what embeddings learned

#### Day 13: Robustness + Error Analysis
**Code (6hr):**
- Sensitivity analysis (λ, spline knots, embedding dim)
- Per-program error analysis
- Guardrails for sparse programs
- Drift detection logic
**Verify (2hr):**
- Stress test edge cases
- Document limitations

#### Day 14: Documentation + Demo
**Code (4hr):**
- Clean up code, add docstrings
- Create demo notebook/script
- Write README with usage examples
**Verify (4hr):**
- Record demo
- Write 2-page math/systems summary
- Plan Pyko integration

---

## Learning Objectives by Topic

### A. Linear Algebra (MAT223)

| Topic | Brief | Project Use | Why Important |
|-------|-------|-------------|---------------|
| A1. Vectors & Norms | Length, angles, ‖x‖ | Feature scaling, gradient magnitudes | Foundation for everything |
| A2. Dot Product | x^T y, orthogonality | Predictions x^T β, similarity | Core of linear models |
| A3. Projections | proj_a(y), closest point | Least squares, IRLS step | Geometric understanding of fitting |
| A4. Column Space | col(X) = {Xβ} | Predictions live here | Understanding model capacity |
| A5. Rank & Nullspace | Dimension of col(X) | Drop reference levels | Prevent singular matrices |
| A6. Conditioning | cond(X), stability | Diagnose ill-conditioning | Explains exploding coefficients |
| A7. QR Factorization | X = QR, Q^T Q = I | Stable solver for IRLS | Never invert matrices |
| A8. Ridge | (X^T X + λI)^{-1} | Regularization | Guarantees SPD, better generalization |
| A9. Eigenvalues | Spectrum of X^T W X | Diagnose conditioning | Informs λ selection |
| A10. SVD | X = UΣV^T | Low-rank approximation, embeddings | Most powerful decomposition |

### B. Statistics & ML

| Topic | Brief | Project Use | Why Important |
|-------|-------|-------------|---------------|
| B1. Beta-Binomial | Posterior with prior | Baseline with uncertainty | Honest small-sample estimates |
| B2. Logistic Regression | p = σ(Xβ), IRLS | Core probability model | Clean convex optimization |
| B3. Calibration | Predicted ≈ observed | Brier, ECE, reliability | Trustworthiness |
| B4. Discrete Hazard | Weekly hazards h_t | "When will I hear back?" | Time-to-event prediction |
| B5. Embeddings | u^T v bilinear | Share info across rare cells | Handles sparsity |

### C. Python / Programming (NumPy, Pandas, etc.)

| Topic | Brief | Project Use | Why Important |
|-------|-------|-------------|---------------|
| C1. NumPy | Arrays, vectorization | All numerical code | Performance + clarity |
| C2. Pandas | DataFrames, ETL | Data loading, cleaning | Industry standard |
| C3. SciPy | linalg, stats | Reference implementations | Verify your code |
| C4. Matplotlib | Plotting | Math visualization | Learning reinforcement |
| C5. MongoDB | Document storage | Store applications | Flexible schema |
| C6. Weaviate | Vector search | Program embeddings | Semantic similarity |

---

## CSC148 Concepts Integration

This section maps CSC148 course concepts to specific implementations in the project.

### D1. Python Memory Model (CSC148 §1.1-1.2)

**Brief:** Understanding how Python stores objects in memory, how variables are references to objects, and how function parameters work.

**Project Implementation:**
```python
# Understanding that numpy arrays are mutable objects
# Changes to X inside a function affect the original!

def normalize_features(X):
    """
    BAD: Modifies X in place - caller's array changes!
    """
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    return X

def normalize_features_safe(X):
    """
    GOOD: Creates a copy - caller's array unchanged
    """
    X_normalized = X.copy()
    X_normalized[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    return X_normalized
```

**Why Important:**
- Avoid accidental data mutation when preprocessing
- Understand why `X.copy()` is sometimes necessary
- Debug issues where "my data changed mysteriously"

**Exercise:** Trace through what happens in memory when you do:
```python
X = np.array([[1, 2], [3, 4]])
Y = X  # Y is an alias, not a copy!
Y[0, 0] = 999  # This changes X too!
```

---

### D2. Function Design Recipe (CSC148 §1.3-1.5)

**Brief:** Systematic approach to writing functions with clear type annotations, docstrings, and preconditions.

**Project Implementation:**
```python
# In src/math/projections.py

def project_onto_subspace(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Project vector y onto the column space of X.

    Uses the normal equations: proj = X @ (X^T X)^{-1} @ X^T @ y

    Args:
        y: Target vector of shape (n,)
        X: Design matrix of shape (n, p) with full column rank

    Returns:
        The orthogonal projection of y onto col(X), shape (n,)

    Preconditions:
        - y.shape[0] == X.shape[0]  (same number of samples)
        - X has full column rank (rank(X) == X.shape[1])

    Raises:
        ValueError: If dimensions don't match
        LinAlgError: If X is rank-deficient

    Example:
        >>> X = np.array([[1, 0], [0, 1], [1, 1]])
        >>> y = np.array([1, 2, 3])
        >>> proj = project_onto_subspace(y, X)
        >>> np.allclose(proj, [1, 2, 3])  # y is in col(X)
        True
    """
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"Dimension mismatch: y has {y.shape[0]} samples, X has {X.shape[0]}")

    Q, R = np.linalg.qr(X)
    return Q @ (Q.T @ y)
```

**Apply to all functions:**
- Type annotations on every function
- Docstrings with Args, Returns, Preconditions, Example
- Input validation that raises meaningful errors

---

### D3. Object-Oriented Programming (CSC148 §3.1-3.4)

**Brief:** Organizing code into classes that bundle data (attributes) with behavior (methods).

**Project Implementation - Model Hierarchy:**
```python
# In src/models/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Representation Invariants:
        - self._is_fitted is True iff fit() has been called successfully
        - If fitted, self._feature_names contains the names of features used
    """

    def __init__(self):
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of admission for each sample."""
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _check_is_fitted(self) -> None:
        """Raise error if model hasn't been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")


# In src/models/baseline.py
class BetaBinomialBaseline(BaseModel):
    """
    Baseline model using Beta-binomial smoothing.

    Representation Invariants:
        - self._alpha > 0 and self._beta > 0 (prior parameters)
        - self._bin_stats maps (uni, program, avg_bin) -> (admits, rejects)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._bin_stats: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BetaBinomialBaseline':
        # Implementation...
        self._is_fitted = True
        return self


# In src/models/logistic.py
class RidgeLogistic(BaseModel):
    """
    Logistic regression with ridge regularization, trained via IRLS.

    Representation Invariants:
        - self._lambda >= 0 (regularization strength)
        - If fitted, self._beta has shape (n_features,)
        - self._max_iter > 0 and self._tol > 0
    """

    def __init__(self, lambda_: float = 0.1, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self._lambda = lambda_
        self._max_iter = max_iter
        self._tol = tol
        self._beta: np.ndarray | None = None
        self._training_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeLogistic':
        # IRLS implementation using your math functions...
        pass
```

---

### D4. Representation Invariants (CSC148 §3.2)

**Brief:** Conditions that must always be true about an object's internal state.

**Project Implementation:**
```python
# In src/features/design_matrix.py

class DesignMatrix:
    """
    Encapsulates the design matrix X with its metadata.

    Representation Invariants:
        - self._X.shape[1] == len(self._feature_names)
        - All columns of self._X are finite (no NaN or Inf)
        - self._X has full column rank (checked on construction)
        - If one-hot encoded, exactly one category is dropped per group
    """

    def __init__(self, X: np.ndarray, feature_names: list[str]):
        self._X = X
        self._feature_names = feature_names
        self._check_rep_invariants()

    def _check_rep_invariants(self) -> None:
        """Verify all representation invariants hold."""
        # Invariant 1: shapes match
        assert self._X.shape[1] == len(self._feature_names), \
            f"Mismatch: {self._X.shape[1]} columns but {len(self._feature_names)} names"

        # Invariant 2: no invalid values
        assert np.all(np.isfinite(self._X)), \
            "Design matrix contains NaN or Inf values"

        # Invariant 3: full column rank
        rank = np.linalg.matrix_rank(self._X)
        assert rank == self._X.shape[1], \
            f"Rank deficient: rank={rank}, columns={self._X.shape[1]}"

    @property
    def shape(self) -> Tuple[int, int]:
        return self._X.shape

    @property
    def condition_number(self) -> float:
        """Compute condition number of X^T X."""
        return np.linalg.cond(self._X.T @ self._X)
```

---

### D5. Inheritance (CSC148 §3.5-3.8)

**Brief:** Creating class hierarchies where subclasses inherit and extend behavior from parent classes.

**Project Implementation - Feature Encoders:**
```python
# In src/features/encoders.py

from abc import ABC, abstractmethod

class FeatureEncoder(ABC):
    """Abstract base class for feature encoders."""

    def __init__(self):
        self._is_fitted = False
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(self, values: list) -> 'FeatureEncoder':
        """Learn the encoding from training data."""
        pass

    @abstractmethod
    def transform(self, values: list) -> np.ndarray:
        """Transform values to encoded representation."""
        pass

    def fit_transform(self, values: list) -> np.ndarray:
        """Convenience method to fit and transform in one step."""
        return self.fit(values).transform(values)


class OneHotEncoder(FeatureEncoder):
    """
    One-hot encode categorical values, dropping one reference category.
    """

    def __init__(self, drop_first: bool = True):
        super().__init__()
        self._drop_first = drop_first
        self._categories: list[str] = []

    def fit(self, values: list) -> 'OneHotEncoder':
        self._categories = sorted(set(values))
        if self._drop_first and len(self._categories) > 1:
            self._categories = self._categories[1:]  # Drop first as reference
        self._feature_names = [f"is_{cat}" for cat in self._categories]
        self._is_fitted = True
        return self

    def transform(self, values: list) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted")
        result = np.zeros((len(values), len(self._categories)))
        for i, val in enumerate(values):
            if val in self._categories:
                result[i, self._categories.index(val)] = 1.0
        return result


class SplineEncoder(FeatureEncoder):
    """
    Piecewise linear spline basis for continuous features.
    """

    def __init__(self, n_knots: int = 5):
        super().__init__()
        self._n_knots = n_knots
        self._knots: np.ndarray | None = None

    def fit(self, values: list) -> 'SplineEncoder':
        values_arr = np.array(values)
        # Place knots at quantiles
        percentiles = np.linspace(0, 100, self._n_knots + 2)[1:-1]
        self._knots = np.percentile(values_arr, percentiles)
        self._feature_names = ['avg_linear'] + [f'avg_spline_{i}' for i in range(len(self._knots))]
        self._is_fitted = True
        return self

    def transform(self, values: list) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted")
        values_arr = np.array(values)
        # Linear term + piecewise terms
        result = np.zeros((len(values_arr), 1 + len(self._knots)))
        result[:, 0] = values_arr
        for i, knot in enumerate(self._knots):
            result[:, i + 1] = np.maximum(0, values_arr - knot)
        return result
```

---

### D6. Abstract Data Types (CSC148 §4.1-4.4)

**Brief:** Defining interfaces (what operations are available) separately from implementation (how they work).

**Project Implementation - Model Interface:**
```python
# In src/models/interface.py

from typing import Protocol, Tuple
import numpy as np

class AdmissionPredictor(Protocol):
    """
    Protocol (interface) for admission prediction models.
    Any class implementing these methods can be used as a predictor.
    """

    def predict_admission(
        self,
        average: float,
        university: str,
        program: str
    ) -> Tuple[float, float, float]:
        """
        Predict admission probability.

        Returns:
            (probability, ci_lower, ci_upper) - point estimate with 80% CI
        """
        ...

    def predict_timing(
        self,
        average: float,
        university: str,
        program: str
    ) -> Tuple[int, int, int]:
        """
        Predict decision timing in weeks from January.

        Returns:
            (median_week, iqr_lower, iqr_upper)
        """
        ...


# Now both Baseline and Logistic models can implement this interface
class AdmissionPredictorImpl:
    """Concrete implementation combining probability and timing models."""

    def __init__(
        self,
        prob_model: BaseModel,
        timing_model: BaseModel,
        feature_builder: 'FeatureBuilder'
    ):
        self._prob_model = prob_model
        self._timing_model = timing_model
        self._feature_builder = feature_builder

    def predict_admission(
        self,
        average: float,
        university: str,
        program: str
    ) -> Tuple[float, float, float]:
        X = self._feature_builder.build_single(average, university, program)
        prob = self._prob_model.predict_proba(X)[0]
        # Bootstrap or analytical CI
        ci_lower, ci_upper = self._compute_ci(X)
        return (prob, ci_lower, ci_upper)
```

---

### D7. Exceptions (CSC148 §5.1-5.4)

**Brief:** Handling errors gracefully using try/except/finally, raising meaningful exceptions.

**Project Implementation:**
```python
# In src/models/logistic.py

class ConvergenceError(Exception):
    """Raised when IRLS fails to converge."""
    pass

class RankDeficiencyError(Exception):
    """Raised when design matrix is rank-deficient."""
    pass


def fit_irls(X: np.ndarray, y: np.ndarray, lambda_: float = 0.1,
             max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Fit logistic regression using IRLS.

    Raises:
        RankDeficiencyError: If X is rank-deficient even with regularization
        ConvergenceError: If IRLS doesn't converge within max_iter
        ValueError: If inputs have invalid shapes
    """
    # Input validation
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1")

    # Check rank with regularization
    XtX_reg = X.T @ X + lambda_ * np.eye(X.shape[1])
    try:
        cond = np.linalg.cond(XtX_reg)
        if cond > 1e10:
            raise RankDeficiencyError(
                f"Design matrix is nearly singular (cond={cond:.2e}). "
                "Try increasing lambda or dropping collinear features."
            )
    except np.linalg.LinAlgError as e:
        raise RankDeficiencyError(f"SVD failed: {e}")

    # IRLS loop
    beta = np.zeros(X.shape[1])
    for iteration in range(max_iter):
        try:
            # Compute weights and working response
            eta = X @ beta
            p = 1 / (1 + np.exp(-eta))
            p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
            W = np.diag(p * (1 - p))
            z = eta + (y - p) / (p * (1 - p))

            # Solve weighted least squares
            beta_new = solve_weighted_ls(X, z, W, lambda_)

            # Check convergence
            if np.linalg.norm(beta_new - beta) < tol:
                return beta_new

            beta = beta_new

        except np.linalg.LinAlgError as e:
            raise ConvergenceError(f"IRLS failed at iteration {iteration}: {e}")

    raise ConvergenceError(
        f"IRLS did not converge after {max_iter} iterations. "
        f"Final change: {np.linalg.norm(beta_new - beta):.2e}"
    )


# Usage in API
def predict_admission_safe(average: float, uni: str, program: str) -> dict:
    """Safe prediction with error handling."""
    try:
        prob, ci_low, ci_high = predictor.predict_admission(average, uni, program)
        return {
            "status": "success",
            "probability": prob,
            "ci": [ci_low, ci_high]
        }
    except RankDeficiencyError:
        return {
            "status": "error",
            "message": "Model error - please contact support"
        }
    except ValueError as e:
        return {
            "status": "invalid_input",
            "message": str(e)
        }
```

---

### D8. Trees - Program Hierarchy (CSC148 §8.1-8.3)

**Brief:** Trees as hierarchical data structures with parent-child relationships.

**Project Implementation - University Program Tree:**
```python
# In src/models/program_tree.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProgramNode:
    """
    A node in the university/faculty/program hierarchy.

    Representation Invariants:
        - All children have self as their parent
        - self._admit_rate is None if no applications, else in [0, 1]
    """
    name: str
    level: str  # 'root', 'university', 'faculty', 'program'
    parent: Optional[ProgramNode] = None
    children: list[ProgramNode] = None
    admit_count: int = 0
    reject_count: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def admit_rate(self) -> Optional[float]:
        """Empirical admit rate, or None if no data."""
        total = self.admit_count + self.reject_count
        return self.admit_count / total if total > 0 else None

    def add_child(self, child: ProgramNode) -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def find_program(self, uni_name: str, program_name: str) -> Optional[ProgramNode]:
        """
        Recursively search for a program in this tree.
        Uses depth-first search.
        """
        if self.level == 'program' and self.name == program_name:
            # Check if ancestor is the right university
            node = self.parent
            while node is not None:
                if node.level == 'university' and node.name == uni_name:
                    return self
                node = node.parent
            return None

        for child in self.children:
            result = child.find_program(uni_name, program_name)
            if result is not None:
                return result

        return None

    def aggregate_rates(self) -> None:
        """
        Recursively aggregate admit/reject counts from children.
        Post-order traversal: process children first, then self.
        """
        for child in self.children:
            child.aggregate_rates()
            self.admit_count += child.admit_count
            self.reject_count += child.reject_count


# Build tree from database
def build_program_tree(applications: list[dict]) -> ProgramNode:
    """
    Build hierarchical tree from flat application records.

    Example structure:
        Root
        ├── University of Toronto
        │   ├── Engineering
        │   │   ├── Computer Engineering
        │   │   └── Electrical Engineering
        │   └── Arts & Science
        │       ├── Computer Science
        │       └── Mathematics
        └── University of Waterloo
            └── ...
    """
    root = ProgramNode(name="Canada", level="root")
    unis = {}

    for app in applications:
        uni_name = app['university']
        program_name = app['program']
        faculty_name = app.get('faculty', 'General')

        # Get or create university node
        if uni_name not in unis:
            uni_node = ProgramNode(name=uni_name, level="university")
            root.add_child(uni_node)
            unis[uni_name] = {'node': uni_node, 'faculties': {}}

        # Get or create faculty node
        if faculty_name not in unis[uni_name]['faculties']:
            fac_node = ProgramNode(name=faculty_name, level="faculty")
            unis[uni_name]['node'].add_child(fac_node)
            unis[uni_name]['faculties'][faculty_name] = {'node': fac_node, 'programs': {}}

        # Get or create program node
        if program_name not in unis[uni_name]['faculties'][faculty_name]['programs']:
            prog_node = ProgramNode(name=program_name, level="program")
            unis[uni_name]['faculties'][faculty_name]['node'].add_child(prog_node)
            unis[uni_name]['faculties'][faculty_name]['programs'][program_name] = prog_node

        # Update counts
        prog_node = unis[uni_name]['faculties'][faculty_name]['programs'][program_name]
        if app['decision'] == 'Accepted':
            prog_node.admit_count += 1
        else:
            prog_node.reject_count += 1

    # Aggregate rates up the tree
    root.aggregate_rates()
    return root
```

**Use for partial pooling:** When a program has few samples, borrow strength from faculty/university level rates.

---

### D9. Recursion (CSC148 §7.1-7.8)

**Brief:** Functions that call themselves, operating on recursively-defined data structures.

**Project Implementation - Tree Traversal and Aggregation:**
```python
# In src/models/program_tree.py (continued)

def collect_all_programs(node: ProgramNode) -> list[ProgramNode]:
    """
    Recursively collect all program-level nodes.

    Base case: If node is a program, return [node]
    Recursive case: Collect from all children and concatenate
    """
    if node.level == 'program':
        return [node]

    result = []
    for child in node.children:
        result.extend(collect_all_programs(child))
    return result


def compute_smoothed_rate(node: ProgramNode, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Compute Beta-smoothed admit rate with hierarchical prior.

    Uses parent's rate as informative prior (empirical Bayes).
    Recursively computes parent rates first.
    """
    # Base case: root uses uninformative prior
    if node.parent is None:
        prior_rate = 0.5
    else:
        # Recursive case: get parent's smoothed rate
        prior_rate = compute_smoothed_rate(node.parent, alpha, beta)

    # Compute posterior mean with prior centered on parent's rate
    n = node.admit_count + node.reject_count
    if n == 0:
        return prior_rate

    # Shrink toward parent: more shrinkage when n is small
    shrinkage = alpha + beta
    posterior_rate = (node.admit_count + shrinkage * prior_rate) / (n + shrinkage)
    return posterior_rate


def tree_to_dict(node: ProgramNode, depth: int = 0) -> dict:
    """
    Recursively convert tree to nested dictionary for JSON serialization.
    """
    result = {
        'name': node.name,
        'level': node.level,
        'admit_rate': node.admit_rate,
        'sample_size': node.admit_count + node.reject_count
    }

    if node.children:
        result['children'] = [tree_to_dict(child, depth + 1) for child in node.children]

    return result
```

---

### D10. Algorithm Analysis (CSC148 §4.4, §8.8, §9.2)

**Brief:** Analyzing time complexity using Big-O notation.

**Project Implementation - Complexity Analysis:**
```python
# Document complexity of key operations

class DesignMatrix:
    """
    Time Complexity Analysis:

    build_X(n_samples, n_universities, n_programs):
        - One-hot encoding: O(n_samples * n_universities + n_samples * n_programs)
        - Spline basis: O(n_samples * n_knots)
        - Total: O(n_samples * (n_universities + n_programs + n_knots))
        - Space: O(n_samples * n_features) where n_features ~ n_universities + n_programs

    For our data: n_samples ~ 5000, n_universities ~ 50, n_programs ~ 200
        - Design matrix: 5000 × 250 ≈ 1.25M entries (fits in memory easily)
    """
    pass


class RidgeLogistic:
    """
    Time Complexity Analysis:

    fit(X: n×p, y: n):
        - Each IRLS iteration:
            - Compute p = sigmoid(Xβ): O(np)
            - Form weighted X^T W X: O(np²) [dominating term]
            - QR factorization: O(np²)
            - Back-substitution: O(p²)
        - Total per iteration: O(np²)
        - K iterations: O(K * np²)

    For our data: n=5000, p=250, K~10
        - Per fit: ~10 * 5000 * 250² ≈ 3 billion operations
        - Takes ~1-2 seconds on modern CPU

    predict_proba(X_new: m×p):
        - Just matrix-vector multiply: O(mp)
        - Very fast for single predictions
    """
    pass


# When to use which data structure

"""
Lookup by (university, program, avg_bin):
    - Dict/HashMap: O(1) average
    - Tree traversal: O(log n) to O(n)
    → Use dict for prediction cache

Find similar programs by embedding:
    - Brute force: O(n * d) for n programs, d dimensions
    - Weaviate (HNSW index): O(log n) approximate
    → Use Weaviate for semantic search
"""
```

---

### D11. Testing Your Code (CSC148 §2.1-2.4)

**Brief:** Writing comprehensive test cases, understanding code coverage, property-based testing.

**Project Implementation:**
```python
# In tests/test_math/test_projections.py

import pytest
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from src.math.projections import project_onto_subspace, solve_via_qr


class TestProjections:
    """Test suite for projection functions."""

    # --- Basic test cases ---

    def test_project_onto_single_column(self):
        """Project onto a 1D subspace (a line)."""
        X = np.array([[1], [0], [0]])
        y = np.array([1, 1, 1])
        proj = project_onto_subspace(y, X)
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(proj, expected)

    def test_project_y_already_in_column_space(self):
        """If y is in col(X), projection should equal y."""
        X = np.array([[1, 0], [0, 1], [0, 0]])
        y = np.array([3, 4, 0])  # = 3*col1 + 4*col2
        proj = project_onto_subspace(y, X)
        np.testing.assert_array_almost_equal(proj, y)

    def test_residual_orthogonal_to_columns(self):
        """Core property: residual must be orthogonal to all columns."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        proj = project_onto_subspace(y, X)
        residual = y - proj
        for j in range(X.shape[1]):
            dot_product = np.dot(residual, X[:, j])
            assert abs(dot_product) < 1e-10, f"Residual not orthogonal to column {j}"

    # --- Edge cases ---

    def test_empty_matrix_raises(self):
        """Empty design matrix should raise error."""
        with pytest.raises(ValueError):
            X = np.array([]).reshape(0, 0)
            y = np.array([])
            project_onto_subspace(y, X)

    def test_rank_deficient_matrix_raises(self):
        """Rank-deficient matrix should raise error."""
        with pytest.raises(Exception):  # Could be LinAlgError or custom
            X = np.array([[1, 1], [2, 2], [3, 3]])  # Rank 1, not 2
            y = np.array([1, 2, 3])
            project_onto_subspace(y, X)

    # --- Property-based testing ---

    @given(arrays(np.float64, shape=(50, 5), elements=st.floats(-100, 100, allow_nan=False)))
    def test_projection_is_idempotent(self, X):
        """Projecting twice should give same result as projecting once."""
        # Skip if rank-deficient
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return

        y = np.random.randn(X.shape[0])
        proj1 = project_onto_subspace(y, X)
        proj2 = project_onto_subspace(proj1, X)
        np.testing.assert_array_almost_equal(proj1, proj2)

    @given(arrays(np.float64, shape=(50, 5), elements=st.floats(-100, 100, allow_nan=False)))
    def test_projection_minimizes_distance(self, X):
        """Projection should be the closest point in col(X) to y."""
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return

        y = np.random.randn(X.shape[0])
        proj = project_onto_subspace(y, X)
        dist_to_proj = np.linalg.norm(y - proj)

        # Any other point in col(X) should be farther
        for _ in range(10):
            beta_random = np.random.randn(X.shape[1])
            other_point = X @ beta_random
            dist_to_other = np.linalg.norm(y - other_point)
            assert dist_to_proj <= dist_to_other + 1e-10


# In tests/test_models/test_logistic.py

class TestLogisticRegression:
    """Test logistic regression against sklearn."""

    def test_matches_sklearn(self):
        """Our IRLS should match sklearn's results."""
        from sklearn.linear_model import LogisticRegression
        from src.models.logistic import RidgeLogistic

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(500, 5)
        true_beta = np.array([1, -1, 0.5, 0, 0.2])
        p = 1 / (1 + np.exp(-X @ true_beta))
        y = (np.random.rand(500) < p).astype(int)

        # Fit both models
        lambda_ = 0.1
        our_model = RidgeLogistic(lambda_=lambda_).fit(X, y)
        sklearn_model = LogisticRegression(C=1/lambda_, penalty='l2', solver='lbfgs')
        sklearn_model.fit(X, y)

        # Compare predictions (not coefficients, due to different conventions)
        our_probs = our_model.predict_proba(X)
        sklearn_probs = sklearn_model.predict_proba(X)[:, 1]

        np.testing.assert_array_almost_equal(our_probs, sklearn_probs, decimal=4)
```

---

## Database Schema

### MongoDB Collections

```javascript
// universities collection
{
  _id: ObjectId,
  name: "University of Toronto",
  aliases: ["UofT", "U of T"],
  province: "Ontario"
}

// programs collection
{
  _id: ObjectId,
  name: "Computer Science",
  ouac_code: "TCS",
  university_id: ObjectId,
  faculty: "Arts & Science"
}

// applications collection
{
  _id: ObjectId,
  university_id: ObjectId,
  program_id: ObjectId,
  cycle: "2023-2024",
  decision: "Accepted",  // Accepted, Rejected, Waitlisted, Deferred
  grade_11_avg: 92.5,
  grade_12_midterm: 94.0,
  top_6_avg: 93.5,
  application_date: ISODate,
  decision_date: ISODate,
  citizenship: "Canadian",
  province: "Ontario",
  supplementary: true
}

// predictions collection (cache)
{
  _id: ObjectId,
  university_id: ObjectId,
  program_id: ObjectId,
  avg_bin: "90-92",
  p_admit: 0.73,
  ci_lower: 0.65,
  ci_upper: 0.81,
  median_week: 12,
  iqr_weeks: [10, 15],
  model_version: "v1.0",
  computed_at: ISODate
}
```

### Weaviate Schema

```javascript
// Program class with embedding
{
  class: "Program",
  properties: [
    { name: "name", dataType: ["text"] },
    { name: "university", dataType: ["text"] },
    { name: "faculty", dataType: ["text"] },
    { name: "avg_admit_rate", dataType: ["number"] },
    { name: "avg_decision_weeks", dataType: ["number"] }
  ],
  vectorizer: "none"  // We'll provide our own learned embeddings
}
```

---

## Critical Files to Create

### Day 1-2: Foundation
1. `src/math/vectors.py` - Vector operations from scratch
2. `src/math/projections.py` - Projection and least squares
3. `src/db/mongo.py` - MongoDB connection
4. `src/db/etl.py` - CSV loading pipeline

### Day 3-4: Design Matrix + Baseline
5. `src/features/design_matrix.py` - Build X with splines, one-hots
6. `src/math/qr.py` - QR factorization
7. `src/models/baseline.py` - Beta-binomial baseline

### Day 5-7: Core Model
8. `src/models/logistic.py` - IRLS logistic regression
9. `src/math/ridge.py` - Ridge solver

### Day 8-10: Evaluation + Timing
10. `src/evaluation/calibration.py` - Brier, ECE, plots
11. `src/evaluation/validation.py` - Temporal split
12. `src/models/hazard.py` - Decision timing

### Day 11-14: Embeddings + API
13. `src/db/weaviate_client.py` - Vector DB operations
14. `src/models/embeddings.py` - Low-rank embeddings
15. `src/api/predictor.py` - Main prediction interface

---

## Key Mathematical Proofs to Understand

1. **Cauchy-Schwarz:** |x^T y| ≤ ‖x‖ ‖y‖
2. **Projection orthogonality:** Residual r = y - proj(y) is orthogonal to subspace
3. **Normal equations:** X^T(y - Xβ) = 0 gives least squares solution
4. **Ridge SPD:** X^T WX + λI is symmetric positive definite for λ > 0
5. **SVD shrinkage:** Ridge solution β_λ = Σ_i (σ_i²/(σ_i² + λ)) <y, u_i> v_i
6. **Logistic convexity:** Hessian X^T W X is PSD → unique minimizer
7. **Isotonic monotonicity:** Projection onto monotone cone preserves order
8. **Hazard → survival:** S(t) = Π_{k≤t}(1 - h_k)

---

## Detailed Day 1-2 MAT223 Learning Breakdown

### Day 1: Vectors, Dot Products, Projections (Foundation Day)

**Goal:** Think in vectors, compute projections, understand least squares as "best approximation"

---

#### Block 1: Vectors and Basic Operations (60 min)
**MAT223 Reference:** Section 4.1

**Learn:**
- Vector as ordered tuple: x = (x₁, x₂, ..., xₙ) ∈ ℝⁿ
- Vector addition: x + y = (x₁ + y₁, ..., xₙ + yₙ)
- Scalar multiplication: αx = (αx₁, ..., αxₙ)
- Linear combination: αx + βy

**Project Connection:**
- Each student application is a feature vector x = (avg, is_toronto, is_waterloo, ..., is_cs, is_eng, ...)
- Your design matrix X has rows that are these vectors
- Model predictions Xβ are linear combinations of feature columns

**Do (by hand + code):**
```python
# Implement from scratch in vectors.py
def add(x, y): ...
def scale(alpha, x): ...
def linear_combination(coeffs, vectors): ...
```

**Verify:**
- Test against numpy: `np.allclose(your_result, np_result)`
- Visualize 2D vectors with matplotlib arrows

---

#### Block 2: Dot Product, Norms, Distance (60 min)
**MAT223 Reference:** Section 4.2, 4.2.1

**Learn:**
- Dot product: ⟨x, y⟩ = x^T y = Σᵢ xᵢyᵢ
- Euclidean norm: ‖x‖ = √(x^T x)
- Distance: d(x, y) = ‖x - y‖
- Orthogonality: x ⊥ y ⟺ x^T y = 0
- Angle: cos(θ) = (x^T y) / (‖x‖ ‖y‖)

**Project Connection:**
- Prediction: ŷ = x^T β is a dot product!
- Loss measures distance: ‖y - Xβ‖² is sum of squared distances
- Embeddings: similarity(program_a, program_b) = u_a^T u_b

**Do:**
```python
# In vectors.py
def dot(x, y): ...
def norm(x): ...
def distance(x, y): ...
def angle(x, y): ...
def is_orthogonal(x, y, tol=1e-10): ...
```

**Key Theorem - Cauchy-Schwarz:**
|x^T y| ≤ ‖x‖ ‖y‖

**Verify:**
```python
# Test Cauchy-Schwarz on random vectors
x, y = np.random.randn(10), np.random.randn(10)
assert np.abs(dot(x, y)) <= norm(x) * norm(y) + 1e-10
```

---

#### Block 3: Projection onto a Line (60 min)
**MAT223 Reference:** Section 4.2.2

**Learn:**
- Projection of y onto nonzero vector a:
  ```
  proj_a(y) = (y^T a / a^T a) * a
  ```
- Residual: r = y - proj_a(y)
- Key property: r ⊥ a (residual is orthogonal to a)

**Project Connection:**
- This is the simplest case of least squares!
- If X has one column a, then β̂ = y^T a / a^T a
- Generalizes to: project y onto column space of X

**Do:**
```python
# In projections.py
def proj_onto_vector(y, a):
    """Project y onto the line spanned by a"""
    return (dot(y, a) / dot(a, a)) * a

def residual(y, a):
    """Compute y - proj_a(y)"""
    return y - proj_onto_vector(y, a)
```

**Verify:**
```python
# Residual must be orthogonal to a
r = residual(y, a)
assert is_orthogonal(r, a)  # Should be True!
```

**Visualize:**
- Plot y, a, proj_a(y), and r in 2D
- Show the right angle between r and a

---

#### Block 4: Column Space and Spans (45 min)
**MAT223 Reference:** Section 1.4.2, 4.4

**Learn:**
- Span of vectors: all linear combinations
  ```
  span{v₁, ..., vₖ} = {α₁v₁ + ... + αₖvₖ : αᵢ ∈ ℝ}
  ```
- Column space of matrix X:
  ```
  col(X) = {Xβ : β ∈ ℝᵖ} = span of columns of X
  ```
- Predictions Xβ live in col(X)

**Project Connection:**
- Your model can only predict values in the column space
- More features (columns) = larger column space = more expressive
- But too many columns → overfitting (tomorrow's topic)

**Do:**
- Take X with 2 columns, visualize col(X) as a plane in ℝ³
- Show that any Xβ lies on this plane

---

#### Block 5: Least Squares as Closest Point (45 min)
**MAT223 Reference:** Section 4.8 (intro)

**Learn:**
- Least squares problem: min_β ‖Xβ - y‖²
- Geometric interpretation: find the closest point in col(X) to y
- Solution: project y onto col(X)

**Project Connection:**
- This is how we'll fit logistic regression (via IRLS)
- Each IRLS step solves a weighted least squares problem
- Understanding this geometry is crucial!

**Do:**
- Derive for single-column case (reduces to Block 3)
- Sketch the geometry: y above a plane, projection is closest point

**Preview (for tomorrow):**
- Normal equations: X^T(y - Xβ̂) = 0
- This says: residual is orthogonal to ALL columns of X

---

### Day 2: Normal Equations, Rank, QR Factorization

**Goal:** Solve least squares stably, understand when solutions exist/are unique

---

#### Block 1: Projection onto Subspaces (60 min)
**MAT223 Reference:** Section 4.2.2, 4.4.1

**Learn:**
- Generalize projection from line to subspace
- Project y onto col(X) where X has multiple columns
- Key insight: residual must be orthogonal to EVERY column

**Derive the Normal Equations:**
```
r = y - Xβ̂  must satisfy  r ⊥ col(X)
⟺ X^T r = 0
⟺ X^T (y - Xβ̂) = 0
⟺ X^T X β̂ = X^T y
```

**Project Connection:**
- This is THE equation we solve in least squares
- Ridge: (X^T X + λI) β̂ = X^T y
- IRLS: (X^T W X + λI) β̂ = X^T W z (weighted version)

**Do:**
```python
# In projections.py
def solve_normal_equations(X, y):
    """Solve X^T X β = X^T y (naive version)"""
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)
```

**Verify:**
```python
# Residual should be orthogonal to all columns
beta = solve_normal_equations(X, y)
residual = y - X @ beta
for j in range(X.shape[1]):
    assert is_orthogonal(residual, X[:, j])
```

---

#### Block 2: Rank and Linear Independence (60 min)
**MAT223 Reference:** Section 1.3.2, 4.6

**Learn:**
- Rank = dimension of column space = number of independent columns
- Full column rank: rank(X) = p (number of columns)
- Rank deficient: some columns are linear combinations of others

**When does X^T X have an inverse?**
- Only when X has full column rank
- Otherwise: infinitely many solutions (non-unique β̂)

**Project Connection:**
- One-hot encoding with intercept: columns sum to 1 → rank deficient!
- Example: if is_toronto + is_waterloo + is_mcgill = 1 (always), then:
  - The intercept column is a linear combination
  - X^T X is singular (no unique inverse)
- Solution: drop one reference category per group

**Do:**
```python
# In matrices.py
def compute_rank(X, tol=1e-10):
    """Compute rank via SVD"""
    _, s, _ = np.linalg.svd(X)
    return np.sum(s > tol)

def check_full_column_rank(X):
    return compute_rank(X) == X.shape[1]
```

**Example to try:**
```python
# Rank-deficient design matrix
X = np.array([
    [1, 1, 0],  # intercept, is_A, is_B
    [1, 0, 1],
    [1, 1, 0],
    [1, 0, 1]
])
# Column 0 = Column 1 + Column 2! Rank = 2, not 3
```

---

#### Block 3: Conditioning and Stability (45 min)
**MAT223 Reference:** Section 4.6

**Learn:**
- Condition number: cond(A) = ‖A‖ · ‖A⁻¹‖
- For symmetric PD: cond(A) = λ_max / λ_min
- High condition number → small input changes cause large output changes

**Why this matters:**
- If cond(X^T X) is huge, solving normal equations is unstable
- Small numerical errors get amplified
- Ridge helps: cond(X^T X + λI) ≤ (λ_max + λ) / λ

**Project Connection:**
- Near-collinear features (e.g., grade_11 and grade_12 highly correlated)
- Rare programs with few samples → nearly zero columns
- Ridge regularization fixes both problems!

**Do:**
```python
# In matrices.py
def condition_number(A):
    """Compute condition number"""
    _, s, _ = np.linalg.svd(A)
    return s[0] / s[-1] if s[-1] > 1e-15 else np.inf

# Check conditioning of your design matrix
print(f"cond(X^T X) = {condition_number(X.T @ X)}")
print(f"cond(X^T X + λI) = {condition_number(X.T @ X + 0.1 * np.eye(X.shape[1]))}")
```

---

#### Block 4: QR Factorization (90 min)
**MAT223 Reference:** Section 4.7, 4.8

**Learn:**
- Factor X = QR where:
  - Q has orthonormal columns: Q^T Q = I
  - R is upper triangular
- Why orthonormal columns are great:
  - Q^T Q = I makes inverses trivial
  - Projection: proj_{col(Q)}(y) = Q Q^T y

**Solving least squares with QR:**
```
X β = y  (approximately)
Q R β = y
R β = Q^T y  (multiply both sides by Q^T, using Q^T Q = I)
```
Now solve triangular system R β = Q^T y by back-substitution!

**Why QR is better than normal equations:**
- Never form X^T X (which squares the condition number)
- cond(R) = cond(X), not cond(X)²
- More numerically stable

**Project Connection:**
- This is how we'll solve each IRLS step
- Never call `np.linalg.inv(X.T @ X)` — use QR instead!

**Do:**
```python
# In qr.py - implement Householder QR
def householder_qr(A):
    """
    Compute QR factorization using Householder reflections
    Returns Q, R such that A = QR
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for j in range(min(m, n)):
        # Compute Householder vector for column j
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply Householder reflection to R
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])

        # Accumulate Q
        Qj = np.eye(m)
        Qj[j:, j:] -= 2 * np.outer(v, v)
        Q = Q @ Qj

    return Q, R

def solve_via_qr(X, y):
    """Solve least squares using QR factorization"""
    Q, R = householder_qr(X)
    # Solve R β = Q^T y
    Qty = Q.T @ y
    # Back substitution for upper triangular R
    return back_substitute(R[:X.shape[1], :], Qty[:X.shape[1]])

def back_substitute(R, b):
    """Solve upper triangular system Rx = b"""
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    return x
```

**Verify:**
```python
# Compare QR solution to numpy
X = np.random.randn(100, 5)
y = np.random.randn(100)

beta_qr = solve_via_qr(X, y)
beta_np = np.linalg.lstsq(X, y, rcond=None)[0]

assert np.allclose(beta_qr, beta_np)
print("QR solution matches numpy!")
```

---

#### Block 5: MongoDB Setup + ETL (remaining time)

**Do:**
1. Install MongoDB locally or use MongoDB Atlas (free tier)
2. Create database `admissions_db`
3. Create collections: `universities`, `programs`, `applications`
4. Write ETL script to load the 3 CSV files

```python
# In db/etl.py
from pymongo import MongoClient
import pandas as pd

def load_csv_to_mongo(csv_path, collection, transform_fn):
    """Load CSV data into MongoDB collection"""
    df = pd.read_csv(csv_path)
    df = transform_fn(df)  # Clean and transform
    records = df.to_dict('records')
    collection.insert_many(records)
    return len(records)
```

---

### Day 1-2 Exit Checklist

**Math Understanding:**
- [ ] Can compute dot products and norms by hand
- [ ] Can project a vector onto a line and verify orthogonality
- [ ] Understand column space as "where predictions live"
- [ ] Can derive normal equations from orthogonality condition
- [ ] Understand why rank deficiency breaks least squares
- [ ] Know why QR is more stable than normal equations

**Code Artifacts:**
- [ ] `vectors.py` with dot, norm, distance, is_orthogonal
- [ ] `projections.py` with proj_onto_vector, solve_normal_equations
- [ ] `matrices.py` with compute_rank, condition_number
- [ ] `qr.py` with householder_qr, solve_via_qr, back_substitute
- [ ] `db/mongo.py` with connection setup
- [ ] `db/etl.py` with CSV loading functions

**Numerical Verifications:**
- [ ] All functions match numpy equivalents
- [ ] Cauchy-Schwarz holds on random vectors
- [ ] Residuals are orthogonal to columns of X
- [ ] QR solution matches np.linalg.lstsq

---

## Success Criteria

### Technical
- [ ] Baseline model produces admit rates with 80% credible intervals
- [ ] IRLS logistic matches sklearn to 4 decimal places
- [ ] Brier score < 0.20 on 2024-25 holdout
- [ ] ECE < 0.05 within average bins
- [ ] Decision timing median within 2 weeks of actual for 70% of cases
- [ ] Embeddings reduce model parameters by >50% without hurting calibration

### Learning
- [ ] Can explain projection as "closest point in column space"
- [ ] Can derive gradient and Hessian of logistic loss
- [ ] Can explain why ridge prevents overfitting geometrically
- [ ] Can implement QR factorization and explain stability benefit
- [ ] Comfortable with numpy/pandas for numerical computing
- [ ] Can query MongoDB and Weaviate programmatically

---

## Dependencies

```
# requirements.txt

# PyTorch (PRIMARY)
torch>=2.0.0
torchvision>=0.15.0  # For transforms, if needed

# Data & Numerical
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0

# Databases
pymongo>=4.3.0
weaviate-client>=3.21.0

# ML Utilities
scikit-learn>=1.2.0  # For metrics, comparison

# Development
pyyaml>=6.0
pytest>=7.3.0
jupyter>=1.0.0
tqdm>=4.65.0  # Progress bars for training
tensorboard>=2.12.0  # Training visualization (optional)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Data quality issues | Day 1-2: Thorough EDA, handle missing values |
| Math implementation bugs | Compare every function to numpy/scipy |
| Sparse program cells | Use Beta smoothing + partial pooling |
| IRLS convergence issues | Add line search, convergence checks |
| MongoDB setup problems | Fallback: Use local JSON files temporarily |
| Time pressure | Baseline model works by Day 4; rest is enhancement |

---

## Pyko Integration Notes

For future integration into Pyko:
1. Export trained model weights as JSON/pickle
2. Expose prediction API that matches Pyko's university model structure
3. Store embeddings in Weaviate for semantic program search
4. Add drift monitoring to trigger recalibration alerts
5. Include calibration badge in API responses

---

## Validated MAT223 Concepts for This Project

Based on the MAT223 lecture notes table of contents, here are the **exact sections** you need:

### Must Learn (Core to Project)

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| **4.1** | Vectors | Feature vectors, predictions | Day 1 |
| **4.2** | Dot Product and Projections | Predictions x^T β, similarity | Day 1 |
| **4.2.2** | Projections onto Lines | Simplest least squares case | Day 1 |
| **4.4** | Subspaces | Column space col(X) | Day 1 |
| **4.4.1** | Span | Understanding model expressiveness | Day 1 |
| **4.5** | Linear Independence | Dropping reference categories | Day 2 |
| **4.6** | Rank of a Matrix | Checking full column rank | Day 2 |
| **4.7** | Orthonormal Bases | QR factorization foundation | Day 2 |
| **4.7.1** | Projections Onto Subspaces | Normal equations derivation | Day 2 |
| **4.7.2** | Gram-Schmidt (conceptual) | Understanding QR | Day 2 |
| **4.8** | Approximating Solutions | Least squares, IRLS | Day 2 |
| **1.3.2** | Rank of a Matrix | Singular matrix diagnosis | Day 2 |
| **1.4.2** | Column Vectors | Design matrix columns | Day 1 |
| **1.4.4** | Matrix Multiplication | Xβ predictions | Day 1 |

### Should Learn (Improves Understanding)

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| **3.4** | Eigenvalues/Eigenvectors | SVD understanding, conditioning | Day 6-7 |
| **3.6** | Applications | Ridge shrinkage geometry | Day 6-7 |
| **1.6** | Matrix Inversion | Why NOT to use it (prefer QR) | Day 2 |
| **1.3.1** | Gaussian Elimination | Row reduction understanding | Optional |

### Can Skip for This Project

| Section | Topic | Why Skip |
|---------|-------|----------|
| 2.x | Complex Numbers | Not used in real-valued ML |
| 3.1-3.3 | Determinants | Not directly needed for IRLS/ridge |
| 3.5 | Diagonalization | Nice to know but not essential |
| 4.3 | Cross Product and Planes | 3D geometry, not ML relevant |
| 1.5 | Linear Transformations | Conceptually useful but not core |

---

## Validated CSC148 Concepts for This Project

Based on the CSC148 course notes, here are the sections to focus on:

### Must Implement

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| **1.3-1.5** | Function Design Recipe | All functions with docstrings, types | Day 1+ |
| **3.1-3.4** | OOP + Representation Invariants | Model classes, DesignMatrix | Day 3+ |
| **3.5-3.8** | Inheritance | BaseModel → LogisticModel hierarchy | Day 5+ |
| **5.1-5.4** | Exceptions | ConvergenceError, RankDeficiencyError | Day 6+ |
| **2.1-2.4** | Testing | Unit tests for math functions | Throughout |

### Should Implement

| Section | Topic | Project Use | Priority |
|---------|-------|-------------|----------|
| **8.1-8.3** | Trees | Program hierarchy for partial pooling | Day 4 |
| **7.1-7.6** | Recursion | Tree traversal, aggregation | Day 4 |
| **4.1-4.2** | ADTs | Model interface/protocol | Day 10 |
| **4.4** | Running Time Analysis | Documenting O(np²) for IRLS | Day 6 |

### Can Reference but Not Implement from Scratch

| Section | Topic | Why Reference Only |
|---------|-------|-------------------|
| 6.x | Linked Lists | Use Python lists/numpy instead |
| 9.x | Recursive Sorting | Use numpy's built-in sorting |
| 4.2-4.3 | Stacks/Queues | Not core to this project |

---

## Quick Start Checklist (Day 1 Morning)

```bash
# 1. Create project structure
mkdir -p Grade_Prediction_Project/{data/raw,data/processed,src/{math,models,features,evaluation,db,visualization,api},notebooks,tests/{test_math,test_models},docs}
cd Grade_Prediction_Project

# 2. Initialize virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install numpy pandas scipy matplotlib plotly pymongo weaviate-client scikit-learn pytest hypothesis jupyter pyyaml

# 4. Copy CSV data
cp ~/Downloads/2022_2023_Canadian_University_Results.csv data/raw/
cp ~/Downloads/2023_2024_Canadian_University_Results.csv data/raw/
cp ~/Downloads/2024_2025_Canadian_University_Results.csv data/raw/

# 5. Create config.yaml
cat > config.yaml << 'EOF'
database:
  mongodb_uri: "mongodb://localhost:27017"
  db_name: "admissions_db"
  weaviate_url: "http://localhost:8080"

model:
  lambda: 0.1
  max_iter: 100
  n_spline_knots: 5

validation:
  train_cycles: ["2022-2023", "2023-2024"]
  test_cycle: "2024-2025"
EOF

# 6. Start with notebook exploration
jupyter notebook
```

---

## Files to Create (Ordered)

| Order | File | Purpose | Day |
|-------|------|---------|-----|
| 1 | `src/math/vectors.py` | Vector ops from scratch | 1 |
| 2 | `src/db/etl.py` | Load CSVs to MongoDB | 1 |
| 3 | `src/math/projections.py` | Projections, least squares | 2 |
| 4 | `src/math/qr.py` | QR solver | 2 |
| 5 | `src/features/design_matrix.py` | Build X with splines, one-hots | 3 |
| 6 | `src/models/baseline.py` | Beta baseline | 4 |
| 7 | `src/models/logistic.py` | IRLS logistic | 5-6 |
| 8 | `src/evaluation/calibration.py` | Brier, ECE, plots | 8 |
| 9 | `src/models/hazard.py` | Decision timing | 9 |
| 10 | `src/models/embeddings.py` | Low-rank UV^T | 11-12 |
| 11 | `src/api/predictor.py` | Main interface | 14 |

---

## Key Success Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Brier Score | < 0.20 | `brier_score_loss(y_test, p_pred)` |
| ECE | < 0.05 | Custom calibration function |
| ROC-AUC | > 0.75 | `roc_auc_score(y_test, p_pred)` |
| IRLS vs sklearn | Match to 4 decimals | `np.allclose(our_probs, sklearn_probs, atol=1e-4)` |
| Timing accuracy | 70% within 2 weeks | `np.mean(np.abs(pred_week - actual_week) <= 2)` |

---

## Research Validation Report

*Comprehensive validation of all project components based on external research (December 2025)*

### 1. Ontario University Admissions System ✅ VALIDATED

**Top 6 Average Calculation:**
- The "Top 6" average varies by university—some use prerequisites in calculations, others don't
- Students must present OSSD and complete six 4U/M courses
- Universities use the six highest U/M level courses (not always identical calculation methods)
- Grade submission timeline: Mid-November (midterms) → February-April (finals) → July (summer/late grades)

**OUAC Process:**
- Ontario high school students under 21 apply through OUAC (centralized portal)
- Application opens ~October 1st, deadline January 15th
- OUAC continues processing after deadline but university-specific deadlines apply

**Sources:**
- [OUInfo FAQ on Admission Averages](https://www.ouinfo.ca/faqs/admission-average)
- [CUDO Secondary School Averages Data](https://cudo.ouac.on.ca/page.php?id=7&table=49)
- [OUAC Statistics](https://www.ouac.on.ca/statistics/)

**Recommendation:** Document that "Top 6" calculation varies by institution. Consider adding university-specific calculation methods as metadata.

---

### 2. ML Architecture: Embeddings for Categorical Variables ✅ VALIDATED

**Entity Embeddings - Industry Standard:**
- "A key technique to making the most of deep learning for tabular data is to use embeddings for categorical variables. This approach allows relationships between categories to be captured."
- Used at Google, Pinterest, Instacart, and winning Kaggle competitions
- Embeddings capture richer relationships than one-hot encoding and avoid sparse matrices

**Embedding Size Formula:**
- Common heuristic: `min(600, round(1.6 * num_classes ** 0.56))`
- For 50 universities: `round(1.6 * 50 ** 0.56)` ≈ 12-16 dimensions ✓
- For 200 programs: `round(1.6 * 200 ** 0.56)` ≈ 25-32 dimensions ✓

**Transfer Learning Potential:**
- Once trained, embeddings can be reused for other models (including tree-based)
- Pinterest's Pin2Vec and Instacart's embeddings demonstrate real-world success

**Sources:**
- [fast.ai: Introduction to Deep Learning for Tabular Data](https://www.fast.ai/posts/2018-04-29-categorical-embeddings.html)
- [arXiv: On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556)
- [Towards Data Science: Entity Embedding](https://towardsdatascience.com/the-right-way-to-use-deep-learning-for-tabular-data-entity-embedding-b5c4aaf1423a/)

---

### 3. ML Architecture: Attention Mechanism ⚠️ VALIDATED WITH CAVEATS

**Key Models:**
- **TabNet:** Sequential attention for feature selection, interpretable
- **SAINT:** Row + column attention, competitive with XGBoost/CatBoost on some benchmarks

**Critical Finding - Deep Learning vs Traditional ML:**
- Mixed research findings on whether DL beats tree-based models for tabular data
- February 2024 study: "Deep Learning methods outperform classical approaches" across 68 datasets
- Earlier influential study: "XGBoost outperforms deep models across datasets"
- Attention mechanisms offer interpretability but don't always improve accuracy over simpler methods

**Recommendation for This Project:**
1. Always benchmark against XGBoost/LightGBM baseline
2. Use attention for interpretability (visualize which programs are similar)
3. Don't expect automatic performance gains from attention alone

**Sources:**
- [Tabular Data: Deep Learning is Not All You Need](https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360)
- [SAINT: Improved Neural Networks for Tabular Data](https://openreview.net/forum?id=nL2lDlsrZU)
- [arXiv: Tabular Data - Is Deep Learning All You Need? (2024)](https://arxiv.org/abs/2402.03970)

---

### 4. Probability Calibration ✅ VALIDATED

**Two Main Methods:**

| Method | When to Use | Data Requirements |
|--------|-------------|-------------------|
| **Platt Scaling** | Sigmoid-shaped distortion, limited data | Works with small calibration sets |
| **Isotonic Regression** | Any monotonic distortion, sufficient data | Needs >1000 samples to avoid overfitting |

**For This Project:**
- With ~4,900 samples total and ~2,800 for training, isotonic regression is appropriate
- scikit-learn's `CalibratedClassifierCV` supports both methods
- Logistic regression is typically well-calibrated already, but embeddings/attention may introduce distortion

**Important Caveat:**
- "Calibration might lead to overfitting on small datasets"
- "In cases of severe class imbalance, calibration methods might struggle"

**Sources:**
- [scikit-learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [ABZU: Calibration Introduction Part 2](https://www.abzu.ai/data-science/calibration-introduction-part-2/)
- [FastML: Classifier Calibration](http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/)

---

### 5. Self-Reported Data Validity ⚠️ CONCERNS IDENTIFIED

**Key Bias Types:**

1. **Social Desirability Bias:** Respondents may overreport favorable outcomes or inflate grades
2. **Reference Bias:** "Systematic error arising from differences in implicit standards" - even truthful responses can be biased
3. **Recall Bias:** Students may not accurately remember exact averages or dates

**Empirical Finding:**
- "Validation with records from university recreation facility admissions uncovered high rates of overreporting on surveys; half of the respondents overreported their frequency of exercise over the past week"
- This suggests self-reported admission data may similarly have inflation

**Mitigation Strategies:**
1. **Cross-validation with official data:** Compare reported cutoffs to university-published admission ranges
2. **Outlier detection:** Flag implausible combinations (e.g., 75% average admitted to UofT CS)
3. **Uncertainty quantification:** Wider confidence intervals for predictions in sparse data regions
4. **Document limitations:** Acknowledge self-reporting bias in model documentation

**Sources:**
- [PMC: Measuring Bias in Self-Reported Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC4224297/)
- [Nature: Reference Bias in Self-Report Measures](https://www.nature.com/articles/s41598-022-23373-9)
- [Wikipedia: Self-Report Study](https://en.wikipedia.org/wiki/Self-report_study)

---

### 6. Sample Size (~4,900 Samples) ✅ VALIDATED

**Research Findings:**

- "When sample sizes from 120 to 2500 reduced discrepancy in accuracy between 85 and 99%"
- "ML, specifically algorithms like RF and XGB, can still outperform traditional methods even if sample size is limited (under n=5000)"
- For classification: "75-100 samples will usually be needed to test a good classifier" per class

**For This Project:**
- ~4,900 total samples across 3 years
- Train: ~2,800 (2022-2024), Test: ~2,074 (2024-2025)
- With binary classification (admit/reject), this exceeds minimum requirements
- Per-program sample sizes may be small for rare programs (need >20 per program filter)

**Deep Learning Consideration:**
- "Deep learning models are data hungry"
- Consider transfer learning or simpler models for programs with <50 samples
- Embedding sharing across similar programs can help (hierarchical structure)

**Sources:**
- [Keras: Estimating Required Sample Size](https://keras.io/examples/keras_recipes/sample_size_estimate/)
- [BMC: Predicting Sample Size Required for Classification](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8)
- [PMC: Sample Size Requirements for Popular Classification Algorithms](https://pmc.ncbi.nlm.nih.gov/articles/PMC11688588/)

---

### 7. AUC-ROC Target (>0.75) ✅ VALIDATED - REALISTIC

**Admission Prediction Benchmarks from Literature:**

| Study | Model | AUC | Accuracy |
|-------|-------|-----|----------|
| MDPI 2024 (Fair Admission) | Logistic Regression | 0.87 | 91% |
| MDPI 2024 (Fair Admission) | Naive Bayes | 0.66 | 70% |
| ResearchGate (College Commitment) | Logistic Regression | **0.796** | - |
| PMC 2023 (Graduate + LOR) | Random Forest | - | 89% |
| arXiv 2025 (LLM-Augmented) | Stacked Ensemble | - | 91% |

**Interpretation:**
- AUC > 0.75 is achievable and realistic for admission prediction
- Best models achieve AUC 0.80-0.87 with rich features
- Self-reported data may limit ceiling compared to official records

**Important Caveat:**
- "A high ROC AUC, such as 0.9, might correspond to low values of precision"
- Should also track Brier score, calibration error, and precision/recall

**Sources:**
- [MDPI: Fair and Transparent Student Admission Prediction](https://www.mdpi.com/1999-4893/17/12/572)
- [ResearchGate: Applying Data Mining to College Admissions](https://www.researchgate.net/publication/227604977_Applying_Data_Mining_to_Predict_College_Admissions_Yield_A_Case_Study)
- [arXiv: Admission Prediction Deep Learning](https://arxiv.org/html/2401.11698v1)

---

### 8. Tech Stack Validation

#### 8a. PyTorch ✅ VALIDATED
- Industry standard for deep learning research and production
- nn.Embedding, autograd, DataLoader all well-documented
- Strong community support and tutorials

#### 8b. MongoDB ✅ VALIDATED
- Appropriate for semi-structured admission data with varying fields
- Good Python integration via pymongo
- Flexible schema handles different data formats per year

**Best Practice:** "Keep each operation atomic so multiple instances can run independently for parallelism"

**Sources:**
- [MongoDB: ETL Best Practices](https://www.mongodb.com/partners/partner-program/technology/certification/etl-best-practices)
- [Neptune.ai: Build ETL Pipeline in ML](https://neptune.ai/blog/build-etl-data-pipeline-in-ml)

#### 8c. Weaviate ✅ VALIDATED
- Open-source vector database with excellent ML integration
- Supports both automatic vectorization and pre-computed embeddings
- Hybrid search (semantic + keyword) perfect for "similar programs" feature
- Recent (Dec 2024): Weaviate Embeddings service launched with no rate limits

**Integration Pattern:**
```python
# Store pre-computed PyTorch embeddings in Weaviate
embedding = model.get_program_embedding(prog_id)  # From PyTorch
weaviate_client.data_object.create({"program_name": "UofT CS"}, vector=embedding.numpy())
```

**Sources:**
- [Weaviate: Vector Embeddings Explained](https://weaviate.io/blog/vector-embeddings-explained)
- [Weaviate Documentation](https://docs.weaviate.io/weaviate)

#### 8d. FastAPI ✅ VALIDATED
- Usage grew from 21% (2021) to 29% (2023) among Python developers
- "Benchmarked as one of the fastest Python frameworks, on par with NodeJS and Go"
- Built-in OpenAPI documentation, async support, data validation

**Production Pattern:**
```python
@app.on_event("startup")
def load_model():
    global model
    model = torch.load("admission_model.pt")

@app.post("/predict")
def predict(avg: float, uni: str, prog: str):
    # Model already loaded at startup
    return {"probability": model.predict(avg, uni, prog)}
```

**Sources:**
- [JetBrains: How to Use FastAPI for ML](https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/)
- [TestDriven.io: FastAPI ML Deployment](https://testdriven.io/blog/fastapi-machine-learning/)

---

### 9. Linear Algebra Concepts (MAT223) ✅ VALIDATED

#### 9a. QR Factorization vs Normal Equations ✅ CRITICAL INSIGHT

| Method | Condition Number | Numerical Stability |
|--------|------------------|---------------------|
| Normal Equations (X^TX)^{-1}X^Ty | κ(X)² | **Poor** - squares condition number |
| QR Factorization | κ(X) | **Good** - preserves condition number |
| SVD | Best handles ill-conditioning | **Best** - slowest |

**Key Quote:** "Using the QR decomposition yields a better least-squares estimate than the Normal Equations in terms of solution quality."

**For This Project:** Plan correctly recommends QR over normal equations. This is validated best practice.

**Sources:**
- [John Lambert: Solving Least-Squares with QR](https://johnwlambert.github.io/least-squares/)
- [Cornell CS: Normal Equations](https://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf)
- [Wikipedia: QR Decomposition](https://en.wikipedia.org/wiki/QR_decomposition)

#### 9b. IRLS for Logistic Regression ✅ VALIDATED

- "IRLS is Newton's method applied to maximizing likelihood"
- Standard method for fitting GLMs including logistic regression
- scikit-learn uses IRLS internally for logistic regression

**Numerical Stability Tip:**
```python
# Add small constant to avoid log(0) in initialization
y_init = np.log((y + 0.5) / (1 - y + 0.5))  # "+0.5 for stability"
```

**Sources:**
- [Wikipedia: IRLS](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
- [CMU: Logistic Regression Example](https://www.cs.cmu.edu/~ggordon/IRLS-example/)

#### 9c. Ridge Regression (L2 Regularization) ✅ VALIDATED

- Shrinks coefficients toward zero but never exactly zero
- Handles multicollinearity by shrinking correlated feature coefficients
- In PyTorch: `weight_decay` parameter in optimizer

**Caveat for Categorical Variables:**
- "Ridge Regression assumes continuous predictors and may not perform well with categorical variables"
- **Solution:** Use embeddings (which are continuous) rather than one-hot encoding with ridge

**Sources:**
- [IBM: What is Ridge Regression](https://www.ibm.com/think/topics/ridge-regression)
- [PMC: Ridge Regularization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9410599/)

---

### 10. Temporal Train/Test Split ✅ VALIDATED - CRITICAL

**Why It Matters:**
- "Data leakage occurs when information from the future leaks into the past during training"
- "A 2023 review found data leakage to be a widespread failure mode in ML-based science, affecting 294+ publications"

**Project's Approach is Correct:**
- Train: 2022-2023, 2023-2024 cycles
- Test: 2024-2025 cycle
- This prevents future leakage

**Additional Best Practices:**
1. **Split before preprocessing:** "Always split first. Do fitting only on training set"
2. **No shuffling:** "Traditional k-fold is inappropriate for time series"
3. **Consider walk-forward validation:** Train on 2022-23 → validate on 2023-24, then train on 2022-24 → validate on 2024-25

**Sources:**
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [arXiv: Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932v1)
- [CodeCut: Avoiding Data Leakage with TimeSeriesSplit](https://codecut.ai/cross-validation-with-time-series/)

---

### Summary: Validation Status

| Component | Status | Key Finding |
|-----------|--------|-------------|
| Ontario Admissions System | ✅ Validated | Top 6 calculation varies by university |
| Embeddings for Categories | ✅ Validated | Industry standard, size formula confirmed |
| Attention Mechanism | ⚠️ Caveats | Benchmark against XGBoost; interpretability value |
| Probability Calibration | ✅ Validated | Isotonic regression appropriate for sample size |
| Self-Reported Data | ⚠️ Concerns | Document bias; validate against published ranges |
| Sample Size (~4,900) | ✅ Validated | Sufficient for binary classification |
| AUC-ROC > 0.75 Target | ✅ Validated | Achievable; literature shows 0.79-0.87 |
| PyTorch | ✅ Validated | Standard choice |
| MongoDB | ✅ Validated | Good for semi-structured data |
| Weaviate | ✅ Validated | Excellent for similarity search |
| FastAPI | ✅ Validated | Fast, production-ready |
| QR Factorization | ✅ Validated | Correct choice over normal equations |
| IRLS | ✅ Validated | Standard for logistic regression |
| Ridge/L2 | ✅ Validated | Use with embeddings, not one-hot |
| Temporal Split | ✅ Validated | Critical for preventing leakage |

### Recommended Adjustments

1. **Add XGBoost/LightGBM baseline** to benchmark against neural network approaches
2. **Document self-reporting bias** in model limitations section
3. **Implement walk-forward validation** in addition to single temporal split
4. **Track precision/recall** alongside AUC-ROC for imbalanced programs
5. **Add data validation** against published university admission ranges

---

## Deep Analysis & Resolution of Concerns

*For a first ML project, here's how to think about each concern pragmatically.*

### Concern 1: Attention Mechanism - Is It Worth the Complexity?

#### The Real Question

The research shows mixed results on whether attention/deep learning beats simpler models for tabular data. Should you even bother with attention on your first project?

#### Deep Analysis

**Why the research is "mixed":**
- Studies showing DL wins often use large datasets (100k+ samples) with complex feature interactions
- Studies showing XGBoost wins often use smaller datasets with simpler patterns
- Your dataset (~4,900 samples) is on the smaller side where simpler models often win

**What attention actually gives you:**
1. **Interpretability** - "The model paid 35% attention to Waterloo CS when predicting for UofT CS" (cool for understanding)
2. **Learned similarities** - Programs that behave similarly get similar embeddings (useful)
3. **Complexity** - More code, more hyperparameters, more things to debug (costly for learning)

**The honest truth for a first project:**
- A well-tuned logistic regression with good features will likely perform within 2-3% AUC of a complex attention model
- The learning value of attention is high (it's used everywhere in modern ML)
- But debugging attention when you're still learning PyTorch basics is frustrating

#### Resolution: Progressive Complexity Strategy

**Don't abandon attention—delay it.** Structure your project in phases:

```
Phase 1 (Days 1-5): Foundation
├── Simple logistic regression with sklearn
├── Learn: What is AUC? What is calibration?
├── Target: Get ANY working model with AUC > 0.65
└── This is your "baseline" to beat

Phase 2 (Days 6-9): PyTorch Basics
├── Reimplement logistic regression in PyTorch
├── Add embeddings for university/program
├── Learn: Tensors, autograd, nn.Module
└── Target: Match or beat sklearn baseline

Phase 3 (Days 10-14): Advanced (Optional)
├── Add attention IF Phase 2 is solid
├── Compare: Did attention actually help?
├── Learn: Attention mechanism, visualization
└── Target: Understand WHY it helped (or didn't)
```

**Key insight:** Attention becomes your "dessert"—you earn it after mastering the basics. If you skip Phase 1-2, you won't understand what attention is actually doing.

**Practical code checkpoint:**
```python
# You should NOT attempt attention until you can answer:
# 1. What does model.parameters() return?
# 2. Why do we call optimizer.zero_grad()?
# 3. What's the difference between model.train() and model.eval()?
# 4. What does loss.backward() actually compute?
```

#### Updated Recommendation

| Original Plan | Revised for First Project |
|---------------|---------------------------|
| Days 8-11: Attention | Days 10-14: Attention (optional stretch goal) |
| Required component | Bonus if time permits |
| Complex from start | Earn it through progression |

**Bottom line:** The concern about "DL vs XGBoost" is valid but irrelevant for learning. Build both, compare them, understand why one wins. That's the learning.

---

### Concern 2: Self-Reported Data Validity - How Much Should You Worry?

#### The Real Question

Research shows people overreport favorable outcomes. If students inflate their grades or misremember, doesn't that make your predictions worthless?

#### Deep Analysis

**Why this concern is overblown for your project:**

1. **The bias is likely systematic, not random**
   - If everyone inflates by ~3-5%, the relative rankings stay intact
   - Model learns: "When someone reports 92%, they probably have 88-92%"
   - Predictions are still useful for ranking/comparing applicants

2. **Your model predicts outcomes, not ground truth**
   - You're predicting: "Given someone who *reports* X average, what happened to them?"
   - Not: "Given someone's *true* average, what would happen?"
   - The self-reported nature is baked into both training and inference

3. **Users will query with self-reported data too**
   - A student using your model will input their self-perceived average
   - Training on self-reported → predicting from self-reported = consistent

4. **Ontario averages are relatively verifiable**
   - Unlike "exercise frequency," grades are concrete numbers students see on transcripts
   - Inflation is likely modest (rounding 89.7 → 90) rather than fabricated

**Where bias actually matters:**
- **Outcome reporting:** "Did you get admitted?" - More reliable (binary, memorable, verifiable through OUAC)
- **Average reporting:** Modest inflation possible, but bounded by reality
- **Timing data:** Most uncertain (people forget exact dates)

#### Resolution: Proportional Response

**Don't over-engineer for a first project.** Here's what's appropriate:

**Do (5 minutes of work):**
```python
# Basic sanity check during data loading
def validate_record(avg, outcome, program):
    # Flag obvious impossibilities
    if avg < 50 or avg > 100:
        return False, "Average out of range"
    if avg < 70 and outcome == "admitted" and program in COMPETITIVE_PROGRAMS:
        return False, "Suspicious: low average admitted to competitive program"
    return True, "OK"
```

**Don't do (over-engineering for first project):**
- ❌ Building complex outlier detection systems
- ❌ Bayesian models to "correct" for bias
- ❌ Weighted training based on "credibility scores"
- ❌ Spending weeks validating against external data

**Document and move on:**
```python
"""
Model Limitations:
- Training data is self-reported; averages may have ~2-5% inflation
- Predictions represent outcomes for students who REPORTED similar profiles
- For competitive programs (CS, Engineering), treat predictions as optimistic
"""
```

#### The Learning Perspective

This concern teaches an important ML lesson: **All real-world data is messy.** Professional ML engineers deal with:
- Missing data
- Label noise
- Distribution shift
- Sampling bias

Your response shouldn't be "fix all bias before training." It should be:
1. Understand the bias exists
2. Document it
3. Build anyway
4. Interpret results with appropriate humility

**This is actually great training** for real ML work where perfect data doesn't exist.

---

### Concern 3 (Implicit): Is the Timeline Realistic?

#### Analysis

The 14-day timeline in the plan assumes focused, full-time work. For a first ML project while learning:
- PyTorch basics (new)
- MongoDB/Weaviate (new)
- Linear algebra concepts (new)
- ML evaluation metrics (new)

This is ambitious.

#### Resolution: Flexible Milestones, Not Fixed Days

Replace the rigid 14-day timeline with **milestone-based progress:**

| Milestone | You're Ready When... | Typical Time |
|-----------|---------------------|--------------|
| M1: Data Ready | Can load data, print sample, know feature types | 1-2 days |
| M2: First Model | sklearn logistic regression runs, prints AUC | 1-2 days |
| M3: PyTorch Basics | Can explain tensors, autograd, train loop | 3-5 days |
| M4: Embeddings Work | Model uses embeddings, AUC ≥ sklearn baseline | 3-5 days |
| M5: Calibration | Can plot calibration curve, compute ECE | 1-2 days |
| M6: API Works | FastAPI endpoint returns prediction | 2-3 days |
| **Stretch:** Attention | Implemented, compared to simpler model | 3-5 days |

**Total realistic time: 3-5 weeks** for a first project with learning.

---

### Concern 4 (Resolved): Tech Stack is Intentional for Learning ✅

#### Analysis

The plan includes:
- MongoDB (document database)
- Weaviate (vector database)
- FastAPI (API framework)
- PyTorch (deep learning)

For ~4,900 records, a CSV + pandas would technically work. However, **learning these technologies is an explicit study goal**, making this an intentional choice rather than over-engineering.

#### Resolution: Intentional Full-Stack ML Learning

**Why MongoDB + Weaviate ARE worth including:**

1. **Real-world relevance:** Production ML systems use databases, not CSVs
2. **Transferable skills:** MongoDB and vector databases are in high demand
3. **Portfolio value:** Makes this a complete, impressive project
4. **Separation of concerns:** Data storage vs ML training vs vector search are distinct skills worth learning

**Staged approach (all stages are learning goals):**

```
Stage 1: Data Foundation (Days 1-2)
├── Load CSVs into MongoDB (learn: documents, collections, pymongo)
├── Basic queries: find by university, filter by year
├── Learn: Why document DB vs relational? When to use each?
└── Deliverable: Data accessible via MongoDB

Stage 2: ML Pipeline (Days 3-7)
├── PyTorch Dataset pulls from MongoDB (not CSV!)
├── Train models, evaluate, iterate
├── Learn: DataLoader, batching, train/eval split
└── Deliverable: Working model with good AUC

Stage 3: Vector Search (Days 8-10)
├── Export trained embeddings to Weaviate
├── Implement "similar programs" search
├── Learn: Vector indexing, semantic search, hybrid queries
└── Deliverable: Query "UofT CS" → returns similar programs

Stage 4: API Layer (Days 11-14)
├── FastAPI wraps everything
├── /predict endpoint → MongoDB lookup → Model inference
├── /similar endpoint → Weaviate vector search
└── Deliverable: Working API others can call
```

**What each technology teaches:**

| Technology | Core Skill | Industry Relevance |
|------------|------------|-------------------|
| **MongoDB** | NoSQL data modeling, document queries, ETL | Standard for flexible schemas |
| **Weaviate** | Vector embeddings, semantic search | RAG, recommendations, GenAI apps |
| **PyTorch** | Deep learning fundamentals, autograd | Research & production ML |
| **FastAPI** | API design, async Python, Pydantic | Modern Python web services |

**This is a legitimate full-stack ML project.** The tech stack isn't over-engineered—it's a deliberate learning curriculum that covers the complete ML application lifecycle:

```
Data (MongoDB) → Model (PyTorch) → Search (Weaviate) → API (FastAPI)
```

---

### Summary: Concerns Resolved

| Concern | Severity | Resolution |
|---------|----------|------------|
| Attention vs XGBoost | **Low** | Build both, compare, learn from the comparison. Delay attention until basics are solid. |
| Self-Reported Data | **Low** | Bias is systematic and consistent train→inference. Basic sanity checks + documentation is sufficient. |
| Timeline Realism | **Medium** | Switch to milestone-based; expect 3-5 weeks for thorough learning. |
| Tech Stack (MongoDB + Weaviate) | **Resolved** ✅ | Intentional learning goals, not over-engineering. Full-stack ML is the curriculum. |

### Mindset Shift for First Project

**Don't aim for:**
- ❌ Production-ready system
- ❌ Perfect accuracy
- ❌ Handling every edge case
- ❌ Using all the technologies

**Do aim for:**
- ✅ Understanding what each component does
- ✅ Being able to explain your model's predictions
- ✅ Knowing why your AUC is what it is
- ✅ Building something that works end-to-end

**The real deliverable isn't the model—it's your understanding.**

A simple logistic regression you fully understand beats a complex attention model you copied from Stack Overflow.

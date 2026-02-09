import { useState } from 'react'
import { Link } from 'react-router-dom'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import Breadcrumb from '@/components/shared/Breadcrumb'
import PrevNextNav from '@/components/shared/PrevNextNav'
import TableOfContents, { type TocItem } from '@/components/shared/TableOfContents'
import M from '@/components/shared/Math'
import SigmoidExplorer from '@/components/interactive/SigmoidExplorer'
import IRLSStepThrough from '@/components/interactive/IRLSStepThrough'
import SurvivalCurveExplorer from '@/components/interactive/SurvivalCurveExplorer'
import BetaShrinkageDemo from '@/components/interactive/BetaShrinkageDemo'
import AttentionHeatmap from '@/components/interactive/AttentionHeatmap'

/* ─────────────────────────────────────────────
   Table of Contents items
   ───────────────────────────────────────────── */
const TOC_ITEMS: TocItem[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'beta-binomial', label: 'Beta-Binomial Baseline' },
  { id: 'irls', label: 'IRLS Logistic Regression' },
  { id: 'hazard', label: 'Hazard Model' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'attention', label: 'Attention' },
  { id: 'fairness', label: 'Fairness' },
]

/* ─────────────────────────────────────────────
   Coefficient table data (expanded)
   ───────────────────────────────────────────── */
const COEFFICIENTS = [
  { feature: 'bias', coefficient: '-0.50', oddsRatio: '0.61', interpretation: 'Base log-odds (intercept)' },
  { feature: 'top_6_average', coefficient: '+0.08', oddsRatio: '1.08', interpretation: 'Each 1% \u2192 +8% odds increase' },
  { feature: 'grade_12_average', coefficient: '+0.03', oddsRatio: '1.03', interpretation: 'Secondary grade predictor' },
  { feature: 'is_ontario', coefficient: '+0.10', oddsRatio: '1.11', interpretation: 'Ontario applicant advantage' },
  { feature: 'is_bc', coefficient: '+0.05', oddsRatio: '1.05', interpretation: 'BC applicant advantage' },
  { feature: 'is_quebec', coefficient: '+0.08', oddsRatio: '1.08', interpretation: 'Quebec applicant advantage' },
  { feature: 'is_alberta', coefficient: '+0.05', oddsRatio: '1.05', interpretation: 'Alberta applicant advantage' },
]

/* ─────────────────────────────────────────────
   IRLS algorithm steps
   ───────────────────────────────────────────── */
const IRLS_STEPS = [
  { step: 'Compute probabilities', formula: 'p = \\sigma(X\\beta)' },
  { step: 'Compute weights', formula: 'W = \\text{diag}(p(1-p))' },
  { step: 'Compute working response', formula: 'z = X\\beta + (y - p) / W' },
  { step: 'Solve weighted ridge', formula: '\\beta_{\\text{new}} = (X^TWX + \\lambda I)^{-1}X^TWz' },
  { step: 'Check convergence', formula: '\\text{if } \\|\\beta_{\\text{new}} - \\beta\\| < \\text{tol, stop}' },
]

/* ─────────────────────────────────────────────
   Person-period expansion example
   ───────────────────────────────────────────── */
const HAZARD_ROWS = [
  { week: 1, features: '[92.5, ...]', decision: '0' },
  { week: 2, features: '[92.5, ...]', decision: '0' },
  { week: 3, features: '[92.5, ...]', decision: '0' },
  { week: 4, features: '[92.5, ...]', decision: '0' },
  { week: 5, features: '[92.5, ...]', decision: '1 \u2190 decision here' },
]

/* ─────────────────────────────────────────────
   Model overview cards data
   ───────────────────────────────────────────── */
const MODEL_CARDS = [
  {
    name: 'Beta-Binomial Baseline',
    question: 'What is this program\'s base admission rate?',
    status: 'Production' as const,
  },
  {
    name: 'IRLS Logistic Regression',
    question: 'Given your full profile, what is your probability?',
    status: 'Production (Core)' as const,
  },
  {
    name: 'Discrete-Time Hazard',
    question: 'When will you hear back?',
    status: 'Production' as const,
  },
  {
    name: 'Embeddings + Attention',
    question: 'Which similar programs influenced the prediction?',
    status: 'Planned' as const,
  },
]

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function Models() {
  const [grade, setGrade] = useState(92.5)

  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <Breadcrumb
            items={[
              { label: 'About', to: '/about' },
              { label: 'Models' },
            ]}
          />

          <div className="mt-8">
            <SectionHeader
              label="Models"
              title="Four models, one"
              accent="pipeline."
              description="From Bayesian priors to attention mechanisms -- each model answers a different question about university admissions."
            />
          </div>
        </div>
      </section>

      {/* ── TOC + Main content flex layout ──── */}
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="flex gap-6 lg:gap-12">
          {/* Table of Contents sidebar */}
          <TableOfContents items={TOC_ITEMS} />

          {/* Main content */}
          <div className="flex-1 min-w-0">

            {/* ════════════════════════════════════
                Section 1: Model Overview
                ════════════════════════════════════ */}
            <section id="overview" className="py-16">
              <div className="space-y-10">
                <h3 className="text-2xl font-bold text-white">Model Overview</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {MODEL_CARDS.map((card) => (
                    <GlassCard key={card.name}>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <h4 className="text-white font-bold">{card.name}</h4>
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full ${
                              card.status.startsWith('Production')
                                ? 'bg-emerald-400/10 text-emerald-400'
                                : 'bg-yellow-400/10 text-yellow-400'
                            }`}
                          >
                            {card.status}
                          </span>
                        </div>
                        <p className="text-gray-400 text-sm">{card.question}</p>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 2: Beta-Binomial Baseline
                ════════════════════════════════════ */}
            <section id="beta-binomial" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Baseline Model"
                  title="Beta-Binomial"
                  accent="prior."
                />

                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    The Beta-Binomial model uses{' '}
                    <strong className="text-white">Bayesian conjugate updating</strong>{' '}
                    to estimate program-level admission rates. We start with a prior
                    belief about the admission rate, then update it with observed data.
                  </p>
                  <p>
                    Prior{' '}
                    <M>{"\\text{Beta}(\\alpha_0, \\beta_0)"}</M>{' '}
                    + Data (k admits, n apps) &rarr; Posterior{' '}
                    <M>{"\\text{Beta}(\\alpha_0 + k, \\beta_0 + n - k)"}</M>
                  </p>
                </div>

                {/* Shrinkage story */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    The Shrinkage Story
                  </h4>
                  <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
                    <p>
                      The posterior mean is a weighted average of the prior and the data:
                    </p>
                    <div className="bg-black/30 rounded-lg px-4 py-3 overflow-x-auto">
                      <M display>{"\\hat{\\mu} = \\frac{\\alpha_0 + \\beta_0}{\\alpha_0 + \\beta_0 + n} \\cdot \\mu_{\\text{prior}} + \\frac{n}{\\alpha_0 + \\beta_0 + n} \\cdot \\mu_{\\text{data}}"}</M>
                    </div>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                        <span>
                          <M>{"n = 2"}</M>:{' '}
                          heavy shrinkage toward the prior (unreliable data)
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                        <span>
                          <M>{"n = 500"}</M>:{' '}
                          posterior &asymp; data mean (reliable data)
                        </span>
                      </li>
                    </ul>
                  </div>
                </GlassCard>

                {/* Why Bayesian */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Why Bayesian?
                  </h4>
                  <div className="space-y-3 text-gray-300 text-sm leading-relaxed">
                    <p>
                      Queens Computing has 2 applications (1 admit, 1 reject). The maximum
                      likelihood estimate says 50% -- but that's based on almost no data.
                      The Bayesian approach shrinks this toward the global prior, producing
                      a more reasonable estimate.
                    </p>
                    <p>
                      <strong className="text-white">Credible intervals</strong> quantify
                      the uncertainty:{' '}
                      <M>{"65\\% \\text{ admission probability } [52\\%, 78\\%]"}</M>{' '}
                      -- wide intervals flag uncertain predictions.
                    </p>
                  </div>
                </GlassCard>

                <GlassCard>
                  <BetaShrinkageDemo />
                </GlassCard>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 3: IRLS Logistic Regression
                ════════════════════════════════════ */}
            <section id="irls" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Core Model"
                  title="IRLS logistic"
                  accent="regression."
                />

                {/* Why not linear regression */}
                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">
                    Why not linear regression?
                  </h4>
                  <p>
                    Linear regression could predict{' '}
                    <M>{"P = x^T\\beta"}</M>
                    , but this can produce values outside [0, 1] -- invalid
                    probabilities! The sigmoid wrapper ensures valid probabilities.
                  </p>
                </div>

                {/* The logistic model */}
                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">The logistic model</h4>
                  <div className="bg-black/30 rounded-lg px-4 py-3 overflow-x-auto">
                    <M display>{"P(\\text{admit} | x) = \\sigma(x^T\\beta) = \\frac{1}{1 + e^{-x^T\\beta}}"}</M>
                  </div>
                </div>

                {/* Maximum Likelihood */}
                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">Maximum likelihood estimation</h4>
                  <p>
                    We choose <M>{"\\beta"}</M> to maximize the probability of the observed data.
                    Taking the log and negating gives us the loss to minimize:
                  </p>
                  <div className="bg-black/30 rounded-lg px-4 py-3 overflow-x-auto space-y-2">
                    <M display>{"\\ell(\\beta) = \\sum \\left[ y \\cdot \\log(p) + (1-y) \\cdot \\log(1-p) \\right]"}</M>
                    <M display>{"L(\\beta) = -\\ell(\\beta) + \\frac{\\lambda}{2} \\beta^T\\beta"}</M>
                  </div>
                  <p className="text-sm text-gray-500">
                    The ridge penalty <M>{"\\frac{\\lambda}{2} \\beta^T\\beta"}</M> prevents
                    overfitting by shrinking coefficients toward zero.
                  </p>
                </div>

                {/* Gradient and Hessian */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Gradient &amp; Hessian
                  </h4>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <p className="text-gray-300 text-sm font-medium">Gradient:</p>
                      <div className="bg-black/30 rounded-lg px-4 py-3">
                        <M display>{"\\nabla L = X^T(p - y) + \\lambda\\beta"}</M>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <p className="text-gray-300 text-sm font-medium">Hessian:</p>
                      <div className="bg-black/30 rounded-lg px-4 py-3">
                        <M display>{"H = X^TWX + \\lambda I"}</M>
                        <p className="text-gray-400 text-sm mt-2">where <M>{"W = \\text{diag}(p(1-p))"}</M></p>
                      </div>
                    </div>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      The Hessian is positive semidefinite &rarr; the loss is{' '}
                      <strong className="text-white">CONVEX</strong> &rarr; gradient
                      descent finds the global optimum.
                    </p>
                  </div>
                </GlassCard>

                {/* The IRLS Algorithm */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    The IRLS Algorithm
                  </h4>
                  <ol className="space-y-3">
                    {IRLS_STEPS.map((item, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold shrink-0 mt-0.5">
                          {i + 1}
                        </span>
                        <div>
                          <span className="text-gray-300 text-sm font-medium">
                            {item.step}:
                          </span>{' '}
                          <M>{item.formula}</M>
                        </div>
                      </li>
                    ))}
                  </ol>
                  <div className="mt-6 p-4 bg-emerald-400/5 border border-emerald-400/20 rounded-lg">
                    <p className="text-gray-300 text-sm leading-relaxed">
                      <strong className="text-white">Key insight:</strong> Newton's
                      method for logistic regression is equivalent to solving a{' '}
                      <strong className="text-emerald-400">
                        weighted least squares
                      </strong>{' '}
                      problem at each step!
                    </p>
                  </div>
                  <div className="mt-4">
                    <Link
                      to="/about/math#ridge"
                      className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors inline-flex items-center gap-1"
                    >
                      See how weighted_ridge_solve works
                      <span aria-hidden="true">&rarr;</span>
                    </Link>
                  </div>
                </GlassCard>

                {/* Numerically stable sigmoid */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Numerically Stable Sigmoid
                  </h4>
                  <div className="bg-black/30 rounded-lg px-4 py-3 space-y-2">
                    <M display>{"\\sigma(z) = \\frac{1}{1 + e^{-z}} \\quad \\text{if } z \\geq 0"}</M>
                    <M display>{"\\sigma(z) = \\frac{e^z}{1 + e^z} \\quad \\text{if } z < 0"}</M>
                  </div>
                  <p className="mt-3 text-gray-400 text-sm leading-relaxed">
                    Avoids overflow:{' '}
                    <M>{"e^{700}"}</M>{' '}
                    overflows, but{' '}
                    <M>{"\\sigma(700) \\approx 1"}</M>
                    .
                  </p>
                </GlassCard>

                {/* Numerical safeguards */}
                <div className="space-y-4">
                  <h4 className="text-lg font-bold text-white">Numerical safeguards</h4>
                  <ul className="space-y-2 text-gray-400 text-sm leading-relaxed">
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        Clip probabilities to{' '}
                        <M>{"[10^{-15},\\; 1 - 10^{-15}]"}</M>{' '}
                        to avoid <M>{"\\log(0)"}</M>
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        Clip weights to{' '}
                        <M>{"[10^{-10},\\; 0.25]"}</M>{' '}
                        to prevent ill-conditioned systems
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        Ridge regularization prevents perfect separation (infinite coefficients)
                      </span>
                    </li>
                  </ul>
                </div>

                {/* Grade slider + SigmoidExplorer */}
                <div className="space-y-6">
                  <h4 className="text-lg font-bold text-white">
                    Interactive: Sigmoid Explorer
                  </h4>
                  <GlassCard>
                    <div className="space-y-4">
                      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
                        <div className="flex-1 space-y-1">
                          <p className="text-gray-300 text-sm">
                            Top-6 average:{' '}
                            <span className="text-white font-bold tabular-nums">
                              {grade.toFixed(1)}%
                            </span>
                          </p>
                        </div>
                        <div className="w-full sm:w-64 space-y-1">
                          <label className="text-xs text-gray-500">
                            Adjust Grade
                          </label>
                          <input
                            type="range"
                            min={70}
                            max={100}
                            step={0.5}
                            value={grade}
                            onChange={(e) => setGrade(Number(e.target.value))}
                            className="w-full accent-emerald-400"
                          />
                          <div className="flex justify-between text-xs text-gray-600 tabular-nums">
                            <span>70%</span>
                            <span>100%</span>
                          </div>
                        </div>
                      </div>
                      <SigmoidExplorer grade={grade} />
                    </div>
                  </GlassCard>
                </div>

                {/* IRLSStepThrough */}
                <div className="space-y-6">
                  <h4 className="text-lg font-bold text-white">
                    Interactive: IRLS Step-Through
                  </h4>
                  <GlassCard>
                    <IRLSStepThrough />
                  </GlassCard>
                </div>

                {/* Coefficient table */}
                <div className="space-y-6">
                  <h4 className="text-lg font-bold text-white">
                    Model Coefficients
                  </h4>
                  <GlassCard padding="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Feature</th>
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Coefficient</th>
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Odds Ratio</th>
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Interpretation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {COEFFICIENTS.map((row) => (
                            <tr
                              key={row.feature}
                              className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                            >
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                                {row.feature}
                              </td>
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-emerald-400 font-mono tabular-nums">
                                {row.coefficient}
                              </td>
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300 font-mono tabular-nums">
                                {row.oddsRatio}
                              </td>
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                                {row.interpretation}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </GlassCard>
                  <p className="text-sm text-gray-500 max-w-3xl">
                    These are from the simplified 8-feature API representation. The full
                    training pipeline uses 250+ features including interaction terms,
                    program-specific encodings, and temporal features.
                  </p>
                </div>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 4: Hazard Model
                ════════════════════════════════════ */}
            <section id="hazard" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Timing Model"
                  title="When you'll"
                  accent="hear back."
                />

                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    The discrete-time hazard model predicts{' '}
                    <strong className="text-white">when</strong> you will receive a
                    decision, not just whether you will be admitted. It models the
                    probability of receiving a decision in each time period, conditional
                    on not having received one yet.
                  </p>
                  <p>The hazard function:</p>
                  <div className="bg-black/30 rounded-lg px-4 py-3">
                    <M display>{"h(t|x) = \\sigma(\\alpha_t + x^T\\beta)"}</M>
                  </div>
                </div>

                {/* Person-period expansion */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Person-Period Expansion
                  </h4>
                  <p className="text-gray-400 text-sm leading-relaxed mb-4">
                    Each applicant's timeline is expanded into one row per time period.
                    The binary outcome indicates whether a decision was received in that
                    period.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-4 py-3 font-medium">Week</th>
                          <th className="px-4 py-3 font-medium">Features</th>
                          <th className="px-4 py-3 font-medium">Decision?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {HAZARD_ROWS.map((row) => (
                          <tr
                            key={row.week}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-4 py-3 text-white font-mono tabular-nums">
                              {row.week}
                            </td>
                            <td className="px-4 py-3 text-gray-300 font-mono text-sm">
                              {row.features}
                            </td>
                            <td className="px-4 py-3 text-gray-300 font-mono text-sm">
                              {row.decision}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-4 text-gray-400 text-sm leading-relaxed">
                    After expansion, this becomes standard logistic regression -- solved
                    with IRLS.
                  </p>
                </GlassCard>

                {/* Survival and CDF */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Survival &amp; CDF
                  </h4>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <p className="text-gray-300 text-sm font-medium">
                        Survival function (no decision by time t):
                      </p>
                      <div className="bg-black/30 rounded-lg px-4 py-3">
                        <M display>{"S(t) = \\prod_{k=1}^{t}(1 - h(k))"}</M>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <p className="text-gray-300 text-sm font-medium">
                        CDF (decision by time t):
                      </p>
                      <div className="bg-black/30 rounded-lg px-4 py-3">
                        <M display>{"F(t) = 1 - S(t)"}</M>
                      </div>
                    </div>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      Enables predictions like:{' '}
                      <strong className="text-white">
                        "You'll likely hear by mid-March"
                      </strong>
                    </p>
                  </div>
                </GlassCard>

                <GlassCard>
                  <SurvivalCurveExplorer />
                </GlassCard>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 5: Embeddings
                ════════════════════════════════════ */}
            <section id="embeddings" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Planned"
                  title="Learned"
                  accent="embeddings."
                />

                <div className="flex items-center gap-3">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-400/10 text-yellow-400">
                    Planned
                  </span>
                  <span className="text-gray-500 text-sm">
                    Not yet in production
                  </span>
                </div>

                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">
                    One-hot vs. dense embeddings
                  </h4>
                  <p>
                    Currently, categorical features like university are one-hot encoded.{' '}
                    <code className="text-emerald-400 font-mono text-sm">
                      "UofT" &rarr; [1,0,0,...,0]
                    </code>{' '}
                    (50 dimensions). This is sparse and treats every university as
                    equally different from every other.
                  </p>
                  <p>
                    Dense embeddings learn a compact representation:{' '}
                    <code className="text-emerald-400 font-mono text-sm">
                      "UofT" &rarr; [0.2,&minus;0.5,...,0.8]
                    </code>{' '}
                    (16 dimensions). Similar universities end up with similar vectors.
                  </p>
                </div>

                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Dimension Sizing
                  </h4>
                  <div className="space-y-3 text-gray-300 text-sm leading-relaxed">
                    <div className="bg-black/30 rounded-lg px-4 py-3">
                      <M display>{"\\text{embed\\_dim} = \\min\\left(50,\\; \\left\\lfloor \\frac{n_{\\text{categories}} + 1}{2} \\right\\rfloor\\right)"}</M>
                    </div>
                    <p className="text-gray-400">
                      This heuristic balances expressiveness against overfitting. 50
                      universities &rarr; 25-dimensional embeddings. 10 provinces &rarr;
                      5-dimensional embeddings.
                    </p>
                  </div>
                </GlassCard>

                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Training Loop
                  </h4>
                  <ol className="space-y-3">
                    {[
                      'Embed categorical features into dense vectors',
                      'Concatenate embeddings with numerical features',
                      'Forward pass: predict admission probability',
                      'Backpropagate gradients through the embedding layer',
                      'Update embedding weights via gradient descent',
                    ].map((step, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold shrink-0 mt-0.5">
                          {i + 1}
                        </span>
                        <span className="text-gray-300 text-sm">{step}</span>
                      </li>
                    ))}
                  </ol>
                </GlassCard>

                <div className="space-y-3 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">Benefits</h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        <strong className="text-white">Fewer parameters</strong> --{' '}
                        <M>{"50 \\times 16 = 800"}</M> vs. <M>{"50 \\times 50 = 2{,}500"}</M> one-hot features
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        <strong className="text-white">Captures similarity</strong> --
                        UofT and Waterloo end up near each other in embedding space
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 shrink-0">&bull;</span>
                      <span>
                        <strong className="text-white">Exportable to Weaviate</strong> --
                        trained embeddings can power vector search for similar programs
                      </span>
                    </li>
                  </ul>
                </div>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 6: Attention
                ════════════════════════════════════ */}
            <section id="attention" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Planned"
                  title="Attention"
                  accent="mechanism."
                />

                <div className="flex items-center gap-3">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-400/10 text-yellow-400">
                    Planned
                  </span>
                  <span className="text-gray-500 text-sm">
                    Not yet in production
                  </span>
                </div>

                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    Attention allows the model to dynamically focus on the most relevant
                    historical applications when making a prediction. Instead of treating
                    all data equally, it learns which past applicants are most informative
                    for the current query.
                  </p>
                  <div className="bg-black/30 rounded-lg px-4 py-3 overflow-x-auto">
                    <M display>{"\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right) V"}</M>
                  </div>
                </div>

                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Q / K / V Analogy
                  </h4>
                  <div className="space-y-3 text-gray-300 text-sm leading-relaxed">
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-400 font-bold shrink-0 w-6">Q</span>
                        <span>
                          <strong className="text-white">Query</strong> -- "What am I
                          looking for?" (the current applicant's profile)
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-400 font-bold shrink-0 w-6">K</span>
                        <span>
                          <strong className="text-white">Key</strong> -- "What's
                          available?" (features of historical applicants)
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-400 font-bold shrink-0 w-6">V</span>
                        <span>
                          <strong className="text-white">Value</strong> -- "What to
                          retrieve?" (their outcomes and features)
                        </span>
                      </li>
                    </ul>
                  </div>
                </GlassCard>

                <div className="space-y-5 text-gray-400 leading-relaxed max-w-3xl">
                  <h4 className="text-lg font-bold text-white">Multi-head attention</h4>
                  <p>
                    Multiple attention heads run in parallel, each learning to focus on
                    different aspects: one head might attend to grade similarity, another
                    to geographic proximity, a third to program competitiveness.
                  </p>
                  <h4 className="text-lg font-bold text-white">
                    Self vs. cross attention
                  </h4>
                  <p>
                    <strong className="text-white">Self-attention</strong> lets features
                    within a single application interact (e.g., grade &times; program
                    competitiveness).{' '}
                    <strong className="text-white">Cross-attention</strong> compares the
                    current applicant against historical records.
                  </p>
                </div>

                {/* Example output */}
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Example Attention Weights
                  </h4>
                  <p className="text-gray-400 text-sm mb-4">
                    For a query "UBC Computer Science, 92.5% average":
                  </p>
                  <div className="space-y-2">
                    {[
                      { program: 'Waterloo CS', weight: 0.35 },
                      { program: 'UofT CS', weight: 0.25 },
                      { program: 'McGill CS', weight: 0.15 },
                      { program: 'Other', weight: 0.25 },
                    ].map((item) => (
                      <div key={item.program} className="flex items-center gap-3">
                        <span className="text-gray-300 text-sm w-28 shrink-0">
                          {item.program}
                        </span>
                        <div className="flex-1 bg-white/5 rounded-full h-4 overflow-hidden">
                          <div
                            className="h-full bg-emerald-400/30 rounded-full"
                            style={{ width: `${item.weight * 100}%` }}
                          />
                        </div>
                        <span className="text-emerald-400 font-mono text-sm tabular-nums w-12 text-right">
                          {item.weight.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </GlassCard>

                <GlassCard>
                  <AttentionHeatmap />
                </GlassCard>
              </div>
            </section>

            {/* ════════════════════════════════════
                Section 7: Fairness
                ════════════════════════════════════ */}
            <section id="fairness" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Ethics"
                  title="Fairness"
                  accent="monitoring."
                />

                <GlassCard>
                  <div className="space-y-5 text-gray-400 leading-relaxed">
                    <h4 className="text-lg font-bold text-white">
                      Provincial subgroup calibration
                    </h4>
                    <p>
                      We evaluate model fairness across provinces by computing calibration
                      metrics (Brier score, ECE) for each subgroup independently. If a
                      provincial subgroup shows significantly worse calibration, we flag
                      it and investigate whether the training data is representative.
                    </p>

                    <h4 className="text-lg font-bold text-white">
                      Data representativeness
                    </h4>
                    <p>
                      Self-reported data inherently reflects the demographics of the
                      communities it comes from, which may not represent all applicant
                      populations equally. Provinces with fewer data points receive wider
                      credible intervals from the Beta-Binomial baseline, naturally
                      flagging uncertainty.
                    </p>

                    <div className="p-4 bg-emerald-400/5 border border-emerald-400/20 rounded-lg">
                      <p className="text-gray-300 text-sm leading-relaxed">
                        <strong className="text-white">Our commitment:</strong> Treat
                        predictions as one data point among many, not as definitive
                        outcomes. We continuously monitor subgroup performance and
                        transparently report calibration gaps when they arise.
                      </p>
                    </div>
                  </div>
                </GlassCard>
              </div>
            </section>

            {/* ── PrevNextNav ──────────────────── */}
            <section className="pb-24">
              <div className="max-w-3xl">
                <PrevNextNav
                  prev={{ label: 'Math Foundation', to: '/about/math' }}
                  next={{ label: 'Evaluation', to: '/about/evaluation' }}
                />
              </div>
            </section>

          </div>
        </div>
      </div>
    </div>
  )
}

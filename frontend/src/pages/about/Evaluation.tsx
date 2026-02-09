import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import StatCard from '@/components/shared/StatCard'
import Breadcrumb from '@/components/shared/Breadcrumb'
import PrevNextNav from '@/components/shared/PrevNextNav'
import TableOfContents, { type TocItem } from '@/components/shared/TableOfContents'
import M from '@/components/shared/Math'
import ReliabilityDiagram from '@/components/viz/ReliabilityDiagram'
import CalibrationPlayground from '@/components/interactive/CalibrationPlayground'
import BrierDecomposition from '@/components/interactive/BrierDecomposition'
import TemporalSplitDiagram from '@/components/interactive/TemporalSplitDiagram'
import ROCCurveExplorer from '@/components/interactive/ROCCurveExplorer'

/* ─────────────────────────────────────────────
   Table of Contents items
   ───────────────────────────────────────────── */
const tocItems: TocItem[] = [
  { id: 'pillars', label: 'Three Pillars' },
  { id: 'calibration', label: 'Calibration Metrics' },
  { id: 'platt-scaling', label: 'Platt Scaling' },
  { id: 'discrimination', label: 'Discrimination' },
  { id: 'validation', label: 'Validation Strategy' },
]

/* ─────────────────────────────────────────────
   Mock calibration data
   ───────────────────────────────────────────── */
const mockCalibration = [
  { predicted: 0.1, observed: 0.12, count: 50 },
  { predicted: 0.2, observed: 0.18, count: 80 },
  { predicted: 0.3, observed: 0.32, count: 120 },
  { predicted: 0.4, observed: 0.38, count: 150 },
  { predicted: 0.5, observed: 0.52, count: 180 },
  { predicted: 0.6, observed: 0.58, count: 160 },
  { predicted: 0.7, observed: 0.72, count: 140 },
  { predicted: 0.8, observed: 0.78, count: 100 },
  { predicted: 0.9, observed: 0.88, count: 60 },
]

/* ─────────────────────────────────────────────
   AUC interpretation table data
   ───────────────────────────────────────────── */
const AUC_TABLE = [
  { auc: '0.5', quality: 'Random guessing' },
  { auc: '0.7-0.8', quality: 'Good' },
  { auc: '0.8-0.9', quality: 'Very good' },
  { auc: '0.9-1.0', quality: 'Excellent' },
]

/* ─────────────────────────────────────────────
   Validation strategies table data
   ───────────────────────────────────────────── */
const VALIDATION_STRATEGIES = [
  {
    strategy: 'Temporal Split',
    description: 'Train on past, test on future',
    when: 'Default',
  },
  {
    strategy: 'Expanding Window',
    description: 'Training grows over time',
    when: 'More data always helps',
  },
  {
    strategy: 'Sliding Window',
    description: 'Fixed window slides forward',
    when: 'Recent data matters most',
  },
  {
    strategy: 'Stratified K-Fold',
    description: 'Random, preserving balance',
    when: 'Only if truly i.i.d.',
  },
]

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function Evaluation() {
  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <Breadcrumb
            items={[
              { label: 'About', to: '/about' },
              { label: 'Evaluation' },
            ]}
          />

          <div className="mt-6 space-y-4">
            <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
              Evaluation
            </p>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
              Measuring what{' '}
              <span className="accent-serif">matters.</span>
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl">
              Calibration, discrimination, and validation -- three axes of
              model quality.
            </p>
          </div>
        </div>
      </section>

      {/* ── Layout: TOC sidebar + main content ── */}
      <div className="container mx-auto px-6 max-w-6xl flex gap-6 lg:gap-12">
        <TableOfContents items={tocItems} />

        <main className="flex-1 min-w-0">
          {/* ── Three Pillars ───────────────────── */}
          <section id="pillars" className="py-16">
            <div className="space-y-10">
              <SectionHeader
                label="Overview"
                title="Three pillars of"
                accent="quality."
              />

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                <StatCard
                  value="0.12"
                  label="Calibration"
                  description="Does 70% mean 70%?"
                />
                <StatCard
                  value="0.82"
                  label="Discrimination"
                  description="Can it separate admits from rejects?"
                />
                <StatCard
                  value="Temporal"
                  label="Validation"
                  description="Will it work next year?"
                />
              </div>
            </div>
          </section>

          {/* ── Calibration Metrics ─────────────── */}
          <section id="calibration" className="py-16">
            <div className="space-y-10">
              <SectionHeader
                label="Calibration"
                title="When we say 70%,"
                accent="we mean it."
              />

              {/* Brier Score */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">Brier Score</h3>
                <p>
                  The mean squared error of probability predictions. It measures
                  how close predicted probabilities are to actual outcomes.
                </p>
                <div className="my-4">
                  <M display>{"\\text{Brier} = \\frac{1}{n} \\sum_{i}(p_i - y_i)^2"}</M>
                </div>
                <p>
                  Range: <M>{"0"}</M>{' '}
                  (perfect) to{' '}
                  <M>{"1"}</M>{' '}
                  (worst). A Brier score of 0.12 means our predictions are close
                  to actual outcomes.
                </p>
              </div>

              {/* Brier Decomposition */}
              <GlassCard>
                <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                  Brier Decomposition
                </h4>
                <p className="text-gray-400 text-sm mb-6">
                  The breakdown nobody else visualizes.
                </p>

                <div className="mb-6">
                  <M display>{"\\text{Brier} = \\text{Uncertainty} - \\text{Resolution} + \\text{Reliability}"}</M>
                </div>

                <div className="space-y-5">
                  <div className="space-y-1">
                    <p className="text-white font-medium">Uncertainty</p>
                    <M>{"\\text{Var}(y) = \\bar{y}(1 - \\bar{y})"}</M>
                    <p className="text-gray-400 text-sm mt-1">
                      How inherently unpredictable? Fixed for dataset.
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-white font-medium">Resolution</p>
                    <M>{"\\frac{1}{n}\\sum_k n_k(\\bar{y}_k - \\bar{y})^2"}</M>
                    <p className="text-gray-400 text-sm mt-1">
                      How much do predicted groups differ? Higher is better.
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-white font-medium">Reliability</p>
                    <M>{"\\frac{1}{n}\\sum_k n_k(\\bar{p}_k - \\bar{y}_k)^2"}</M>
                    <p className="text-gray-400 text-sm mt-1">
                      How close are predictions to reality? Lower is better.
                    </p>
                  </div>
                </div>

                <BrierDecomposition />
              </GlassCard>

              {/* ECE */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  ECE (Expected Calibration Error)
                </h3>
                <div className="my-4">
                  <M display>{"\\text{ECE} = \\sum_k \\frac{n_k}{n} |\\bar{p}_k - \\bar{y}_k|"}</M>
                </div>
                <p>
                  Bins predictions into groups, measures average gap. Our ECE of
                  3.2% means predictions are off by 3.2% on average.
                </p>
              </div>

              {/* MCE */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  MCE (Maximum Calibration Error)
                </h3>
                <div className="my-4">
                  <M display>{"\\text{MCE} = \\max_k |\\bar{p}_k - \\bar{y}_k|"}</M>
                </div>
                <p>
                  Worst-case bin error. Useful for detecting specific ranges
                  where the model fails.
                </p>
              </div>

              {/* Reliability diagram */}
              <GlassCard className="shrink-0">
                <ReliabilityDiagram data={mockCalibration} />
              </GlassCard>
            </div>
          </section>

          {/* ── Platt Scaling ───────────────────── */}
          <section id="platt-scaling" className="py-16">
            <div className="space-y-10">
              <SectionHeader
                label="Post-Hoc Calibration"
                title="Making probabilities"
                accent="honest."
              />

              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <p>
                  Raw logistic regression can be overconfident or
                  underconfident. Platt scaling corrects this by fitting a
                  secondary sigmoid transformation on a held-out validation set.
                </p>
                <div className="my-4 space-y-2">
                  <M display>{"p_{\\text{cal}} = \\sigma(A \\cdot \\text{logit}(p_{\\text{raw}}) + B)"}</M>
                  <M display>{"\\text{logit}(p) = \\log\\left(\\frac{p}{1-p}\\right)"}</M>
                </div>
              </div>

              {/* How A and B are fitted */}
              <GlassCard>
                <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                  How A and B Are Fitted
                </h4>
                <div className="space-y-4 text-gray-400 leading-relaxed">
                  <p>
                    Gradient descent on a held-out validation set minimizes the
                    negative log-likelihood:
                  </p>
                  <div className="space-y-2">
                    <M display>{"\\mathcal{L} = -\\frac{1}{n}\\sum\\left[y\\log(p_{\\text{cal}}) + (1-y)\\log(1-p_{\\text{cal}})\\right]"}</M>
                  </div>
                  <p>Parameter updates at each step:</p>
                  <div className="space-y-1">
                    <M display>{"A \\leftarrow A - \\eta \\cdot \\frac{\\partial \\mathcal{L}}{\\partial A}"}</M>
                    <M display>{"B \\leftarrow B - \\eta \\cdot \\frac{\\partial \\mathcal{L}}{\\partial B}"}</M>
                  </div>
                </div>
              </GlassCard>

              {/* Interpreting A and B */}
              <GlassCard>
                <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-6">
                  Interpreting A and B
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <h5 className="text-white font-medium">
                      Slope parameter A
                    </h5>
                    <ul className="space-y-3 text-gray-400 text-sm">
                      <li className="flex items-start gap-3">
                        <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
                        <span>
                          <M>{"A \\approx 1.0"}</M>
                          {' '}&mdash; predictions already well-calibrated
                        </span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
                        <span>
                          <M>{"A > 1.0"}</M>
                          {' '}&mdash; raw predictions too uncertain (under-confident)
                        </span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
                        <span>
                          <M>{"A < 1.0"}</M>
                          {' '}&mdash; raw predictions too confident (over-confident)
                        </span>
                      </li>
                    </ul>
                  </div>
                  <div className="space-y-4">
                    <h5 className="text-white font-medium">
                      Intercept parameter B
                    </h5>
                    <ul className="space-y-3 text-gray-400 text-sm">
                      <li className="flex items-start gap-3">
                        <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
                        <span>
                          <M>{"B > 0"}</M>
                          {' '}&mdash; shifts probabilities up
                        </span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
                        <span>
                          <M>{"B < 0"}</M>
                          {' '}&mdash; shifts probabilities down
                        </span>
                      </li>
                    </ul>
                  </div>
                </div>
              </GlassCard>

              {/* Interactive playground */}
              <GlassCard>
                <CalibrationPlayground />
              </GlassCard>
            </div>
          </section>

          {/* ── Discrimination Metrics ──────────── */}
          <section id="discrimination" className="py-16">
            <div className="space-y-10">
              <SectionHeader
                label="Discrimination"
                title="Separating admits from"
                accent="rejects."
              />

              {/* ROC-AUC */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">ROC-AUC</h3>
                <p>
                  The ROC curve plots True Positive Rate vs False Positive Rate
                  at every threshold. The area under this curve (AUC) summarizes
                  overall discrimination ability.
                </p>
                <p>
                  <strong className="text-white">Algorithm:</strong> sweep the
                  decision threshold from 1.0 to 0.0, computing TPR and FPR at
                  each step. The curve traced out reveals how the model trades
                  off sensitivity for specificity.
                </p>
              </div>

              {/* AUC interpretation table */}
              <GlassCard padding="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                        <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">AUC</th>
                        <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Quality</th>
                      </tr>
                    </thead>
                    <tbody>
                      {AUC_TABLE.map((row) => (
                        <tr
                          key={row.auc}
                          className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                        >
                          <td className="px-3 sm:px-6 py-3 sm:py-4 text-emerald-400 font-mono tabular-nums">
                            {row.auc}
                          </td>
                          <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                            {row.quality}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>

              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <p>
                  <strong className="text-white">
                    Probabilistic interpretation:
                  </strong>{' '}
                  AUC = probability that a random admitted student has higher
                  predicted probability than a random rejected student.
                </p>
              </div>

              <GlassCard>
                <ROCCurveExplorer />
              </GlassCard>

              {/* Precision-Recall */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  Precision-Recall
                </h3>
                <p>
                  More informative than ROC when classes are imbalanced.
                  Precision focuses on the quality of positive predictions,
                  while recall measures coverage of actual positives.
                </p>
                <div className="space-y-2 my-4">
                  <M display>{"\\text{Precision} = \\frac{TP}{TP + FP}"}</M>
                  <M display>{"\\text{Recall} = \\frac{TP}{TP + FN}"}</M>
                </div>
                <p>
                  <strong className="text-white">Average Precision (AP)</strong>{' '}
                  = area under the PR curve. It summarizes the precision-recall
                  trade-off across all thresholds.
                </p>
              </div>

              {/* Lift Analysis */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  Lift Analysis
                </h3>
                <div className="my-4">
                  <M display>{"\\text{Lift}_{k\\%} = \\frac{\\text{positive rate in top } k\\%}{\\text{overall positive rate}}"}</M>
                </div>
                <p>
                  Lift = 2.0 means our top picks are 2x better than
                  random. This metric is especially valuable for understanding
                  how well the model prioritizes the most likely admits.
                </p>
              </div>

              {/* DeLong Test */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">DeLong Test</h3>
                <p>
                  A statistical test to compare two models' AUC values and
                  determine whether the difference is significant.
                </p>
                <div className="space-y-2 my-4">
                  <M display>{"H_0: \\text{AUC}_A = \\text{AUC}_B"}</M>
                  <p className="text-gray-400">
                    <M>{"p < 0.05"}</M> means significant difference
                  </p>
                </div>
                <p>
                  We use DeLong's test to validate that model improvements (e.g.
                  adding interaction features) produce statistically meaningful
                  gains in discrimination, not just noise.
                </p>
              </div>
            </div>
          </section>

          {/* ── Validation Strategy ─────────────── */}
          <section id="validation" className="py-16">
            <div className="space-y-10">
              <SectionHeader
                label="Validation"
                title="Will it work"
                accent="next year?"
              />

              {/* Why temporal splitting */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  Why Temporal Splitting
                </h3>
                <p>
                  Admission data is time-series. Random splits leak future
                  information! The model could memorize patterns from future
                  cycles, producing unrealistically optimistic evaluation
                  metrics.
                </p>
              </div>

              {/* Wrong vs Right approach */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-red-400 font-medium mb-3">
                    Wrong
                  </h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Train on 2024, test on 2022 -- the model sees the future.
                    Random splits across years create temporal leakage that
                    inflates accuracy.
                  </p>
                </GlassCard>
                <GlassCard>
                  <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-3">
                    Right
                  </h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Train 2022-23, Validate 2023-24, Test 2024-25. Strictly
                    chronological ordering ensures the model never sees future
                    data.
                  </p>
                </GlassCard>
              </div>

              {/* Validation strategies table */}
              <GlassCard padding="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                        <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Strategy</th>
                        <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Description</th>
                        <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">When to Use</th>
                      </tr>
                    </thead>
                    <tbody>
                      {VALIDATION_STRATEGIES.map((row) => (
                        <tr
                          key={row.strategy}
                          className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                        >
                          <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-medium">
                            {row.strategy}
                          </td>
                          <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                            {row.description}
                          </td>
                          <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-400">
                            {row.when}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>

              {/* Expanding Window CV */}
              <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                <h3 className="text-2xl font-bold text-white">
                  Expanding Window CV
                </h3>
                <p>
                  Our cross-validation strategy grows the training window over
                  time, ensuring the model always trains on past data and
                  validates on the immediate future.
                </p>
              </div>

              <GlassCard>
                <h4 className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                  Folds
                </h4>
                <div className="space-y-4">
                  <div className="flex items-center gap-2 sm:gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0">
                      1
                    </span>
                    <div className="flex-1 flex items-center gap-2 sm:gap-3">
                      <div className="flex-1 rounded-lg bg-emerald-400/10 border border-emerald-400/20 px-2 sm:px-4 py-2 text-center">
                        <p className="text-xs text-gray-500 uppercase tracking-wider">
                          Train
                        </p>
                        <p className="text-xs sm:text-sm text-emerald-400 font-mono">
                          2022-23
                        </p>
                      </div>
                      <span className="text-gray-600 shrink-0">&rarr;</span>
                      <div className="flex-1 rounded-lg bg-white/5 border border-white/10 px-2 sm:px-4 py-2 text-center">
                        <p className="text-xs text-gray-500 uppercase tracking-wider">
                          Validate
                        </p>
                        <p className="text-xs sm:text-sm text-white font-mono">
                          2023-24
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 sm:gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0">
                      2
                    </span>
                    <div className="flex-1 flex items-center gap-2 sm:gap-3">
                      <div className="flex-1 rounded-lg bg-emerald-400/10 border border-emerald-400/20 px-2 sm:px-4 py-2 text-center">
                        <p className="text-xs text-gray-500 uppercase tracking-wider">
                          Train
                        </p>
                        <p className="text-xs sm:text-sm text-emerald-400 font-mono">
                          2022-23, 2023-24
                        </p>
                      </div>
                      <span className="text-gray-600 shrink-0">&rarr;</span>
                      <div className="flex-1 rounded-lg bg-white/5 border border-white/10 px-2 sm:px-4 py-2 text-center">
                        <p className="text-xs text-gray-500 uppercase tracking-wider">
                          Validate
                        </p>
                        <p className="text-xs sm:text-sm text-white font-mono">
                          2024-25
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <TemporalSplitDiagram />
              </GlassCard>
            </div>
          </section>

          {/* ── PrevNextNav ──────────────────────── */}
          <section className="pb-24">
            <div className="max-w-6xl">
              <PrevNextNav
                prev={{ label: 'Models', to: '/about/models' }}
                next={{ label: 'API Pipeline', to: '/about/api' }}
              />
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

import { useState } from 'react'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import Breadcrumb from '@/components/shared/Breadcrumb'
import PrevNextNav from '@/components/shared/PrevNextNav'
import TableOfContents, { type TocItem } from '@/components/shared/TableOfContents'
import WorkedExampleTracer from '@/components/interactive/WorkedExampleTracer'
import M from '@/components/shared/Math'
import DesignMatrixBuilder from '@/components/interactive/DesignMatrixBuilder'

/* ─────────────────────────────────────────────
   Table of contents items
   ───────────────────────────────────────────── */
const tocItems: TocItem[] = [
  { id: 'encoders', label: 'Feature Engineering' },
  { id: 'design-matrix', label: 'Design Matrix' },
  { id: 'endpoints', label: 'API Endpoints' },
  { id: 'pipeline', label: '8-Step Pipeline' },
  { id: 'example', label: 'Worked Example' },
]

/* ─────────────────────────────────────────────
   Encoder table data
   ───────────────────────────────────────────── */
const ENCODERS = [
  { encoder: 'GPAEncoder', input: '92.5', output: 'z-score + bucket flags', method: 'Z-score normalization, percentile binning' },
  { encoder: 'UniversityEncoder', input: '"UofT"', output: 'Binary indicators', method: 'One-hot / dummy encoding' },
  { encoder: 'ProgramEncoder', input: '"Computer Science"', output: 'Binary indicators', method: 'Clustered encoding' },
  { encoder: 'TermEncoder', input: '"Fall 2023"', output: 'Cyclical features', method: 'Sin/cos encoding' },
  { encoder: 'DateEncoder', input: 'Application dates', output: 'Days elapsed, month', method: 'Temporal extraction' },
  { encoder: 'FrequencyEncoder', input: 'Category name', output: 'Category frequency', method: 'Target-independent' },
  { encoder: 'WOEEncoder', input: 'Category name', output: 'Weight of Evidence', method: 'ln(% events / % non-events)' },
  { encoder: 'CompositeEncoder', input: 'Multiple fields', output: 'Combined vector', method: 'Chains encoders' },
]

/* ─────────────────────────────────────────────
   Validation utilities data
   ───────────────────────────────────────────── */
const VALIDATION_UTILS = [
  { fn: 'validate_design_matrix()', check: 'NaN/Inf, constant columns, rank, condition number, outliers' },
  { fn: 'check_column_rank()', check: 'SVD-based rank deficiency' },
  { fn: 'identify_collinear_features()', check: 'High-correlation feature pairs' },
]

/* ─────────────────────────────────────────────
   API endpoints data
   ───────────────────────────────────────────── */
const ENDPOINTS = [
  { method: 'POST', path: '/predict', purpose: 'Single prediction' },
  { method: 'POST', path: '/predict/batch', purpose: 'Batch (up to 1000)' },
  { method: 'POST', path: '/explain', purpose: 'Feature importance' },
  { method: 'GET', path: '/health', purpose: 'Service health' },
  { method: 'GET', path: '/model/info', purpose: 'Model metadata' },
  { method: 'GET', path: '/universities', purpose: 'Supported universities' },
  { method: 'GET', path: '/programs', purpose: 'Programs (filterable)' },
]

/* ─────────────────────────────────────────────
   Pipeline steps data
   ───────────────────────────────────────────── */
const PIPELINE_STEPS = [
  {
    step: 1,
    label: 'VALIDATE',
    summary: 'Check required fields, range 0-100, valid province',
    detail:
      'Every incoming request is validated against the schema before any computation begins. Required fields (top_6_average, university, program) must be present. Numeric grades must fall within the 0-100 range. Province must be one of the recognized Canadian provinces. Invalid requests return a 422 with a detailed error message identifying which fields failed validation.',
  },
  {
    step: 2,
    label: 'BUILD FEATURES',
    summary: 'Normalize names, construct the feature vector x',
    detail:
      'University and program names are fuzzy-matched to canonical forms using RapidFuzz. Grade averages are divided by 100 to produce [0, 1] features. Province is one-hot encoded into indicator variables for Ontario, BC, Alberta, and Quebec (other provinces fall into the implicit reference category). The final vector includes a bias term: x = [1.0, avg/100, g11/100, g12/100, is_ON, is_BC, is_AB, is_QC].',
  },
  {
    step: 3,
    label: 'RAW PREDICTION',
    summary: 'Compute z = x\u1D40\u03B2, then p_raw = sigmoid(z)',
    detail:
      'The dot product of the feature vector x and the trained coefficient vector \u03B2 produces the log-odds z. The sigmoid function 1/(1+exp(-z)) transforms this into a raw probability between 0 and 1. This is a single vector operation -- the core of every prediction.',
  },
  {
    step: 4,
    label: 'CALIBRATION',
    summary: 'Platt scaling: logit \u2192 calibrated probability',
    detail:
      'Raw probabilities from logistic regression can be systematically over- or under-confident. Platt scaling fits two parameters (A, B) on a held-out calibration set. The logit of the raw probability is computed as log(p/(1-p)), then the calibrated probability is sigmoid(A \u00D7 logit + B). This ensures that when the model outputs 70%, approximately 70% of applicants in that bucket were actually admitted.',
  },
  {
    step: 5,
    label: 'CONFIDENCE INTERVAL',
    summary: 'SE = \u221A(p(1-p)/n), then 95% CI',
    detail:
      'The standard error is estimated using the calibrated probability and the effective sample size for the university-program combination. The 95% confidence interval is [p - 1.96\u00D7SE, p + 1.96\u00D7SE], clipped to [0, 1]. Wider intervals indicate less certainty -- typically for programs with fewer historical data points.',
  },
  {
    step: 6,
    label: 'EXPLANATION',
    summary: 'Compute feature contributions, sort by |contribution|',
    detail:
      'Each feature\'s contribution is computed as x_j \u00D7 \u03B2_j -- how much that feature pushes the log-odds up or down. Contributions are sorted by absolute value so the most influential features appear first. This powers the "why" behind every prediction, making the model interpretable at the individual level.',
  },
  {
    step: 7,
    label: 'LABEL',
    summary: 'Assign categorical label based on probability thresholds',
    detail:
      'The calibrated probability is mapped to one of three labels: LIKELY_ADMIT (\u2265 0.70), UNCERTAIN (0.40 -- 0.70), or UNLIKELY_ADMIT (< 0.40). These thresholds are chosen to give applicants clear, actionable information while acknowledging the inherent uncertainty in admissions predictions.',
  },
  {
    step: 8,
    label: 'RETURN',
    summary: 'Assemble and return the PredictionResponse JSON',
    detail:
      'All computed values -- probability, confidence interval, label, feature importance, similar programs, and model version -- are assembled into the response payload. The response is serialized as JSON and returned with appropriate cache headers. Batch requests return an array of individual responses.',
  },
]

/* ─────────────────────────────────────────────
   Expandable pipeline step component
   ───────────────────────────────────────────── */
function PipelineStep({
  step,
  label,
  summary,
  detail,
  open,
  onToggle,
}: {
  step: number
  label: string
  summary: string
  detail: string
  open: boolean
  onToggle: () => void
}) {
  return (
    <div className="border-b border-white/10">
      <button
        onClick={onToggle}
        className="flex w-full items-start gap-4 py-5 text-left"
      >
        <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
          {step}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-white">
            <span className="text-emerald-400 font-mono text-sm mr-2">
              {label}
            </span>
            <span className="text-gray-300 font-normal text-sm">
              {summary}
            </span>
          </p>
        </div>
        <span className="ml-4 shrink-0 text-gray-500 text-xl leading-none mt-1">
          {open ? '\u2212' : '+'}
        </span>
      </button>
      {open && (
        <div className="pb-5 pl-12 pr-8">
          <p className="text-gray-400 leading-relaxed text-sm">{detail}</p>
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function ApiPipeline() {
  const [openStep, setOpenStep] = useState<number | null>(null)

  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <Breadcrumb
            items={[
              { label: 'About', to: '/about' },
              { label: 'API Pipeline' },
            ]}
          />

          <div className="mt-6">
            <SectionHeader
              label="API Pipeline"
              title="From input to"
              accent="probability."
              description="Feature engineering, the design matrix, and the 8-step serving pipeline that turns your application into a prediction."
            />
          </div>
        </div>
      </section>

      {/* ── Layout: sidebar + content ─────────── */}
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="flex gap-6 lg:gap-12">
          <TableOfContents items={tocItems} />

          <main className="min-w-0 flex-1">
            {/* ── Feature Engineering ───────────── */}
            <section id="encoders" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Feature Engineering"
                  title="Eight specialized"
                  accent="encoders."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  Raw applicant data -- grades, university names, program
                  strings, dates -- cannot be fed directly into a linear model.
                  Each field requires a specialized encoder that transforms it
                  into a numeric representation the model can consume. Together,
                  these eight encoders form the feature engineering layer.
                </p>

                {/* Encoder table */}
                <GlassCard padding="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Encoder</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Input</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Output</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Method</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ENCODERS.map((row) => (
                          <tr
                            key={row.encoder}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                              {row.encoder}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.input}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.output}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-400 text-sm">
                              {row.method}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>

                {/* GPAEncoder worked example */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    GPAEncoder worked example
                  </h3>
                  <GlassCard>
                    <pre className="font-mono text-xs sm:text-sm text-gray-300 leading-relaxed overflow-x-auto">
{`Input: top_6_average = 92.5

Step 1 - Z-score: (92.5 - 87.3) / 4.2 = 1.24
Step 2 - Bucket:  92.5 \u2208 [90, 95) \u2192 is_90_95 = 1
Step 3 - Output:  [1.24, 0, 0, 0, 1, 0]
                   z    <80 80-85 85-90 90-95 95+`}
                    </pre>
                    <p className="mt-4 text-gray-500 text-sm">
                      The z-score captures relative standing. The bucket flags
                      let the model learn non-linear cutoff effects -- a 90%
                      average behaves differently from an 85% average in ways
                      that a single linear term cannot capture.
                    </p>
                  </GlassCard>
                </div>
              </div>
            </section>

            {/* ── Design Matrix ─────────────────── */}
            <section id="design-matrix" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Design Matrix"
                  title="Building the"
                  accent="feature vector."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  The bridge between raw features and the numeric matrix{' '}
                  <M>{"X"}</M>{' '}
                  that models consume. The{' '}
                  <code className="text-emerald-400 font-mono text-sm">
                    DesignMatrixBuilder
                  </code>{' '}
                  uses a builder pattern to chain transformers together, fitting
                  on training data and transforming new observations
                  consistently.
                </p>

                {/* Builder pattern flow */}
                <GlassCard>
                  <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Builder Pattern
                  </p>
                  <pre className="font-mono text-xs sm:text-sm text-gray-300 leading-relaxed overflow-x-auto">
{`Raw dicts \u2192 DesignMatrixBuilder.fit(data) \u2192 .transform(data) \u2192 X (n \u00D7 p) + feature names`}
                  </pre>
                  <p className="mt-4 text-gray-500 text-sm">
                    Fit learns statistics (means, standard deviations, category
                    sets) from training data. Transform applies those learned
                    parameters to any new data, ensuring consistency between
                    training and serving.
                  </p>
                </GlassCard>

                {/* Transformer classes */}
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-white">
                    Transformer classes
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <GlassCard>
                      <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                        NumericScaler
                      </p>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Z-score:{' '}
                        <M>{"(x - \\mu) / \\sigma"}</M>{' '}
                        or Min-max:{' '}
                        <M>{"(x - \\min) / (\\max - \\min)"}</M>
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        If std = 0, outputs 0.0 to avoid division by zero.
                      </p>
                    </GlassCard>

                    <GlassCard>
                      <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                        OneHotEncoder
                      </p>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Category &rarr; binary columns. Each unique category
                        becomes its own 0/1 column.
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        Three unknown-handling strategies: error, ignore,
                        encode.
                      </p>
                    </GlassCard>

                    <GlassCard>
                      <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                        DummyEncoder
                      </p>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Like OneHotEncoder but drops one category to avoid
                        multicollinearity -- essential for linear models.
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        The dropped category becomes the implicit reference
                        level.
                      </p>
                    </GlassCard>

                    <GlassCard>
                      <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                        OrdinalEncoder
                      </p>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        Ordered categories &rarr; integers preserving order.
                        Suitable when categories have a natural ranking.
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        Maps each level to its position: 0, 1, 2, ...
                      </p>
                    </GlassCard>
                  </div>
                </div>

                {/* InteractionBuilder */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    InteractionBuilder
                  </h3>
                  <GlassCard>
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                          Multiplicative
                        </p>
                        <p className="text-gray-300 text-sm leading-relaxed">
                          <M>{"x_{\\text{avg}} \\times x_{\\text{competitive}}"}</M>{' '}
                          -- a high average matters more for competitive
                          programs.
                        </p>
                      </div>
                      <div>
                        <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                          Polynomial
                        </p>
                        <p className="text-gray-300 text-sm leading-relaxed">
                          <M>{"x_{\\text{avg}},\\; x_{\\text{avg}}^2,\\; x_{\\text{avg}}^3"}</M>{' '}
                          -- captures non-linear GPA effects where the
                          relationship between average and admission probability
                          is not a straight line.
                        </p>
                      </div>
                    </div>
                  </GlassCard>
                </div>

                {/* Validation utilities */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Validation utilities
                  </h3>
                  <GlassCard padding="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Function</th>
                            <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">
                              What it checks
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {VALIDATION_UTILS.map((row) => (
                            <tr
                              key={row.fn}
                              className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                            >
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                                {row.fn}
                              </td>
                              <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                                {row.check}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </GlassCard>
                </div>

                {/* Condition number note */}
                <GlassCard className="border-emerald-400/20">
                  <div className="flex items-start gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      !
                    </span>
                    <div>
                      <p className="text-white font-medium">
                        Condition number threshold
                      </p>
                      <p className="text-gray-400 text-sm leading-relaxed mt-1">
                        If{' '}
                        <M>{"\\kappa(X) > 10^3"}</M>{' '}
                        (i.e.,{' '}
                        <M>{"\\kappa(X^TX) > 10^6"}</M>
                        ), ridge regularization is essential. Without it,
                        coefficient estimates become numerically unstable and
                        predictions are unreliable.
                      </p>
                    </div>
                  </div>
                </GlassCard>

                <GlassCard>
                  <DesignMatrixBuilder />
                </GlassCard>
              </div>
            </section>

            {/* ── API Endpoints ─────────────────── */}
            <section id="endpoints" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="API"
                  title="REST"
                  accent="endpoints."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  The prediction service exposes a RESTful API. All prediction
                  endpoints accept JSON and return JSON. Authentication is not
                  required for public endpoints.
                </p>

                {/* Endpoints table */}
                <GlassCard padding="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Method</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Path</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Purpose</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ENDPOINTS.map((row) => (
                          <tr
                            key={row.path}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-3 sm:px-6 py-3 sm:py-4">
                              <span
                                className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                                  row.method === 'POST'
                                    ? 'bg-emerald-400/10 text-emerald-400'
                                    : 'bg-blue-400/10 text-blue-400'
                                }`}
                              >
                                {row.method}
                              </span>
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                              {row.path}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.purpose}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>

                {/* Request schema */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Request schema
                  </h3>
                  <GlassCard>
                    <pre className="text-xs sm:text-sm font-mono text-gray-300 overflow-x-auto">
{`{
  "top_6_average": 92.5,
  "grade_11_average": 88.0,
  "grade_12_average": 91.0,
  "university": "University of Toronto",
  "program": "Computer Science",
  "province": "Ontario",
  "country": "Canada"
}`}
                    </pre>
                  </GlassCard>
                </div>

                {/* Response schema */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Response schema
                  </h3>
                  <GlassCard>
                    <pre className="text-xs sm:text-sm font-mono text-gray-300 overflow-x-auto">
{`{
  "probability": 0.73,
  "confidence_interval": { "lower": 0.65, "upper": 0.80 },
  "prediction": "LIKELY_ADMIT",
  "feature_importance": [...],
  "similar_programs": [...],
  "model_version": "v1.0.0"
}`}
                    </pre>
                  </GlassCard>
                </div>
              </div>
            </section>

            {/* ── 8-Step Pipeline ───────────────── */}
            <section id="pipeline" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Serving"
                  title="The 8-step"
                  accent="pipeline."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  Every prediction request passes through the same eight steps
                  in order. Click any step to expand its details.
                </p>

                {/* Expandable steps */}
                <GlassCard>
                  {PIPELINE_STEPS.map((s) => (
                    <PipelineStep
                      key={s.step}
                      step={s.step}
                      label={s.label}
                      summary={s.summary}
                      detail={s.detail}
                      open={openStep === s.step}
                      onToggle={() =>
                        setOpenStep(openStep === s.step ? null : s.step)
                      }
                    />
                  ))}
                </GlassCard>

                {/* Label thresholds visual */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Decision thresholds
                  </h3>
                  <GlassCard>
                    <div className="space-y-3">
                      <div className="flex h-10 rounded-lg overflow-hidden text-[10px] sm:text-xs font-bold">
                        <div className="flex items-center justify-center bg-red-500/20 text-red-400 border-r border-white/10 w-[40%]">
                          UNLIKELY_ADMIT
                        </div>
                        <div className="flex items-center justify-center bg-yellow-500/20 text-yellow-400 border-r border-white/10 w-[30%]">
                          UNCERTAIN
                        </div>
                        <div className="flex items-center justify-center bg-emerald-500/20 text-emerald-400 w-[30%]">
                          LIKELY_ADMIT
                        </div>
                      </div>
                      <div className="flex text-xs text-gray-500 font-mono">
                        <span className="w-[40%] text-left">0.00</span>
                        <span className="w-[30%] text-left">0.40</span>
                        <span className="w-[30%] text-left">0.70</span>
                        <span className="text-right">1.00</span>
                      </div>
                    </div>
                  </GlassCard>
                </div>
              </div>
            </section>

            {/* ── Worked Example ────────────────── */}
            <section id="example" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Worked Example"
                  title="Tracing one"
                  accent="prediction."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  Ontario student, 92.5% average &rarr; UofT Computer Science.
                  Let's trace every step of the pipeline with real numbers.
                </p>

                {/* Step 1: Normalization */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      1
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Normalization
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`"University of Toronto" \u2192 exact match (confidence: 1.00)`}
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 2: Feature vector */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      2
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Feature vector
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`x = [1.0,   0.925,  0.88,   0.91,   1.0,   0.0,   0.0,   0.0]
      bias   avg     g11     g12     is_ON  is_BC  is_AB  is_QC`}
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 3: Linear combination */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      3
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Linear combination
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto leading-relaxed">
{`z = x\u1D40\u03B2
  = 1.0\u00D7(-0.5) + 0.925\u00D70.08 + 0.88\u00D70.03 + 0.91\u00D70.03
    + 1.0\u00D70.10 + 0.0\u00D70.05 + 0.0\u00D70.05 + 0.0\u00D70.08
  = -0.500 + 0.074 + 0.026 + 0.027 + 0.100 + 0.0 + 0.0 + 0.0
  = `}<span className="text-emerald-400">-0.2730</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 4: Sigmoid */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      4
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">Sigmoid</p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`p_raw = sigmoid(-0.2730)
      = 1 / (1 + exp(0.2730))
      = 1 / (1 + 1.3140)
      = `}<span className="text-emerald-400">0.432</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 5: Platt scaling */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      5
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Platt scaling
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`logit  = log(0.432 / (1 - 0.432)) = -0.273
p_cal  = sigmoid(1.0 \u00D7 (-0.273) + 0.0)
       = sigmoid(-0.273)
       = `}<span className="text-emerald-400">0.432</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 6: Confidence interval */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      6
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Confidence interval
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`SE = \u221A(0.432 \u00D7 0.568 / 10000) = 0.00495
CI = [0.432 - 1.96 \u00D7 0.00495,  0.432 + 1.96 \u00D7 0.00495]
   = `}<span className="text-emerald-400">[0.422, 0.442]</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 7: Feature importance */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      7
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">
                        Feature importance
                      </p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto leading-relaxed">
{`contribution_j = x_j \u00D7 \u03B2_j

  bias        1.0   \u00D7 -0.50  = `}<span className="text-red-400">-0.500</span>{`  (largest magnitude)
  is_ontario  1.0   \u00D7 +0.10  = `}<span className="text-emerald-400">+0.100</span>{`
  top_6_avg   0.925 \u00D7 +0.08  = `}<span className="text-emerald-400">+0.074</span>{`
  grade_12    0.91  \u00D7 +0.03  = `}<span className="text-emerald-400">+0.027</span>{`
  grade_11    0.88  \u00D7 +0.03  = `}<span className="text-emerald-400">+0.026</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Step 8: Label */}
                <GlassCard>
                  <div className="flex items-start gap-4">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      8
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-medium mb-2">Label</p>
                      <pre className="font-mono text-xs sm:text-sm text-gray-300 overflow-x-auto">
{`0.432 \u2208 [0.40, 0.70) \u2192 `}<span className="text-yellow-400 font-bold">"UNCERTAIN"</span>
                      </pre>
                    </div>
                  </div>
                </GlassCard>

                {/* Final JSON response */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Final response
                  </h3>
                  <GlassCard>
                    <pre className="text-xs sm:text-sm font-mono text-gray-300 overflow-x-auto">
{`{
  "probability": 0.432,
  "confidence_interval": { "lower": 0.422, "upper": 0.442 },
  "prediction": "UNCERTAIN",
  "feature_importance": [
    { "feature": "bias",       "contribution": -0.500 },
    { "feature": "is_ontario", "contribution": +0.100 },
    { "feature": "top_6_avg",  "contribution": +0.074 },
    { "feature": "grade_12",   "contribution": +0.027 },
    { "feature": "grade_11",   "contribution": +0.026 }
  ],
  "similar_programs": [
    { "program": "UofT Data Science",     "probability": 0.48 },
    { "program": "UofT Math & Statistics", "probability": 0.61 }
  ],
  "model_version": "v1.0.0"
}`}
                    </pre>
                  </GlassCard>
                </div>

                {/* Calibration note */}
                <GlassCard className="border-emerald-400/20">
                  <div className="flex items-start gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0 mt-0.5">
                      !
                    </span>
                    <div>
                      <p className="text-white font-medium">
                        Note on Platt scaling
                      </p>
                      <p className="text-gray-400 text-sm leading-relaxed mt-1">
                        With A = 1.0, B = 0.0, Platt scaling is an identity
                        transform. A trained calibrator would have fitted A and B
                        on a held-out calibration set, potentially adjusting the
                        raw probabilities to better match observed admission
                        rates.
                      </p>
                    </div>
                  </div>
                </GlassCard>

                <WorkedExampleTracer />
              </div>
            </section>

            {/* ── PrevNextNav ─────────────────── */}
            <section className="pb-24">
              <PrevNextNav
                prev={{ label: 'Evaluation', to: '/about/evaluation' }}
              />
            </section>
          </main>
        </div>
      </div>
    </div>
  )
}

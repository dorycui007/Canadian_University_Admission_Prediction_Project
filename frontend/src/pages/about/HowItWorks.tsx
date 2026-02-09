import { useState } from 'react'
import { Link } from 'react-router-dom'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import useScrollProgress from '@/hooks/useScrollProgress'
import useReducedMotion from '@/hooks/useReducedMotion'
import FuzzyMatchDemo from '@/components/interactive/FuzzyMatchDemo'
import GaussianBellCurve from '@/components/interactive/GaussianBellCurve'
import SigmoidExplorer from '@/components/interactive/SigmoidExplorer'
import IRLSStepThrough from '@/components/interactive/IRLSStepThrough'
import CalibrationPlayground from '@/components/interactive/CalibrationPlayground'
import PipelineTracer from '@/components/interactive/PipelineTracer'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Fade-in wrapper
   ───────────────────────────────────────────── */
function FadeSection({ children }: { children: React.ReactNode }) {
  const { ref, isVisible } = useScrollProgress(0.15)
  const reducedMotion = useReducedMotion()

  return (
    <div
      ref={ref}
      className={
        reducedMotion
          ? ''
          : `transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`
      }
    >
      {children}
    </div>
  )
}

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function HowItWorks() {
  const [exampleGrade, setExampleGrade] = useState(92.5)

  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-16">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="space-y-4 mb-12">
            <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
              Interactive Guide
            </p>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
              How the prediction engine{' '}
              <span className="accent-serif">works.</span>
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl">
              Follow one example student through every layer of the pipeline.
              Drag the grade slider to see how one input change cascades through
              the entire system.
            </p>
          </div>

          <GlassCard>
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
              <div className="flex-1 space-y-1">
                <h3 className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
                  Example Student
                </h3>
                <p className="text-gray-300 text-sm">
                  Ontario &middot; UofT Computer Science &middot; Top-6 average:{' '}
                  <span className="text-white font-bold tabular-nums">
                    {exampleGrade.toFixed(1)}%
                  </span>
                </p>
              </div>
              <div className="w-full sm:w-64 space-y-1">
                <label className="text-xs text-gray-500">
                  Top-6 Average
                </label>
                <input
                  type="range"
                  min={70}
                  max={100}
                  step={0.5}
                  value={exampleGrade}
                  onChange={(e) => setExampleGrade(Number(e.target.value))}
                  className="w-full accent-emerald-400"
                />
                <div className="flex justify-between text-xs text-gray-600 tabular-nums">
                  <span>70%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* ── Section 1: Fuzzy Matching ─────────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 1 &middot; Normalization"
              title="Fuzzy matching"
              accent="names."
              description="Raw user input like 'uoft' or 'waterloo eng' needs to match canonical university names. We use token-sort similarity scoring to find the best match."
            />
            <div className="flex flex-col lg:flex-row gap-8">
              <div className="lg:w-2/5 space-y-4 text-gray-400 leading-relaxed">
                <p>
                  University names come in many forms: abbreviations, nicknames,
                  misspellings. The normalization layer computes a similarity
                  score against every canonical name and picks the best match
                  above an 85% threshold.
                </p>
                <p>
                  Try typing a university name below to see how the matching
                  works in real-time.
                </p>
                <Link to="/about/api#encoders" className="inline-flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300 transition-colors">
                  Learn more about feature engineering <span aria-hidden="true">&rarr;</span>
                </Link>
              </div>
              <div className="lg:w-3/5">
                <GlassCard>
                  <FuzzyMatchDemo />
                </GlassCard>
              </div>
            </div>
          </div>
        </section>
      </FadeSection>

      {/* ── Section 2: Feature Engineering ────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 2 &middot; Feature Engineering"
              title="From grades to"
              accent="z-scores."
              description="Raw percentages are standardized using the training set's mean and standard deviation, producing z-scores that the model can work with."
            />
            <div className="flex flex-col lg:flex-row gap-8">
              <div className="lg:w-2/5 space-y-4 text-gray-400 leading-relaxed">
                <p>
                  The GPA encoder transforms raw grades into standardized
                  z-scores. A student with a 92.5% average in a distribution
                  with mean 87.3% and standard deviation 4.2% gets a z-score of{' '}
                  <span className="text-white font-mono">
                    {((exampleGrade - 87.3) / 4.2).toFixed(2)}
                  </span>
                  .
                </p>
                <p>
                  The bell curve shows where the example student falls in the
                  grade distribution. Move the slider above to see the marker
                  shift.
                </p>
                <Link to="/about/api#design-matrix" className="inline-flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300 transition-colors">
                  Learn more about the design matrix <span aria-hidden="true">&rarr;</span>
                </Link>
              </div>
              <div className="lg:w-3/5">
                <GlassCard>
                  <GaussianBellCurve
                    mean={87.3}
                    std={4.2}
                    value={exampleGrade}
                  />
                </GlassCard>
              </div>
            </div>
          </div>
        </section>
      </FadeSection>

      {/* ── Section 3: Sigmoid ─── CENTERPIECE ── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 3 &middot; The Core Model"
              title="Logistic regression"
              accent="visualized."
              description="Each feature contributes to a linear sum z, which the sigmoid function squashes into a probability between 0 and 1."
            />
            <GlassCard>
              <SigmoidExplorer grade={exampleGrade} />
            </GlassCard>
            <Link to="/about/models#irls" className="inline-flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300 transition-colors">
              Deep dive into IRLS logistic regression <span aria-hidden="true">&rarr;</span>
            </Link>
          </div>
        </section>
      </FadeSection>

      {/* ── Section 4: IRLS ──────────────────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 4 &middot; Training"
              title="How the model"
              accent="learns."
              description="IRLS iteratively adjusts coefficients to minimize prediction error. Step through the iterations to watch convergence."
            />
            <div className="flex flex-col lg:flex-row gap-8">
              <div className="lg:w-2/5 space-y-4 text-gray-400 leading-relaxed">
                <p>
                  Iteratively Reweighted Least Squares (IRLS) is an elegant
                  algorithm: at each step, it converts the nonlinear logistic
                  problem into a weighted least squares problem, solved via QR
                  decomposition.
                </p>
                <p>
                  Watch how the coefficients stabilize and the loss drops as the
                  algorithm converges over 8 iterations.
                </p>
                <Link to="/about/math#qr" className="inline-flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300 transition-colors">
                  How QR decomposition solves each step <span aria-hidden="true">&rarr;</span>
                </Link>
              </div>
              <div className="lg:w-3/5">
                <GlassCard>
                  <IRLSStepThrough />
                </GlassCard>
              </div>
            </div>
          </div>
        </section>
      </FadeSection>

      {/* ── Section 5: Calibration ───────────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 5 &middot; Calibration"
              title="Making probabilities"
              accent="honest."
              description="Platt scaling adjusts raw model outputs so that a predicted 70% really means 70% of similar applicants were admitted."
            />
            <div className="flex flex-col lg:flex-row gap-8">
              <div className="lg:w-2/5 space-y-4 text-gray-400 leading-relaxed">
                <p>
                  Raw logistic regression probabilities can be overconfident or
                  underconfident. Platt scaling fits a secondary sigmoid{' '}
                  <M>{"p = \\sigma(A \\cdot \\mathrm{logit}(p_{\\mathrm{raw}}) + B)"}</M>{' '}
                  to correct the calibration.
                </p>
                <p>
                  Drag the A and B sliders to see how they reshape the
                  probability mapping. Watch the ECE and Brier score change in
                  real time.
                </p>
                <Link to="/about/evaluation#platt-scaling" className="inline-flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300 transition-colors">
                  Full Platt scaling derivation <span aria-hidden="true">&rarr;</span>
                </Link>
              </div>
              <div className="lg:w-3/5">
                <GlassCard>
                  <CalibrationPlayground />
                </GlassCard>
              </div>
            </div>
          </div>
        </section>
      </FadeSection>

      {/* ── Section 6: Pipeline ──────────────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-10">
            <SectionHeader
              label="Step 6 &middot; End-to-End"
              title="The full"
              accent="pipeline."
              description="From raw input to calibrated probability, trace every intermediate value through the six-layer prediction engine."
            />
            <PipelineTracer grade={exampleGrade} />
          </div>
        </section>
      </FadeSection>

      {/* ── Further Reading ────────────────────── */}
      <FadeSection>
        <section className="py-16">
          <div className="container mx-auto px-6 max-w-6xl space-y-8">
            <h3 className="text-2xl font-bold text-white">
              Want to go deeper?
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[
                { to: '/about/math', label: 'Math Foundation', desc: 'Vectors, QR, SVD, and ridge regression from scratch' },
                { to: '/about/models', label: 'Models', desc: 'Beta-Binomial, IRLS, hazard, embeddings, attention' },
                { to: '/about/evaluation', label: 'Evaluation', desc: 'Brier score, ROC-AUC, and validation strategy' },
                { to: '/about/api', label: 'API Pipeline', desc: 'Encoders, design matrix, and end-to-end example' },
              ].map((card) => (
                <Link key={card.to} to={card.to} className="group">
                  <GlassCard className="hover:border-white/20 transition-colors">
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-1">
                      {card.label}
                    </p>
                    <p className="text-gray-400 text-sm">{card.desc}</p>
                  </GlassCard>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </FadeSection>

      {/* ── CTA ──────────────────────────────── */}
      <section className="py-16 pb-24">
        <div className="container mx-auto px-6 max-w-6xl text-center space-y-6">
          <h2 className="text-2xl font-bold text-white">
            Ready to try it yourself?
          </h2>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/predict"
              className="px-6 py-3 rounded-full bg-emerald-400 text-black font-semibold text-sm hover:bg-emerald-300 transition-colors"
            >
              Get Your Prediction
            </Link>
            <Link
              to="/about/models"
              className="px-6 py-3 rounded-full border border-white/10 text-gray-300 font-medium text-sm hover:border-white/20 hover:text-white transition-colors"
            >
              Model Deep Dive
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}

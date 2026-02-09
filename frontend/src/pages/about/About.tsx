import { useState } from 'react'
import { Link } from 'react-router-dom'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import StatCard from '@/components/shared/StatCard'
import ReliabilityDiagram from '@/components/viz/ReliabilityDiagram'

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
   FAQ data
   ───────────────────────────────────────────── */
const FAQ_ITEMS = [
  {
    question: 'Is my data stored?',
    answer:
      'No. Predictions are computed in real-time and nothing is saved. We do not collect, log, or retain any input you provide.',
  },
  {
    question: 'How accurate is this?',
    answer:
      'Our Brier score of 0.12 means predictions closely match actual outcomes. When we predict a 70% chance of admission, approximately 70 out of 100 applicants in that bucket are indeed admitted.',
  },
  {
    question: 'Why are my chances so low?',
    answer:
      'Competitive programs have very high averages. For example, McMaster Health Sciences admits fewer than 10% of applicants, with median averages above 95%. A "low" predicted probability reflects the reality of the admissions landscape, not a judgment on your application.',
  },
  {
    question: 'What about supplementary applications?',
    answer:
      'Our model cannot score essays, interviews, or other supplementary components. For programs that weigh these heavily (like McMaster Health Sciences or Waterloo Engineering), the prediction reflects grade-based probability only.',
  },
]

/* ─────────────────────────────────────────────
   FAQ Item component
   ───────────────────────────────────────────── */
function FaqItem({ question, answer }: { question: string; answer: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border-b border-white/10">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between py-5 text-left"
      >
        <span className="font-medium text-white">{question}</span>
        <span className="ml-4 shrink-0 text-gray-500 text-xl leading-none">
          {open ? '\u2212' : '+'}
        </span>
      </button>
      {open && (
        <p className="pb-5 text-gray-400 leading-relaxed">{answer}</p>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function About() {
  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-16">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="space-y-4">
            <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
              About
            </p>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
              How this{' '}
              <span className="accent-serif">actually</span>{' '}
              works.
            </h2>
          </div>
        </div>
      </section>

      {/* ── Data section ─────────────────────── */}
      <section className="py-16">
        <div className="container mx-auto px-6 max-w-6xl space-y-10">
          <SectionHeader
            label="Data"
            title="~4,900 real applications."
          />

          <p className="text-gray-400 leading-relaxed max-w-3xl">
            Our dataset consists of self-reported application outcomes from the
            2022-2025 admissions cycles, sourced from Canadian student
            communities. Each data point includes the applicant's top-6 average,
            target university and program, province of origin, and the final
            admissions decision. We clean, normalize, and deduplicate every
            record before training.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <StatCard value="981" label="2022-23" />
            <StatCard value="1,845" label="2023-24" />
            <StatCard value="2,074" label="2024-25" />
          </div>
        </div>
      </section>

      {/* ── Deep Dive Navigation ──────────────── */}
      <section className="py-16">
        <div className="container mx-auto px-6 max-w-6xl space-y-10">
          <SectionHeader
            label="Deep Dives"
            title="Built from scratch, fully auditable."
          />

          <p className="text-gray-400 leading-relaxed max-w-3xl">
            Instead of using off-the-shelf ML libraries, we implemented the
            entire prediction pipeline from scratch in Python: vector and
            matrix operations, QR decomposition, IRLS logistic regression, and
            Platt scaling for calibration. Every line of code is auditable.
          </p>

          <Link
            to="/about/how-it-works"
            className="block group"
          >
            <GlassCard className="border-emerald-400/20 hover:border-emerald-400/40 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-1">
                    Interactive Guide
                  </p>
                  <p className="text-white font-bold text-lg">
                    See the prediction engine in action
                  </p>
                  <p className="text-gray-400 text-sm mt-1">
                    Follow one student through every layer of the pipeline with
                    interactive visualizations.
                  </p>
                </div>
                <span className="text-emerald-400 text-xl group-hover:translate-x-1 transition-transform" aria-hidden="true">
                  &rarr;
                </span>
              </div>
            </GlassCard>
          </Link>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {[
              {
                to: '/about/math',
                label: 'Math Foundation',
                description:
                  'Vectors, matrices, QR decomposition, SVD, and ridge regression -- the linear algebra behind every prediction.',
              },
              {
                to: '/about/models',
                label: 'Models',
                description:
                  'Beta-Binomial baseline, IRLS logistic regression, hazard model, embeddings, and attention.',
              },
              {
                to: '/about/evaluation',
                label: 'Evaluation',
                description:
                  'Brier score, ECE, ROC-AUC, Platt scaling, and temporal validation strategy.',
              },
              {
                to: '/about/api',
                label: 'API Pipeline',
                description:
                  'Feature engineering, the design matrix, 8-step serving pipeline, and an end-to-end worked example.',
              },
            ].map((card) => (
              <Link key={card.to} to={card.to} className="group">
                <GlassCard className="h-full hover:border-white/20 transition-colors">
                  <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                    {card.label}
                  </p>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    {card.description}
                  </p>
                  <span className="inline-flex items-center gap-1 text-sm text-emerald-400 mt-3 group-hover:gap-2 transition-all">
                    Read more <span aria-hidden="true">&rarr;</span>
                  </span>
                </GlassCard>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── Calibration section ──────────────── */}
      <section className="py-16">
        <div className="container mx-auto px-6 max-w-6xl space-y-10">
          <SectionHeader
            label="Calibration"
            title="When we say 70%, we mean it."
          />

          <div className="flex flex-col md:flex-row gap-8 items-start">
            <GlassCard className="shrink-0">
              <ReliabilityDiagram data={mockCalibration} />
            </GlassCard>

            <div className="flex-1 space-y-4 text-gray-400 leading-relaxed">
              <p>
                The reliability diagram shows predicted probabilities on the
                x-axis and actual observed admission rates on the y-axis. A
                perfectly calibrated model follows the diagonal. Our model
                tracks it closely, meaning the probabilities it outputs are
                trustworthy.
              </p>
              <p>
                Bubble size indicates the number of predictions in each bin --
                larger bubbles mean more data and more confidence.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <StatCard
              value="0.12"
              label="Brier Score"
              description="Lower is better"
            />
            <StatCard
              value="3.2%"
              label="ECE"
              description="Expected calibration error"
            />
            <StatCard
              value="0.82"
              label="ROC-AUC"
              description="Discrimination ability"
            />
          </div>
        </div>
      </section>

      {/* ── Limitations section ──────────────── */}
      <section className="py-16">
        <div className="container mx-auto px-6 max-w-6xl space-y-10">
          <SectionHeader
            label="Limitations"
            title="What we can't predict."
          />

          <ul className="space-y-3 text-gray-400 leading-relaxed max-w-3xl">
            <li className="flex items-start gap-3">
              <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gray-500" />
              <span>Self-reported data with potential selection bias</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gray-500" />
              <span>Cannot capture supplementary application quality</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gray-500" />
              <span>Historical data may not reflect future criteria</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gray-500" />
              <span>Small sample sizes for some programs</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="mt-2 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gray-500" />
              <span>Does not distinguish 101 vs 105 applicant pools</span>
            </li>
          </ul>
        </div>
      </section>

      {/* ── FAQ section ──────────────────────── */}
      <section className="py-16 pb-24">
        <div className="container mx-auto px-6 max-w-6xl space-y-10">
          <SectionHeader
            label="FAQ"
            title="Common questions."
          />

          <GlassCard>
            {FAQ_ITEMS.map((item) => (
              <FaqItem
                key={item.question}
                question={item.question}
                answer={item.answer}
              />
            ))}
          </GlassCard>
        </div>
      </section>
    </div>
  )
}

import { Link } from 'react-router-dom'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import StatCard from '@/components/shared/StatCard'


/* ─────────────────────────────────────────────
   Data for the competitive-programs table
   ───────────────────────────────────────────── */
const competitivePrograms = [
  { program: 'McMaster Health Sciences', avg: '90%+', rate: '5\u201310%' },
  { program: 'UofT Engineering Science', avg: 'High 90s', rate: '~10%' },
  { program: 'UofT Rotman Commerce', avg: 'High 90s', rate: '~15%' },
  { program: 'Waterloo Software Eng', avg: 'Mid-90s', rate: '~20%' },
  { program: 'UofT Computer Science', avg: 'Low 90s', rate: '~30%' },
  { program: 'Waterloo CS', avg: 'Mid-high 80s', rate: '~40%' },
  { program: 'Western / Carleton CS', avg: 'Mid 80s', rate: '~50%' },
  { program: 'General Arts / Science', avg: '70\u201375%', rate: '70\u201380%' },
]

/* ─────────────────────────────────────────────
   Landing page
   ───────────────────────────────────────────── */
export default function Landing() {
  return (
    <div>
      {/* ── 1. Hero ─────────────────────────────── */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Emerald orb background */}
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-emerald-orb pointer-events-none"
        />

        <div className="relative z-10 container mx-auto px-6 max-w-6xl text-center flex flex-col items-center gap-6">
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold tracking-tight text-white leading-tight">
            Know your chances{' '}
            <span className="accent-serif">before</span> you apply.
          </h1>

          <p className="text-lg md:text-xl text-gray-400 max-w-2xl">
            Data from 50+ Canadian universities and ~4,900 real applications,
            distilled into a single calibrated probability.
          </p>

          <Link
            to="/predict"
            className="mt-4 inline-flex items-center gap-2 bg-emerald-400 hover:bg-emerald-300 text-black font-semibold px-8 py-3.5 rounded-full transition-colors"
          >
            Get Your Prediction
          </Link>

          <p className="text-sm text-gray-500">
            Free. No sign-up. No data stored.
          </p>
        </div>
      </section>

      {/* ── 2. Stats Strip ──────────────────────── */}
      <section className="py-24">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <StatCard value="4,900+" label="Applications Analyzed" />
            <StatCard value="50+" label="Universities Covered" />
            <StatCard
              value="Calibrated"
              label="Probabilities"
              description="When we say 70%, ~70% get in"
            />
          </div>
        </div>
      </section>

      {/* ── 3. How It Works ─────────────────────── */}
      <section className="py-24">
        <div className="container mx-auto px-6 max-w-6xl space-y-16">
          <SectionHeader
            label="How it works"
            title="Three steps to"
            accent="clarity."
          />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Step 1 */}
            <GlassCard>
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold">
                  01
                </div>
                <h3 className="text-lg font-semibold text-white">
                  Enter your grades
                </h3>
                <p className="text-sm text-gray-400">
                  Multi-format support: %, GPA, IB, letter grades
                </p>
              </div>
            </GlassCard>

            {/* Step 2 */}
            <GlassCard>
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold">
                  02
                </div>
                <h3 className="text-lg font-semibold text-white">
                  Pick your target
                </h3>
                <p className="text-sm text-gray-400">
                  50+ universities, 200+ programs with fuzzy search
                </p>
              </div>
            </GlassCard>

            {/* Step 3 */}
            <GlassCard>
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold">
                  03
                </div>
                <h3 className="text-lg font-semibold text-white">
                  Get your prediction
                </h3>
                <p className="text-sm text-gray-400">
                  Calibrated probability with confidence interval
                </p>
              </div>
            </GlassCard>
          </div>
        </div>
      </section>

      {/* ── 4. What You'll Get (Sample Prediction) */}
      <section className="py-24">
        <div className="container mx-auto px-6 max-w-6xl space-y-16">
          <SectionHeader
            label="See it in action"
            title="What you'll"
            accent="get."
          />

          <GlassCard padding="p-0">
            <div className="grid grid-cols-1 md:grid-cols-2">
              {/* Left: probability */}
              <div className="flex flex-col items-center justify-center gap-4 p-10 md:border-r border-white/10">
                <p className="text-7xl sm:text-8xl font-bold tracking-tight text-white">
                  73<span className="text-4xl sm:text-5xl text-gray-500">%</span>
                </p>
                <span className="inline-block px-4 py-1.5 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-semibold uppercase tracking-wide">
                  Likely Admit
                </span>
              </div>

              {/* Right: details */}
              <div className="flex flex-col justify-center gap-5 p-10">
                <DetailBullet>
                  Your <strong className="text-white">92%</strong> is{' '}
                  <strong className="text-white">35th percentile</strong> for
                  UofT CS
                </DetailBullet>
                <DetailBullet>
                  EC Tier:{' '}
                  <strong className="text-white">Tier 2 (High)</strong> — Score
                  14/20
                </DetailBullet>
                <DetailBullet>
                  Top factor:{' '}
                  <strong className="text-white">
                    Top 6 Average (+0.8 contribution)
                  </strong>
                </DetailBullet>
                <DetailBullet>
                  ~<strong className="text-white">12 weeks</strong> until
                  decision
                </DetailBullet>
              </div>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* ── 5. Competitive Programs Table ────────── */}
      <section className="py-24">
        <div className="container mx-auto px-6 max-w-6xl space-y-16">
          <SectionHeader
            label="The landscape"
            title="Know the"
            accent="competition."
          />

          <GlassCard padding="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                    <th className="px-6 py-4 font-medium">Program</th>
                    <th className="px-6 py-4 font-medium">Avg Required</th>
                    <th className="px-6 py-4 font-medium">Acceptance Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {competitivePrograms.map((row) => (
                    <tr
                      key={row.program}
                      className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                    >
                      <td className="px-6 py-4 text-white font-medium">
                        {row.program}
                      </td>
                      <td className="px-6 py-4 text-gray-300">{row.avg}</td>
                      <td className="px-6 py-4 text-gray-300">{row.rate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* ── 6. Methodology Teaser ────────────────── */}
      <section className="py-24">
        <div className="container mx-auto px-6 max-w-6xl space-y-16">
          <SectionHeader
            label="Under the hood"
            title="Transparent by"
            accent="design."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left: prose */}
            <div className="space-y-5 text-gray-400 leading-relaxed">
              <p>
                Our model uses{' '}
                <strong className="text-white">logistic regression</strong> with
                IRLS estimation, trained on thousands of self-reported
                application outcomes from Canadian universities.
              </p>
              <p>
                Every predicted probability is{' '}
                <strong className="text-white">calibrated</strong>: when we say
                there is a 70% chance of admission, roughly 70 out of 100
                applicants with that score are admitted. We verify this with
                reliability diagrams and Brier scores.
              </p>
              <p>
                The entire pipeline -- from linear algebra to model evaluation
                -- is implemented{' '}
                <strong className="text-white">from scratch</strong> in Python,
                with no black-box ML libraries. Every line is auditable.
              </p>

              <Link
                to="/about"
                className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
              >
                Read more about our methodology
                <span aria-hidden="true">&rarr;</span>
              </Link>
            </div>

            {/* Right: reliability diagram placeholder */}
            <GlassCard className="flex items-center justify-center min-h-[300px]">
              <div className="flex flex-col items-center gap-3 text-gray-500">
                <svg
                  className="w-10 h-10 text-gray-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"
                  />
                </svg>
                <p className="text-sm font-medium">Reliability Diagram</p>
                <p className="text-xs">Interactive D3 chart coming soon</p>
              </div>
            </GlassCard>
          </div>
        </div>
      </section>

      {/* ── 7. Final CTA ─────────────────────────── */}
      <section className="relative py-32 overflow-hidden">
        {/* Emerald orb background */}
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-emerald-orb pointer-events-none"
        />

        <div className="relative z-10 container mx-auto px-6 max-w-6xl text-center flex flex-col items-center gap-6">
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
            Ready to check your chances?
          </h2>

          <Link
            to="/predict"
            className="mt-2 inline-flex items-center gap-2 bg-emerald-400 hover:bg-emerald-300 text-black font-semibold px-8 py-3.5 rounded-full transition-colors"
          >
            Get Your Prediction
          </Link>
        </div>
      </section>
    </div>
  )
}

/* ─────────────────────────────────────────────
   Small helper for the sample-prediction bullets
   ───────────────────────────────────────────── */
function DetailBullet({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 text-sm text-gray-400">
      <span className="mt-1.5 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-400" />
      <span>{children}</span>
    </div>
  )
}

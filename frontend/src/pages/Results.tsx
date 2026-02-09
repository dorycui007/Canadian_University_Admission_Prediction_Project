import { useState, useEffect, useMemo } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import type {
  ApplicationRequest,
  PredictionResponse,
  SimilarCase,
  ProgramAnalytics,
  DistributionData,
} from '@/types/api'
import { api } from '@/lib/api'
import GlassCard from '@/components/shared/GlassCard'
import ProbabilityGauge from '@/components/viz/ProbabilityGauge'
import FeatureImportanceChart from '@/components/viz/FeatureImportanceChart'
import SimilarCasesScatter from '@/components/viz/SimilarCasesScatter'
import GradeHistogram from '@/components/viz/GradeHistogram'
import YearTrendChart from '@/components/viz/YearTrendChart'
import OfferTimelineChart from '@/components/viz/OfferTimelineChart'
import TimelineGraphic from '@/components/viz/TimelineGraphic'
import { formatPercent, formatPercentile } from '@/lib/formatters'

// ---------- Location state from Predict page ----------

interface ResultsState {
  request: ApplicationRequest
  response: PredictionResponse
}

// ---------- Helpers ----------

function predictionBadge(prediction: string) {
  if (prediction.includes('LIKELY') && !prediction.includes('UNLIKELY')) {
    return { label: 'Likely Admit', classes: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' }
  }
  if (prediction.includes('UNLIKELY')) {
    return { label: 'Unlikely', classes: 'bg-red-500/20 text-red-400 border-red-500/30' }
  }
  return { label: 'Uncertain', classes: 'bg-amber-500/20 text-amber-400 border-amber-500/30' }
}

/**
 * Compute what percentile a grade falls at within a histogram distribution.
 * Walks the bins cumulatively and interpolates within the bin containing the grade.
 */
function computePercentile(
  grade: number,
  bins: number[],
  counts: number[],
): number {
  const total = counts.reduce((s, c) => s + c, 0)
  if (total === 0) return 50

  let cumulative = 0
  for (let i = 0; i < counts.length; i++) {
    const binLow = bins[i]
    const binHigh = bins[i + 1]

    if (grade < binLow) {
      // Below the entire distribution
      return Math.round((cumulative / total) * 100)
    }

    if (grade >= binLow && grade < binHigh) {
      // Interpolate within this bin
      const fraction = (grade - binLow) / (binHigh - binLow)
      cumulative += counts[i] * fraction
      return Math.round((cumulative / total) * 100)
    }

    cumulative += counts[i]
  }

  // Above the entire distribution
  return 100
}

/**
 * Classify the application as Safety / Target / Reach
 */
function classifyAdmission(
  probability: number,
  grade: number,
  analytics: ProgramAnalytics | null,
): { tier: 'safety' | 'target' | 'reach'; label: string; message: string; classes: string } {
  const accepted = analytics?.grade_statistics?.accepted
  const p75 = accepted?.p75 ?? 95
  const p25 = accepted?.p25 ?? 80

  if (probability >= 0.75 && grade >= p75) {
    return {
      tier: 'safety',
      label: 'Safety',
      message: 'Your profile is well above typical admits for this program.',
      classes: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
    }
  }
  if (probability < 0.40 || grade < p25) {
    return {
      tier: 'reach',
      label: 'Reach',
      message: 'This program is a stretch goal — consider pairing it with target and safety choices.',
      classes: 'bg-orange-500/15 border-orange-500/30 text-orange-400',
    }
  }
  return {
    tier: 'target',
    label: 'Target',
    message: "You're competitive for this program — it's a solid match for your profile.",
    classes: 'bg-amber-500/15 border-amber-500/30 text-amber-400',
  }
}

/**
 * Estimate admission probability from distribution data at a given grade.
 * Uses the acceptance rate in the bin surrounding the grade.
 */
function estimateProbabilityAtGrade(
  grade: number,
  distribution: DistributionData,
): number | null {
  const { bins, counts_accepted, counts_rejected } = distribution
  if (bins.length < 2) return null

  for (let i = 0; i < bins.length - 1; i++) {
    if (grade >= bins[i] && grade < bins[i + 1]) {
      const acc = counts_accepted[i] ?? 0
      const rej = counts_rejected[i] ?? 0
      const total = acc + rej
      return total > 0 ? acc / total : null
    }
  }

  // If grade is above all bins, use the last bin
  if (grade >= bins[bins.length - 1]) {
    const last = bins.length - 2
    const acc = counts_accepted[last] ?? 0
    const rej = counts_rejected[last] ?? 0
    const total = acc + rej
    return total > 0 ? acc / total : null
  }

  return null
}

// ---------- Analytics loading skeleton ----------

function AnalyticsSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-10 w-64 bg-white/5 rounded-xl" />
      <div className="h-64 bg-white/5 rounded-2xl" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="h-56 bg-white/5 rounded-2xl" />
        <div className="h-56 bg-white/5 rounded-2xl" />
      </div>
    </div>
  )
}

// ---------- Component ----------

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const state = location.state as ResultsState | null

  // Analytics state
  const [analytics, setAnalytics] = useState<ProgramAnalytics | null>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState(false)
  const [analyticsError, setAnalyticsError] = useState<string | null>(null)

  // What-if slider state
  const [whatIfGrade, setWhatIfGrade] = useState<number | null>(null)

  // Fetch program analytics when page loads
  const request = state?.request
  useEffect(() => {
    if (!request?.university || !request?.program) return
    setAnalyticsLoading(true)
    setAnalyticsError(null)
    api
      .getProgramAnalytics(request.university, request.program)
      .then(setAnalytics)
      .catch((e) => setAnalyticsError(e.message))
      .finally(() => setAnalyticsLoading(false))
  }, [request?.university, request?.program])

  // Initialize what-if slider to current grade
  useEffect(() => {
    if (request?.top_6_average && whatIfGrade === null) {
      setWhatIfGrade(request.top_6_average)
    }
  }, [request?.top_6_average, whatIfGrade])

  // No state -- user navigated directly to /results
  if (!state?.response) {
    return (
      <section className="min-h-[60vh] flex flex-col items-center justify-center px-6 text-center">
        <GlassCard className="max-w-md w-full space-y-4">
          <h2 className="text-xl font-semibold text-white">
            No prediction data found
          </h2>
          <p className="text-gray-400 text-sm">
            Please submit a prediction first.
          </p>
          <Link
            to="/predict"
            className="
              inline-block px-6 py-2.5 rounded-full
              bg-emerald-500 hover:bg-emerald-400
              text-white text-sm font-semibold
              transition-colors
            "
          >
            Go to Predict
          </Link>
        </GlassCard>
      </section>
    )
  }

  const { request: req, response: result } = state
  const badge = predictionBadge(result.prediction)

  const universityName = req.university || 'University of Toronto'
  const programName = req.program || 'Computer Science'
  const studentGrade = req.top_6_average ?? 92
  const ec = result.ec_assessment
  const studentECScore = ec?.score ?? 0

  // Similar cases from EC scoring (real data from backend)
  const similarCases: SimilarCase[] = result.similar_cases ?? []
  const hasSimilarCases = similarCases.length > 0

  // Summary counts for the similar cases section
  const acceptedCount = similarCases.filter((c) => c.outcome === 'accepted').length
  const waitlistedCount = similarCases.filter((c) => c.outcome === 'waitlisted').length
  const rejectedCount = similarCases.filter((c) => c.outcome === 'rejected').length
  const gradeMin = hasSimilarCases ? Math.min(...similarCases.map((c) => c.grade)) : 0
  const gradeMax = hasSimilarCases ? Math.max(...similarCases.map((c) => c.grade)) : 0

  // Compute real percentiles from analytics distribution data
  const dist = analytics?.distribution
  const hasDistribution = dist && dist.bins.length >= 2

  const overallPercentile = useMemo(() => {
    if (!dist || !hasDistribution) return null
    const allCounts = dist.counts_accepted.map(
      (a, i) => a + (dist.counts_rejected[i] ?? 0),
    )
    return computePercentile(studentGrade, dist.bins, allCounts)
  }, [dist, hasDistribution, studentGrade])

  const acceptedPercentile = useMemo(() => {
    if (!dist || !hasDistribution) return null
    return computePercentile(studentGrade, dist.bins, dist.counts_accepted)
  }, [dist, hasDistribution, studentGrade])

  // Admission classification
  const classification = classifyAdmission(result.probability, studentGrade, analytics)

  // Year trend data
  const yearTrendData = useMemo(() => {
    if (!analytics?.year_trend) return []
    return analytics.year_trend.map((t) => ({
      year: t.year,
      median_grade: t.median_grade,
    }))
  }, [analytics?.year_trend])

  // What-If estimated probability
  const whatIfProbability = useMemo(() => {
    if (!dist || !hasDistribution || whatIfGrade === null) return null
    return estimateProbabilityAtGrade(whatIfGrade, dist)
  }, [dist, hasDistribution, whatIfGrade])

  // Accepted stats shorthand
  const acceptedStats = analytics?.grade_statistics?.accepted
  const competitiveness = analytics?.competitiveness

  return (
    <section className="py-12 md:py-20 px-6">
      <div className="max-w-5xl mx-auto space-y-10">

        {/* ====== 1. Header row ====== */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-1">
              Prediction Results
            </p>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">
              {universityName}
            </h1>
            <p className="text-lg text-gray-400 mt-1">{programName}</p>
          </div>

          <Link
            to="/predict"
            className="
              self-start sm:self-auto
              inline-flex items-center gap-2
              px-5 py-2 rounded-full
              bg-white/5 border border-white/10
              text-sm text-gray-300 hover:text-white hover:border-white/20
              transition-colors
            "
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Prediction
          </Link>
        </div>

        {/* ====== 2. Primary stats row (3 cards) ====== */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* --- Card A: Probability --- */}
          <GlassCard className="flex flex-col items-center text-center space-y-4">
            <ProbabilityGauge
              probability={result.probability}
              confidenceInterval={result.confidence_interval}
            />

            <div className="space-y-2">
              <p className="text-4xl font-bold text-white">
                {Math.round(result.probability * 100)}%
              </p>
              <span
                className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${badge.classes}`}
              >
                {badge.label}
              </span>
              <p className="text-xs text-gray-500">
                CI: {Math.round(result.confidence_interval.lower * 100)}% &ndash;{' '}
                {Math.round(result.confidence_interval.upper * 100)}%
              </p>
            </div>
          </GlassCard>

          {/* --- Card B: Grade Percentile (real data) --- */}
          <GlassCard className="space-y-4">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
              Where You Stand
            </h3>

            {overallPercentile !== null ? (
              <>
                <p className="text-sm text-gray-400">
                  Your <span className="text-white font-medium">{studentGrade}%</span> is at the{' '}
                  <span className="text-white font-medium">{formatPercentile(overallPercentile)}</span> percentile
                  of all applicants to this program.
                </p>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/5 rounded-lg px-3 py-2 text-center">
                    <p className="text-lg font-bold text-white">
                      {overallPercentile}<span className="text-xs text-gray-500">th</span>
                    </p>
                    <p className="text-[11px] text-gray-500">vs. All</p>
                  </div>
                  <div className="bg-white/5 rounded-lg px-3 py-2 text-center">
                    <p className="text-lg font-bold text-white">
                      {acceptedPercentile ?? '—'}<span className="text-xs text-gray-500">{acceptedPercentile !== null ? 'th' : ''}</span>
                    </p>
                    <p className="text-[11px] text-gray-500">vs. Accepted</p>
                  </div>
                  <div className="bg-white/5 rounded-lg px-3 py-2 text-center">
                    <p className="text-lg font-bold text-white">
                      {acceptedStats ? `${acceptedStats.p25}–${acceptedStats.p75}` : '—'}
                    </p>
                    <p className="text-[11px] text-gray-500">Competitive Range</p>
                  </div>
                  <div className="bg-white/5 rounded-lg px-3 py-2 text-center">
                    <p className="text-lg font-bold text-white">
                      {acceptedStats ? String(acceptedStats.median) : '—'}
                    </p>
                    <p className="text-[11px] text-gray-500">Median Admitted</p>
                  </div>
                </div>
              </>
            ) : analyticsLoading ? (
              <div className="space-y-3 animate-pulse">
                <div className="h-4 w-48 bg-white/10 rounded" />
                <div className="grid grid-cols-2 gap-3">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-16 bg-white/5 rounded-lg" />
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                Grade percentile data is not available for this program.
              </p>
            )}
          </GlassCard>

          {/* --- Card C: EC Assessment --- */}
          <GlassCard className="space-y-4">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
              EC Assessment
            </h3>

            {ec ? (
              <>
                {/* Tier badge */}
                <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${
                  ec.tier <= 2
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                    : 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                }`}>
                  Tier {ec.tier} ({ec.tier_label})
                </span>

                {/* Score */}
                <div className="space-y-1.5">
                  <div className="flex items-baseline justify-between">
                    <span className="text-2xl font-bold text-white">
                      {Math.round(ec.score)}<span className="text-sm text-gray-500 font-normal">/20</span>
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all"
                      style={{ width: `${(ec.score / 20) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Category pills */}
                <div className="flex flex-wrap gap-2">
                  {ec.category_breakdown
                    .filter((cat) => cat.active)
                    .map((cat) => (
                      <span
                        key={cat.category}
                        className="px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
                      >
                        {cat.category.replace(/ & /g, '/').replace(/ \/ /g, '/')}
                      </span>
                    ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-gray-500">No extracurriculars provided.</p>
            )}
          </GlassCard>
        </div>

        {/* ====== 3. Reach / Target / Safety Classification ====== */}
        <GlassCard className={`flex items-center gap-4 border ${classification.classes}`}>
          <span className="text-3xl shrink-0">
            {classification.tier === 'safety' && (
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            )}
            {classification.tier === 'target' && (
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            )}
            {classification.tier === 'reach' && (
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            )}
          </span>
          <div>
            <p className="text-lg font-semibold">{classification.label}</p>
            <p className="text-sm opacity-80">{classification.message}</p>
          </div>
        </GlassCard>

        {/* ====== 4. Grade Distribution Histogram ====== */}
        {analyticsLoading && <AnalyticsSkeleton />}

        {!analyticsLoading && hasDistribution && dist && (
          <GlassCard className="space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Grade distribution</h2>
              <p className="text-sm text-gray-400 mt-1">
                Where your <span className="text-white font-medium">{studentGrade}%</span> falls
                among {dist.statistics.n} reported applicants. The dashed line marks your position.
              </p>
            </div>
            <GradeHistogram data={dist} studentGrade={studentGrade} />
          </GlassCard>
        )}

        {/* ====== 5. How You Compare — Narrative Card ====== */}
        {!analyticsLoading && analytics && acceptedStats && (
          <GlassCard className="space-y-3">
            <h2 className="text-lg font-semibold text-white">How you compare</h2>
            <ul className="space-y-2 text-sm text-gray-400">
              {acceptedPercentile !== null && (
                <li>
                  Your <span className="text-white font-medium">{studentGrade}%</span> average is
                  higher than <span className="text-white font-medium">{acceptedPercentile}%</span> of
                  admitted students to this program.
                </li>
              )}
              <li>
                The competitive range is{' '}
                <span className="text-white font-medium">{acceptedStats.p25}%&ndash;{acceptedStats.p75}%</span>{' '}
                (25th&ndash;75th percentile of admits).
              </li>
              {competitiveness && (
                <li>
                  This program is rated{' '}
                  <span className="text-white font-medium">{competitiveness.difficulty}</span>{' '}
                  with{' '}
                  <span className="text-white font-medium">{competitiveness.sample_size}</span>{' '}
                  reported applications
                  {competitiveness.confidence_level !== 'high' && (
                    <span className="text-gray-500"> (confidence: {competitiveness.confidence_level})</span>
                  )}
                  .
                </li>
              )}
              {studentGrade >= acceptedStats.median ? (
                <li className="text-emerald-400">
                  Your average is above the median of admitted students ({acceptedStats.median}%). You are in a strong position.
                </li>
              ) : (
                <li className="text-amber-400">
                  Your average is below the median of admitted students ({acceptedStats.median}%).
                  Strong extracurriculars and supplementary applications can strengthen your candidacy.
                </li>
              )}
            </ul>
          </GlassCard>
        )}

        {/* ====== 6. Year Trend ====== */}
        {!analyticsLoading && yearTrendData.length >= 2 && (
          <GlassCard className="space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Admission average trend</h2>
              <p className="text-sm text-gray-400 mt-1">
                Year-over-year median grade for this program. The horizontal line shows your grade.
                Ontario has experienced significant grade inflation in recent years.
              </p>
            </div>
            <YearTrendChart data={yearTrendData} studentGrade={studentGrade} />
          </GlassCard>
        )}

        {/* ====== 7. Offer Timeline ====== */}
        {!analyticsLoading && analytics?.offer_timeline?.by_month?.length ? (
          <GlassCard className="space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">When to expect your decision</h2>
              <p className="text-sm text-gray-400 mt-1">
                Based on {analytics.offer_timeline.total_with_dates} reported decision dates.
                {analytics.offer_timeline.median_month && (
                  <> Most decisions arrive around <span className="text-white font-medium">{analytics.offer_timeline.median_month}</span>.</>
                )}
              </p>
            </div>
            <OfferTimelineChart data={analytics.offer_timeline.by_month} />
          </GlassCard>
        ) : !analyticsLoading ? (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">Expected timeline</h2>
            <TimelineGraphic weeksUntilDecision={12} />
          </GlassCard>
        ) : null}

        {/* ====== 8. Feature Importance ====== */}
        <GlassCard className="space-y-4">
          <h2 className="text-lg font-semibold text-white">What mattered most</h2>
          <FeatureImportanceChart features={result.feature_importance} />
        </GlassCard>

        {/* ====== 9. What-If Scenario Slider ====== */}
        {hasDistribution && dist && whatIfGrade !== null && (
          <GlassCard className="space-y-5">
            <div>
              <h2 className="text-lg font-semibold text-white">What if your average changes?</h2>
              <p className="text-sm text-gray-400 mt-1">
                Drag the slider to see how a different top-6 average would affect your estimated chance.
              </p>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-500 tabular-nums w-12 text-right">
                  {Math.max(50, studentGrade - 10)}%
                </span>
                <input
                  type="range"
                  min={Math.max(50, studentGrade - 10)}
                  max={Math.min(100, studentGrade + 5)}
                  step={1}
                  value={whatIfGrade}
                  onChange={(e) => setWhatIfGrade(Number(e.target.value))}
                  className="flex-1 h-2 rounded-full appearance-none bg-white/10 accent-emerald-500 cursor-pointer"
                />
                <span className="text-sm text-gray-500 tabular-nums w-12">
                  {Math.min(100, studentGrade + 5)}%
                </span>
              </div>

              <div className="flex items-center justify-center gap-6">
                <div className="text-center">
                  <p className="text-3xl font-bold text-white tabular-nums">{whatIfGrade}%</p>
                  <p className="text-xs text-gray-500 mt-1">Hypothetical Average</p>
                </div>
                <svg className="w-6 h-6 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5" />
                </svg>
                <div className="text-center">
                  <p className={`text-3xl font-bold tabular-nums ${
                    whatIfProbability !== null && whatIfProbability >= 0.6
                      ? 'text-emerald-400'
                      : whatIfProbability !== null && whatIfProbability >= 0.4
                        ? 'text-amber-400'
                        : 'text-red-400'
                  }`}>
                    {whatIfProbability !== null
                      ? `${Math.round(whatIfProbability * 100)}%`
                      : '—'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Estimated Chance</p>
                </div>
              </div>

              {whatIfGrade !== studentGrade && (
                <p className="text-xs text-gray-500 text-center">
                  {whatIfGrade > studentGrade
                    ? `A ${whatIfGrade - studentGrade}% increase could improve your chances.`
                    : `A ${studentGrade - whatIfGrade}% decrease would reduce your chances.`}
                </p>
              )}
            </div>
          </GlassCard>
        )}

        {/* ====== 10. Similar Cases ====== */}
        {hasSimilarCases && (
          <GlassCard className="space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Similar applications</h2>
              <p className="text-sm text-gray-400 mt-1">
                {similarCases.length} students with {gradeMin}&ndash;{gradeMax}% average applied to{' '}
                {universityName.replace('University of ', 'U').replace('University', 'U')} {programName.replace('Computer Science', 'CS')}.{' '}
                {acceptedCount} were accepted, {waitlistedCount} waitlisted, {rejectedCount} rejected.
              </p>
            </div>
            <SimilarCasesScatter
              cases={similarCases}
              studentGrade={studentGrade}
              studentECScore={studentECScore}
            />
          </GlassCard>
        )}

        {/* ====== 11. Similar Programs ====== */}
        <GlassCard className="space-y-4">
          <h2 className="text-lg font-semibold text-white">Programs with similar patterns</h2>

          <div className="flex gap-4 overflow-x-auto pb-2 -mx-2 px-2">
            {result.similar_programs.map((sp) => (
              <button
                key={`${sp.university}-${sp.program}`}
                type="button"
                onClick={() =>
                  navigate('/predict', {
                    state: {
                      top_6_average: req.top_6_average,
                      university: sp.university,
                      program: sp.program,
                    },
                  })
                }
                className="
                  flex-shrink-0 w-56
                  bg-white/5 border border-white/10 rounded-xl p-4
                  text-left space-y-2
                  hover:border-emerald-500/30 hover:bg-white/[0.07]
                  transition-colors cursor-pointer
                "
              >
                <p className="text-sm font-medium text-white truncate">{sp.university}</p>
                <p className="text-xs text-gray-400">{sp.program}</p>
                <div className="flex items-center justify-between text-xs pt-1">
                  <span className="text-emerald-400 font-medium">
                    {formatPercent(sp.similarity)} match
                  </span>
                  {sp.historical_admit_rate !== null && (
                    <span className="text-gray-500">
                      {formatPercent(sp.historical_admit_rate)} admit
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </GlassCard>

        {/* ====== 12. Warnings ====== */}
        {result.warnings.length > 0 && (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3 space-y-1">
            {result.warnings.map((w, i) => (
              <p key={i} className="text-sm text-amber-400">{w}</p>
            ))}
          </div>
        )}

        {/* ====== Explore Link ====== */}
        {analytics && (
          <div className="text-center">
            <Link
              to={`/explore/programs/${encodeURIComponent(universityName)}/${encodeURIComponent(programName)}`}
              className="
                inline-flex items-center gap-2
                px-5 py-2 rounded-full
                bg-white/5 border border-white/10
                text-sm text-gray-300 hover:text-white hover:border-white/20
                transition-colors
              "
            >
              View full program analytics
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
          </div>
        )}

        {/* ====== Disclaimer ====== */}
        <p className="text-xs text-gray-600 text-center max-w-2xl mx-auto leading-relaxed">
          This prediction is generated by a statistical model trained on historical
          application data and is intended for informational purposes only. It does not
          guarantee admission outcomes. Actual decisions are made by university admissions
          offices and may consider factors not captured by this model.
        </p>
      </div>
    </section>
  )
}

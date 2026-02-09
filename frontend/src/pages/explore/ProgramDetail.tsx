import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import type { ProgramAnalytics, DifficultyLabel } from '@/types/api'
import { api } from '@/lib/api'
import GlassCard from '@/components/shared/GlassCard'
import StatCard from '@/components/shared/StatCard'
import GradeHistogram from '@/components/viz/GradeHistogram'
import YearTrendChart from '@/components/viz/YearTrendChart'
import DecisionPieChart from '@/components/viz/DecisionPieChart'
import AdmittedGradeChart from '@/components/viz/AdmittedGradeChart'
import OfferTimelineChart from '@/components/viz/OfferTimelineChart'

/* ─────────────────────────────────────────────
   Loading skeleton
   ───────────────────────────────────────────── */
function LoadingSkeleton() {
  return (
    <section className="py-12 md:py-20 px-6">
      <div className="max-w-5xl mx-auto space-y-8">
        <div className="space-y-3 animate-pulse">
          <div className="h-4 w-48 bg-white/10 rounded" />
          <div className="h-8 w-96 bg-white/10 rounded" />
          <div className="h-5 w-64 bg-white/10 rounded" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 bg-white/5 rounded-2xl animate-pulse" />
          ))}
        </div>
        <div className="h-72 bg-white/5 rounded-2xl animate-pulse" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="h-64 bg-white/5 rounded-2xl animate-pulse" />
          <div className="h-64 bg-white/5 rounded-2xl animate-pulse" />
        </div>
      </div>
    </section>
  )
}

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */

function difficultyColor(d: DifficultyLabel) {
  switch (d) {
    case 'Very Competitive': return 'bg-red-500/20 text-red-400'
    case 'Competitive':      return 'bg-amber-500/20 text-amber-400'
    case 'Moderate':         return 'bg-emerald-500/20 text-emerald-400'
    case 'Accessible':       return 'bg-sky-500/20 text-sky-400'
    default:                 return 'bg-white/10 text-gray-400'
  }
}

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function ProgramDetail() {
  const { university, program } = useParams<{
    university: string
    program: string
  }>()
  const [data, setData] = useState<ProgramAnalytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!university || !program) return
    setLoading(true)
    setError(null)
    api
      .getProgramAnalytics(
        decodeURIComponent(university),
        decodeURIComponent(program)
      )
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [university, program])

  if (loading) return <LoadingSkeleton />

  if (error) {
    return (
      <section className="min-h-[60vh] flex flex-col items-center justify-center px-6 text-center">
        <GlassCard className="max-w-md w-full space-y-4">
          <h2 className="text-xl font-semibold text-white">
            Program not found
          </h2>
          <p className="text-gray-400 text-sm">{error}</p>
          <Link
            to="/explore/programs"
            className="
              inline-block px-6 py-2.5 rounded-full
              bg-emerald-500 hover:bg-emerald-400
              text-white text-sm font-semibold
              transition-colors
            "
          >
            Back to Programs
          </Link>
        </GlassCard>
      </section>
    )
  }

  if (!data) return null

  const decodedUni = decodeURIComponent(university || '')
  const decodedProg = decodeURIComponent(program || '')
  const isSparse = data.total_records < 3
  const acceptedStats = data.grade_statistics.accepted
  const allStats = data.grade_statistics.all
  const rejectedStats = data.grade_statistics.rejected
  const { difficulty, admitted_range: ar, sample_size, confidence_level } = data.competitiveness

  // Build year_trend in format YearTrendChart expects
  const yearTrendData = data.year_trend.map((t) => ({
    year: t.year,
    median_grade: t.median_grade,
  }))

  return (
    <section className="py-12 md:py-20 px-6">
      <div className="max-w-5xl mx-auto space-y-10">
        {/* ====== Breadcrumb ====== */}
        <nav className="flex items-center gap-2 text-sm text-gray-500">
          <Link to="/explore/programs" className="hover:text-gray-300 transition-colors">
            Explore
          </Link>
          <span>/</span>
          <Link to="/explore/programs" className="hover:text-gray-300 transition-colors">
            Programs
          </Link>
          <span>/</span>
          <span className="text-gray-400">{decodedUni}</span>
        </nav>

        {/* ====== 1. Header ====== */}
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">
              {decodedProg}
            </h1>
            <p className="text-lg text-gray-400 mt-1">{decodedUni}</p>
            <p className="text-sm text-gray-500 mt-2">
              {data.cycle_years.length} application cycle{data.cycle_years.length !== 1 ? 's' : ''}
              {' '}&middot;{' '}
              {data.total_records} total record{data.total_records !== 1 ? 's' : ''}
              {' '}&middot;{' '}
              {data.cycle_years[0]}&#8211;{data.cycle_years[data.cycle_years.length - 1]?.split('-')[1]}
            </p>
          </div>

          <Link
            to="/predict"
            state={{ university: decodedUni, program: decodedProg }}
            className="
              self-start
              inline-flex items-center gap-2
              bg-emerald-400 hover:bg-emerald-300
              text-black font-semibold text-sm
              px-6 py-2.5 rounded-lg transition-colors shrink-0
            "
          >
            Get Prediction
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>

        {/* ====== Sparse Data Warning ====== */}
        {data.data_quality.sparse_warning && (
          <GlassCard className="border-amber-500/30">
            <div className="flex items-center gap-3">
              <span className="text-amber-400 text-xl shrink-0">!</span>
              <div>
                <p className="text-sm font-medium text-amber-400">
                  Limited Data Available
                </p>
                <p className="text-xs text-gray-400">
                  {data.data_quality.sparse_warning}
                </p>
              </div>
            </div>
          </GlassCard>
        )}

        {/* ====== 2. Overview Stats ====== */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Competitive Range — the #1 thing students want to know */}
          <GlassCard>
            <div className="flex items-center gap-2">
              <p className="text-3xl font-bold text-white tabular-nums">
                {ar ? `${ar.p25}–${ar.p75}` : '—'}
              </p>
            </div>
            <p className="mt-1 text-sm text-gray-400">Competitive Range</p>
            <p className="mt-1 text-xs text-gray-500">
              Grades where most admitted students fall
            </p>
          </GlassCard>

          {/* Difficulty badge */}
          <GlassCard>
            <span
              className={`
                inline-block px-2 py-1 rounded text-sm font-semibold
                ${difficultyColor(difficulty)}
              `}
            >
              {difficulty}
            </span>
            <p className="mt-2 text-sm text-gray-400">Difficulty</p>
            <p className="mt-1 text-xs text-gray-500">
              Based on median admitted grade
            </p>
          </GlassCard>

          {/* Minimum admitted grade */}
          <StatCard
            value={ar ? String(ar.min) : '—'}
            label="Lowest Admitted Grade"
            description="Minimum reported"
          />

          {/* Sample size with confidence */}
          <GlassCard>
            <div className="flex items-center gap-2">
              <p className="text-3xl font-bold text-white tabular-nums">
                {sample_size}
              </p>
              <span
                className={`
                  px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide
                  ${confidence_level === 'high'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : confidence_level === 'moderate'
                      ? 'bg-amber-500/20 text-amber-400'
                      : 'bg-red-500/20 text-red-400'
                  }
                `}
              >
                {confidence_level}
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-400">Total Reports</p>
            <p className="mt-1 text-xs text-gray-500">
              Self-reported results
            </p>
          </GlassCard>
        </div>

        {/* ====== 3. Grade Distribution ====== */}
        {!isSparse && data.distribution.bins.length >= 2 && (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Grade Distribution
            </h2>
            <p className="text-sm text-gray-400">
              Distribution of Top 6 averages for accepted vs rejected applicants.
            </p>
            <GradeHistogram data={data.distribution} />
          </GlassCard>
        )}

        {/* ====== 4 & 5. Admitted Grade Range by Year + Decision Breakdown ====== */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Admitted Grade Range by Year */}
          {data.competitiveness.by_year.length > 0 && (
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Admitted Grades by Cycle
              </h2>
              <p className="text-xs text-gray-500">
                Median grade with 25th–75th percentile range for admitted students.
              </p>
              <AdmittedGradeChart data={data.competitiveness.by_year} />
            </GlassCard>
          )}

          {/* Decision Breakdown */}
          {Object.keys(data.decision_breakdown).length > 0 && (
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Decision Breakdown
              </h2>
              <DecisionPieChart data={data.decision_breakdown} />
            </GlassCard>
          )}
        </div>

        {/* ====== 5b. Offer Timeline — "When will I hear back?" ====== */}
        {data.offer_timeline.by_month.length > 0 && (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">
              When Do Offers Come Out?
            </h2>
            <p className="text-xs text-gray-500">
              Based on {data.offer_timeline.total_with_dates} reported decision dates.
              {data.offer_timeline.median_month && (
                <> Most decisions arrive around <span className="text-white font-medium">{data.offer_timeline.median_month}</span>.</>
              )}
            </p>
            <OfferTimelineChart data={data.offer_timeline.by_month} />
          </GlassCard>
        )}

        {/* ====== 5c. Grade Progression + Applicant Pool ====== */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Grade Progression */}
          {data.grade_progression && (
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Grade Trajectory (Accepted Students)
              </h2>
              <p className="text-xs text-gray-500">
                How admitted students' averages progressed from Grade 11 to Grade 12.
                Only available for 2023-2024 cycle.
              </p>
              <div className="flex items-end justify-around gap-2 pt-2 pb-4">
                {data.grade_progression.g11_avg != null && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white tabular-nums">
                      {data.grade_progression.g11_avg}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">G11 Final</p>
                    <p className="text-[10px] text-gray-600">n={data.grade_progression.g11_n}</p>
                  </div>
                )}
                {data.grade_progression.g11_avg != null && data.grade_progression.g12_midterm_avg != null && (
                  <svg className="w-6 h-6 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5" />
                  </svg>
                )}
                {data.grade_progression.g12_midterm_avg != null && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-emerald-400 tabular-nums">
                      {data.grade_progression.g12_midterm_avg}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">G12 Midterm</p>
                    <p className="text-[10px] text-gray-600">n={data.grade_progression.g12_midterm_n}</p>
                  </div>
                )}
                {data.grade_progression.g12_midterm_avg != null && data.grade_progression.g12_final_avg != null && (
                  <svg className="w-6 h-6 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5" />
                  </svg>
                )}
                {data.grade_progression.g12_final_avg != null && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-emerald-400 tabular-nums">
                      {data.grade_progression.g12_final_avg}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">G12 Final</p>
                    <p className="text-[10px] text-gray-600">n={data.grade_progression.g12_final_n}</p>
                  </div>
                )}
              </div>
            </GlassCard>
          )}

          {/* Applicant Pool (101 vs 105) */}
          {(data.applicant_type['101'] > 0 || data.applicant_type['105'] > 0) && (
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Applicant Pool
              </h2>
              <p className="text-xs text-gray-500">
                101 = Ontario high school applicants. 105 = out-of-province / international.
              </p>
              <div className="space-y-3 pt-2">
                {(['101', '105'] as const).map((type) => {
                  const count = data.applicant_type[type]
                  const total = data.applicant_type['101'] + data.applicant_type['105'] + data.applicant_type.unknown
                  const pct = total > 0 ? (count / total) * 100 : 0
                  return (
                    <div key={type} className="flex items-center gap-3">
                      <span className="text-sm text-gray-300 w-10 shrink-0 font-medium">{type}</span>
                      <div className="flex-1 h-3 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${type === '101' ? 'bg-emerald-500/60' : 'bg-sky-500/60'}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 tabular-nums w-20 text-right shrink-0">
                        {count} ({Math.round(pct)}%)
                      </span>
                    </div>
                  )
                })}
                {data.applicant_type.unknown > 0 && (
                  <p className="text-[10px] text-gray-600">
                    {data.applicant_type.unknown} with unknown applicant type
                  </p>
                )}
              </div>
            </GlassCard>
          )}
        </div>

        {/* ====== 6. Year Trend ====== */}
        {!isSparse && yearTrendData.length >= 2 && (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Median Grade Trend
            </h2>
            <p className="text-sm text-gray-400">
              Year-over-year median grade of all applicants.
            </p>
            <YearTrendChart data={yearTrendData} />
          </GlassCard>
        )}

        {/* ====== 7. Grade Stats Table ====== */}
        {allStats && (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Detailed Grade Statistics
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left text-gray-400 font-medium py-2 pr-4">
                      Metric
                    </th>
                    <th className="text-right text-gray-400 font-medium py-2 px-4">
                      All
                    </th>
                    {acceptedStats && (
                      <th className="text-right text-emerald-400/80 font-medium py-2 px-4">
                        Accepted
                      </th>
                    )}
                    {rejectedStats && (
                      <th className="text-right text-red-400/80 font-medium py-2 px-4">
                        Rejected
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ['Mean', 'mean'],
                      ['Median', 'median'],
                      ['P25', 'p25'],
                      ['P75', 'p75'],
                      ['Min', 'min'],
                      ['Max', 'max'],
                      ['Count', 'n'],
                    ] as const
                  ).map(([label, key]) => (
                    <tr
                      key={key}
                      className="border-b border-white/5"
                    >
                      <td className="py-2 pr-4 text-gray-400">
                        {label}
                      </td>
                      <td className="py-2 px-4 text-right text-white tabular-nums">
                        {allStats[key]}
                      </td>
                      {acceptedStats && (
                        <td className="py-2 px-4 text-right text-emerald-400 tabular-nums">
                          {acceptedStats[key]}
                        </td>
                      )}
                      {rejectedStats && (
                        <td className="py-2 px-4 text-right text-red-400 tabular-nums">
                          {rejectedStats[key]}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        )}

        {/* ====== 8. Province Breakdown ====== */}
        {Object.keys(data.province_breakdown).length > 0 && (
          <GlassCard className="space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Applicants by Province
            </h2>
            <div className="space-y-3">
              {Object.entries(data.province_breakdown)
                .sort(([, a], [, b]) => b - a)
                .map(([province, count]) => {
                  const pct = (count / data.total_records) * 100
                  return (
                    <div key={province} className="flex items-center gap-3">
                      <span className="text-sm text-gray-300 w-28 shrink-0 truncate">
                        {province}
                      </span>
                      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500/60 rounded-full transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 tabular-nums w-16 text-right shrink-0">
                        {count} ({Math.round(pct)}%)
                      </span>
                    </div>
                  )
                })}
            </div>
          </GlassCard>
        )}

        {/* ====== CTA + Disclaimer ====== */}
        <div className="text-center space-y-4 pt-4">
          <p className="text-lg font-semibold text-white">
            Ready to check your chances?
          </p>
          <Link
            to="/predict"
            state={{ university: decodedUni, program: decodedProg }}
            className="
              inline-flex items-center gap-2
              bg-emerald-400 hover:bg-emerald-300
              text-black font-semibold text-sm
              px-8 py-3 rounded-lg transition-colors
            "
          >
            Get Your Prediction
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>

        <p className="text-xs text-gray-600 text-center max-w-2xl mx-auto leading-relaxed">
          This data is from self-reported application results collected from
          Reddit and Discord communities. It is intended for informational
          purposes only and may not represent the full applicant pool. Actual
          admission rates and requirements are determined by university
          admissions offices.
        </p>
      </div>
    </section>
  )
}

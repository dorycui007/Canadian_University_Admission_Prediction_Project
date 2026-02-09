import { useState } from 'react'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import StatCard from '@/components/shared/StatCard'
import GradeHistogram from '@/components/viz/GradeHistogram'
import YearTrendChart from '@/components/viz/YearTrendChart'
import UniversityCombobox from '@/components/form/UniversityCombobox'
import ProgramCombobox from '@/components/form/ProgramCombobox'
import GradeInput from '@/components/form/GradeInput'
import type { GradeFormat, DistributionData } from '@/types/api'

/* ─────────────────────────────────────────────
   Mock distribution data
   ───────────────────────────────────────────── */
const mockDistribution: DistributionData = {
  bins: [70, 75, 80, 85, 90, 95, 100],
  counts_accepted: [2, 5, 15, 30, 45, 35, 10],
  counts_rejected: [8, 12, 18, 15, 8, 3, 1],
  statistics: { mean: 89.5, median: 91.0, p25: 85, p75: 95, min: 72, max: 100, n: 207 },
  year_trend: [
    { year: 2022, median_grade: 88 },
    { year: 2023, median_grade: 90 },
    { year: 2024, median_grade: 91 },
    { year: 2025, median_grade: 92 },
  ],
}

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function Distributions() {
  const [university, setUniversity] = useState('')
  const [program, setProgram] = useState('')
  const [grade, setGrade] = useState<number | undefined>(undefined)
  const [gradeFormat, setGradeFormat] = useState<GradeFormat>('percentage')
  const [showData, setShowData] = useState(false)

  function handleShow() {
    setShowData(true)
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <SectionHeader label="Explore" title="Grade Distributions" />
        </div>
      </section>

      {/* Selection panel */}
      <section className="pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <GlassCard>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <UniversityCombobox value={university} onChange={setUniversity} />
              <ProgramCombobox value={program} onChange={setProgram} />
              <GradeInput
                label="Your Grade (optional)"
                value={grade}
                onChange={setGrade}
                format={gradeFormat}
                onFormatChange={setGradeFormat}
              />
            </div>
            <div className="mt-6">
              <button
                onClick={handleShow}
                className="
                  inline-flex items-center gap-2
                  bg-emerald-400 hover:bg-emerald-300
                  text-black font-semibold text-sm
                  px-6 py-2.5 rounded-lg transition-colors
                "
              >
                Show
              </button>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* Distribution results */}
      {showData && (
        <section className="pb-24">
          <div className="container mx-auto px-6 max-w-6xl space-y-8">
            {/* Grade histogram */}
            <GlassCard>
              <h3 className="text-lg font-semibold text-white mb-4">
                Grade Distribution
              </h3>
              <GradeHistogram data={mockDistribution} studentGrade={grade} />
            </GlassCard>

            {/* Statistics panel */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Statistics
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <StatCard
                  value={mockDistribution.statistics.mean.toFixed(1)}
                  label="Mean"
                />
                <StatCard
                  value={mockDistribution.statistics.median.toFixed(1)}
                  label="Median"
                />
                <StatCard
                  value={String(mockDistribution.statistics.p25)}
                  label="P25"
                />
                <StatCard
                  value={String(mockDistribution.statistics.p75)}
                  label="P75"
                />
                <StatCard
                  value={String(mockDistribution.statistics.min)}
                  label="Min"
                />
                <StatCard
                  value={String(mockDistribution.statistics.max)}
                  label="Max"
                />
              </div>
              <p className="mt-3 text-sm text-gray-500">
                Sample size: {mockDistribution.statistics.n} applications
              </p>
            </div>

            {/* Year trend chart */}
            {mockDistribution.year_trend && (
              <GlassCard>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Year-over-Year Trend
                </h3>
                <YearTrendChart
                  data={mockDistribution.year_trend}
                  studentGrade={grade}
                />
                <p className="mt-4 text-sm text-gray-400">
                  Grade requirements are rising. The median has increased 4% since
                  2022.
                </p>
              </GlassCard>
            )}
          </div>
        </section>
      )}
    </div>
  )
}

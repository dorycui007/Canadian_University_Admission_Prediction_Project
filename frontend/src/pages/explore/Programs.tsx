import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { ProgramListingEntry, DifficultyLabel } from '@/types/api'
import { api } from '@/lib/api'
import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'

/* ─────────────────────────────────────────────
   Types
   ───────────────────────────────────────────── */

type DifficultyFilter = 'All' | 'Very Competitive' | 'Competitive' | 'Moderate' | 'Accessible'

const DIFFICULTY_OPTIONS: DifficultyFilter[] = [
  'All',
  'Very Competitive',
  'Competitive',
  'Moderate',
  'Accessible',
]

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */

function difficultyColor(d: DifficultyLabel): string {
  switch (d) {
    case 'Very Competitive': return 'bg-red-400'
    case 'Competitive':      return 'bg-amber-400'
    case 'Moderate':         return 'bg-emerald-400'
    case 'Accessible':       return 'bg-sky-400'
    default:                 return 'bg-gray-400'
  }
}

function difficultyBadgeClass(d: DifficultyLabel): string {
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
export default function Programs() {
  const [programs, setPrograms] = useState<ProgramListingEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [difficulty, setDifficulty] = useState<DifficultyFilter>('All')

  useEffect(() => {
    api
      .getProgramListing()
      .then(setPrograms)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim()
    return programs.filter((p) => {
      if (q && !p.program.toLowerCase().includes(q) && !p.university.toLowerCase().includes(q)) {
        return false
      }
      if (difficulty !== 'All' && p.difficulty !== difficulty) {
        return false
      }
      return true
    })
  }, [programs, search, difficulty])

  return (
    <div className="min-h-screen">
      {/* Header */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <SectionHeader label="Explore" title="Program Explorer" />
          <p className="mt-2 text-sm text-gray-500">
            {programs.length} programs from self-reported admission data.
          </p>
        </div>
      </section>

      {/* Filter bar — sticky below nav */}
      <div className="sticky top-16 z-30 bg-[#050505]/80 backdrop-blur-md border-b border-white/5">
        <div className="container mx-auto px-6 max-w-6xl py-4">
          <GlassCard padding="p-4">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              {/* Text search */}
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search programs or universities..."
                className="
                  flex-1 min-w-0 px-3 py-2 rounded-lg text-sm text-white
                  bg-white/5 border border-white/10
                  outline-none transition-colors
                  focus:border-emerald-400
                  placeholder-gray-500
                "
              />

              {/* Difficulty pill toggles */}
              <div className="flex gap-1 shrink-0 flex-wrap">
                {DIFFICULTY_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setDifficulty(opt)}
                    className={`
                      px-3 py-2 rounded-full text-xs font-medium transition-colors
                      ${
                        difficulty === opt
                          ? 'bg-emerald-400/20 text-emerald-400 border border-emerald-400/40'
                          : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
                      }
                    `}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          </GlassCard>
        </div>
      </div>

      {/* Program grid */}
      <section className="py-8">
        <div className="container mx-auto px-6 max-w-6xl">
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="h-36 bg-white/5 rounded-2xl animate-pulse" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-16 text-center">
              <p className="text-gray-500">No programs match your filters.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filtered.map((p, idx) => (
                <Link
                  key={`${p.university}-${p.program}-${idx}`}
                  to={`/explore/programs/${encodeURIComponent(p.university)}/${encodeURIComponent(p.program)}`}
                  className="block group"
                >
                  <GlassCard className="transition-colors group-hover:border-emerald-500/30">
                    <div className="flex flex-col gap-3">
                      {/* Program name */}
                      <h3 className="font-semibold text-white">{p.program}</h3>

                      {/* University */}
                      <p className="text-sm text-gray-400">{p.university}</p>

                      {/* Difficulty + median grade */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <span
                          className={`
                            inline-block px-2 py-0.5 rounded text-xs font-semibold
                            ${difficultyBadgeClass(p.difficulty)}
                          `}
                        >
                          {p.difficulty}
                        </span>
                        {p.median_grade_accepted != null && (
                          <span className="text-xs text-gray-400">
                            Median admitted: <span className="text-white font-medium">{p.median_grade_accepted}%</span>
                          </span>
                        )}
                      </div>

                      {/* Reports + confidence */}
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span
                          className={`inline-block h-1.5 w-1.5 rounded-full ${difficultyColor(p.difficulty)}`}
                        />
                        {p.total_records} report{p.total_records !== 1 ? 's' : ''}
                        <span className="text-gray-600">&middot;</span>
                        <span className={
                          p.confidence_level === 'high' ? 'text-emerald-500' :
                          p.confidence_level === 'moderate' ? 'text-amber-500' :
                          'text-red-500'
                        }>
                          {p.confidence_level} confidence
                        </span>
                      </div>

                      {/* View Details link */}
                      <span className="
                        mt-1 inline-flex items-center gap-1
                        text-emerald-400 text-sm font-medium
                      ">
                        View Details
                        <svg
                          className="w-4 h-4 transition-transform group-hover:translate-x-1"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                      </span>
                    </div>
                  </GlassCard>
                </Link>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

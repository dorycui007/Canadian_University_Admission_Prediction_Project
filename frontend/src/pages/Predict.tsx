import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Activity, GradeFormat } from '@/types/api'
import { toPercentage } from '@/lib/gradeConverter'
import { api } from '@/lib/api'
import GlassCard from '@/components/shared/GlassCard'
import GradeInput from '@/components/form/GradeInput'
import UniversityCombobox from '@/components/form/UniversityCombobox'
import ProgramCombobox from '@/components/form/ProgramCombobox'
import ECSection from '@/components/form/ECSection'

const PROVINCES = [
  'Ontario',
  'British Columbia',
  'Alberta',
  'Quebec',
  'Manitoba',
  'Saskatchewan',
  'Nova Scotia',
  'New Brunswick',
  'Newfoundland and Labrador',
  'Prince Edward Island',
  'Northwest Territories',
  'Nunavut',
  'Yukon',
]

const PROVINCE_CODES: Record<string, string> = {
  'Ontario': 'ON',
  'British Columbia': 'BC',
  'Alberta': 'AB',
  'Quebec': 'QC',
  'Manitoba': 'MB',
  'Saskatchewan': 'SK',
  'Nova Scotia': 'NS',
  'New Brunswick': 'NB',
  'Newfoundland and Labrador': 'NL',
  'Prince Edward Island': 'PE',
  'Northwest Territories': 'NT',
  'Nunavut': 'NU',
  'Yukon': 'YT',
}

export default function Predict() {
  const navigate = useNavigate()

  // Grade state
  const [top6, setTop6] = useState<number | undefined>(undefined)
  const [gr11, setGr11] = useState<number | undefined>(undefined)
  const [gradeFormat, setGradeFormat] = useState<GradeFormat>('percentage')

  // Program state
  const [university, setUniversity] = useState('')
  const [program, setProgram] = useState('')

  // Province
  const [province, setProvince] = useState('')

  // Extracurriculars
  const [ecOpen, setEcOpen] = useState(false)
  const [activities, setActivities] = useState<Activity[]>([])

  // Validation & API state
  const [submitted, setSubmitted] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)

  const missingTop6 = submitted && top6 === undefined
  const missingUni = submitted && university.trim() === ''
  const missingProg = submitted && program.trim() === ''

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitted(true)
    setApiError(null)

    if (top6 === undefined || university.trim() === '' || program.trim() === '') {
      return
    }

    const top6Pct = toPercentage(top6, gradeFormat)
    const gr11Pct = gr11 !== undefined ? toPercentage(gr11, gradeFormat) : undefined

    const request = {
      top_6_average: top6Pct,
      university,
      program,
      grade_11_average: gr11Pct,
      province: province ? PROVINCE_CODES[province] : undefined,
      activities: activities.length > 0 ? activities : undefined,
    }

    setIsLoading(true)
    try {
      const response = await api.predict(request)
      navigate('/results', {
        state: { request, response },
      })
    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Prediction failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen selection:bg-emerald-500/30">
      <div className="relative z-10">
        <section className="pt-4 pb-16 md:pt-6 md:pb-24 px-6">
          <div className="max-w-2xl mx-auto space-y-8">
            {/* Page header */}
            <div className="relative z-10 text-center mb-8">
              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tighter text-white drop-shadow-2xl">
                Your prediction.
              </h1>

              {/* Subtext */}
              <p className="mt-4 text-gray-400 max-w-xl mx-auto text-lg">
                Enter your academic profile below to see where you stand.
              </p>
            </div>

            {/* Form */}
            <GlassCard padding="p-6 md:p-8">
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* ---- Academic Profile ---- */}
                <fieldset className="space-y-5">
                  <legend className="text-lg font-semibold text-white mb-1">
                    Academic Profile
                  </legend>

                  <GradeInput
                    label="Top 6 Average"
                    value={top6}
                    onChange={setTop6}
                    format={gradeFormat}
                    onFormatChange={setGradeFormat}
                    required
                  />
                  {missingTop6 && (
                    <p className="text-xs text-red-400 -mt-3">
                      Top 6 average is required.
                    </p>
                  )}

                  <GradeInput
                    label="Grade 11 Average"
                    value={gr11}
                    onChange={setGr11}
                    format={gradeFormat}
                    onFormatChange={setGradeFormat}
                  />

                </fieldset>

                {/* ---- Target Program ---- */}
                <fieldset className="space-y-5">
                  <legend className="text-lg font-semibold text-white mb-1">
                    Target Program
                  </legend>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <UniversityCombobox value={university} onChange={setUniversity} />
                      {missingUni && (
                        <p className="text-xs text-red-400 mt-1">
                          University is required.
                        </p>
                      )}
                    </div>
                    <div>
                      <ProgramCombobox value={program} onChange={setProgram} />
                      {missingProg && (
                        <p className="text-xs text-red-400 mt-1">
                          Program is required.
                        </p>
                      )}
                    </div>
                  </div>
                </fieldset>

                {/* ---- Province ---- */}
                <div className="space-y-1.5">
                  <label className="block text-sm font-medium text-gray-300">
                    Province / Territory
                  </label>
                  <select
                    value={province}
                    onChange={(e) => setProvince(e.target.value)}
                    className="
                      w-full px-3 py-2.5 rounded-lg text-sm text-white
                      bg-white/5 border border-white/10
                      outline-none transition-colors
                      focus:border-emerald-400
                      cursor-pointer
                    "
                  >
                    <option value="" className="bg-[#111] text-white">
                      Select province (optional)
                    </option>
                    {PROVINCES.map((p) => (
                      <option key={p} value={p} className="bg-[#111] text-white">
                        {p}
                      </option>
                    ))}
                  </select>
                </div>

                {/* ---- Extracurriculars (collapsible) ---- */}
                <div className="space-y-3">
                  <button
                    type="button"
                    onClick={() => setEcOpen(!ecOpen)}
                    className="
                      flex items-center gap-2 text-sm font-medium text-gray-300
                      hover:text-white transition-colors
                    "
                  >
                    <svg
                      className={`w-4 h-4 transition-transform ${ecOpen ? 'rotate-90' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    Extracurriculars
                    <span className="text-xs text-gray-500">(optional)</span>
                  </button>

                  {ecOpen && <ECSection activities={activities} onChange={setActivities} />}
                </div>

                {/* ---- Error message ---- */}
                {apiError && (
                  <p className="text-sm text-red-400 text-center bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
                    {apiError}
                  </p>
                )}

                {/* ---- Submit ---- */}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="
                    w-full sm:w-auto sm:mx-auto sm:block
                    px-8 py-3 rounded-full
                    bg-emerald-500 hover:bg-emerald-400
                    disabled:bg-emerald-500/50 disabled:cursor-not-allowed
                    text-white font-semibold text-sm
                    transition-colors cursor-pointer
                  "
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Predicting...
                    </span>
                  ) : (
                    'Predict My Chances'
                  )}
                </button>
              </form>
            </GlassCard>
          </div>
        </section>
      </div>
    </div>
  )
}

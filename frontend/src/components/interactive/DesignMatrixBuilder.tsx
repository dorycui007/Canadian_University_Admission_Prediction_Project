import { useState, useMemo } from 'react'
import GlassCard from '@/components/shared/GlassCard'

/* ─────────────────────────────────────────────
   Constants
   ───────────────────────────────────────────── */
const RAW_APPLICATION = {
  university: 'uoft',
  program: 'comp sci',
  grade: 92.5,
  province: 'Ontario',
  term: 'Fall 2025',
}

const MEAN = 87.3
const STD = 4.2
const Z_SCORE = (RAW_APPLICATION.grade - MEAN) / STD

const STEP_LABELS = [
  'Raw Input',
  'Numeric Scaling',
  'Categorical Encoding',
  'Interaction Features',
  'Final Row',
]

const FEATURE_TYPES = {
  numeric: { label: 'Numeric', color: 'bg-emerald-400/20 text-emerald-400 border-emerald-400/30' },
  oneHot: { label: 'One-Hot', color: 'bg-blue-400/20 text-blue-400 border-blue-400/30' },
  dummy: { label: 'Dummy', color: 'bg-pink-400/20 text-pink-400 border-pink-400/30' },
  interaction: { label: 'Interaction', color: 'bg-amber-400/20 text-amber-400 border-amber-400/30' },
} as const

type FeatureType = keyof typeof FEATURE_TYPES

interface FeatureCell {
  label: string
  value: number
  type: FeatureType
}

const FINAL_FEATURES: FeatureCell[] = [
  { label: 'intercept', value: 1, type: 'numeric' },
  { label: 'z_grade', value: Z_SCORE, type: 'numeric' },
  { label: 'is_uoft', value: 1, type: 'oneHot' },
  { label: 'is_waterloo', value: 0, type: 'oneHot' },
  { label: 'is_mcgill', value: 0, type: 'oneHot' },
  { label: 'is_ubc', value: 0, type: 'oneHot' },
  { label: 'is_ontario', value: 1, type: 'dummy' },
  { label: 'is_bc', value: 0, type: 'dummy' },
  { label: 'is_alberta', value: 0, type: 'dummy' },
  { label: 'comp_rank', value: 8, type: 'numeric' },
  { label: 'z_grade\u00d7comp', value: Z_SCORE * 1, type: 'interaction' },
]

const BETA = [-0.50, 0.08, 0.12, -0.02, 0.04, 0.03, 0.10, 0.05, 0.04, 0.015, 0.06]

/* ─────────────────────────────────────────────
   Sub-components
   ───────────────────────────────────────────── */
function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center justify-center gap-3">
      {Array.from({ length: total }, (_, i) => (
        <button
          key={i}
          className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
            i === current
              ? 'bg-emerald-400 text-[#050505] scale-110 shadow-lg shadow-emerald-400/30'
              : i < current
                ? 'bg-emerald-400/20 text-emerald-400 border border-emerald-400/40'
                : 'bg-white/5 text-gray-600 border border-white/10'
          }`}
          aria-label={`Step ${i + 1}: ${STEP_LABELS[i]}`}
          tabIndex={-1}
        >
          {i + 1}
        </button>
      ))}
    </div>
  )
}

function FeatureCellBox({ feature }: { feature: FeatureCell }) {
  const style = FEATURE_TYPES[feature.type]
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className={`px-3 py-2 rounded-lg border font-mono text-sm font-semibold ${style.color}`}
      >
        {Number.isInteger(feature.value)
          ? feature.value.toString()
          : feature.value.toFixed(3)}
      </div>
      <span className="text-[10px] text-gray-500 font-mono truncate max-w-[5rem] text-center">
        {feature.label}
      </span>
    </div>
  )
}

function Legend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs">
      {(Object.entries(FEATURE_TYPES) as [FeatureType, (typeof FEATURE_TYPES)[FeatureType]][]).map(
        ([key, { label, color }]) => (
          <div key={key} className="flex items-center gap-1.5">
            <div className={`w-3 h-3 rounded-sm border ${color}`} />
            <span className="text-gray-400">{label}</span>
          </div>
        ),
      )}
    </div>
  )
}

function NumberLine({ value, min, max }: { value: number; min: number; max: number }) {
  const pct = ((value - min) / (max - min)) * 100
  const ticks = []
  for (let t = min; t <= max; t++) {
    ticks.push(t)
  }

  return (
    <div className="relative w-full h-12 mt-2">
      {/* Axis line */}
      <div className="absolute top-4 left-0 right-0 h-px bg-white/20" />

      {/* Tick marks */}
      {ticks.map((t) => {
        const tickPct = ((t - min) / (max - min)) * 100
        return (
          <div
            key={t}
            className="absolute top-2"
            style={{ left: `${tickPct}%`, transform: 'translateX(-50%)' }}
          >
            <div
              className={`w-px h-4 ${t === 0 ? 'bg-white/40' : 'bg-white/10'}`}
            />
            <span className="text-[9px] text-gray-600 font-mono mt-0.5 block text-center">
              {t}
            </span>
          </div>
        )
      })}

      {/* Marker */}
      <div
        className="absolute top-0 flex flex-col items-center"
        style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
      >
        <span className="text-[10px] text-emerald-400 font-mono font-bold -mt-1">
          {value.toFixed(3)}
        </span>
        <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 mt-0.5 shadow-lg shadow-emerald-400/40" />
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   Step renderers
   ───────────────────────────────────────────── */
function StepRawInput() {
  const fields: { key: string; value: string }[] = [
    { key: 'university', value: `"${RAW_APPLICATION.university}"` },
    { key: 'program', value: `"${RAW_APPLICATION.program}"` },
    { key: 'grade', value: RAW_APPLICATION.grade.toString() },
    { key: 'province', value: `"${RAW_APPLICATION.province}"` },
    { key: 'term', value: `"${RAW_APPLICATION.term}"` },
  ]

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-400">
        A raw application arrives as unstructured key-value data. This is the starting point
        before any feature engineering.
      </p>
      <div className="bg-black/30 rounded-xl p-5 space-y-2.5 font-mono text-sm">
        <span className="text-gray-600">{'{'}</span>
        {fields.map((f) => (
          <div key={f.key} className="pl-4 flex gap-2">
            <span className="text-emerald-400">{f.key}</span>
            <span className="text-gray-600">:</span>
            <span className="text-white">{f.value}</span>
          </div>
        ))}
        <span className="text-gray-600">{'}'}</span>
      </div>
    </div>
  )
}

function StepNumericScaling() {
  const raw = RAW_APPLICATION.grade
  const z = Z_SCORE

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-400">
        The <span className="text-white">NumericScaler</span> standardizes continuous features to
        z-scores, centering them around zero with unit variance.
      </p>

      {/* Formula */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <div className="text-sm text-gray-400">
          Formula: <span className="font-mono text-white">z = (x - {'\u03BC'}) / {'\u03C3'}</span>
        </div>
        <div className="font-mono text-sm space-y-1.5">
          <div className="text-gray-400">
            <span className="text-emerald-400">grade</span>
            {' = '}
            <span className="text-white">{raw}</span>
          </div>
          <div className="text-gray-400">
            {'\u03BC'} = <span className="text-white">{MEAN}</span>
            {' , '}
            {'\u03C3'} = <span className="text-white">{STD}</span>
          </div>
        </div>

        <div className="border-t border-white/10 pt-3">
          <div className="font-mono text-sm">
            <span className="text-gray-400">z = </span>
            <span className="text-white">({raw} - {MEAN}) / {STD}</span>
            <span className="text-gray-400"> = </span>
            <span className="text-white">{(raw - MEAN).toFixed(1)} / {STD}</span>
            <span className="text-gray-400"> = </span>
            <span className="text-emerald-400 font-bold">{z.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Transformation arrow */}
      <div className="flex items-center gap-4 justify-center text-sm">
        <div className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 font-mono">
          <span className="text-gray-400">raw: </span>
          <span className="text-white">{raw}</span>
        </div>
        <span className="text-emerald-400 text-lg">{'\u2192'}</span>
        <div className="px-3 py-1.5 rounded-lg bg-emerald-400/10 border border-emerald-400/30 font-mono">
          <span className="text-gray-400">z: </span>
          <span className="text-emerald-400 font-bold">{z.toFixed(3)}</span>
        </div>
      </div>

      {/* Number line */}
      <div className="px-4">
        <p className="text-xs text-gray-500 mb-1">z-score on standard normal scale</p>
        <NumberLine value={z} min={-3} max={3} />
      </div>
    </div>
  )
}

function StepCategoricalEncoding() {
  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-400">
        Categorical features are converted to numeric representations using different encoding
        strategies depending on the variable type.
      </p>

      {/* One-hot: university */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2">
          <div className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-blue-400/20 text-blue-400 border border-blue-400/30">
            One-Hot
          </div>
          <span className="text-sm text-white font-medium">university</span>
        </div>
        <div className="font-mono text-sm space-y-1">
          <div className="text-gray-400">
            <span className="text-emerald-400">"uoft"</span>
            <span className="text-gray-600 mx-2">{'\u2192'}</span>
            <span className="text-gray-400">one-hot vector:</span>
          </div>
          <div className="flex items-center gap-1 mt-2 flex-wrap">
            {['UofT', 'Waterloo', 'McGill', 'UBC'].map((uni, i) => (
              <div key={uni} className="flex flex-col items-center gap-1">
                <div
                  className={`w-10 h-8 rounded border flex items-center justify-center font-mono text-sm font-semibold ${
                    i === 0
                      ? 'bg-blue-400/20 text-blue-400 border-blue-400/30'
                      : 'bg-white/5 text-gray-600 border-white/10'
                  }`}
                >
                  {i === 0 ? '1' : '0'}
                </div>
                <span className="text-[9px] text-gray-500">{uni}</span>
              </div>
            ))}
            <span className="text-gray-600 font-mono text-xs ml-1">...</span>
          </div>
        </div>
      </div>

      {/* Dummy: province */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2">
          <div className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-pink-400/20 text-pink-400 border border-pink-400/30">
            Dummy
          </div>
          <span className="text-sm text-white font-medium">province</span>
        </div>
        <div className="font-mono text-sm space-y-1">
          <div className="text-gray-400">
            <span className="text-emerald-400">"Ontario"</span>
            <span className="text-gray-600 mx-2">{'\u2192'}</span>
            <span className="text-gray-400">dummy variables:</span>
          </div>
          <div className="flex items-center gap-1 mt-2">
            {[
              { label: 'is_ontario', val: 1 },
              { label: 'is_bc', val: 0 },
              { label: 'is_alberta', val: 0 },
            ].map((d) => (
              <div key={d.label} className="flex flex-col items-center gap-1">
                <div
                  className={`w-14 h-8 rounded border flex items-center justify-center font-mono text-sm font-semibold ${
                    d.val === 1
                      ? 'bg-pink-400/20 text-pink-400 border-pink-400/30'
                      : 'bg-white/5 text-gray-600 border-white/10'
                  }`}
                >
                  {d.val}
                </div>
                <span className="text-[9px] text-gray-500">{d.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Ordinal: program */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2">
          <div className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-400/20 text-emerald-400 border border-emerald-400/30">
            Ordinal
          </div>
          <span className="text-sm text-white font-medium">program</span>
        </div>
        <div className="font-mono text-sm space-y-2">
          <div className="text-gray-400">
            <span className="text-emerald-400">"comp sci"</span>
            <span className="text-gray-600 mx-2">{'\u2192'}</span>
            <span className="text-gray-400">competitive_rank = </span>
            <span className="text-emerald-400 font-bold">8</span>
            <span className="text-gray-500"> (out of 10)</span>
          </div>
          {/* Mini bar */}
          <div className="flex items-center gap-1 mt-1">
            {Array.from({ length: 10 }, (_, i) => (
              <div
                key={i}
                className={`h-4 flex-1 rounded-sm ${
                  i < 8 ? 'bg-emerald-400/40' : 'bg-white/5'
                }`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[9px] text-gray-600">
            <span>1 (low)</span>
            <span>10 (high)</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function StepInteractionFeatures() {
  const interactionVal = Z_SCORE * 1

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-400">
        Interaction features capture non-linear relationships by multiplying existing features
        together. A high grade in a competitive program matters more than in a non-competitive one.
      </p>

      {/* Interaction computation */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <div className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-400/20 text-amber-400 border border-amber-400/30">
            Interaction
          </div>
        </div>
        <div className="font-mono text-sm space-y-2">
          <div className="text-gray-400">
            interaction_1 = z_grade {'\u00d7'} is_competitive
          </div>
          <div className="text-gray-400">
            interaction_1 ={' '}
            <span className="text-emerald-400">{Z_SCORE.toFixed(3)}</span>
            <span className="text-gray-500"> {'\u00d7'} </span>
            <span className="text-pink-400">1</span>
            <span className="text-gray-500"> = </span>
            <span className="text-amber-400 font-bold">{interactionVal.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Final feature vector */}
      <div className="space-y-3">
        <p className="text-xs text-gray-500">Assembled feature vector:</p>
        <div className="flex flex-wrap gap-2 justify-center">
          {FINAL_FEATURES.map((f) => (
            <FeatureCellBox key={f.label} feature={f} />
          ))}
        </div>
      </div>

      <Legend />
    </div>
  )
}

function StepFinalRow() {
  const dotProduct = useMemo(() => {
    return FINAL_FEATURES.reduce((sum, f, i) => sum + f.value * BETA[i], 0)
  }, [])

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-400">
        The complete design matrix row is a{' '}
        <span className="text-white font-mono">1{'\u00d7'}{FINAL_FEATURES.length}</span>{' '}
        vector ready for matrix multiplication with the coefficient vector{' '}
        <span className="text-white font-mono">{'\u03B2'}</span>.
      </p>

      {/* Full row with color coding */}
      <div className="bg-black/30 rounded-xl p-5 overflow-x-auto">
        <p className="text-xs text-gray-500 mb-3 font-mono">
          x = design matrix row (1{'\u00d7'}{FINAL_FEATURES.length})
        </p>
        <div className="flex gap-1.5 min-w-max justify-center">
          {FINAL_FEATURES.map((f) => (
            <FeatureCellBox key={f.label} feature={f} />
          ))}
        </div>
      </div>

      <Legend />

      {/* Dot product computation */}
      <div className="bg-black/30 rounded-xl p-5 space-y-3">
        <p className="text-xs text-gray-500 font-mono mb-2">
          z = x{'\u1D40'}{'\u03B2'} = sum of element-wise products:
        </p>
        <div className="space-y-1 font-mono text-xs">
          {FINAL_FEATURES.map((f, i) => {
            const product = f.value * BETA[i]
            const style = FEATURE_TYPES[f.type]
            return (
              <div key={f.label} className="flex items-center gap-2">
                <span className={`w-20 text-right truncate ${style.color.split(' ')[1]}`}>
                  {f.label}
                </span>
                <span className="text-gray-600">:</span>
                <span className="text-gray-400 w-16 text-right">
                  {Number.isInteger(f.value) ? f.value.toString() : f.value.toFixed(3)}
                </span>
                <span className="text-gray-600">{'\u00d7'}</span>
                <span className="text-gray-400 w-14 text-right">
                  {BETA[i].toFixed(3)}
                </span>
                <span className="text-gray-600">=</span>
                <span className="text-white w-16 text-right">
                  {product.toFixed(4)}
                </span>
              </div>
            )
          })}
        </div>
        <div className="border-t border-white/10 pt-3 mt-2">
          <div className="font-mono text-sm">
            <span className="text-gray-400">z = x</span>
            <span className="text-gray-400">{'\u1D40'}{'\u03B2'}</span>
            <span className="text-gray-400"> = </span>
            <span className="text-emerald-400 font-bold">{dotProduct.toFixed(4)}</span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            This linear predictor z is then passed through {'\u03C3'}(z) to produce a probability.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   Main Component
   ───────────────────────────────────────────── */
export default function DesignMatrixBuilder() {
  const [step, setStep] = useState(0)

  const totalSteps = STEP_LABELS.length

  const handlePrev = () => setStep((s) => Math.max(0, s - 1))
  const handleNext = () => setStep((s) => Math.min(totalSteps - 1, s + 1))

  const renderStep = () => {
    switch (step) {
      case 0:
        return <StepRawInput />
      case 1:
        return <StepNumericScaling />
      case 2:
        return <StepCategoricalEncoding />
      case 3:
        return <StepInteractionFeatures />
      case 4:
        return <StepFinalRow />
      default:
        return null
    }
  }

  return (
    <div className="bg-[#050505] rounded-2xl p-6 space-y-6">
      {/* Step indicator */}
      <div className="space-y-2">
        <StepIndicator current={step} total={totalSteps} />
        <p className="text-center text-sm text-white font-medium">
          {STEP_LABELS[step]}
        </p>
      </div>

      {/* Step content with opacity transition */}
      <GlassCard>
        <div
          key={step}
          className="animate-[fadeIn_300ms_ease-in-out]"
          style={{
            animation: 'fadeIn 300ms ease-in-out',
          }}
        >
          {renderStep()}
        </div>
      </GlassCard>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={handlePrev}
          disabled={step <= 0}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {'\u2190'} Previous
        </button>

        <span className="text-xs text-gray-500 font-mono tabular-nums">
          {step + 1} / {totalSteps}
        </span>

        <button
          onClick={handleNext}
          disabled={step >= totalSteps - 1}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Next {'\u2192'}
        </button>
      </div>

      {/* Global keyframes style */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}

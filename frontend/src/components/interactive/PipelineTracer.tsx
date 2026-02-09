import { useState, useEffect, useRef, useMemo } from 'react'
import GlassCard from '@/components/shared/GlassCard'
import useReducedMotion from '@/hooks/useReducedMotion'

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

interface PipelineStep {
  num: number
  name: string
  getDetail: (grade: number) => string
}

const STEPS: PipelineStep[] = [
  {
    num: 1,
    name: 'Normalization',
    getDetail: (g) =>
      `Input: "${g.toFixed(1)}%, Ontario, UofT CS"\nFuzzy match → "University of Toronto" (97%)\nProgram → "Computer Science" (exact)`,
  },
  {
    num: 2,
    name: 'Feature Engineering',
    getDetail: (g) => {
      const z = ((g - 87.3) / 4.2).toFixed(2)
      return `GPA z-score: (${g.toFixed(1)} - 87.3) / 4.2 = ${z}\nis_ontario = 1, is_bc = 0, is_alberta = 0\ninteraction = ${z} × 0.8 = ${(parseFloat(z) * 0.8).toFixed(3)}`
    },
  },
  {
    num: 3,
    name: 'Logistic Regression',
    getDetail: (g) => {
      const z = (g - 87.3) / 4.2
      const linear =
        -0.5 * 1 + 0.08 * z + 0.02 * ((g - 85) / 5) + 0.03 * (z * 0.8) + 0.1 * 1
      return `z = β₀ + β₁x₁ + ... + β₈x₈\nz = ${linear.toFixed(4)}\nσ(z) = 1/(1+e^(-z)) = ${sigmoid(linear).toFixed(4)}`
    },
  },
  {
    num: 4,
    name: 'IRLS Training',
    getDetail: () =>
      `8 iterations until convergence\nFinal loss: 0.230\nRidge λ = 0.01 (regularization)\nQR decomposition for numerical stability`,
  },
  {
    num: 5,
    name: 'Platt Calibration',
    getDetail: (g) => {
      const z = (g - 87.3) / 4.2
      const linear =
        -0.5 * 1 + 0.08 * z + 0.02 * ((g - 85) / 5) + 0.03 * (z * 0.8) + 0.1 * 1
      const raw = sigmoid(linear)
      const cal = raw // Platt with A=1, B=0 is identity
      return `p_raw = ${raw.toFixed(4)}\nPlatt scaling: A=1.0, B=0.0\np_calibrated = σ(1.0 × logit(${raw.toFixed(3)}) + 0.0) = ${cal.toFixed(4)}`
    },
  },
  {
    num: 6,
    name: 'Prediction Output',
    getDetail: (g) => {
      const z = (g - 87.3) / 4.2
      const linear =
        -0.5 * 1 + 0.08 * z + 0.02 * ((g - 85) / 5) + 0.03 * (z * 0.8) + 0.1 * 1
      const prob = sigmoid(linear)
      const label =
        prob >= 0.7 ? 'LIKELY_ADMIT' : prob >= 0.4 ? 'UNCERTAIN' : 'UNLIKELY_ADMIT'
      return `Probability: ${(prob * 100).toFixed(1)}%\nPrediction: ${label}\nCI: [${((prob - 0.08) * 100).toFixed(1)}%, ${((prob + 0.08) * 100).toFixed(1)}%]`
    },
  },
]

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
interface PipelineTracerProps {
  grade: number
}

export default function PipelineTracer({ grade }: PipelineTracerProps) {
  const reducedMotion = useReducedMotion()
  const [expanded, setExpanded] = useState<Record<number, boolean>>({})
  const [pulseStep, setPulseStep] = useState(-1)
  const intervalRef = useRef<ReturnType<typeof setInterval>>(null)

  // Pulse animation: emerald dot travels down the pipeline
  useEffect(() => {
    if (reducedMotion) return

    intervalRef.current = setInterval(() => {
      setPulseStep((s) => {
        if (s >= STEPS.length) return -1
        return s + 1
      })
    }, 600)

    return () => clearInterval(intervalRef.current ?? undefined)
  }, [reducedMotion])

  const toggle = (num: number) => {
    setExpanded((prev) => ({ ...prev, [num]: !prev[num] }))
  }

  const details = useMemo(
    () => STEPS.map((s) => s.getDetail(grade)),
    [grade],
  )

  return (
    <div className="relative">
      {/* Connecting line */}
      <div className="absolute left-6 top-0 bottom-0 w-px bg-white/10" />

      <div className="space-y-4 relative">
        {STEPS.map((step, i) => {
          const isActive = pulseStep === i

          return (
            <div key={step.num} className="relative pl-14">
              {/* Step indicator dot */}
              <div
                className={`absolute left-4 top-5 w-5 h-5 rounded-full border-2 flex items-center justify-center text-[10px] font-bold transition-all duration-300 ${
                  isActive
                    ? 'border-emerald-400 bg-emerald-400/20 text-emerald-400 scale-125'
                    : 'border-white/20 bg-[#050505] text-gray-500'
                }`}
              >
                {step.num}
              </div>

              {/* Card */}
              <GlassCard padding="p-4">
                <button
                  onClick={() => toggle(step.num)}
                  className="w-full text-left flex items-center justify-between"
                >
                  <div>
                    <h4 className="text-sm font-bold text-white">
                      {step.name}
                    </h4>
                    {!expanded[step.num] && (
                      <p className="text-xs text-gray-500 mt-0.5">
                        Click to see intermediate values
                      </p>
                    )}
                  </div>
                  <span className="text-gray-500 text-lg leading-none shrink-0 ml-4">
                    {expanded[step.num] ? '\u2212' : '+'}
                  </span>
                </button>

                {expanded[step.num] && (
                  <pre className="mt-3 text-xs text-gray-300 font-mono whitespace-pre-wrap leading-relaxed bg-white/[0.03] rounded-lg p-3">
                    {details[i]}
                  </pre>
                )}
              </GlassCard>
            </div>
          )
        })}
      </div>
    </div>
  )
}

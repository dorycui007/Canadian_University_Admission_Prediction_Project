import { useState, useMemo } from 'react'
import GlassCard from '@/components/shared/GlassCard'

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function logit(p: number): number {
  const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p))
  return Math.log(clamped / (1 - clamped))
}

/* ─────────────────────────────────────────────
   Types
   ───────────────────────────────────────────── */
interface WorkedExampleTracerProps {
  grade?: number
}

interface StepData {
  num: number
  title: string
  preview: (vals: ComputedValues) => string
  renderExpanded: (vals: ComputedValues) => React.ReactNode
}

interface ComputedValues {
  grade: number
  z_score: number
  interaction: number
  featureVector: number[]
  beta: number[]
  terms: number[]
  z_linear: number
  p_raw: number
  A: number
  B: number
  p_cal: number
  se: number
  ci_lower: number
  ci_upper: number
  label: string
}

/* ─────────────────────────────────────────────
   Compute all pipeline values from grade
   ───────────────────────────────────────────── */
function computeValues(grade: number): ComputedValues {
  const z_score = (grade - 87.3) / 4.2
  const interaction = z_score * 0.8
  const featureVector = [1, z_score, interaction, 1, 0, 0, 0]

  const beta = [-0.50, 0.08, 0.02, 0.03, 0.10, 0.05, 0.05]
  const terms = featureVector.map((x, i) => x * beta[i])
  const z_linear = terms.reduce((sum, t) => sum + t, 0)

  const p_raw = sigmoid(z_linear)

  const A = 1.0
  const B = 0.0
  const p_cal = sigmoid(A * logit(p_raw) + B)

  const n = 150
  const se = Math.sqrt((p_cal * (1 - p_cal)) / n)
  const ci_lower = Math.max(0, p_cal - 1.96 * se)
  const ci_upper = Math.min(1, p_cal + 1.96 * se)

  const label = p_cal >= 0.7 ? 'LIKELY' : p_cal >= 0.4 ? 'UNCERTAIN' : 'UNLIKELY'

  return {
    grade,
    z_score,
    interaction,
    featureVector,
    beta,
    terms,
    z_linear,
    p_raw,
    A,
    B,
    p_cal,
    se,
    ci_lower,
    ci_upper,
    label,
  }
}

/* ─────────────────────────────────────────────
   Inline sub-components
   ───────────────────────────────────────────── */
function Label({ children }: { children: React.ReactNode }) {
  return <span className="text-gray-500 text-xs">{children}</span>
}

function Val({ children }: { children: React.ReactNode }) {
  return <span className="text-emerald-400 font-mono">{children}</span>
}

function Computation({ children }: { children: React.ReactNode }) {
  return <div className="text-gray-300 font-mono text-sm">{children}</div>
}

function JsonBlock({ children }: { children: string }) {
  return (
    <pre className="bg-black/30 rounded-lg p-4 font-mono text-sm text-gray-300 whitespace-pre-wrap overflow-x-auto">
      {children}
    </pre>
  )
}

function Arrow() {
  return <span className="text-gray-600 mx-2">{'\u2192'}</span>
}

/* ─────────────────────────────────────────────
   Step definitions
   ───────────────────────────────────────────── */
const STEPS: StepData[] = [
  {
    num: 1,
    title: 'Normalize Input',
    preview: (v) => `"uoft" \u2192 "University of Toronto", grade: ${v.grade}`,
    renderExpanded: (v) => (
      <div className="space-y-3">
        <div>
          <Label>Input:</Label>
          <JsonBlock>
            {`{ university: "uoft", program: "comp sci", grade: ${v.grade} }`}
          </JsonBlock>
        </div>
        <div className="flex items-center">
          <Label>Fuzzy match</Label>
          <Arrow />
        </div>
        <div>
          <Label>Output:</Label>
          <JsonBlock>
            {`{ university: "University of Toronto", program: "Computer Science", grade: ${v.grade} }`}
          </JsonBlock>
        </div>
        <Computation>
          Fuzzy match score: <Val>96.2%</Val>
        </Computation>
      </div>
    ),
  },
  {
    num: 2,
    title: 'Feature Engineering',
    preview: (v) => `z_score = ${v.z_score.toFixed(3)}, feature vector [1, ${v.z_score.toFixed(2)}, ...]`,
    renderExpanded: (v) => (
      <div className="space-y-3">
        <Computation>
          z_score = (grade - mean) / std = ({v.grade} - 87.3) / 4.2 ={' '}
          <Val>{v.z_score.toFixed(4)}</Val>
        </Computation>
        <Computation>
          is_ontario = <Val>1</Val>, is_competitive = <Val>1</Val>
        </Computation>
        <Computation>
          interaction = z_score {'\u00d7'} 0.8 = {v.z_score.toFixed(4)} {'\u00d7'} 0.8 ={' '}
          <Val>{v.interaction.toFixed(4)}</Val>
        </Computation>
        <div className="mt-2">
          <Label>Output feature vector:</Label>
          <div className="mt-1">
            <Val>[{v.featureVector.map((x) => x.toFixed(3)).join(', ')}]</Val>
          </div>
        </div>
      </div>
    ),
  },
  {
    num: 3,
    title: 'Design Matrix Row',
    preview: (v) => `1\u00d77 row: [1, ${v.z_score.toFixed(3)}, ${v.interaction.toFixed(3)}, 1, 0, 0, 0]`,
    renderExpanded: (v) => {
      const labels = ['intercept', 'top6_z', 'interaction', 'ontario', 'bc', 'alberta', 'quebec']
      return (
        <div className="space-y-3">
          <Computation>Design matrix row x (1{'\u00d7'}7):</Computation>
          <div className="bg-black/30 rounded-lg p-4 overflow-x-auto">
            <table className="text-sm font-mono">
              <thead>
                <tr>
                  {labels.map((l) => (
                    <th key={l} className="px-3 py-1 text-gray-500 font-normal text-xs text-center">
                      {l}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {v.featureVector.map((val, i) => (
                    <td key={i} className="px-3 py-1 text-center text-emerald-400">
                      {val.toFixed(3)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <div className="space-y-1">
            <Computation>
              intercept = <Val>1</Val>
            </Computation>
            <Computation>
              top6_z = <Val>{v.z_score.toFixed(3)}</Val>
            </Computation>
            <Computation>
              interaction = <Val>{v.interaction.toFixed(3)}</Val>
            </Computation>
            <Computation>
              ontario = <Val>1</Val>, bc = <Val>0</Val>, alberta = <Val>0</Val>, quebec = <Val>0</Val>
            </Computation>
          </div>
        </div>
      )
    },
  },
  {
    num: 4,
    title: 'Linear Prediction',
    preview: (v) => `z = x\u1d40\u03b2 = ${v.z_linear.toFixed(4)}`,
    renderExpanded: (v) => {
      const labels = ['intercept', 'top6_z', 'interact', 'ontario', 'bc', 'alberta', 'quebec']
      return (
        <div className="space-y-3">
          <Computation>
            {'\u03b2'} = [{v.beta.map((b) => b.toFixed(2)).join(', ')}]
          </Computation>
          <div className="mt-2">
            <Computation>z = x{'\u1d40'}{'\u03b2'} = sum of element-wise products:</Computation>
          </div>
          <div className="bg-black/30 rounded-lg p-4 space-y-1 overflow-x-auto">
            {v.featureVector.map((x, i) => (
              <Computation key={i}>
                {labels[i]}: {x.toFixed(3)} {'\u00d7'} {v.beta[i].toFixed(2)} ={' '}
                <Val>{v.terms[i].toFixed(5)}</Val>
              </Computation>
            ))}
            <div className="border-t border-white/10 mt-2 pt-2">
              <Computation>
                z = {v.terms.map((t) => t.toFixed(5)).join(' + ')} = <Val>{v.z_linear.toFixed(5)}</Val>
              </Computation>
            </div>
          </div>
        </div>
      )
    },
  },
  {
    num: 5,
    title: 'Sigmoid Transform',
    preview: (v) => `\u03c3(${v.z_linear.toFixed(4)}) = ${v.p_raw.toFixed(4)}`,
    renderExpanded: (v) => (
      <div className="space-y-3">
        <Computation>
          p_raw = {'\u03c3'}(z) = 1 / (1 + exp(-z))
        </Computation>
        <div className="bg-black/30 rounded-lg p-4 space-y-1">
          <Computation>
            z = <Val>{v.z_linear.toFixed(5)}</Val>
          </Computation>
          <Computation>
            exp(-z) = exp({(-v.z_linear).toFixed(5)}) = <Val>{Math.exp(-v.z_linear).toFixed(5)}</Val>
          </Computation>
          <Computation>
            1 + exp(-z) = <Val>{(1 + Math.exp(-v.z_linear)).toFixed(5)}</Val>
          </Computation>
          <Computation>
            p_raw = 1 / {(1 + Math.exp(-v.z_linear)).toFixed(5)} = <Val>{v.p_raw.toFixed(5)}</Val>
          </Computation>
        </div>
        <Computation>
          p_raw = <Val>{(v.p_raw * 100).toFixed(2)}%</Val>
        </Computation>
      </div>
    ),
  },
  {
    num: 6,
    title: 'Platt Scaling',
    preview: (v) => `p_cal = ${v.p_cal.toFixed(4)} (A=1.0, B=0.0)`,
    renderExpanded: (v) => (
      <div className="space-y-3">
        <Computation>
          p_cal = {'\u03c3'}(A {'\u00b7'} logit(p_raw) + B)
        </Computation>
        <div className="bg-black/30 rounded-lg p-4 space-y-1">
          <Computation>
            A = <Val>{v.A.toFixed(1)}</Val>, B = <Val>{v.B.toFixed(1)}</Val>
          </Computation>
          <Computation>
            logit(p_raw) = logit({v.p_raw.toFixed(5)}) = <Val>{logit(v.p_raw).toFixed(5)}</Val>
          </Computation>
          <Computation>
            A {'\u00b7'} logit(p_raw) + B = {v.A.toFixed(1)} {'\u00d7'} {logit(v.p_raw).toFixed(5)} + {v.B.toFixed(1)} ={' '}
            <Val>{(v.A * logit(v.p_raw) + v.B).toFixed(5)}</Val>
          </Computation>
          <Computation>
            p_cal = {'\u03c3'}({(v.A * logit(v.p_raw) + v.B).toFixed(5)}) = <Val>{v.p_cal.toFixed(5)}</Val>
          </Computation>
        </div>
        <p className="text-xs text-gray-500 italic">
          With fitted calibrator, A and B would adjust the probability. Currently identity (A=1.0, B=0.0).
        </p>
      </div>
    ),
  },
  {
    num: 7,
    title: 'Confidence Interval',
    preview: (v) => `CI = [${v.ci_lower.toFixed(4)}, ${v.ci_upper.toFixed(4)}]`,
    renderExpanded: (v) => (
      <div className="space-y-3">
        <Computation>
          CI = [p_cal - 1.96 {'\u00b7'} se, p_cal + 1.96 {'\u00b7'} se]
        </Computation>
        <div className="bg-black/30 rounded-lg p-4 space-y-1">
          <Computation>
            n = <Val>150</Val> (sample size)
          </Computation>
          <Computation>
            se = sqrt(p {'\u00b7'} (1-p) / n)
          </Computation>
          <Computation>
            se = sqrt({v.p_cal.toFixed(5)} {'\u00d7'} {(1 - v.p_cal).toFixed(5)} / 150)
          </Computation>
          <Computation>
            se = sqrt({(v.p_cal * (1 - v.p_cal)).toFixed(5)} / 150)
          </Computation>
          <Computation>
            se = sqrt({(v.p_cal * (1 - v.p_cal) / 150).toFixed(7)}) = <Val>{v.se.toFixed(5)}</Val>
          </Computation>
          <div className="border-t border-white/10 mt-2 pt-2">
            <Computation>
              lower = {v.p_cal.toFixed(5)} - 1.96 {'\u00d7'} {v.se.toFixed(5)} ={' '}
              <Val>{v.ci_lower.toFixed(5)}</Val>
            </Computation>
            <Computation>
              upper = {v.p_cal.toFixed(5)} + 1.96 {'\u00d7'} {v.se.toFixed(5)} ={' '}
              <Val>{v.ci_upper.toFixed(5)}</Val>
            </Computation>
          </div>
        </div>
        <Computation>
          95% CI: <Val>[{(v.ci_lower * 100).toFixed(2)}%, {(v.ci_upper * 100).toFixed(2)}%]</Val>
        </Computation>
      </div>
    ),
  },
  {
    num: 8,
    title: 'Final Response',
    preview: (v) => `{ probability: ${v.p_cal.toFixed(4)}, label: "${v.label}" }`,
    renderExpanded: (v) => {
      const response = JSON.stringify(
        {
          probability: parseFloat(v.p_cal.toFixed(4)),
          label: v.label,
          confidence_interval: [
            parseFloat(v.ci_lower.toFixed(4)),
            parseFloat(v.ci_upper.toFixed(4)),
          ],
          calibrated: true,
          features_used: 7,
        },
        null,
        2,
      )
      return (
        <div className="space-y-3">
          <Label>API response payload:</Label>
          <JsonBlock>{response}</JsonBlock>
          <div className="flex items-center gap-4">
            <Computation>
              Label:{' '}
              <span
                className={`font-bold ${
                  v.label === 'LIKELY'
                    ? 'text-emerald-400'
                    : v.label === 'UNCERTAIN'
                      ? 'text-amber-400'
                      : 'text-red-400'
                }`}
              >
                {v.label}
              </span>
            </Computation>
            <Computation>
              Probability: <Val>{(v.p_cal * 100).toFixed(2)}%</Val>
            </Computation>
          </div>
        </div>
      )
    },
  },
]

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function WorkedExampleTracer({ grade = 92.5 }: WorkedExampleTracerProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const vals = useMemo(() => computeValues(grade), [grade])

  const toggle = (num: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(num)) {
        next.delete(num)
      } else {
        next.add(num)
      }
      return next
    })
  }

  return (
    <div className="bg-[#050505] rounded-2xl p-6 space-y-3">
      {STEPS.map((step) => {
        const isExpanded = expandedSteps.has(step.num)

        return (
          <div key={step.num} className="relative">
            <GlassCard padding="p-0">
              {/* Header - clickable */}
              <button
                onClick={() => toggle(step.num)}
                className="w-full text-left px-5 py-4 flex items-start gap-4 group"
              >
                {/* Step number badge */}
                <div className="shrink-0 w-7 h-7 rounded-full bg-emerald-400/20 border border-emerald-400/40 flex items-center justify-center mt-0.5">
                  <span className="text-xs font-bold text-white">{step.num}</span>
                </div>

                {/* Title and preview */}
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-bold text-white group-hover:text-emerald-400 transition-colors">
                    {step.title}
                  </h4>
                  {!isExpanded && (
                    <p className="text-xs text-gray-500 font-mono mt-1 truncate">
                      {step.preview(vals)}
                    </p>
                  )}
                </div>

                {/* Chevron */}
                <span className="text-gray-500 text-sm shrink-0 mt-1 transition-transform duration-200">
                  {isExpanded ? '\u25bc' : '\u25b6'}
                </span>
              </button>

              {/* Expanded content */}
              {isExpanded && (
                <div className="px-5 pb-5 pt-0 ml-11">
                  {step.renderExpanded(vals)}
                </div>
              )}
            </GlassCard>
          </div>
        )
      })}
    </div>
  )
}

import { useState, useCallback } from 'react'
import GlassCard from '@/components/shared/GlassCard'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Pre-computed Householder QR factorization
   of A = [[2, -1, 0], [1, 3, 1], [0, 1, 2]]
   ───────────────────────────────────────────── */

const A: number[][] = [
  [2, -1, 0],
  [1, 3, 1],
  [0, 1, 2],
]

const H1: number[][] = [
  [-0.894, -0.447, 0.0],
  [-0.447, 0.894, 0.0],
  [0.0, 0.0, 1.0],
]

const H1A: number[][] = [
  [-2.236, -0.447, -0.447],
  [0.0, 3.13, 0.894],
  [0.0, 1.0, 2.0],
]

const H2: number[][] = [
  [1.0, 0.0, 0.0],
  [0.0, -0.953, -0.304],
  [0.0, -0.304, 0.953],
]

const R: number[][] = [
  [-2.236, -0.447, -0.447],
  [0.0, -3.286, -1.461],
  [0.0, 0.0, 1.633],
]

const Q: number[][] = [
  [-0.894, 0.426, 0.136],
  [-0.447, -0.852, -0.272],
  [0.0, -0.304, 0.953],
]

/* ─────────────────────────────────────────────
   Step definitions
   ───────────────────────────────────────────── */

interface Step {
  title: string
  explanation: string
}

const STEPS: Step[] = [
  {
    title: 'Original Matrix A',
    explanation:
      'We begin with a 3\u00d73 matrix A. The goal is to decompose it into A = QR where Q is orthogonal and R is upper triangular, using Householder reflections to zero out below-diagonal entries column by column.',
  },
  {
    title: 'Reflecting column 1 to zero below the diagonal',
    explanation:
      'The first column [2, 1, 0]\u1d40 has norm \u221a5 \u2248 2.236. We construct a Householder reflector H\u2081 = I \u2212 2vv\u1d40/v\u1d40v that maps this column onto [\u22122.236, 0, 0]\u1d40. Applying H\u2081 to A zeros out the subdiagonal entries in column 1.',
  },
  {
    title: 'Reflecting column 2 of the submatrix',
    explanation:
      'In the lower-right 2\u00d72 submatrix of H\u2081A, column 1 is [3.130, 1.000]\u1d40 with norm \u2248 3.286. We build H\u2082 (identity in the first row/column) to zero out below the (2,2) position. The result H\u2082H\u2081A is the upper triangular matrix R.',
  },
  {
    title: 'Final Q and R matrices',
    explanation:
      'Since H\u2082H\u2081A = R, we have A = (H\u2081H\u2082)\u1d40\u207b\u00b9R. Because Householder matrices are symmetric and orthogonal, Q = H\u2081H\u2082 and A = QR. The product Q\u00d7R reconstructs the original matrix A.',
  },
]

/* ─────────────────────────────────────────────
   Highlight specifications per step.
   Each entry maps "row,col" to a highlight kind.
   ───────────────────────────────────────────── */

type CellHighlight = 'zeroed' | 'diagonal'

type HighlightMap = Record<string, CellHighlight>

/* ─────────────────────────────────────────────
   Matrix display component
   ───────────────────────────────────────────── */

function MatrixDisplay({
  label,
  data,
  highlights = {},
}: {
  label: string
  data: number[][]
  highlights?: HighlightMap
}) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-gray-500 font-medium">{label}</p>
      <div className="inline-flex items-center gap-0">
        {/* Left bracket */}
        <div className="w-1.5 border-l-2 border-t-2 border-b-2 border-gray-600 rounded-l-sm self-stretch" />

        <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 py-1 px-2">
          {data.map((row, i) =>
            row.map((val, j) => {
              const key = `${i},${j}`
              const hl = highlights[key]
              let cellClass = 'text-gray-300 font-mono text-sm tabular-nums text-right px-1 py-0.5 rounded'
              if (hl === 'zeroed') {
                cellClass =
                  'text-emerald-400 font-mono text-sm tabular-nums text-right px-1 py-0.5 rounded bg-emerald-400/20 font-semibold'
              } else if (hl === 'diagonal') {
                cellClass =
                  'text-amber-400 font-mono text-sm tabular-nums text-right px-1 py-0.5 rounded font-semibold'
              }
              return (
                <span key={key} className={cellClass}>
                  {val.toFixed(3)}
                </span>
              )
            }),
          )}
        </div>

        {/* Right bracket */}
        <div className="w-1.5 border-r-2 border-t-2 border-b-2 border-gray-600 rounded-r-sm self-stretch" />
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   Step content renderers
   ───────────────────────────────────────────── */

function StepZero() {
  return <MatrixDisplay label="A" data={A} />
}

function StepOne() {
  const hlH1A: HighlightMap = {
    '1,0': 'zeroed',
    '2,0': 'zeroed',
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <p className="text-xs text-gray-500 font-medium">
            Reflection vector v<sub>1</sub> (normalized)
          </p>
          <p className="font-mono text-sm text-emerald-400">
            [0.973, 0.230, 0.000]
          </p>
        </div>
        <div className="space-y-1.5">
          <p className="text-xs text-gray-500 font-medium">Column norm</p>
          <p className="font-mono text-sm text-white">
            ||col<sub>1</sub>|| = <span className="text-emerald-400">2.236</span>
          </p>
        </div>
      </div>

      <MatrixDisplay label="H\u2081 = I \u2212 2vv\u1d40 / v\u1d40v" data={H1} />
      <MatrixDisplay label="H\u2081A" data={H1A} highlights={hlH1A} />
    </div>
  )
}

function StepTwo() {
  const hlR: HighlightMap = {
    '1,0': 'zeroed',
    '2,0': 'zeroed',
    '2,1': 'zeroed',
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <p className="text-xs text-gray-500 font-medium">
            Reflection vector v<sub>2</sub> (normalized)
          </p>
          <p className="font-mono text-sm text-emerald-400">
            [0.988, 0.154]
          </p>
        </div>
        <div className="space-y-1.5">
          <p className="text-xs text-gray-500 font-medium">Subcolumn norm</p>
          <p className="font-mono text-sm text-white">
            ||subcol<sub>2</sub>|| = <span className="text-emerald-400">3.286</span>
          </p>
        </div>
      </div>

      <MatrixDisplay label="H\u2082 (embedded in 3\u00d73)" data={H2} />
      <MatrixDisplay label="R = H\u2082H\u2081A" data={R} highlights={hlR} />
    </div>
  )
}

function StepThree() {
  const hlR: HighlightMap = {
    '0,0': 'diagonal',
    '1,1': 'diagonal',
    '2,2': 'diagonal',
    '1,0': 'zeroed',
    '2,0': 'zeroed',
    '2,1': 'zeroed',
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MatrixDisplay label="Q = H\u2081H\u2082" data={Q} />
        <MatrixDisplay label="R (upper triangular)" data={R} highlights={hlR} />
      </div>

      <div className="space-y-1.5">
        <p className="text-xs text-gray-500 font-medium">
          Verification: <M>{"Q \\times R \\approx A"}</M>
        </p>
        <MatrixDisplay label="Q \u00d7 R" data={A} />
        <p className="text-xs text-emerald-400 mt-1">
          Reconstruction matches the original matrix A.
        </p>
      </div>
    </div>
  )
}

const STEP_RENDERERS = [StepZero, StepOne, StepTwo, StepThree]

/* ─────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────── */

export default function HouseholderStepThrough() {
  const [step, setStep] = useState(0)
  const total = STEPS.length

  const handlePrev = useCallback(() => setStep((s) => Math.max(0, s - 1)), [])
  const handleNext = useCallback(
    () => setStep((s) => Math.min(total - 1, s + 1)),
    [total],
  )

  const currentStep = STEPS[step]
  const StepContent = STEP_RENDERERS[step]

  return (
    <div className="space-y-4">
      {/* Step indicator dots and navigation */}
      <GlassCard padding="px-4 py-3">
        <div className="space-y-3">
          {/* Step dots */}
          <div className="flex items-center justify-center gap-2">
            {STEPS.map((_, i) => (
              <button
                key={i}
                onClick={() => setStep(i)}
                className={`w-7 h-7 rounded-full text-xs font-mono font-medium transition-all duration-200 ${
                  i === step
                    ? 'bg-emerald-400/20 text-emerald-400 ring-1 ring-emerald-400/40'
                    : 'bg-white/5 text-gray-500 hover:bg-white/10 hover:text-gray-400'
                }`}
              >
                {i}
              </button>
            ))}
          </div>

          {/* Prev / Next buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={handlePrev}
              disabled={step <= 0}
              className="px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Prev
            </button>

            <span className="text-sm text-gray-400 tabular-nums min-w-[4rem] text-center">
              {step + 1} / {total}
            </span>

            <button
              onClick={handleNext}
              disabled={step >= total - 1}
              className="px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      </GlassCard>

      {/* Step title and explanation */}
      <div className="space-y-1">
        <h3 className="text-white text-sm font-semibold">
          Step {step}: {currentStep.title}
        </h3>
        <p className="text-gray-400 text-xs leading-relaxed">
          {currentStep.explanation}
        </p>
      </div>

      {/* Step content with opacity transition */}
      <div
        key={step}
        className="animate-[fadeIn_300ms_ease-in-out]"
        style={{
          animation: 'fadeIn 300ms ease-in-out',
        }}
      >
        <StepContent />
      </div>

      {/* Inline keyframes for fade animation */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  )
}

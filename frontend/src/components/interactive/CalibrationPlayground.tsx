import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Mock calibration data (same as About.tsx)
   ───────────────────────────────────────────── */
const CALIBRATION_DATA = [
  { predicted: 0.1, observed: 0.12, count: 50 },
  { predicted: 0.2, observed: 0.18, count: 80 },
  { predicted: 0.3, observed: 0.32, count: 120 },
  { predicted: 0.4, observed: 0.38, count: 150 },
  { predicted: 0.5, observed: 0.52, count: 180 },
  { predicted: 0.6, observed: 0.58, count: 160 },
  { predicted: 0.7, observed: 0.72, count: 140 },
  { predicted: 0.8, observed: 0.78, count: 100 },
  { predicted: 0.9, observed: 0.88, count: 60 },
]

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function logit(p: number): number {
  const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p))
  return Math.log(clamped / (1 - clamped))
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function CalibrationPlayground() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [plattA, setPlattA] = useState(1.0)
  const [plattB, setPlattB] = useState(0.0)

  // Apply Platt scaling: p_cal = sigmoid(A * logit(p_raw) + B)
  const calibrated = useMemo(() => {
    return CALIBRATION_DATA.map((d) => ({
      ...d,
      calibrated: sigmoid(plattA * logit(d.predicted) + plattB),
    }))
  }, [plattA, plattB])

  // Compute ECE & Brier
  const metrics = useMemo(() => {
    const totalCount = calibrated.reduce((s, d) => s + d.count, 0)
    let ece = 0
    let brier = 0

    calibrated.forEach((d) => {
      const pCal = d.calibrated
      ece += (d.count / totalCount) * Math.abs(pCal - d.observed)
      // Brier for the bin: mean (p - outcome)^2 approximated
      brier +=
        (d.count / totalCount) *
        (d.observed * (1 - pCal) ** 2 + (1 - d.observed) * pCal ** 2)
    })

    return { ece, brier }
  }, [calibrated])

  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const size = Math.min(340, container.clientWidth)
    const margin = { top: 16, right: 16, bottom: 40, left: 44 }

    svg.attr('width', size).attr('height', size)

    const x = d3.scaleLinear().domain([0, 1]).range([margin.left, size - margin.right])
    const y = d3.scaleLinear().domain([0, 1]).range([size - margin.bottom, margin.top])

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${size - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Y axis
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Axis labels
    svg
      .append('text')
      .attr('x', (margin.left + size - margin.right) / 2)
      .attr('y', size - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Predicted Probability')

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + size - margin.bottom) / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Observed Frequency')

    // Perfect calibration diagonal
    svg
      .append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(1))
      .attr('y2', y(1))
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '6,4')

    // Platt scaling sigmoid overlay
    const plattLine = d3.range(0.01, 1.0, 0.01).map((pRaw) => ({
      pRaw,
      pCal: sigmoid(plattA * logit(pRaw) + plattB),
    }))

    svg
      .append('path')
      .datum(plattLine)
      .attr(
        'd',
        d3
          .line<{ pRaw: number; pCal: number }>()
          .x((d) => x(d.pRaw))
          .y((d) => y(d.pCal))
          .curve(d3.curveBasis),
      )
      .attr('fill', 'none')
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,2')
      .attr('opacity', 0.8)

    // Calibration line
    const sorted = [...CALIBRATION_DATA].sort((a, b) => a.predicted - b.predicted)
    svg
      .append('path')
      .datum(sorted)
      .attr(
        'd',
        d3
          .line<(typeof CALIBRATION_DATA)[0]>()
          .x((d) => x(d.predicted))
          .y((d) => y(d.observed)),
      )
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    // Point size scale
    const maxCount = d3.max(sorted, (d) => d.count) ?? 1
    const rScale = d3.scaleSqrt().domain([0, maxCount]).range([3, 10])

    // Data points
    svg
      .selectAll('circle.point')
      .data(sorted)
      .join('circle')
      .attr('class', 'point')
      .attr('cx', (d) => x(d.predicted))
      .attr('cy', (d) => y(d.observed))
      .attr('r', (d) => rScale(d.count))
      .attr('fill', '#10b981')
      .attr('stroke', '#050505')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.85)
  }, [plattA, plattB])

  return (
    <div className="space-y-4">
      <div ref={containerRef} className="flex justify-center">
        <svg ref={svgRef} />
      </div>

      {/* Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>A (slope)</span>
            <span className="text-white font-mono tabular-nums">{plattA.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0.5}
            max={2.0}
            step={0.05}
            value={plattA}
            onChange={(e) => setPlattA(Number(e.target.value))}
            className="w-full accent-emerald-400"
          />
        </div>
        <div className="space-y-1">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>B (intercept)</span>
            <span className="text-white font-mono tabular-nums">{plattB.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={-1.0}
            max={1.0}
            step={0.05}
            value={plattB}
            onChange={(e) => setPlattB(Number(e.target.value))}
            className="w-full accent-emerald-400"
          />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={() => {
            setPlattA(1.0)
            setPlattB(0.0)
          }}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:border-white/20 transition-colors"
        >
          Reset to fitted
        </button>

        <div className="flex items-center gap-6 text-sm">
          <div>
            <span className="text-gray-500">ECE: </span>
            <span className="text-white font-mono font-bold tabular-nums">
              {(metrics.ece * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-gray-500">Brier: </span>
            <span className="text-white font-mono font-bold tabular-nums">
              {metrics.brier.toFixed(3)}
            </span>
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-600">
        Amber dashed line shows the Platt scaling transform. Green line shows
        actual calibration. Perfect calibration follows the gray diagonal.
      </p>
    </div>
  )
}

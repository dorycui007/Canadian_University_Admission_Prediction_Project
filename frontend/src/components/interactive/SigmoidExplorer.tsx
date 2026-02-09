import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'
import useReducedMotion from '@/hooks/useReducedMotion'

/* ─────────────────────────────────────────────
   Feature definitions & coefficients
   (from prediction_engine.md)
   ───────────────────────────────────────────── */
const FEATURES = [
  { key: 'bias', label: 'Bias (intercept)', beta: -0.5, getValue: () => 1 },
  { key: 'top6', label: 'Top-6 Average', beta: 0.08, getValue: (g: number) => (g - 87.3) / 4.2 },
  { key: 'gr12', label: 'Grade 12 Avg', beta: 0.02, getValue: (g: number) => (g - 85.0) / 5.0 },
  { key: 'interaction', label: 'Grade x Program', beta: 0.03, getValue: (g: number) => ((g - 87.3) / 4.2) * 0.8 },
  { key: 'ontario', label: 'is_ontario', beta: 0.1, getValue: () => 1 },
  { key: 'bc', label: 'is_bc', beta: 0.05, getValue: () => 0 },
  { key: 'alberta', label: 'is_alberta', beta: 0.05, getValue: () => 0 },
  { key: 'quebec', label: 'is_quebec', beta: 0.08, getValue: () => 0 },
]

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
interface SigmoidExplorerProps {
  grade: number
}

export default function SigmoidExplorer({ grade }: SigmoidExplorerProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const reducedMotion = useReducedMotion()

  const [enabled, setEnabled] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(FEATURES.map((f) => [f.key, true])),
  )

  // Compute contributions
  const contributions = useMemo(() => {
    return FEATURES.map((f) => {
      const x = f.getValue(grade)
      const contrib = enabled[f.key] ? x * f.beta : 0
      return { ...f, x, contrib }
    })
  }, [grade, enabled])

  const z = useMemo(
    () => contributions.reduce((s, c) => s + c.contrib, 0),
    [contributions],
  )
  const prob = sigmoid(z)

  // D3 rendering
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 340
    const waterfallWidth = Math.round(totalWidth * 0.35)
    const sigmoidWidth = totalWidth - waterfallWidth
    const margin = { top: 24, right: 24, bottom: 36, left: 16 }

    svg.attr('width', totalWidth).attr('height', height)

    // ─── Probability bands (sigmoid area) ───
    const sigG = svg.append('g').attr('transform', `translate(${waterfallWidth}, 0)`)

    const xScale = d3.scaleLinear().domain([-6, 6]).range([margin.left + 40, sigmoidWidth - margin.right])
    const yScale = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top])

    // Background probability zones
    const zones = [
      { y0: 0, y1: 0.4, color: 'rgba(239,68,68,0.15)', label: 'UNLIKELY', labelColor: 'rgba(239,68,68,0.5)' },
      { y0: 0.4, y1: 0.7, color: 'rgba(251,191,36,0.12)', label: 'UNCERTAIN', labelColor: 'rgba(251,191,36,0.5)' },
      { y0: 0.7, y1: 1.0, color: 'rgba(16,185,129,0.15)', label: 'LIKELY', labelColor: 'rgba(16,185,129,0.5)' },
    ]

    zones.forEach((zone) => {
      sigG
        .append('rect')
        .attr('x', xScale(-6))
        .attr('y', yScale(zone.y1))
        .attr('width', xScale(6) - xScale(-6))
        .attr('height', yScale(zone.y0) - yScale(zone.y1))
        .attr('fill', zone.color)

      sigG
        .append('text')
        .attr('x', sigmoidWidth - margin.right - 4)
        .attr('y', yScale((zone.y0 + zone.y1) / 2) + 4)
        .attr('text-anchor', 'end')
        .attr('fill', zone.labelColor)
        .attr('font-size', '11px')
        .attr('font-weight', '700')
        .attr('letter-spacing', '0.05em')
        .text(zone.label)
    })

    // X axis
    sigG
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    sigG
      .append('text')
      .attr('x', (xScale(-6) + xScale(6)) / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#d1d5db')
      .attr('font-size', '11px')
      .text('z (linear predictor)')

    // Y axis
    sigG
      .append('g')
      .attr('transform', `translate(${margin.left + 40},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    // Sigmoid curve
    const lineData = d3.range(-6, 6.05, 0.1).map((zv) => ({ z: zv, p: sigmoid(zv) }))
    const line = d3
      .line<{ z: number; p: number }>()
      .x((d) => xScale(d.z))
      .y((d) => yScale(d.p))
      .curve(d3.curveBasis)

    sigG
      .append('path')
      .datum(lineData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#34d399')
      .attr('stroke-width', 3)

    // Current point on sigmoid
    const clampedZ = Math.max(-6, Math.min(6, z))
    const pointX = xScale(clampedZ)
    const pointY = yScale(prob)

    // Vertical + horizontal reference lines
    sigG
      .append('line')
      .attr('x1', pointX)
      .attr('y1', height - margin.bottom)
      .attr('x2', pointX)
      .attr('y2', pointY)
      .attr('stroke', '#34d399')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.7)

    sigG
      .append('line')
      .attr('x1', margin.left + 40)
      .attr('y1', pointY)
      .attr('x2', pointX)
      .attr('y2', pointY)
      .attr('stroke', '#34d399')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.7)

    // Point dot
    sigG
      .append('circle')
      .attr('cx', pointX)
      .attr('cy', pointY)
      .attr('r', 7)
      .attr('fill', '#34d399')
      .attr('stroke', '#050505')
      .attr('stroke-width', 2)

    // Probability label
    sigG
      .append('text')
      .attr('x', pointX + 10)
      .attr('y', pointY - 10)
      .attr('fill', '#fff')
      .attr('font-size', '13px')
      .attr('font-weight', '700')
      .text(`p = ${prob.toFixed(3)}`)

    // ─── Waterfall (left side) ───
    const wG = svg.append('g')
    const barHeight = 20
    const barGap = 4
    const activeContribs = contributions.filter((c) => c.contrib !== 0)
    const totalBars = activeContribs.length
    const waterfallTop = margin.top + 10
    const waterfallH = totalBars * (barHeight + barGap)

    const maxAbsContrib = Math.max(
      0.3,
      d3.max(activeContribs, (d) => Math.abs(d.contrib)) ?? 0.3,
    )
    const barScale = d3
      .scaleLinear()
      .domain([-maxAbsContrib, maxAbsContrib])
      .range([16, waterfallWidth - 16])

    const zeroX = barScale(0)

    // Zero line
    wG.append('line')
      .attr('x1', zeroX)
      .attr('y1', waterfallTop - 4)
      .attr('x2', zeroX)
      .attr('y2', waterfallTop + waterfallH + 4)
      .attr('stroke', 'rgba(255,255,255,0.2)')
      .attr('stroke-width', 1)

    activeContribs.forEach((c, i) => {
      const y = waterfallTop + i * (barHeight + barGap)
      const barX = c.contrib >= 0 ? zeroX : barScale(c.contrib)
      const barW = Math.abs(barScale(c.contrib) - zeroX)
      const color = c.contrib >= 0 ? '#34d399' : '#f87171'

      wG.append('rect')
        .attr('x', barX)
        .attr('y', y)
        .attr('width', Math.max(1, barW))
        .attr('height', barHeight)
        .attr('rx', 3)
        .attr('fill', color)
        .attr('opacity', 0.85)

      wG.append('text')
        .attr('x', 4)
        .attr('y', y + barHeight / 2 + 4)
        .attr('fill', '#d1d5db')
        .attr('font-size', '11px')
        .text(c.label)

      wG.append('text')
        .attr('x', c.contrib >= 0 ? barX + barW + 4 : barX - 4)
        .attr('y', y + barHeight / 2 + 4)
        .attr('text-anchor', c.contrib >= 0 ? 'start' : 'end')
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('font-weight', '700')
        .text(c.contrib >= 0 ? `+${c.contrib.toFixed(3)}` : c.contrib.toFixed(3))
    })

    // Sum label
    wG.append('text')
      .attr('x', waterfallWidth / 2)
      .attr('y', waterfallTop + waterfallH + 24)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(`z = ${z.toFixed(3)}`)

    // Connecting arrow from waterfall sum to sigmoid
    const arrowY = waterfallTop + waterfallH + 20
    sigG
      .append('line')
      .attr('x1', 0)
      .attr('y1', arrowY)
      .attr('x2', pointX)
      .attr('y2', pointY)
      .attr('stroke', 'rgba(255,255,255,0.25)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4')
  }, [contributions, z, prob, reducedMotion])

  return (
    <div className="space-y-4">
      {/* Feature toggles */}
      <div className="flex flex-wrap gap-3">
        {FEATURES.map((f) => (
          <label
            key={f.key}
            className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer select-none"
          >
            <input
              type="checkbox"
              checked={enabled[f.key]}
              onChange={() =>
                setEnabled((prev) => ({ ...prev, [f.key]: !prev[f.key] }))
              }
              className="accent-emerald-400 rounded"
            />
            {f.label}
          </label>
        ))}
      </div>

      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Summary stats */}
      <div className="flex items-center gap-6 text-sm">
        <div>
          <span className="text-gray-500">Linear predictor: </span>
          <span className="text-white font-mono font-bold">{z.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-gray-500">Probability: </span>
          <span className="text-emerald-400 font-mono font-bold">
            {(prob * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="text-gray-500">Prediction: </span>
          <span
            className={`font-bold ${
              prob >= 0.7
                ? 'text-emerald-400'
                : prob >= 0.4
                  ? 'text-amber-400'
                  : 'text-red-400'
            }`}
          >
            {prob >= 0.7 ? 'LIKELY' : prob >= 0.4 ? 'UNCERTAIN' : 'UNLIKELY'}
          </span>
        </div>
      </div>
    </div>
  )
}

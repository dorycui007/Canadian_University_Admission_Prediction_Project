import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Mock singular values from design matrix SVD
   ───────────────────────────────────────────── */
const SINGULAR_VALUES = [12.5, 8.3, 4.1, 2.0, 0.8, 0.3, 0.1]

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function SVDShrinkageExplorer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Lambda on a log scale: slider value is log10(lambda)
  const [logLambda, setLogLambda] = useState(0) // log10(1.0) = 0
  const lambda = useMemo(() => Math.pow(10, logLambda), [logLambda])

  // Shrinkage factors: sigma_i^2 / (sigma_i^2 + lambda)
  const shrinkageFactors = useMemo(() => {
    return SINGULAR_VALUES.map((s) => (s * s) / (s * s + lambda))
  }, [lambda])

  // Effective degrees of freedom = sum of shrinkage factors
  const edf = useMemo(() => {
    return shrinkageFactors.reduce((sum, f) => sum + f, 0)
  }, [shrinkageFactors])

  // Condition number: max(sigma) / min(sigma), and after shrinkage
  const conditionBefore = useMemo(() => {
    const max = Math.max(...SINGULAR_VALUES)
    const min = Math.min(...SINGULAR_VALUES)
    return max / min
  }, [])

  const conditionAfter = useMemo(() => {
    const effectiveSV = SINGULAR_VALUES.map((s) => (s * s) / (s * s + lambda) * s)
    const max = Math.max(...effectiveSV)
    const min = Math.min(...effectiveSV)
    return max / min
  }, [lambda])

  // D3 rendering
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 280
    const margin = { top: 24, right: 60, bottom: 44, left: 50 }

    svg.attr('width', totalWidth).attr('height', height)

    const innerWidth = totalWidth - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const n = SINGULAR_VALUES.length

    // X scale: band scale for groups
    const xScale = d3
      .scaleBand<number>()
      .domain(d3.range(n))
      .range([0, innerWidth])
      .paddingInner(0.3)
      .paddingOuter(0.15)

    // Left Y axis: shrinkage factors (0 to 1)
    const yLeft = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0])

    // Right Y axis: singular values (0 to 15)
    const yRight = d3.scaleLinear().domain([0, 15]).range([innerHeight, 0])

    // Bar widths: each group has 2 bars
    const groupWidth = xScale.bandwidth()
    const barWidth = groupWidth / 2

    // ─── Grid lines ───
    g.append('g')
      .selectAll('line')
      .data(yLeft.ticks(5))
      .join('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', (d) => yLeft(d))
      .attr('y2', (d) => yLeft(d))
      .attr('stroke', 'rgba(255,255,255,0.05)')
      .attr('stroke-width', 1)

    // ─── Bars ───
    SINGULAR_VALUES.forEach((sigma, i) => {
      const groupX = xScale(i)!
      const shrinkage = shrinkageFactors[i]

      // Original singular value bar (emerald, mapped to right Y axis)
      const svBarHeight = innerHeight - yRight(sigma)
      g.append('rect')
        .attr('x', groupX)
        .attr('y', yRight(sigma))
        .attr('width', barWidth)
        .attr('height', Math.max(0, svBarHeight))
        .attr('rx', 2)
        .attr('fill', '#10b981')
        .attr('opacity', 0.85)

      // Shrunk bar (amber, mapped to left Y axis)
      const shrunkBarHeight = innerHeight - yLeft(shrinkage)
      g.append('rect')
        .attr('x', groupX + barWidth)
        .attr('y', yLeft(shrinkage))
        .attr('width', barWidth)
        .attr('height', Math.max(0, shrunkBarHeight))
        .attr('rx', 2)
        .attr('fill', '#fbbf24')
        .attr('opacity', 0.85)
    })

    // ─── X axis ───
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickFormat((_, i) => `\u03C3\u2081`.replace('\u2081', String.fromCharCode(0x2081 + i))),
      )
      .call((ax) => {
        ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        ax.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        ax.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '12px')
      })

    // X axis label
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Singular Values')

    // ─── Left Y axis (shrinkage factor) ───
    g.append('g')
      .call(d3.axisLeft(yLeft).ticks(5).tickFormat(d3.format('.1f')))
      .call((ax) => {
        ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        ax.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        ax.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '10px')
      })

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fbbf24')
      .attr('font-size', '11px')
      .text('Shrinkage Factor')

    // ─── Right Y axis (singular values) ───
    g.append('g')
      .attr('transform', `translate(${innerWidth},0)`)
      .call(d3.axisRight(yRight).ticks(5))
      .call((ax) => {
        ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        ax.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        ax.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '10px')
      })

    g.append('text')
      .attr('transform', 'rotate(90)')
      .attr('x', innerHeight / 2)
      .attr('y', -innerWidth - 46)
      .attr('text-anchor', 'middle')
      .attr('fill', '#10b981')
      .attr('font-size', '11px')
      .text('Singular Value')

    // ─── Legend ───
    const legend = g.append('g').attr('transform', `translate(${innerWidth - 160}, -6)`)

    // Emerald swatch
    legend
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 12)
      .attr('height', 12)
      .attr('rx', 2)
      .attr('fill', '#10b981')
      .attr('opacity', 0.85)

    legend
      .append('text')
      .attr('x', 18)
      .attr('y', 10)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Original \u03C3\u1D62')

    // Amber swatch
    legend
      .append('rect')
      .attr('x', 88)
      .attr('y', 0)
      .attr('width', 12)
      .attr('height', 12)
      .attr('rx', 2)
      .attr('fill', '#fbbf24')
      .attr('opacity', 0.85)

    legend
      .append('text')
      .attr('x', 106)
      .attr('y', 10)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Shrinkage')
  }, [shrinkageFactors])

  return (
    <div className="space-y-4">
      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Lambda slider */}
      <div className="space-y-1">
        <label className="flex items-center justify-between text-xs text-gray-400">
          <span>
            Regularization <M>{"\\lambda"}</M> (log scale)
          </span>
          <span className="text-white font-mono tabular-nums">
            <M>{"\\lambda"}</M> = {lambda.toFixed(lambda < 0.1 ? 3 : lambda < 1 ? 2 : 1)}
          </span>
        </label>
        <input
          type="range"
          min={-2}
          max={1}
          step={0.01}
          value={logLambda}
          onChange={(e) => setLogLambda(Number(e.target.value))}
          className="w-full accent-emerald-400"
        />
        <div className="flex justify-between text-[10px] text-gray-600 font-mono">
          <span>0.01</span>
          <span>0.1</span>
          <span>1.0</span>
          <span>10</span>
        </div>
      </div>

      {/* Stats below chart */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Effective DoF: </span>
          <span className="text-emerald-400 font-mono font-bold tabular-nums">
            {edf.toFixed(2)}
          </span>
          <span className="text-gray-600 text-xs ml-1">
            / {SINGULAR_VALUES.length}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Condition # (before): </span>
          <span className="text-white font-mono font-bold tabular-nums">
            {conditionBefore.toFixed(1)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Condition # (after): </span>
          <span className="text-amber-400 font-mono font-bold tabular-nums">
            {conditionAfter.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Shrinkage factors table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-gray-400">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left py-1 pr-3 font-medium text-gray-500">Index</th>
              <th className="text-right py-1 px-3 font-medium text-gray-500">
                <M>{"\\sigma"}</M>
              </th>
              <th className="text-right py-1 px-3 font-medium text-gray-500">
                <M>{"\\sigma^2 / (\\sigma^2 + \\lambda)"}</M>
              </th>
            </tr>
          </thead>
          <tbody>
            {SINGULAR_VALUES.map((sigma, i) => (
              <tr key={i} className="border-b border-white/[0.03]">
                <td className="py-1 pr-3 font-mono">
                  <M>{`\\sigma_${i + 1}`}</M>
                </td>
                <td className="py-1 px-3 text-right font-mono text-emerald-400/80">
                  {sigma.toFixed(1)}
                </td>
                <td className="py-1 px-3 text-right font-mono text-amber-400/80">
                  {shrinkageFactors[i].toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-600">
        Emerald bars show original singular values (right axis). Amber bars show the
        shrinkage factor <M>{"\\sigma^2 / (\\sigma^2 + \\lambda)"}</M> on the left axis.
        Larger <M>{"\\lambda"}</M> shrinks small singular values more aggressively, reducing effective
        degrees of freedom and improving the condition number.
      </p>
    </div>
  )
}

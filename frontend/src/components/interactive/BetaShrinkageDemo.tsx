import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Beta distribution math utilities
   ───────────────────────────────────────────── */

/** Lanczos approximation of log-gamma */
function logGamma(x: number): number {
  if (x <= 0) return Infinity
  const g = 7
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  let sum = c[0]
  for (let i = 1; i < g + 2; i++) {
    sum += c[i] / (x + i - 1)
  }
  const t = x + g - 0.5
  return 0.5 * Math.log(2 * Math.PI) + (x - 0.5) * Math.log(t) - t + Math.log(sum)
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b)
}

function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0
  if (a <= 0 || b <= 0) return 0
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b))
}

/** Find approximate credible interval by numerical integration (trapezoid rule) */
function credibleInterval(
  a: number,
  b: number,
  level: number = 0.95,
): [number, number] {
  const n = 1000
  const dx = 1 / n
  const tail = (1 - level) / 2

  // Build CDF via trapezoid rule
  const cdf: number[] = [0]
  for (let i = 1; i <= n; i++) {
    const x0 = (i - 1) * dx
    const x1 = i * dx
    const area = 0.5 * (betaPdf(x0, a, b) + betaPdf(x1, a, b)) * dx
    cdf.push(cdf[i - 1] + area)
  }

  // Normalize CDF to account for numerical integration error
  const total = cdf[n]
  for (let i = 0; i <= n; i++) {
    cdf[i] /= total
  }

  let lower = 0
  let upper = 1
  for (let i = 0; i <= n; i++) {
    if (cdf[i] >= tail) {
      lower = i * dx
      break
    }
  }
  for (let i = n; i >= 0; i--) {
    if (cdf[i] <= 1 - tail) {
      upper = i * dx
      break
    }
  }

  return [lower, upper]
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function BetaShrinkageDemo() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [n, setN] = useState(20)
  const [k, setK] = useState(12)
  const [priorStrength, setPriorStrength] = useState(10)

  // Clamp k to n when n changes
  const effectiveK = Math.min(k, n)

  // Prior parameters: uniform-like base rate = 0.5
  const baseRate = 0.5
  const alpha0 = priorStrength * baseRate
  const beta0 = priorStrength * (1 - baseRate)

  // Posterior parameters
  const alphaPost = alpha0 + effectiveK
  const betaPost = beta0 + n - effectiveK

  // Derived statistics
  const mle = useMemo(() => (n > 0 ? effectiveK / n : 0.5), [effectiveK, n])
  const posteriorMean = useMemo(
    () => (alphaPost) / (alphaPost + betaPost),
    [alphaPost, betaPost],
  )
  const ci = useMemo(
    () => credibleInterval(alphaPost, betaPost, 0.95),
    [alphaPost, betaPost],
  )
  const shrinkagePct = useMemo(() => {
    if (n === 0) return 0
    const distance = Math.abs(posteriorMean - mle)
    const totalDistance = Math.abs(mle - baseRate)
    if (totalDistance < 1e-9) return 0
    return (distance / totalDistance) * 100
  }, [posteriorMean, mle, n])

  // D3 rendering
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 280
    const margin = { top: 24, right: 24, bottom: 44, left: 50 }

    svg.attr('width', totalWidth).attr('height', height)

    const innerWidth = totalWidth - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Generate curve data
    const numPoints = 300
    const step = 1 / numPoints

    const priorData: { x: number; y: number }[] = []
    const posteriorData: { x: number; y: number }[] = []

    for (let i = 0; i <= numPoints; i++) {
      const x = i * step
      priorData.push({ x, y: betaPdf(x, alpha0, beta0) })
      posteriorData.push({ x, y: betaPdf(x, alphaPost, betaPost) })
    }

    // X scale: 0 to 1
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerWidth])

    // Y scale: auto to max of both curves
    const yMax =
      Math.max(
        d3.max(priorData, (d) => d.y) ?? 0,
        d3.max(posteriorData, (d) => d.y) ?? 0,
      ) * 1.1

    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerHeight, 0])

    // ─── Grid lines ───
    g.append('g')
      .selectAll('line')
      .data(yScale.ticks(5))
      .join('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', (d) => yScale(d))
      .attr('y2', (d) => yScale(d))
      .attr('stroke', 'rgba(255,255,255,0.05)')
      .attr('stroke-width', 1)

    // ─── 95% credible interval shaded area ───
    const ciData = posteriorData.filter((d) => d.x >= ci[0] && d.x <= ci[1])

    const area = d3
      .area<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y0(yScale(0))
      .y1((d) => yScale(d.y))
      .curve(d3.curveBasis)

    g.append('path')
      .datum(ciData)
      .attr('d', area)
      .attr('fill', 'rgba(16, 185, 129, 0.12)')

    // ─── Prior curve (dashed gray) ───
    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(d3.curveBasis)

    g.append('path')
      .datum(priorData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4')

    // ─── Posterior curve (solid emerald) ───
    g.append('path')
      .datum(posteriorData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    // ─── MLE vertical dashed line (amber) ───
    if (n > 0) {
      g.append('line')
        .attr('x1', xScale(mle))
        .attr('y1', yScale(0))
        .attr('x2', xScale(mle))
        .attr('y2', yScale(yMax * 0.9))
        .attr('stroke', '#fbbf24')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3')

      g.append('text')
        .attr('x', xScale(mle))
        .attr('y', yScale(yMax * 0.9) - 6)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fbbf24')
        .attr('font-size', '10px')
        .text(`MLE = ${mle.toFixed(2)}`)
    }

    // ─── Posterior mean vertical solid line (emerald) ───
    g.append('line')
      .attr('x1', xScale(posteriorMean))
      .attr('y1', yScale(0))
      .attr('x2', xScale(posteriorMean))
      .attr('y2', yScale(yMax * 0.82))
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', xScale(posteriorMean))
      .attr('y', yScale(yMax * 0.82) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#34d399')
      .attr('font-size', '10px')
      .text(`Post. = ${posteriorMean.toFixed(2)}`)

    // ─── Shrinkage arrow from MLE to posterior mean ───
    if (n > 0 && Math.abs(mle - posteriorMean) > 0.005) {
      const arrowY = yScale(yMax * 0.72)
      const mleX = xScale(mle)
      const postX = xScale(posteriorMean)

      // Arrow line
      g.append('line')
        .attr('x1', mleX)
        .attr('y1', arrowY)
        .attr('x2', postX)
        .attr('y2', arrowY)
        .attr('stroke', '#9ca3af')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)')

      // Arrowhead marker definition
      svg
        .append('defs')
        .append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8)
        .attr('refY', 5)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', '#9ca3af')

      // Label
      const midX = (mleX + postX) / 2
      g.append('text')
        .attr('x', midX)
        .attr('y', arrowY - 6)
        .attr('text-anchor', 'middle')
        .attr('fill', '#9ca3af')
        .attr('font-size', '9px')
        .text('shrinkage')
    }

    // ─── X axis ───
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(10).tickFormat(d3.format('.1f')))
      .call((ax) => {
        ax.select('.domain').attr('stroke', 'rgba(255,255,255,0.2)')
        ax.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        ax.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '10px')
      })

    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Probability of Admission')

    // ─── Y axis ───
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.1f')))
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
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Density')

    // ─── Legend ───
    const legend = g.append('g').attr('transform', `translate(${innerWidth - 200}, -6)`)

    // Prior swatch (dashed gray)
    legend
      .append('line')
      .attr('x1', 0)
      .attr('y1', 6)
      .attr('x2', 18)
      .attr('y2', 6)
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4')

    legend
      .append('text')
      .attr('x', 24)
      .attr('y', 10)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Prior')

    // Posterior swatch (solid emerald)
    legend
      .append('line')
      .attr('x1', 60)
      .attr('y1', 6)
      .attr('x2', 78)
      .attr('y2', 6)
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    legend
      .append('text')
      .attr('x', 84)
      .attr('y', 10)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Posterior')

    // MLE swatch (dashed amber)
    legend
      .append('line')
      .attr('x1', 140)
      .attr('y1', 6)
      .attr('x2', 158)
      .attr('y2', 6)
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3')

    legend
      .append('text')
      .attr('x', 164)
      .attr('y', 10)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('MLE')
  }, [n, effectiveK, priorStrength, alpha0, beta0, alphaPost, betaPost, mle, posteriorMean, ci])

  return (
    <div className="space-y-4">
      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Sliders */}
      <div className="space-y-3">
        {/* n slider */}
        <div className="space-y-1">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>
              Observations <span className="font-mono">n</span>
            </span>
            <span className="text-white font-mono tabular-nums">n = {n}</span>
          </label>
          <input
            type="range"
            min={0}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-full accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-gray-600 font-mono">
            <span>0</span>
            <span>50</span>
            <span>100</span>
            <span>150</span>
            <span>200</span>
          </div>
        </div>

        {/* k slider */}
        <div className="space-y-1">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>
              Admits <span className="font-mono">k</span>
            </span>
            <span className="text-white font-mono tabular-nums">
              k = {effectiveK}
              {k > n && (
                <span className="text-amber-400 ml-1">(clamped to n)</span>
              )}
            </span>
          </label>
          <input
            type="range"
            min={0}
            max={Math.max(n, 1)}
            step={1}
            value={effectiveK}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-full accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-gray-600 font-mono">
            <span>0</span>
            <span>{Math.round(n / 2)}</span>
            <span>{n}</span>
          </div>
        </div>

        {/* Prior strength slider */}
        <div className="space-y-1">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>
              Prior strength{' '}
              <M>{"\\alpha_0 + \\beta_0"}</M>
            </span>
            <span className="text-white font-mono tabular-nums">
              {priorStrength}{' '}
              <span className="text-gray-500">
                (<M>{"\\alpha_0"}</M>={alpha0.toFixed(1)}, <M>{"\\beta_0"}</M>=
                {beta0.toFixed(1)})
              </span>
            </span>
          </label>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={priorStrength}
            onChange={(e) => setPriorStrength(Number(e.target.value))}
            className="w-full accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-gray-600 font-mono">
            <span>1 (weak)</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100 (strong)</span>
          </div>
        </div>
      </div>

      {/* Stats below chart */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-gray-500">MLE: </span>
          <span className="text-amber-400 font-mono font-bold tabular-nums">
            {n > 0 ? `${effectiveK}/${n} = ${mle.toFixed(3)}` : 'N/A'}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Posterior mean: </span>
          <span className="text-emerald-400 font-mono font-bold tabular-nums">
            {posteriorMean.toFixed(3)}
          </span>
          <span className="text-gray-600 text-xs ml-1">(shrunk toward prior)</span>
        </div>
        <div>
          <span className="text-gray-500">95% CI: </span>
          <span className="text-emerald-400/80 font-mono font-bold tabular-nums">
            [{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-600">
        With only <span className="text-gray-400 font-mono">{n}</span>{' '}
        observations, the posterior is pulled{' '}
        <span className="text-emerald-400/80 font-mono font-bold">
          {shrinkagePct.toFixed(1)}%
        </span>{' '}
        toward the prior. More data &rarr; less shrinkage.
      </p>
    </div>
  )
}

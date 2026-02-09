import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Feature definitions & mock singular values
   ───────────────────────────────────────────── */
const FEATURES = [
  { name: 'Top-6 Avg', color: '#34d399', beta0: 0.08 },
  { name: 'Program', color: '#60a5fa', beta0: 0.15 },
  { name: 'Province', color: '#f472b6', beta0: 0.10 },
  { name: 'Interaction', color: '#fbbf24', beta0: 0.03 },
  { name: 'Term', color: '#a78bfa', beta0: -0.05 },
]

const SINGULAR_VALUES = [12, 8, 4, 2, 0.5]

/**
 * Ridge shrinkage formula:
 *   beta(lambda) = beta0 * sigma^2 / (sigma^2 + lambda)
 */
function ridgeCoeff(beta0: number, sigma: number, lambda: number): number {
  const s2 = sigma * sigma
  return beta0 * s2 / (s2 + lambda)
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function RidgePathExplorer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [logLambda, setLogLambda] = useState(0)

  const lambda = Math.pow(10, logLambda)

  // Generate path data: ~100 points per feature
  const pathData = useMemo(() => {
    const logMin = -3
    const logMax = 3
    const nPoints = 100

    return FEATURES.map((feat, i) => {
      const sigma = SINGULAR_VALUES[i]
      const points = d3.range(nPoints + 1).map((k) => {
        const logL = logMin + (logMax - logMin) * (k / nPoints)
        const lam = Math.pow(10, logL)
        return { logLambda: logL, beta: ridgeCoeff(feat.beta0, sigma, lam) }
      })
      return { ...feat, sigma, points }
    })
  }, [])

  // Current coefficient values at the dragged lambda
  const currentCoeffs = useMemo(() => {
    return FEATURES.map((feat, i) => ({
      name: feat.name,
      color: feat.color,
      beta: ridgeCoeff(feat.beta0, SINGULAR_VALUES[i], lambda),
    }))
  }, [lambda])

  // D3 rendering
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 300
    const margin = { top: 40, right: 20, bottom: 44, left: 50 }

    svg.attr('width', totalWidth).attr('height', height)

    const plotWidth = totalWidth - margin.left - margin.right
    const plotHeight = height - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales
    const xScale = d3.scaleLinear().domain([-3, 3]).range([0, plotWidth])
    const yExtent = d3.extent(pathData.flatMap((f) => f.points.map((p) => p.beta))) as [number, number]
    const yPadding = (yExtent[1] - yExtent[0]) * 0.15
    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([plotHeight, 0])

    // Gridlines (horizontal, subtle dashed)
    const yTicks = yScale.ticks(6)
    yTicks.forEach((tick) => {
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(tick))
        .attr('x2', plotWidth)
        .attr('y2', yScale(tick))
        .attr('stroke', 'rgba(255,255,255,0.06)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4')
    })

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${plotHeight})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .call((axis) => {
        axis.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        axis.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        axis.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    // X axis label
    g.append('text')
      .attr('x', plotWidth / 2)
      .attr('y', plotHeight + 36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '12px')
      .text('log\u2081\u2080(\u03BB)')

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(6).tickFormat(d3.format('.2f')))
      .call((axis) => {
        axis.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        axis.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.1)')
        axis.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -plotHeight / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '12px')
      .text('\u03B2 coefficient')

    // Zero line
    if (yScale.domain()[0] < 0 && yScale.domain()[1] > 0) {
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', plotWidth)
        .attr('y2', yScale(0))
        .attr('stroke', 'rgba(255,255,255,0.12)')
        .attr('stroke-width', 1)
    }

    // Feature coefficient paths
    const lineGenerator = d3
      .line<{ logLambda: number; beta: number }>()
      .x((d) => xScale(d.logLambda))
      .y((d) => yScale(d.beta))
      .curve(d3.curveBasis)

    pathData.forEach((feat) => {
      g.append('path')
        .datum(feat.points)
        .attr('d', lineGenerator)
        .attr('fill', 'none')
        .attr('stroke', feat.color)
        .attr('stroke-width', 2)
    })

    // Vertical draggable line at current lambda
    const clampedLog = Math.max(-3, Math.min(3, logLambda))
    const lineX = xScale(clampedLog)

    const dragLine = g
      .append('line')
      .attr('x1', lineX)
      .attr('y1', 0)
      .attr('x2', lineX)
      .attr('y2', plotHeight)
      .attr('stroke', 'rgba(255,255,255,0.3)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4')
      .style('cursor', 'ew-resize')

    // Invisible wider hit area for easier dragging
    const dragHitArea = g
      .append('rect')
      .attr('x', lineX - 10)
      .attr('y', 0)
      .attr('width', 20)
      .attr('height', plotHeight)
      .attr('fill', 'transparent')
      .style('cursor', 'ew-resize')

    // Dots on each curve at current lambda
    currentCoeffs.forEach((coeff) => {
      g.append('circle')
        .attr('cx', lineX)
        .attr('cy', yScale(coeff.beta))
        .attr('r', 5)
        .attr('fill', coeff.color)
        .attr('stroke', '#050505')
        .attr('stroke-width', 1.5)
    })

    // Tooltip panel showing coefficient values at current lambda
    const tooltipX = clampedLog > 1 ? lineX - 140 : lineX + 12
    const tooltipY = 4

    const tooltipG = g.append('g').attr('transform', `translate(${tooltipX},${tooltipY})`)

    tooltipG
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 126)
      .attr('height', currentCoeffs.length * 18 + 10)
      .attr('rx', 6)
      .attr('fill', 'rgba(5,5,5,0.85)')
      .attr('stroke', 'rgba(255,255,255,0.1)')
      .attr('stroke-width', 1)

    currentCoeffs.forEach((coeff, i) => {
      const rowY = 16 + i * 18

      tooltipG
        .append('circle')
        .attr('cx', 12)
        .attr('cy', rowY - 3)
        .attr('r', 4)
        .attr('fill', coeff.color)

      tooltipG
        .append('text')
        .attr('x', 22)
        .attr('y', rowY)
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .text(coeff.name)

      tooltipG
        .append('text')
        .attr('x', 120)
        .attr('y', rowY)
        .attr('text-anchor', 'end')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text(coeff.beta.toFixed(4))
    })

    // Legend at top
    let legendX = 0
    pathData.forEach((feat) => {
      const legendG = svg
        .append('g')
        .attr('transform', `translate(${margin.left + legendX},${14})`)

      legendG
        .append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', 4)
        .attr('fill', feat.color)

      legendG
        .append('text')
        .attr('x', 8)
        .attr('y', 4)
        .attr('fill', '#9ca3af')
        .attr('font-size', '11px')
        .text(feat.name)

      legendX += feat.name.length * 7 + 24
    })

    // Drag behavior
    const drag = d3.drag<SVGRectElement, unknown>().on('drag', (event) => {
      const [mouseX] = d3.pointer(event, g.node())
      const newLogLambda = xScale.invert(mouseX)
      const clamped = Math.max(-3, Math.min(3, newLogLambda))
      setLogLambda(clamped)
    })

    dragHitArea.call(drag)
    dragLine.call(
      d3.drag<SVGLineElement, unknown>().on('drag', (event) => {
        const [mouseX] = d3.pointer(event, g.node())
        const newLogLambda = xScale.invert(mouseX)
        const clamped = Math.max(-3, Math.min(3, newLogLambda))
        setLogLambda(clamped)
      }),
    )
  }, [logLambda, pathData, currentCoeffs])

  return (
    <div className="space-y-4">
      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" style={{ background: '#050505' }} />
      </div>

      {/* Info below chart */}
      <div className="flex items-center gap-6 text-sm">
        <div>
          <span className="text-gray-500">Current </span>
          <span className="text-white font-mono font-bold">
            {'\u03BB'} = {lambda.toFixed(4)}
          </span>
          <span className="text-gray-600 ml-2 text-xs">
            (log{'\u2081\u2080'} = {logLambda.toFixed(2)})
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-600">
        All coefficients {'\u2192'} 0 as {'\u03BB'} {'\u2192'} {'\u221E'} (total shrinkage)
      </p>
    </div>
  )
}

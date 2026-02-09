import { useRef, useEffect, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Discrete-time hazard model constants
   ───────────────────────────────────────────── */
const MONTH_LABELS = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
const BASE_HAZARD_ALPHA = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.0, 1.5, 2.0]

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
interface SurvivalCurveExplorerProps {
  grade?: number
}

export default function SurvivalCurveExplorer({ grade = 92.5 }: SurvivalCurveExplorerProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Compute hazard, survival, and CDF at each time point
  const curves = useMemo(() => {
    const gradeEffect = 0.08 * (grade - 87.3) / 4.2
    const points: { t: number; month: string; h: number; S: number; F: number }[] = []

    let survivalProd = 1.0

    for (let i = 0; i < BASE_HAZARD_ALPHA.length; i++) {
      const h = sigmoid(BASE_HAZARD_ALPHA[i] + gradeEffect)
      survivalProd *= (1 - h)
      const S = survivalProd
      const F = 1 - S

      points.push({
        t: i + 1,
        month: MONTH_LABELS[i],
        h,
        S,
        F,
      })
    }

    return points
  }, [grade])

  // Derived summary stats
  const medianMonth = useMemo(() => {
    const pt = curves.find((p) => p.F >= 0.5)
    return pt ? pt.month : 'after Jun'
  }, [curves])

  const eightyPctMonth = useMemo(() => {
    const pt = curves.find((p) => p.F >= 0.8)
    return pt ? pt.month : 'after Jun'
  }, [curves])

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

    // Background
    svg
      .append('rect')
      .attr('width', totalWidth)
      .attr('height', height)
      .attr('fill', '#050505')
      .attr('rx', 4)

    const plotWidth = totalWidth - margin.left - margin.right
    const plotHeight = height - margin.top - margin.bottom

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([1, 8])
      .range([margin.left, margin.left + plotWidth])

    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([margin.top + plotHeight, margin.top])

    // Grid lines
    const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yTicks.forEach((tick) => {
      svg
        .append('line')
        .attr('x1', margin.left)
        .attr('y1', yScale(tick))
        .attr('x2', margin.left + plotWidth)
        .attr('y2', yScale(tick))
        .attr('stroke', 'rgba(255,255,255,0.06)')
        .attr('stroke-width', 1)
    })

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${margin.top + plotHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickValues([1, 2, 3, 4, 5, 6, 7, 8])
          .tickFormat((_d, i) => MONTH_LABELS[i]),
      )
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.08)')
        g.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    // X axis label
    svg
      .append('text')
      .attr('x', margin.left + plotWidth / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Month')

    // Y axis
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.08)')
        g.selectAll('.tick text').attr('fill', '#9ca3af').attr('font-size', '11px')
      })

    // Y axis label
    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + plotHeight / 2))
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Probability')

    // Build data with initial point at t=0 for drawing step curves
    // S(0) = 1.0, F(0) = 0.0
    const survivalData: [number, number][] = [[0, 1.0], ...curves.map((p) => [p.t, p.S] as [number, number])]
    const cdfData: [number, number][] = [[0, 0.0], ...curves.map((p) => [p.t, p.F] as [number, number])]

    // Extend xScale domain to include t=0 for drawing (off-chart left edge)
    const xScaleExtended = d3
      .scaleLinear()
      .domain([0, 8])
      .range([margin.left - plotWidth / 8, margin.left + plotWidth])

    // Clip path so nothing renders outside the plot area
    svg
      .append('defs')
      .append('clipPath')
      .attr('id', 'plot-clip')
      .append('rect')
      .attr('x', margin.left)
      .attr('y', margin.top)
      .attr('width', plotWidth)
      .attr('height', plotHeight)

    const plotG = svg.append('g').attr('clip-path', 'url(#plot-clip)')

    // Area under S(t)
    const areaGen = d3
      .area<[number, number]>()
      .x((d) => xScaleExtended(d[0]))
      .y0(yScale(0))
      .y1((d) => yScale(d[1]))
      .curve(d3.curveStepAfter)

    plotG
      .append('path')
      .datum(survivalData)
      .attr('d', areaGen)
      .attr('fill', '#34d399')
      .attr('opacity', 0.1)

    // S(t) line
    const lineGen = d3
      .line<[number, number]>()
      .x((d) => xScaleExtended(d[0]))
      .y((d) => yScale(d[1]))
      .curve(d3.curveStepAfter)

    plotG
      .append('path')
      .datum(survivalData)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2.5)

    // F(t) line (dashed)
    plotG
      .append('path')
      .datum(cdfData)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '8,4')

    // Dots on S(t) at each time point
    plotG
      .selectAll('circle.survival-dot')
      .data(curves)
      .join('circle')
      .attr('class', 'survival-dot')
      .attr('cx', (d) => xScale(d.t))
      .attr('cy', (d) => yScale(d.S))
      .attr('r', 4)
      .attr('fill', '#34d399')
      .attr('stroke', '#050505')
      .attr('stroke-width', 1.5)

    // Dots on F(t) at each time point
    plotG
      .selectAll('circle.cdf-dot')
      .data(curves)
      .join('circle')
      .attr('class', 'cdf-dot')
      .attr('cx', (d) => xScale(d.t))
      .attr('cy', (d) => yScale(d.F))
      .attr('r', 4)
      .attr('fill', '#fbbf24')
      .attr('stroke', '#050505')
      .attr('stroke-width', 1.5)

    // Legend (top right)
    const legendX = margin.left + plotWidth - 140
    const legendY = margin.top + 6

    // S(t) legend entry
    svg
      .append('line')
      .attr('x1', legendX)
      .attr('y1', legendY)
      .attr('x2', legendX + 20)
      .attr('y2', legendY)
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2.5)

    svg
      .append('text')
      .attr('x', legendX + 26)
      .attr('y', legendY + 4)
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('S(t) Survival')

    // F(t) legend entry
    svg
      .append('line')
      .attr('x1', legendX)
      .attr('y1', legendY + 18)
      .attr('x2', legendX + 20)
      .attr('y2', legendY + 18)
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '8,4')

    svg
      .append('text')
      .attr('x', legendX + 26)
      .attr('y', legendY + 22)
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('F(t) CDF')

    // Tooltip group (hidden by default)
    const tooltip = svg
      .append('g')
      .attr('class', 'tooltip')
      .style('display', 'none')

    const tooltipRect = tooltip
      .append('rect')
      .attr('rx', 6)
      .attr('fill', 'rgba(0,0,0,0.85)')
      .attr('stroke', 'rgba(255,255,255,0.15)')
      .attr('stroke-width', 1)

    const tooltipText = tooltip
      .append('text')
      .attr('fill', '#d1d5db')
      .attr('font-size', '11px')

    const tooltipLine1 = tooltipText.append('tspan').attr('x', 10).attr('dy', '1.2em')
    const tooltipLine2 = tooltipText.append('tspan').attr('x', 10).attr('dy', '1.3em')
    const tooltipLine3 = tooltipText.append('tspan').attr('x', 10).attr('dy', '1.3em')
    const tooltipLine4 = tooltipText.append('tspan').attr('x', 10).attr('dy', '1.3em')

    // Invisible hover targets for each time point
    const hoverWidth = plotWidth / 8
    curves.forEach((p) => {
      svg
        .append('rect')
        .attr('x', xScale(p.t) - hoverWidth / 2)
        .attr('y', margin.top)
        .attr('width', hoverWidth)
        .attr('height', plotHeight)
        .attr('fill', 'transparent')
        .attr('cursor', 'crosshair')
        .on('mouseenter', () => {
          tooltip.style('display', null)

          tooltipLine1.text(p.month).attr('fill', '#fff').attr('font-weight', '700')
          tooltipLine2.text(`h(t) = ${p.h.toFixed(4)}`).attr('fill', '#9ca3af').attr('font-weight', '400')
          tooltipLine3.text(`S(t) = ${p.S.toFixed(4)}`).attr('fill', '#34d399').attr('font-weight', '400')
          tooltipLine4.text(`F(t) = ${p.F.toFixed(4)}`).attr('fill', '#fbbf24').attr('font-weight', '400')

          const tooltipW = 120
          const tooltipH = 78

          // Position tooltip; flip if too close to right edge
          let tx = xScale(p.t) + 12
          if (tx + tooltipW > totalWidth - margin.right) {
            tx = xScale(p.t) - tooltipW - 12
          }
          const ty = margin.top + 10

          tooltip.attr('transform', `translate(${tx},${ty})`)
          tooltipRect.attr('width', tooltipW).attr('height', tooltipH)

          // Vertical guide line
          svg
            .selectAll('.hover-guide')
            .data([p.t])
            .join('line')
            .attr('class', 'hover-guide')
            .attr('x1', xScale(p.t))
            .attr('y1', margin.top)
            .attr('x2', xScale(p.t))
            .attr('y2', margin.top + plotHeight)
            .attr('stroke', 'rgba(255,255,255,0.15)')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '3,3')
        })
        .on('mouseleave', () => {
          tooltip.style('display', 'none')
          svg.selectAll('.hover-guide').remove()
        })
    })
  }, [curves])

  return (
    <div className="space-y-3">
      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Text summary */}
      <div className="flex flex-wrap items-center gap-6 text-sm">
        <div>
          <span className="text-gray-500">Median wait time: </span>
          <span className="text-emerald-400 font-mono font-bold">{medianMonth}</span>
        </div>
        <div>
          <span className="text-gray-500">80% chance of hearing by: </span>
          <span className="text-emerald-400 font-mono font-bold">{eightyPctMonth}</span>
        </div>
      </div>
    </div>
  )
}

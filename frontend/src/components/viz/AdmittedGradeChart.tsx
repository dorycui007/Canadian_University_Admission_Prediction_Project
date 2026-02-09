import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { CompetitivenessByYear } from '@/types/api'
import { injectChartDefs } from '@/lib/chartDefs'
import { styleAxis, applyTextStyle } from '@/lib/chartStyles'
import { useChartTooltip } from '@/components/viz/ChartTooltip'
import useChartDimensions from '@/hooks/useChartDimensions'
import useReducedMotion from '@/hooks/useReducedMotion'

interface AdmittedGradeChartProps {
  data: CompetitivenessByYear[]
}

export default function AdmittedGradeChart({
  data,
}: AdmittedGradeChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const { containerRef, width } = useChartDimensions(220)
  const { containerRef: tooltipRef, show, hide, TooltipElement } = useChartTooltip()
  const reducedMotion = useReducedMotion()

  const wrapperRef = (el: HTMLDivElement | null) => {
    ;(containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el
    ;(tooltipRef as React.MutableRefObject<HTMLDivElement | null>).current = el
  }

  const valid = data.filter((d) => d.admitted_range !== null)

  useEffect(() => {
    const svgEl = svgRef.current
    if (!svgEl || width === 0 || valid.length === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const height = 220
    const margin = { top: 24, right: 16, bottom: 44, left: 48 }

    svg.attr('width', width).attr('height', height)
    injectChartDefs(svg)

    const x = d3
      .scaleBand()
      .domain(valid.map((d) => d.year))
      .range([margin.left, width - margin.right])
      .padding(0.4)

    const y = d3
      .scaleLinear()
      .domain([60, 100])
      .range([height - margin.bottom, margin.top])

    // ── Grid lines ──
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickFormat((d) => `${d}%`)
          .tickSize(-(width - margin.left - margin.right)),
      )
      .call((g) => styleAxis(g, { gridLines: true }))

    // ── X axis ──
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call((g) => styleAxis(g))

    // ── IQR range bars ──
    const iqrBars = svg
      .selectAll('rect.iqr')
      .data(valid)
      .join('rect')
      .attr('class', 'iqr')
      .attr('x', (d) => x(d.year)!)
      .attr('width', x.bandwidth())
      .attr('fill', 'url(#emerald-bar-gradient)')
      .attr('rx', 4)
      .style('cursor', 'pointer')

    if (reducedMotion) {
      iqrBars
        .attr('y', (d) => y(d.admitted_range!.p75))
        .attr('height', (d) =>
          Math.max(0, y(d.admitted_range!.p25) - y(d.admitted_range!.p75)),
        )
        .attr('opacity', (d) => (d.total_reports < 10 ? 0.3 : 0.55))
    } else {
      iqrBars
        .attr('y', (d) => y(d.admitted_range!.median))
        .attr('height', 0)
        .attr('opacity', 0)
        .transition()
        .duration(600)
        .delay((_, i) => i * 100)
        .ease(d3.easeCubicOut)
        .attr('y', (d) => y(d.admitted_range!.p75))
        .attr('height', (d) =>
          Math.max(0, y(d.admitted_range!.p25) - y(d.admitted_range!.p75)),
        )
        .attr('opacity', (d) => (d.total_reports < 10 ? 0.3 : 0.55))
    }

    // ── Median line ──
    const medianLines = svg
      .selectAll('line.median')
      .data(valid)
      .join('line')
      .attr('class', 'median')
      .attr('x1', (d) => x(d.year)!)
      .attr('x2', (d) => x(d.year)! + x.bandwidth())
      .attr('y1', (d) => y(d.admitted_range!.median))
      .attr('y2', (d) => y(d.admitted_range!.median))
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2.5)
      .style('pointer-events', 'none')

    if (reducedMotion) {
      medianLines.attr('opacity', (d) => (d.total_reports < 10 ? 0.5 : 1))
    } else {
      medianLines
        .attr('opacity', 0)
        .transition()
        .duration(400)
        .delay((_, i) => 600 + i * 100)
        .attr('opacity', (d) => (d.total_reports < 10 ? 0.5 : 1))
    }

    // ── Whisker lines ──
    const whiskerDelay = reducedMotion ? 0 : 800

    // Low whisker (min to p25)
    svg
      .selectAll('line.whisker-low')
      .data(valid)
      .join('line')
      .attr('class', 'whisker-low')
      .attr('x1', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('x2', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('y1', (d) => y(d.admitted_range!.min))
      .attr('y2', (d) => y(d.admitted_range!.p25))
      .attr('stroke', '#6ee7b7')
      .attr('stroke-width', 1)
      .attr('opacity', reducedMotion ? 0.5 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 400)
      .delay((_, i) => whiskerDelay + i * 80)
      .attr('opacity', 0.5)

    // High whisker (p75 to max)
    svg
      .selectAll('line.whisker-high')
      .data(valid)
      .join('line')
      .attr('class', 'whisker-high')
      .attr('x1', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('x2', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('y1', (d) => y(d.admitted_range!.p75))
      .attr('y2', (d) => y(d.admitted_range!.max))
      .attr('stroke', '#6ee7b7')
      .attr('stroke-width', 1)
      .attr('opacity', reducedMotion ? 0.5 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 400)
      .delay((_, i) => whiskerDelay + i * 80)
      .attr('opacity', 0.5)

    // Whisker caps (min)
    svg
      .selectAll('line.cap-min')
      .data(valid)
      .join('line')
      .attr('class', 'cap-min')
      .attr('x1', (d) => x(d.year)! + x.bandwidth() / 2 - 4)
      .attr('x2', (d) => x(d.year)! + x.bandwidth() / 2 + 4)
      .attr('y1', (d) => y(d.admitted_range!.min))
      .attr('y2', (d) => y(d.admitted_range!.min))
      .attr('stroke', '#6ee7b7')
      .attr('stroke-width', 1)
      .attr('opacity', reducedMotion ? 0.5 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 400)
      .delay((_, i) => whiskerDelay + i * 80)
      .attr('opacity', 0.5)

    // Whisker caps (max)
    svg
      .selectAll('line.cap-max')
      .data(valid)
      .join('line')
      .attr('class', 'cap-max')
      .attr('x1', (d) => x(d.year)! + x.bandwidth() / 2 - 4)
      .attr('x2', (d) => x(d.year)! + x.bandwidth() / 2 + 4)
      .attr('y1', (d) => y(d.admitted_range!.max))
      .attr('y2', (d) => y(d.admitted_range!.max))
      .attr('stroke', '#6ee7b7')
      .attr('stroke-width', 1)
      .attr('opacity', reducedMotion ? 0.5 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 400)
      .delay((_, i) => whiskerDelay + i * 80)
      .attr('opacity', 0.5)

    // ── Median label above box ──
    svg
      .selectAll('text.median-label')
      .data(valid)
      .join('text')
      .attr('class', 'median-label')
      .attr('x', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('y', (d) => y(d.admitted_range!.max) - 6)
      .attr('text-anchor', 'middle')
      .call((s) => applyTextStyle(s, 'value'))
      .attr('fill', '#10b981')
      .text((d) => `${d.admitted_range!.median}`)

    // ── Sample size labels ──
    svg
      .selectAll('text.n-label')
      .data(valid)
      .join('text')
      .attr('class', 'n-label')
      .attr('x', (d) => x(d.year)! + x.bandwidth() / 2)
      .attr('y', height - margin.bottom + 30)
      .attr('text-anchor', 'middle')
      .call((s) => applyTextStyle(s, 'muted'))
      .text((d) => `n=${d.admitted_range!.n}`)

    // ── Hover targets (invisible rects over each year column) ──
    svg
      .selectAll('rect.hover-target')
      .data(valid)
      .join('rect')
      .attr('class', 'hover-target')
      .attr('x', (d) => x(d.year)! - x.step() * x.padding() / 2)
      .attr('y', margin.top)
      .attr('width', x.step())
      .attr('height', height - margin.top - margin.bottom)
      .attr('fill', 'none')
      .style('pointer-events', 'all')
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) {
        // Dim all IQR bars
        iqrBars.transition().duration(150).attr('opacity', 0.12)
        medianLines.transition().duration(150).attr('opacity', 0.2)

        // Brighten hovered
        const idx = valid.indexOf(d)
        d3.select(iqrBars.nodes()[idx])
          .transition()
          .duration(150)
          .attr('opacity', 0.85)
          .attr('filter', 'url(#emerald-glow)')
        d3.select(medianLines.nodes()[idx])
          .transition()
          .duration(150)
          .attr('opacity', 1)

        const ar = d.admitted_range!
        show(
          event,
          <div className="space-y-0.5">
            <div className="text-white font-semibold text-sm">{d.year}</div>
            <div className="text-gray-400">
              Max: <span className="text-white">{ar.max}%</span>
            </div>
            <div className="text-gray-400">
              P75: <span className="text-white">{ar.p75}%</span>
            </div>
            <div className="text-emerald-400 font-medium">
              Median: {ar.median}%
            </div>
            <div className="text-gray-400">
              P25: <span className="text-white">{ar.p25}%</span>
            </div>
            <div className="text-gray-400">
              Min: <span className="text-white">{ar.min}%</span>
            </div>
            <div className="pt-1 border-t border-white/10 text-gray-500 text-[10px]">
              n = {ar.n} reports
            </div>
          </div>,
        )
      })
      .on('mousemove', function (event, d) {
        const ar = d.admitted_range!
        show(
          event,
          <div className="space-y-0.5">
            <div className="text-white font-semibold text-sm">{d.year}</div>
            <div className="text-gray-400">
              Max: <span className="text-white">{ar.max}%</span>
            </div>
            <div className="text-gray-400">
              P75: <span className="text-white">{ar.p75}%</span>
            </div>
            <div className="text-emerald-400 font-medium">
              Median: {ar.median}%
            </div>
            <div className="text-gray-400">
              P25: <span className="text-white">{ar.p25}%</span>
            </div>
            <div className="text-gray-400">
              Min: <span className="text-white">{ar.min}%</span>
            </div>
            <div className="pt-1 border-t border-white/10 text-gray-500 text-[10px]">
              n = {ar.n} reports
            </div>
          </div>,
        )
      })
      .on('mouseleave', function () {
        iqrBars
          .transition()
          .duration(300)
          .attr('opacity', (d) => (d.total_reports < 10 ? 0.3 : 0.55))
          .attr('filter', null)
        medianLines
          .transition()
          .duration(300)
          .attr('opacity', (d) => (d.total_reports < 10 ? 0.5 : 1))
        hide()
      })
  }, [valid, width, reducedMotion, show, hide])

  if (valid.length === 0) {
    return <p className="text-sm text-gray-500">No admitted grade data available.</p>
  }

  return (
    <div ref={wrapperRef} className="relative w-full" onMouseLeave={hide}>
      <svg ref={svgRef} width="100%" height={220} />
      {TooltipElement}
    </div>
  )
}

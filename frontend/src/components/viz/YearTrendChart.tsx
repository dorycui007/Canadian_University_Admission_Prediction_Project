import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { injectChartDefs } from '@/lib/chartDefs'
import { styleAxis, applyTextStyle } from '@/lib/chartStyles'
import { useChartTooltip } from '@/components/viz/ChartTooltip'
import useChartDimensions from '@/hooks/useChartDimensions'
import useReducedMotion from '@/hooks/useReducedMotion'

interface YearTrendPoint {
  year: number
  median_grade: number
}

interface YearTrendChartProps {
  data: YearTrendPoint[]
  studentGrade?: number
}

export default function YearTrendChart({
  data,
  studentGrade,
}: YearTrendChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const { containerRef, width } = useChartDimensions(200)
  const { containerRef: tooltipRef, show, hide, TooltipElement } = useChartTooltip()
  const reducedMotion = useReducedMotion()

  const wrapperRef = (el: HTMLDivElement | null) => {
    ;(containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el
    ;(tooltipRef as React.MutableRefObject<HTMLDivElement | null>).current = el
  }

  useEffect(() => {
    const svgEl = svgRef.current
    if (!svgEl || width === 0 || data.length === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const height = 200
    const margin = { top: 20, right: 20, bottom: 36, left: 44 }

    svg.attr('width', width).attr('height', height)
    injectChartDefs(svg)

    const sorted = [...data].sort((a, b) => a.year - b.year)

    const xExtent = d3.extent(sorted, (d) => d.year) as [number, number]
    const yExtent = d3.extent(sorted, (d) => d.median_grade) as [number, number]

    let yMin = yExtent[0]
    let yMax = yExtent[1]
    if (studentGrade !== undefined) {
      yMin = Math.min(yMin, studentGrade)
      yMax = Math.max(yMax, studentGrade)
    }
    const yPad = (yMax - yMin) * 0.15 || 2
    yMin = Math.max(0, yMin - yPad)
    yMax = yMax + yPad

    const x = d3
      .scaleLinear()
      .domain(xExtent)
      .range([margin.left, width - margin.right])

    const y = d3
      .scaleLinear()
      .domain([yMin, yMax])
      .range([height - margin.bottom, margin.top])

    // ── Grid lines ──
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickSize(-(width - margin.left - margin.right)),
      )
      .call((g) => styleAxis(g, { gridLines: true }))

    // ── X axis ──
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(
        d3
          .axisBottom(x)
          .ticks(Math.min(sorted.length, 8))
          .tickFormat((d) => String(d))
          .tickSize(0),
      )
      .call((g) => styleAxis(g))

    // ── Axis labels (editorial placement) ──
    svg
      .append('text')
      .attr('x', (margin.left + width - margin.right) / 2)
      .attr('y', height - 2)
      .attr('text-anchor', 'middle')
      .call((s) => applyTextStyle(s, 'axisLabel'))
      .text('Year')

    svg
      .append('text')
      .attr('x', margin.left)
      .attr('y', margin.top - 8)
      .attr('text-anchor', 'start')
      .call((s) => applyTextStyle(s, 'axisLabel'))
      .text('Median Grade')

    // ── Gradient area fill under the line ──
    const area = d3
      .area<YearTrendPoint>()
      .x((d) => x(d.year))
      .y0(height - margin.bottom)
      .y1((d) => y(d.median_grade))
      .curve(d3.curveMonotoneX)

    const areaPath = svg
      .append('path')
      .datum(sorted)
      .attr('d', area)
      .attr('fill', 'url(#emerald-area-gradient)')

    if (!reducedMotion) {
      areaPath
        .attr('opacity', 0)
        .transition()
        .delay(800)
        .duration(400)
        .attr('opacity', 1)
    }

    // ── Line ──
    const line = d3
      .line<YearTrendPoint>()
      .x((d) => x(d.year))
      .y((d) => y(d.median_grade))
      .curve(d3.curveMonotoneX)

    const linePath = svg
      .append('path')
      .datum(sorted)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    if (!reducedMotion) {
      const totalLength = (linePath.node() as SVGPathElement).getTotalLength()
      linePath
        .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(1200)
        .ease(d3.easeCubicInOut)
        .attr('stroke-dashoffset', 0)
    }

    // ── Static dots (fade in after line draws) ──
    svg
      .selectAll('circle.dot')
      .data(sorted)
      .join('circle')
      .attr('class', 'dot')
      .attr('cx', (d) => x(d.year))
      .attr('cy', (d) => y(d.median_grade))
      .attr('fill', '#10b981')
      .attr('stroke', '#050505')
      .attr('stroke-width', 1.5)
      .style('pointer-events', 'none')

    if (reducedMotion) {
      svg.selectAll('circle.dot').attr('r', 4)
    } else {
      svg
        .selectAll('circle.dot')
        .attr('r', 0)
        .transition()
        .duration(300)
        .delay((_, i) => 1200 + i * 100)
        .attr('r', 4)
    }

    // ── Hover: glowing dot + vertical crosshair ──
    const hoverDot = svg
      .append('circle')
      .attr('r', 6)
      .attr('fill', '#10b981')
      .attr('filter', 'url(#emerald-glow-intense)')
      .attr('opacity', 0)
      .style('pointer-events', 'none')

    const hoverLine = svg
      .append('line')
      .attr('stroke', 'rgba(255,255,255,0.12)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4')
      .attr('opacity', 0)
      .style('pointer-events', 'none')

    const bisect = d3.bisector<YearTrendPoint, number>((d) => d.year).left

    svg
      .append('rect')
      .attr('class', 'overlay')
      .attr('x', margin.left)
      .attr('y', margin.top)
      .attr('width', width - margin.left - margin.right)
      .attr('height', height - margin.top - margin.bottom)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .style('cursor', 'crosshair')
      .on('mousemove', function (event) {
        const [mx] = d3.pointer(event)
        const xVal = x.invert(mx)
        const idx = bisect(sorted, xVal, 1)
        const d0 = sorted[idx - 1]
        const d1 = sorted[idx]
        const d = d1 && xVal - d0.year > d1.year - xVal ? d1 : d0
        if (!d) return

        hoverDot
          .attr('cx', x(d.year))
          .attr('cy', y(d.median_grade))
          .attr('opacity', 1)

        hoverLine
          .attr('x1', x(d.year))
          .attr('x2', x(d.year))
          .attr('y1', margin.top)
          .attr('y2', height - margin.bottom)
          .attr('opacity', 1)

        show(
          event,
          <div className="space-y-0.5">
            <div className="text-white font-semibold">{d.year}</div>
            <div className="text-gray-400">
              Median: <span className="text-emerald-400 font-medium">{d.median_grade}%</span>
            </div>
          </div>,
        )
      })
      .on('mouseleave', function () {
        hoverDot.attr('opacity', 0)
        hoverLine.attr('opacity', 0)
        hide()
      })

    // ── Student grade horizontal line ──
    if (studentGrade !== undefined) {
      const gy = y(studentGrade)

      svg
        .append('line')
        .attr('x1', margin.left)
        .attr('x2', width - margin.right)
        .attr('y1', gy)
        .attr('y2', gy)
        .attr('stroke', '#fbbf24')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,4')
        .style('pointer-events', 'none')

      svg
        .append('text')
        .attr('x', width - margin.right - 4)
        .attr('y', gy - 6)
        .attr('text-anchor', 'end')
        .attr('fill', '#fbbf24')
        .attr('font-size', '10px')
        .attr('font-weight', '600')
        .text('Your Grade')
    }
  }, [data, studentGrade, width, reducedMotion, show, hide])

  if (data.length === 0) {
    return (
      <p className="text-sm text-gray-500">No trend data available.</p>
    )
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <svg ref={svgRef} width="100%" height={200} />
      {TooltipElement}
    </div>
  )
}

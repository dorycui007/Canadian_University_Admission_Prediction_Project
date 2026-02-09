import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { DistributionData } from '@/types/api'
import { injectChartDefs, roundedTopRect } from '@/lib/chartDefs'
import { styleAxis, applyTextStyle } from '@/lib/chartStyles'
import { useChartTooltip } from '@/components/viz/ChartTooltip'
import useChartDimensions from '@/hooks/useChartDimensions'
import useReducedMotion from '@/hooks/useReducedMotion'

interface GradeHistogramProps {
  data: DistributionData
  studentGrade?: number
}

export default function GradeHistogram({
  data,
  studentGrade,
}: GradeHistogramProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const { containerRef, width } = useChartDimensions(250)
  const { containerRef: tooltipRef, show, hide, TooltipElement } = useChartTooltip()
  const reducedMotion = useReducedMotion()

  // Merge both refs onto the same div
  const wrapperRef = (el: HTMLDivElement | null) => {
    ;(containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el
    ;(tooltipRef as React.MutableRefObject<HTMLDivElement | null>).current = el
  }

  useEffect(() => {
    const svgEl = svgRef.current
    if (!svgEl || width === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const { bins, counts_accepted, counts_rejected } = data
    if (bins.length < 2) return

    const height = 250
    const margin = { top: 20, right: 16, bottom: 36, left: 44 }

    svg.attr('width', width).attr('height', height)
    injectChartDefs(svg)

    const binCount = bins.length - 1
    const binData = Array.from({ length: binCount }, (_, i) => ({
      x0: bins[i],
      x1: bins[i + 1],
      accepted: counts_accepted[i] ?? 0,
      rejected: counts_rejected[i] ?? 0,
    }))

    if (binData.length === 0) return

    // Dynamic domain from actual data range
    const xMin = binData[0].x0
    const xMax = binData[binData.length - 1].x1

    const x = d3
      .scaleLinear()
      .domain([xMin, xMax])
      .range([margin.left, width - margin.right])

    const yMax = d3.max(binData, (d) => d.accepted + d.rejected) ?? 1
    const y = d3
      .scaleLinear()
      .domain([0, yMax])
      .nice()
      .range([height - margin.bottom, margin.top])

    // ── Grid lines (Y axis extends as dashed lines) ──
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
        d3.axisBottom(x)
          .tickValues(
            (() => {
              const span = xMax - xMin
              const step = span <= 20 ? 2 : span <= 40 ? 4 : 5
              const first = Math.ceil(xMin / step) * step
              return d3.range(first, xMax + 1, step)
            })(),
          )
          .tickSize(0),
      )
      .call((g) => styleAxis(g))

    // ── Axis labels ──
    svg
      .append('text')
      .attr('x', (margin.left + width - margin.right) / 2)
      .attr('y', height - 2)
      .attr('text-anchor', 'middle')
      .call((s) => applyTextStyle(s, 'axisLabel'))
      .text('Grade (%)')

    svg
      .append('text')
      .attr('x', margin.left)
      .attr('y', margin.top - 8)
      .attr('text-anchor', 'start')
      .call((s) => applyTextStyle(s, 'axisLabel'))
      .text('Count')

    // ── Crosshair line (hidden by default) ──
    const crosshair = svg
      .append('line')
      .attr('class', 'crosshair')
      .attr('x1', margin.left)
      .attr('x2', width - margin.right)
      .attr('stroke', 'rgba(255,255,255,0.12)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4')
      .attr('opacity', 0)
      .style('pointer-events', 'none')

    // ── Rejected bars (bottom layer) ──
    const rejectedBars = svg
      .selectAll('rect.rejected')
      .data(binData)
      .join('rect')
      .attr('class', 'rejected')
      .attr('x', (d) => x(d.x0) + 1)
      .attr('width', (d) => Math.max(0, x(d.x1) - x(d.x0) - 2))
      .attr('fill', 'url(#red-bar-gradient)')
      .attr('rx', 2)
      .style('cursor', 'pointer')

    if (reducedMotion) {
      rejectedBars
        .attr('y', (d) => y(d.rejected))
        .attr('height', (d) => Math.max(0, y(0) - y(d.rejected)))
        .attr('opacity', 0.55)
    } else {
      rejectedBars
        .attr('y', height - margin.bottom)
        .attr('height', 0)
        .attr('opacity', 0)
        .transition()
        .duration(600)
        .delay((_, i) => i * 25)
        .ease(d3.easeCubicOut)
        .attr('y', (d) => y(d.rejected))
        .attr('height', (d) => Math.max(0, y(0) - y(d.rejected)))
        .attr('opacity', 0.55)
    }

    // ── Accepted bars (stacked on top, rounded top corners) ──
    const acceptedBars = svg
      .selectAll('path.accepted')
      .data(binData)
      .join('path')
      .attr('class', 'accepted')
      .attr('fill', 'url(#emerald-bar-gradient)')
      .style('cursor', 'pointer')

    if (reducedMotion) {
      acceptedBars
        .attr('d', (d) =>
          roundedTopRect(
            x(d.x0) + 1,
            y(d.accepted + d.rejected),
            Math.max(0, x(d.x1) - x(d.x0) - 2),
            Math.max(0, y(d.rejected) - y(d.accepted + d.rejected)),
            2,
          ),
        )
        .attr('opacity', 0.65)
    } else {
      acceptedBars
        .attr('d', (d) =>
          roundedTopRect(
            x(d.x0) + 1,
            height - margin.bottom,
            Math.max(0, x(d.x1) - x(d.x0) - 2),
            0,
            2,
          ),
        )
        .attr('opacity', 0)
        .transition()
        .duration(600)
        .delay((_, i) => i * 25)
        .ease(d3.easeCubicOut)
        .attr('d', (d) =>
          roundedTopRect(
            x(d.x0) + 1,
            y(d.accepted + d.rejected),
            Math.max(0, x(d.x1) - x(d.x0) - 2),
            Math.max(0, y(d.rejected) - y(d.accepted + d.rejected)),
            2,
          ),
        )
        .attr('opacity', 0.65)
    }

    // ── Hover interaction (invisible rects over each bin) ──
    svg
      .selectAll('rect.hover-target')
      .data(binData)
      .join('rect')
      .attr('class', 'hover-target')
      .attr('x', (d) => x(d.x0))
      .attr('y', margin.top)
      .attr('width', (d) => Math.max(0, x(d.x1) - x(d.x0)))
      .attr('height', height - margin.top - margin.bottom)
      .attr('fill', 'none')
      .style('pointer-events', 'all')
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) {
        // Dim all bars
        rejectedBars.transition().duration(150).attr('opacity', 0.15)
        acceptedBars.transition().duration(150).attr('opacity', 0.15)

        // Brighten this bin's bars
        const idx = binData.indexOf(d)
        d3.select(rejectedBars.nodes()[idx])
          .transition()
          .duration(150)
          .attr('opacity', 0.85)
          .attr('filter', 'url(#emerald-glow)')
        d3.select(acceptedBars.nodes()[idx])
          .transition()
          .duration(150)
          .attr('opacity', 1)
          .attr('filter', 'url(#emerald-glow)')

        // Show crosshair
        crosshair
          .attr('y1', y(d.accepted + d.rejected))
          .attr('y2', y(d.accepted + d.rejected))
          .attr('opacity', 1)

        // Tooltip
        const total = d.accepted + d.rejected
        show(
          event,
          <div className="space-y-1">
            <div className="text-white font-semibold text-sm">
              {d.x0}% &ndash; {d.x1}%
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
              <span className="text-white">Accepted</span>
              <span className="text-white font-medium ml-auto">{d.accepted}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
              <span className="text-white">Rejected</span>
              <span className="text-white font-medium ml-auto">{d.rejected}</span>
            </div>
            <div className="pt-1 border-t border-white/10 text-gray-300">
              {total} total applicant{total !== 1 ? 's' : ''}
            </div>
          </div>,
        )
      })
      .on('mousemove', function (event, d) {
        const total = d.accepted + d.rejected
        show(
          event,
          <div className="space-y-1">
            <div className="text-white font-semibold text-sm">
              {d.x0}% &ndash; {d.x1}%
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
              <span className="text-white">Accepted</span>
              <span className="text-white font-medium ml-auto">{d.accepted}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
              <span className="text-white">Rejected</span>
              <span className="text-white font-medium ml-auto">{d.rejected}</span>
            </div>
            <div className="pt-1 border-t border-white/10 text-gray-300">
              {total} total applicant{total !== 1 ? 's' : ''}
            </div>
          </div>,
        )
      })
      .on('mouseleave', function () {
        rejectedBars
          .transition()
          .duration(300)
          .attr('opacity', 0.55)
          .attr('filter', null)
        acceptedBars
          .transition()
          .duration(300)
          .attr('opacity', 0.65)
          .attr('filter', null)
        crosshair.attr('opacity', 0)
        hide()
      })

    // ── Student grade marker ──
    if (studentGrade !== undefined) {
      const gx = x(studentGrade)

      svg
        .append('line')
        .attr('x1', gx)
        .attr('x2', gx)
        .attr('y1', margin.top)
        .attr('y2', height - margin.bottom)
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '2,4')
        .style('pointer-events', 'none')

      svg
        .append('text')
        .attr('x', gx + 5)
        .attr('y', margin.top + 12)
        .attr('fill', '#ef4444')
        .attr('font-size', '10px')
        .attr('font-weight', '600')
        .text('You')
    }

  }, [data, studentGrade, width, reducedMotion, show, hide])

  if (data.bins.length < 2) {
    return (
      <p className="text-sm text-gray-500">No distribution data available.</p>
    )
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <svg ref={svgRef} width="100%" height={250} />
      <div className="flex justify-center gap-5 mt-2">
        <div className="flex items-center gap-1.5 text-xs">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-emerald-400 opacity-70" />
          <span className="text-gray-400">Accepted</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-400 opacity-70" />
          <span className="text-gray-400">Rejected</span>
        </div>
      </div>
      {TooltipElement}
    </div>
  )
}

import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { OfferByMonth } from '@/types/api'
import { injectChartDefs } from '@/lib/chartDefs'
import { applyTextStyle } from '@/lib/chartStyles'
import { useChartTooltip } from '@/components/viz/ChartTooltip'
import useChartDimensions from '@/hooks/useChartDimensions'
import useReducedMotion from '@/hooks/useReducedMotion'

interface OfferTimelineChartProps {
  data: OfferByMonth[]
}

export default function OfferTimelineChart({ data }: OfferTimelineChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const totalHeight = 16 + data.length * 30
  const { containerRef, width } = useChartDimensions(totalHeight)
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

    const barHeight = 24
    const gap = 6
    const margin = { top: 8, right: 64, bottom: 8, left: 40 }
    const height = margin.top + margin.bottom + data.length * (barHeight + gap)

    svg.attr('width', width).attr('height', height)
    injectChartDefs(svg)

    const total = d3.sum(data, (d) => d.count)
    const maxCount = d3.max(data, (d) => d.count) || 1

    const y = d3
      .scaleBand()
      .domain(data.map((d) => d.month))
      .range([margin.top, height - margin.bottom])
      .padding(0.15)

    const x = d3
      .scaleLinear()
      .domain([0, maxCount])
      .range([margin.left, width - margin.right])

    // ── Month labels ──
    svg
      .selectAll('text.month')
      .data(data)
      .join('text')
      .attr('class', 'month')
      .attr('x', margin.left - 6)
      .attr('y', (d) => y(d.month)! + y.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'central')
      .call((s) => applyTextStyle(s, 'axisLabel'))

      .text((d) => d.month)

    // ── Bars ──
    const bars = svg
      .selectAll('rect.bar')
      .data(data)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', margin.left)
      .attr('y', (d) => y(d.month)!)
      .attr('height', y.bandwidth())
      .attr('fill', 'url(#emerald-bar-gradient)')
      .attr('rx', 4)
      .style('cursor', 'pointer')

    if (reducedMotion) {
      bars
        .attr('width', (d) => Math.max(0, x(d.count) - margin.left))
        .attr('opacity', 0.65)
    } else {
      bars
        .attr('width', 0)
        .attr('opacity', 0.3)
        .transition()
        .duration(500)
        .delay((_, i) => i * 60)
        .ease(d3.easeCubicOut)
        .attr('width', (d) => Math.max(0, x(d.count) - margin.left))
        .attr('opacity', 0.65)
    }

    // ── Count labels ──
    svg
      .selectAll('text.count')
      .data(data)
      .join('text')
      .attr('class', 'count')
      .attr('x', (d) => x(d.count) + 6)
      .attr('y', (d) => y(d.month)! + y.bandwidth() / 2)
      .attr('dominant-baseline', 'central')
      .call((s) => applyTextStyle(s, 'value'))
      .text((d) => d.count)
      .attr('opacity', reducedMotion ? 1 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 300)
      .delay((_, i) => (reducedMotion ? 0 : 500 + i * 60))
      .attr('opacity', 1)

    // ── Percentage labels ──
    svg
      .selectAll('text.pct')
      .data(data)
      .join('text')
      .attr('class', 'pct')
      .attr('x', (d) => x(d.count) + 30)
      .attr('y', (d) => y(d.month)! + y.bandwidth() / 2)
      .attr('dominant-baseline', 'central')
      .call((s) => applyTextStyle(s, 'muted'))
      .text((d) => total > 0 ? `${((d.count / total) * 100).toFixed(0)}%` : '')
      .attr('opacity', reducedMotion ? 1 : 0)
      .transition()
      .duration(reducedMotion ? 0 : 300)
      .delay((_, i) => (reducedMotion ? 0 : 500 + i * 60))
      .attr('opacity', 1)

    // ── Hover interaction ──
    bars
      .on('mouseenter', function (event, d) {
        bars.transition().duration(150).attr('opacity', 0.2)
        d3.select(this)
          .transition()
          .duration(150)
          .attr('opacity', 1)
          .attr('filter', 'url(#emerald-glow)')

        const pct = total > 0 ? ((d.count / total) * 100).toFixed(1) : '0'
        show(
          event,
          <div className="space-y-1">
            <div className="text-white font-semibold">{d.month}</div>
            <div className="text-gray-400">
              Decisions: <span className="text-white font-medium">{d.count}</span>
            </div>
            <div className="text-gray-400">
              Share: <span className="text-white font-medium">{pct}%</span>
            </div>
          </div>,
        )
      })
      .on('mousemove', function (event, d) {
        const pct = total > 0 ? ((d.count / total) * 100).toFixed(1) : '0'
        show(
          event,
          <div className="space-y-1">
            <div className="text-white font-semibold">{d.month}</div>
            <div className="text-gray-400">
              Decisions: <span className="text-white font-medium">{d.count}</span>
            </div>
            <div className="text-gray-400">
              Share: <span className="text-white font-medium">{pct}%</span>
            </div>
          </div>,
        )
      })
      .on('mouseleave', function () {
        bars
          .transition()
          .duration(300)
          .attr('opacity', 0.65)
          .attr('filter', null)
        hide()
      })
  }, [data, width, reducedMotion, show, hide])

  if (data.length === 0) {
    return <p className="text-sm text-gray-500">No decision date data available.</p>
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <svg ref={svgRef} width="100%" height={totalHeight} />
      {TooltipElement}
    </div>
  )
}

import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { injectChartDefs } from '@/lib/chartDefs'
import { useChartTooltip } from '@/components/viz/ChartTooltip'
import useReducedMotion from '@/hooks/useReducedMotion'

interface DecisionPieChartProps {
  data: Record<string, number>
}

const DECISION_COLORS: Record<string, string> = {
  Accepted: '#10b981',
  Rejected: '#f87171',
  Waitlisted: '#fbbf24',
  Deferred: '#60a5fa',
  Pending: '#8b5cf6',
  Withdrawn: '#6b7280',
}

export default function DecisionPieChart({ data }: DecisionPieChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const { containerRef: tooltipRef, show, hide, TooltipElement } = useChartTooltip()
  const reducedMotion = useReducedMotion()

  const wrapperRef = (el: HTMLDivElement | null) => {
    ;(containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el
    ;(tooltipRef as React.MutableRefObject<HTMLDivElement | null>).current = el
  }

  useEffect(() => {
    const container = containerRef.current
    const svgEl = svgRef.current
    if (!container || !svgEl) return

    const entries = Object.entries(data).filter(([, v]) => v > 0)
    if (entries.length === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const size = Math.min(container.clientWidth, 280)
    const radius = size / 2
    const innerRadius = radius * 0.55
    const outerRadius = radius * 0.88

    svg.attr('width', size).attr('height', size)

    const defs = injectChartDefs(svg)

    // ── Per-slice radial gradients ──
    entries.forEach(([label], i) => {
      const base = DECISION_COLORS[label] || '#6b7280'
      const lighter = d3.color(base)?.brighter(0.4)?.formatHex() ?? base

      const grad = defs
        .append('radialGradient')
        .attr('id', `slice-grad-${i}`)

      grad
        .append('stop')
        .attr('offset', '30%')
        .attr('stop-color', lighter)

      grad
        .append('stop')
        .attr('offset', '100%')
        .attr('stop-color', base)
    })

    const g = svg
      .append('g')
      .attr('transform', `translate(${radius},${radius})`)

    const total = entries.reduce((sum, [, v]) => sum + v, 0)

    const pie = d3
      .pie<[string, number]>()
      .value((d) => d[1])
      .sort(null)
      .padAngle(0.02)

    const arc = d3
      .arc<d3.PieArcDatum<[string, number]>>()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .cornerRadius(3)

    const arcHover = d3
      .arc<d3.PieArcDatum<[string, number]>>()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius + 8)
      .cornerRadius(3)

    // ── Slices ──
    const slices = g
      .selectAll('path')
      .data(pie(entries))
      .join('path')
      .attr('fill', (_, i) => `url(#slice-grad-${i})`)
      .attr('opacity', 0.85)
      .style('cursor', 'pointer')

    if (reducedMotion) {
      slices.attr('d', arc)
    } else {
      slices
        .attr('d', arc)
        .transition()
        .duration(800)
        .delay((_, i) => i * 80)
        .ease(d3.easeCubicOut)
        .attrTween('d', function (d) {
          const interpolate = d3.interpolate(
            { startAngle: d.startAngle, endAngle: d.startAngle },
            d,
          )
          return (t: number) => arc(interpolate(t))!
        })
    }

    // ── Slice hover ──
    slices
      .on('mouseenter', function (event, d) {
        // Dim all slices
        slices.transition().duration(150).attr('opacity', 0.35)

        // Expand and brighten hovered
        d3.select(this)
          .transition()
          .duration(200)
          .attr('d', arcHover(d)!)
          .attr('opacity', 1)
          .attr('filter', 'url(#emerald-glow)')

        const pct = ((d.data[1] / total) * 100).toFixed(1)
        show(
          event,
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-sm inline-block"
                style={{ backgroundColor: DECISION_COLORS[d.data[0]] || '#6b7280' }}
              />
              <span className="text-white font-semibold">{d.data[0]}</span>
            </div>
            <div className="text-gray-400">
              Count: <span className="text-white font-medium">{d.data[1]}</span>
            </div>
            <div className="text-gray-400">
              Share: <span className="text-white font-medium">{pct}%</span>
            </div>
          </div>,
        )
      })
      .on('mousemove', function (event, d) {
        const pct = ((d.data[1] / total) * 100).toFixed(1)
        show(
          event,
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-sm inline-block"
                style={{ backgroundColor: DECISION_COLORS[d.data[0]] || '#6b7280' }}
              />
              <span className="text-white font-semibold">{d.data[0]}</span>
            </div>
            <div className="text-gray-400">
              Count: <span className="text-white font-medium">{d.data[1]}</span>
            </div>
            <div className="text-gray-400">
              Share: <span className="text-white font-medium">{pct}%</span>
            </div>
          </div>,
        )
      })
      .on('mouseleave', function (_, d) {
        slices
          .transition()
          .duration(300)
          .attr('opacity', 0.85)
          .attr('filter', null)

        d3.select(this)
          .transition()
          .duration(300)
          .attr('d', arc(d)!)

        hide()
      })

    // ── Center total label with counting animation ──
    const totalText = g
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('dy', '-0.3em')
      .attr('fill', 'white')
      .attr('font-size', '24px')
      .attr('font-weight', '700')
      .attr('font-family', "'Mona Sans', system-ui, sans-serif")
      .attr('letter-spacing', '-0.02em')

    if (reducedMotion) {
      totalText.text(String(total))
    } else {
      totalText.text('0')
      totalText
        .transition()
        .duration(1000)
        .ease(d3.easeCubicOut)
        .tween('text', function () {
          const interp = d3.interpolateNumber(0, total)
          return function (this: SVGTextElement, t: number) {
            this.textContent = String(Math.round(interp(t)))
          }
        })
    }

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('dy', '1.2em')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .attr('font-family', "'Mona Sans', system-ui, sans-serif")
      .text('Total')

    // ── Percentage labels on slices ──
    g.selectAll('text.slice-label')
      .data(pie(entries))
      .join('text')
      .attr('class', 'slice-label')
      .attr('transform', (d) => {
        const labelArc = d3
          .arc<d3.PieArcDatum<[string, number]>>()
          .innerRadius(outerRadius * 0.7)
          .outerRadius(outerRadius * 0.7)
        return `translate(${labelArc.centroid(d)})`
      })
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('fill', 'white')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('font-family', "'Mona Sans', system-ui, sans-serif")
      .attr('opacity', reducedMotion ? 1 : 0)
      .text((d) => {
        const pct = (d.data[1] / total) * 100
        return pct >= 8 ? `${Math.round(pct)}%` : ''
      })
      .transition()
      .duration(reducedMotion ? 0 : 400)
      .delay(reducedMotion ? 0 : 800)
      .attr('opacity', 1)
  }, [data, reducedMotion, show, hide])

  const entries = Object.entries(data).filter(([, v]) => v > 0)

  if (entries.length === 0) {
    return <p className="text-sm text-gray-500">No decision data available.</p>
  }

  return (
    <div ref={wrapperRef} className="relative flex flex-col items-center gap-4">
      <div className="w-full flex justify-center">
        <svg ref={svgRef} />
      </div>
      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1">
        {entries.map(([label, count]) => (
          <div key={label} className="flex items-center gap-1.5 text-xs">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{
                backgroundColor: DECISION_COLORS[label] || '#6b7280',
                opacity: 0.85,
              }}
            />
            <span className="text-gray-400">
              {label}{' '}
              <span className="text-white font-medium">{count}</span>
            </span>
          </div>
        ))}
      </div>
      {TooltipElement}
    </div>
  )
}

import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { FeatureImportance } from '@/types/api'

interface FeatureImportanceChartProps {
  features: FeatureImportance[]
}

function humanize(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function FeatureImportanceChart({
  features,
}: FeatureImportanceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const container = containerRef.current
    const svgEl = svgRef.current
    if (!container || !svgEl) return
    if (features.length === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    // Sort by absolute contribution, largest first
    const sorted = [...features].sort(
      (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
    )

    const width = container.clientWidth
    const rowHeight = 40
    const height = sorted.length * rowHeight
    const margin = { top: 8, right: 60, bottom: 8, left: 140 }

    svg.attr('width', width).attr('height', height)

    const maxAbs =
      d3.max(sorted, (d) => Math.abs(d.contribution)) ?? 1

    const x = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([margin.left, width - margin.right])

    const y = d3
      .scaleBand<number>()
      .domain(sorted.map((_, i) => i))
      .range([margin.top, height - margin.bottom])
      .padding(0.35)

    const zeroX = x(0)

    // Zero line
    svg
      .append('line')
      .attr('x1', zeroX)
      .attr('x2', zeroX)
      .attr('y1', margin.top)
      .attr('y2', height - margin.bottom)
      .attr('stroke', 'rgba(255,255,255,0.15)')
      .attr('stroke-width', 1)

    // Bars
    svg
      .selectAll('rect.bar')
      .data(sorted)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (d) => (d.contribution >= 0 ? zeroX : x(d.contribution)))
      .attr('y', (_, i) => y(i) ?? 0)
      .attr('width', (d) => Math.abs(x(d.contribution) - zeroX))
      .attr('height', y.bandwidth())
      .attr('rx', 3)
      .attr('fill', (d) =>
        d.contribution >= 0 ? '#10b981' : '#f87171'
      )
      .attr('opacity', 0.85)

    // Feature names on the left
    svg
      .selectAll('text.label')
      .data(sorted)
      .join('text')
      .attr('class', 'label')
      .attr('x', margin.left - 8)
      .attr('y', (_, i) => (y(i) ?? 0) + y.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'central')
      .attr('fill', '#d1d5db')
      .attr('font-size', '12px')
      .text((d) => humanize(d.feature_name))

    // Value labels on the right of each bar
    svg
      .selectAll('text.value')
      .data(sorted)
      .join('text')
      .attr('class', 'value')
      .attr('x', (d) => {
        if (d.contribution >= 0) {
          return x(d.contribution) + 6
        }
        // For negative bars, keep label to the left of the bar
        // but clamp so it never overlaps with the feature name area
        const labelX = x(d.contribution) - 6
        return Math.max(margin.left + 4, labelX)
      })
      .attr('y', (_, i) => (y(i) ?? 0) + y.bandwidth() / 2)
      .attr('text-anchor', (d) => {
        if (d.contribution >= 0) return 'start'
        // If clamped, switch to start anchor so text flows right
        const labelX = x(d.contribution) - 6
        return labelX < margin.left + 4 ? 'start' : 'end'
      })
      .attr('dominant-baseline', 'central')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text((d) => {
        const sign = d.contribution >= 0 ? '+' : ''
        return `${sign}${d.contribution.toFixed(3)}`
      })
  }, [features])

  if (features.length === 0) {
    return (
      <p className="text-sm text-gray-500">No feature data available.</p>
    )
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width="100%" height={features.length * 40} />
    </div>
  )
}

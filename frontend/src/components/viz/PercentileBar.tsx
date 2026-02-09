import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

interface PercentileBarProps {
  percentile: number
  label: string
  median?: number
}

export default function PercentileBar({
  percentile,
  label,
  median,
}: PercentileBarProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const container = containerRef.current
    const svgEl = svgRef.current
    if (!container || !svgEl) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const height = 60
    const margin = { left: 12, right: 12, top: 16, bottom: 24 }
    const barHeight = 8
    const barY = margin.top + 4

    svg.attr('width', width).attr('height', height)

    const x = d3
      .scaleLinear()
      .domain([0, 100])
      .range([margin.left, width - margin.right])

    const defs = svg.append('defs')
    const gradientId = `percentile-grad-${label.replace(/\s+/g, '-')}`
    const gradient = defs
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%')

    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#374151')
    gradient
      .append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#10b981')

    // Bar background
    svg
      .append('rect')
      .attr('x', margin.left)
      .attr('y', barY)
      .attr('width', width - margin.left - margin.right)
      .attr('height', barHeight)
      .attr('rx', 4)
      .attr('fill', 'rgba(255,255,255,0.06)')

    // Filled bar
    svg
      .append('rect')
      .attr('x', margin.left)
      .attr('y', barY)
      .attr('width', Math.max(0, x(Math.max(0, Math.min(100, percentile))) - margin.left))
      .attr('height', barHeight)
      .attr('rx', 4)
      .attr('fill', `url(#${gradientId})`)

    // Percentile marker
    const markerX = x(Math.max(0, Math.min(100, percentile)))

    // Vertical line
    svg
      .append('line')
      .attr('x1', markerX)
      .attr('x2', markerX)
      .attr('y1', barY - 6)
      .attr('y2', barY + barHeight + 2)
      .attr('stroke', 'white')
      .attr('stroke-width', 2)

    // Triangle above
    const triangleSize = 5
    svg
      .append('polygon')
      .attr(
        'points',
        `${markerX},${barY - 6} ${markerX - triangleSize},${barY - 6 - triangleSize * 1.5} ${markerX + triangleSize},${barY - 6 - triangleSize * 1.5}`
      )
      .attr('fill', 'white')

    // Percentile label below marker
    svg
      .append('text')
      .attr('x', markerX)
      .attr('y', barY + barHeight + 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#d1d5db')
      .attr('font-size', '11px')
      .text(`${Math.round(percentile)}th percentile`)

    // Optional median marker
    if (median !== undefined) {
      const medianX = x(Math.max(0, Math.min(100, median)))

      svg
        .append('line')
        .attr('x1', medianX)
        .attr('x2', medianX)
        .attr('y1', barY - 2)
        .attr('y2', barY + barHeight + 2)
        .attr('stroke', '#9ca3af')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '3,3')
    }

    // Label on the left
    svg
      .append('text')
      .attr('x', margin.left)
      .attr('y', barY + barHeight + 18)
      .attr('text-anchor', 'start')
      .attr('fill', '#6b7280')
      .attr('font-size', '10px')
      .text(label)
  }, [percentile, label, median])

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width="100%" height={60} />
    </div>
  )
}

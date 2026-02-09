import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

interface CalibrationPoint {
  predicted: number
  observed: number
  count: number
}

interface ReliabilityDiagramProps {
  data: CalibrationPoint[]
}

export default function ReliabilityDiagram({
  data,
}: ReliabilityDiagramProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const svgEl = svgRef.current
    if (!svgEl) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const size = 300
    const margin = { top: 16, right: 16, bottom: 40, left: 44 }

    svg.attr('width', size).attr('height', size)

    const x = d3
      .scaleLinear()
      .domain([0, 1])
      .range([margin.left, size - margin.right])

    const y = d3
      .scaleLinear()
      .domain([0, 1])
      .range([size - margin.bottom, margin.top])

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${size - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Y axis
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Axis labels
    svg
      .append('text')
      .attr('x', (margin.left + size - margin.right) / 2)
      .attr('y', size - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Predicted Probability')

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + size - margin.bottom) / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Observed Frequency')

    // Diagonal reference line (perfect calibration)
    svg
      .append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(1))
      .attr('y2', y(1))
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '6,4')

    if (data.length === 0) return

    // Sort data by predicted value for the line
    const sorted = [...data].sort((a, b) => a.predicted - b.predicted)

    // Line connecting data points
    const line = d3
      .line<CalibrationPoint>()
      .x((d) => x(d.predicted))
      .y((d) => y(d.observed))

    svg
      .append('path')
      .datum(sorted)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    // Point size scale based on count
    const maxCount = d3.max(sorted, (d) => d.count) ?? 1
    const rScale = d3
      .scaleSqrt()
      .domain([0, maxCount])
      .range([3, 10])

    // Data points
    svg
      .selectAll('circle.point')
      .data(sorted)
      .join('circle')
      .attr('class', 'point')
      .attr('cx', (d) => x(d.predicted))
      .attr('cy', (d) => y(d.observed))
      .attr('r', (d) => rScale(d.count))
      .attr('fill', '#10b981')
      .attr('stroke', '#050505')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.85)
  }, [data])

  if (data.length === 0) {
    return (
      <p className="text-sm text-gray-500">No calibration data available.</p>
    )
  }

  return <svg ref={svgRef} width={300} height={300} />
}

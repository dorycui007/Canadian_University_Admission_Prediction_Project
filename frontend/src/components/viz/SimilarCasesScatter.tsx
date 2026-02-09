import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { SimilarCase } from '@/types/api'

interface SimilarCasesScatterProps {
  cases: SimilarCase[]
  studentGrade: number
  studentECScore?: number
}

const OUTCOME_COLORS: Record<string, string> = {
  accepted: '#10b981',
  rejected: '#f87171',
  waitlisted: '#fbbf24',
}

const OUTCOME_LABELS: Array<{ key: string; label: string; color: string }> = [
  { key: 'accepted', label: 'Accepted', color: '#10b981' },
  { key: 'rejected', label: 'Rejected', color: '#f87171' },
  { key: 'waitlisted', label: 'Waitlisted', color: '#fbbf24' },
]

export default function SimilarCasesScatter({
  cases,
  studentGrade,
  studentECScore,
}: SimilarCasesScatterProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const container = containerRef.current
    const svgEl = svgRef.current
    if (!container || !svgEl) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const height = 250
    const margin = { top: 16, right: 20, bottom: 40, left: 44 }

    svg.attr('width', width).attr('height', height)

    const x = d3
      .scaleLinear()
      .domain([0, 100])
      .range([margin.left, width - margin.right])

    const y = d3
      .scaleLinear()
      .domain([0, 20])
      .range([height - margin.bottom, margin.top])

    // Grid lines
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(
        d3
          .axisBottom(x)
          .ticks(5)
          .tickSize(-height + margin.top + margin.bottom)
      )
      .call((g) => {
        g.select('.domain').remove()
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickSize(-width + margin.left + margin.right)
      )
      .call((g) => {
        g.select('.domain').remove()
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Axis labels
    svg
      .append('text')
      .attr('x', (margin.left + width - margin.right) / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Grade')

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + height - margin.bottom) / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('EC Score')

    // Data dots
    if (cases.length > 0) {
      svg
        .selectAll('circle.case')
        .data(cases)
        .join('circle')
        .attr('class', 'case')
        .attr('cx', (d) => x(d.grade))
        .attr('cy', (d) => y(d.ec_score))
        .attr('r', 4)
        .attr('fill', (d) => OUTCOME_COLORS[d.outcome] ?? '#6b7280')
        .attr('opacity', 0.7)
    }

    // Student dot (larger, pulsing via CSS)
    const ecY = studentECScore !== undefined ? studentECScore : 10
    const studentGroup = svg.append('g')

    // Outer ring
    studentGroup
      .append('circle')
      .attr('cx', x(studentGrade))
      .attr('cy', y(ecY))
      .attr('r', 10)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)
      .attr('opacity', 0.6)

    // Inner dot
    studentGroup
      .append('circle')
      .attr('cx', x(studentGrade))
      .attr('cy', y(ecY))
      .attr('r', 5)
      .attr('fill', '#10b981')

    // Pulse animation using D3 transition loop
    function pulse(selection: d3.Selection<SVGCircleElement, unknown, null, undefined>) {
      selection
        .transition()
        .duration(1200)
        .attr('r', 14)
        .attr('opacity', 0.1)
        .transition()
        .duration(0)
        .attr('r', 10)
        .attr('opacity', 0.6)
        .on('end', function () {
          pulse(d3.select(this as SVGCircleElement))
        })
    }

    const pulseCircle = studentGroup
      .append('circle')
      .attr('cx', x(studentGrade))
      .attr('cy', y(ecY))
      .attr('r', 10)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.6)

    pulse(pulseCircle as d3.Selection<SVGCircleElement, unknown, null, undefined>)

    // Legend
    const legendG = svg
      .append('g')
      .attr('transform', `translate(${width - margin.right - 100},${margin.top})`)

    OUTCOME_LABELS.forEach((item, i) => {
      const row = legendG
        .append('g')
        .attr('transform', `translate(0,${i * 18})`)

      row
        .append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', 4)
        .attr('fill', item.color)

      row
        .append('text')
        .attr('x', 10)
        .attr('y', 0)
        .attr('dominant-baseline', 'central')
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .text(item.label)
    })
  }, [cases, studentGrade, studentECScore])

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width="100%" height={250} />
    </div>
  )
}

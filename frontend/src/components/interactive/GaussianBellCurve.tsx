import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Normal PDF
   ───────────────────────────────────────────── */
function normalPdf(x: number, mean: number, std: number): number {
  const z = (x - mean) / std
  return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z)
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
interface GaussianBellCurveProps {
  mean: number
  std: number
  value: number
}

export default function GaussianBellCurve({
  mean,
  std,
  value,
}: GaussianBellCurveProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const height = 220
    const margin = { top: 16, right: 24, bottom: 36, left: 44 }

    svg.attr('width', width).attr('height', height)

    const xMin = mean - 4 * std
    const xMax = mean + 4 * std
    const x = d3.scaleLinear().domain([xMin, xMax]).range([margin.left, width - margin.right])

    const pdfMax = normalPdf(mean, mean, std) * 1.1
    const y = d3.scaleLinear().domain([0, pdfMax]).range([height - margin.bottom, margin.top])

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(8).tickFormat((d) => `${d}%`))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    svg
      .append('text')
      .attr('x', (margin.left + width - margin.right) / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Top-6 Average (%)')

    // Generate curve data
    const step = (xMax - xMin) / 200
    const curveData = d3.range(xMin, xMax + step, step).map((xv) => ({
      x: xv,
      y: normalPdf(xv, mean, std),
    }))

    // Shaded area up to student value
    const clampedValue = Math.max(xMin, Math.min(xMax, value))
    const areaData = curveData.filter((d) => d.x <= clampedValue)

    const area = d3
      .area<{ x: number; y: number }>()
      .x((d) => x(d.x))
      .y0(y(0))
      .y1((d) => y(d.y))
      .curve(d3.curveBasis)

    svg
      .append('path')
      .datum(areaData)
      .attr('d', area)
      .attr('fill', 'rgba(16, 185, 129, 0.15)')

    // Bell curve line
    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => x(d.x))
      .y((d) => y(d.y))
      .curve(d3.curveBasis)

    svg
      .append('path')
      .datum(curveData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    // Mean line
    svg
      .append('line')
      .attr('x1', x(mean))
      .attr('y1', y(0))
      .attr('x2', x(mean))
      .attr('y2', y(normalPdf(mean, mean, std)))
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')

    svg
      .append('text')
      .attr('x', x(mean))
      .attr('y', y(pdfMax) + 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '10px')
      .text(`mean = ${mean}%`)

    // Student marker
    const studentY = normalPdf(clampedValue, mean, std)

    svg
      .append('line')
      .attr('x1', x(clampedValue))
      .attr('y1', y(0))
      .attr('x2', x(clampedValue))
      .attr('y2', y(studentY))
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    svg
      .append('circle')
      .attr('cx', x(clampedValue))
      .attr('cy', y(studentY))
      .attr('r', 5)
      .attr('fill', '#10b981')
      .attr('stroke', '#050505')
      .attr('stroke-width', 2)

    // Z-score label
    const zScore = (clampedValue - mean) / std
    svg
      .append('text')
      .attr('x', x(clampedValue) + 8)
      .attr('y', y(studentY) - 8)
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(`z = ${zScore.toFixed(2)}`)

    svg
      .append('text')
      .attr('x', x(clampedValue) + 8)
      .attr('y', y(studentY) + 6)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text(`${clampedValue.toFixed(1)}%`)
  }, [mean, std, value])

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} className="w-full" />
    </div>
  )
}

import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

interface ProbabilityGaugeProps {
  probability: number
  confidenceInterval?: { lower: number; upper: number }
}

export default function ProbabilityGauge({
  probability,
  confidenceInterval,
}: ProbabilityGaugeProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 240
    const height = 140
    const cx = width / 2
    const cy = height - 10
    const outerRadius = 100
    const innerRadius = 72

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${cx},${cy})`)

    // Angles: -pi/2 (left) to pi/2 (right) for a bottom-opening semicircle
    // But we want the opening at the bottom, so the arc goes from -pi to 0
    const startAngle = -Math.PI / 2
    const endAngle = Math.PI / 2

    const arcGen = d3
      .arc<{ startAngle: number; endAngle: number }>()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .cornerRadius(3)

    // Background track
    g.append('path')
      .datum({ startAngle, endAngle })
      .attr('d', arcGen)
      .attr('fill', 'rgba(255,255,255,0.06)')

    // Confidence interval shading (if provided)
    if (confidenceInterval) {
      const ciLower = Math.max(0, Math.min(1, confidenceInterval.lower))
      const ciUpper = Math.max(0, Math.min(1, confidenceInterval.upper))
      const ciStartAngle = startAngle + ciLower * Math.PI
      const ciEndAngle = startAngle + ciUpper * Math.PI

      g.append('path')
        .datum({ startAngle: ciStartAngle, endAngle: ciEndAngle })
        .attr('d', arcGen)
        .attr('fill', 'rgba(16,185,129,0.15)')
    }

    // Color based on probability
    const clampedProb = Math.max(0, Math.min(1, probability))
    let fillColor: string
    if (clampedProb < 0.4) {
      fillColor = '#ef4444' // red
    } else if (clampedProb < 0.7) {
      fillColor = '#f59e0b' // amber
    } else {
      fillColor = '#10b981' // emerald
    }

    // Value arc
    const valueEndAngle = startAngle + clampedProb * Math.PI

    g.append('path')
      .datum({ startAngle, endAngle: valueEndAngle })
      .attr('d', arcGen)
      .attr('fill', fillColor)

    // Center percentage text
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', -28)
      .attr('fill', 'white')
      .attr('font-size', '32px')
      .attr('font-weight', 'bold')
      .text(`${Math.round(clampedProb * 100)}%`)

    // Label below the percentage
    let label: string
    if (clampedProb >= 0.7) {
      label = 'LIKELY ADMIT'
    } else if (clampedProb >= 0.4) {
      label = 'UNCERTAIN'
    } else {
      label = 'UNLIKELY'
    }

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', -6)
      .attr('fill', fillColor)
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('letter-spacing', '0.1em')
      .text(label)
  }, [probability, confidenceInterval])

  return <svg ref={svgRef} width={240} height={140} />
}

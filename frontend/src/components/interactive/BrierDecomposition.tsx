import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Brier Score Decomposition
   BS = Uncertainty - Resolution + Reliability

   Uncertainty: p_bar(1 - p_bar), fixed property of the data
   Resolution:  how well the model separates groups (higher = better)
   Reliability: calibration error (lower = better)
   ───────────────────────────────────────────── */

export default function BrierDecomposition() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [quality, setQuality] = useState(75)

  // Compute decomposition from slider
  const { unc, res, rel, bs } = useMemo(() => {
    const q = quality / 100
    const unc = 0.25
    const res = 0.22 * Math.pow(q, 1.2)
    const rel = 0.10 * Math.pow(1 - q, 1.5) + 0.005
    const bs = unc - res + rel
    return { unc, res, rel, bs }
  }, [quality])

  // Interpretation text
  const interpretation = useMemo(() => {
    if (bs < 0.15) return { text: 'Well calibrated', color: 'text-emerald-400' }
    if (bs < 0.25) return { text: 'Moderate', color: 'text-amber-400' }
    return { text: 'Poorly calibrated', color: 'text-red-400' }
  }, [bs])

  // D3 rendering
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 200
    const barAreaWidth = Math.round(totalWidth * 0.72)
    const gaugeAreaWidth = totalWidth - barAreaWidth

    svg.attr('width', totalWidth).attr('height', height)

    // ─── LEFT: Horizontal bars ───
    const barG = svg.append('g')

    const margin = { top: 20, right: 16, bottom: 28, left: 120 }
    const barHeight = 30
    const barGap = 12
    const xMax = 0.40

    const xScale = d3
      .scaleLinear()
      .domain([0, xMax])
      .range([margin.left, barAreaWidth - margin.right])

    // Gridlines at 0.1 intervals
    const gridValues = [0.1, 0.2, 0.3, 0.4]
    gridValues.forEach((v) => {
      barG
        .append('line')
        .attr('x1', xScale(v))
        .attr('y1', margin.top - 4)
        .attr('x2', xScale(v))
        .attr('y2', margin.top + 4 * (barHeight + barGap))
        .attr('stroke', 'rgba(255,255,255,0.06)')
        .attr('stroke-width', 1)
    })

    // X axis at bottom
    barG
      .append('g')
      .attr('transform', `translate(0,${margin.top + 4 * (barHeight + barGap) + 4})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickValues([0, 0.1, 0.2, 0.3, 0.4])
          .tickFormat(d3.format('.1f')),
      )
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Bar data
    const bars = [
      { label: 'Uncertainty', value: unc, color: '#6b7280' },
      { label: '\u2212 Resolution', value: res, color: '#34d399' },
      { label: '+ Reliability', value: rel, color: '#f87171' },
      { label: '= Brier Score', value: bs, color: '#ffffff' },
    ]

    bars.forEach((bar, i) => {
      const y = margin.top + i * (barHeight + barGap)
      const barW = Math.max(1, xScale(Math.min(bar.value, xMax)) - xScale(0))

      // Label on the left
      barG
        .append('text')
        .attr('x', margin.left - 8)
        .attr('y', y + barHeight / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#9ca3af')
        .attr('font-size', '12px')
        .text(bar.label)

      // Bar
      barG
        .append('rect')
        .attr('x', xScale(0))
        .attr('y', y)
        .attr('width', barW)
        .attr('height', barHeight)
        .attr('rx', 4)
        .attr('fill', bar.color)
        .attr('opacity', bar.label === '= Brier Score' ? 0.9 : 0.75)

      // Value label at end of bar
      barG
        .append('text')
        .attr('x', xScale(0) + barW + 6)
        .attr('y', y + barHeight / 2 + 4)
        .attr('fill', bar.color)
        .attr('font-size', '12px')
        .attr('font-weight', '700')
        .attr('font-family', 'monospace')
        .text(bar.value.toFixed(3))
    })

    // ─── RIGHT: Circular gauge ───
    const gaugeG = svg
      .append('g')
      .attr('transform', `translate(${barAreaWidth + gaugeAreaWidth / 2},${height / 2})`)

    const radius = Math.min(gaugeAreaWidth * 0.35, 60)
    const arcScale = d3.scaleLinear().domain([0, 0.4]).range([0, 2 * Math.PI]).clamp(true)

    // Background arc
    const bgArc = d3
      .arc<d3.DefaultArcObject>()
      .innerRadius(radius - 10)
      .outerRadius(radius)
      .startAngle(0)
      .endAngle(2 * Math.PI)

    gaugeG
      .append('path')
      .attr('d', bgArc({ innerRadius: radius - 10, outerRadius: radius, startAngle: 0, endAngle: 2 * Math.PI }))
      .attr('fill', 'rgba(255,255,255,0.06)')

    // Value arc with color based on score
    const gaugeColor = bs < 0.15 ? '#34d399' : bs < 0.25 ? '#f59e0b' : '#f87171'

    const valueArc = d3
      .arc<d3.DefaultArcObject>()
      .innerRadius(radius - 10)
      .outerRadius(radius)
      .startAngle(0)
      .endAngle(arcScale(bs))
      .cornerRadius(4)

    gaugeG
      .append('path')
      .attr('d', valueArc({ innerRadius: radius - 10, outerRadius: radius, startAngle: 0, endAngle: arcScale(bs) }))
      .attr('fill', gaugeColor)

    // Center text: Brier Score value
    gaugeG
      .append('text')
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', '20px')
      .attr('font-weight', '700')
      .attr('font-family', 'monospace')
      .text(bs.toFixed(3))

    // Label below
    gaugeG
      .append('text')
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('Brier Score')
  }, [unc, res, rel, bs])

  return (
    <div className="space-y-4">
      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Slider */}
      <div className="space-y-1">
        <label className="flex items-center justify-between text-xs text-gray-400">
          <span>Model Quality</span>
          <span className="text-white font-mono tabular-nums">{quality}</span>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={quality}
          onChange={(e) => setQuality(Number(e.target.value))}
          className="w-full accent-emerald-400"
        />
      </div>

      {/* Equation display */}
      <div className="text-sm font-mono tabular-nums">
        <span className="text-gray-500">BS = </span>
        <span className="text-gray-400">{unc.toFixed(3)}</span>
        <span className="text-gray-500"> &minus; </span>
        <span className="text-emerald-400">{res.toFixed(3)}</span>
        <span className="text-gray-500"> + </span>
        <span className="text-red-400">{rel.toFixed(3)}</span>
        <span className="text-gray-500"> = </span>
        <span className="text-white font-bold">{bs.toFixed(3)}</span>
      </div>

      {/* Interpretation */}
      <p className="text-xs text-gray-600">
        Interpretation:{' '}
        <span className={`font-bold ${interpretation.color}`}>{interpretation.text}</span>
        {' \u2014 '}
        {bs < 0.15
          ? 'The model separates groups effectively with low calibration error.'
          : bs < 0.25
            ? 'The model has some discriminative power but calibration could improve.'
            : 'The model struggles to separate groups and/or is poorly calibrated.'}
      </p>
    </div>
  )
}

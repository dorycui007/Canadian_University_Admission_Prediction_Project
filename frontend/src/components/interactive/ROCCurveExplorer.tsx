import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Mock ROC data (realistic for AUC ~ 0.82)
   ───────────────────────────────────────────── */
const ROC_DATA = [
  { threshold: 0.00, fpr: 1.00, tpr: 1.00 },
  { threshold: 0.05, fpr: 0.92, tpr: 0.99 },
  { threshold: 0.10, fpr: 0.82, tpr: 0.97 },
  { threshold: 0.15, fpr: 0.72, tpr: 0.95 },
  { threshold: 0.20, fpr: 0.60, tpr: 0.92 },
  { threshold: 0.30, fpr: 0.42, tpr: 0.85 },
  { threshold: 0.40, fpr: 0.28, tpr: 0.78 },
  { threshold: 0.50, fpr: 0.18, tpr: 0.70 },
  { threshold: 0.60, fpr: 0.10, tpr: 0.58 },
  { threshold: 0.70, fpr: 0.05, tpr: 0.45 },
  { threshold: 0.80, fpr: 0.02, tpr: 0.30 },
  { threshold: 0.90, fpr: 0.01, tpr: 0.15 },
  { threshold: 1.00, fpr: 0.00, tpr: 0.00 },
]

/* ─────────────────────────────────────────────
   Compute AUC via trapezoidal rule
   Points are sorted by descending FPR (threshold ascending),
   so we integrate from right to left (FPR 0 -> 1).
   ───────────────────────────────────────────── */
function computeAUC(data: typeof ROC_DATA): number {
  // Sort by FPR ascending for integration
  const sorted = [...data].sort((a, b) => a.fpr - b.fpr)
  let auc = 0
  for (let i = 1; i < sorted.length; i++) {
    const dx = sorted[i].fpr - sorted[i - 1].fpr
    const avgY = (sorted[i].tpr + sorted[i - 1].tpr) / 2
    auc += dx * avgY
  }
  return auc
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function ROCCurveExplorer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [selectedIdx, setSelectedIdx] = useState(7) // threshold = 0.50

  const auc = useMemo(() => computeAUC(ROC_DATA), [])
  const aucPct = Math.round(auc * 100)

  const selected = ROC_DATA[selectedIdx]
  const precision =
    selected.tpr + selected.fpr > 0
      ? selected.tpr / (selected.tpr + selected.fpr)
      : 0

  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const size = Math.min(container.clientWidth, 380)
    const margin = { top: 16, right: 16, bottom: 44, left: 48 }

    svg.attr('width', size).attr('height', size)

    // Background
    svg
      .append('rect')
      .attr('width', size)
      .attr('height', size)
      .attr('fill', '#050505')
      .attr('rx', 4)

    const plotWidth = size - margin.left - margin.right
    const plotHeight = size - margin.top - margin.bottom

    // Scales
    const x = d3.scaleLinear().domain([0, 1]).range([margin.left, margin.left + plotWidth])
    const y = d3.scaleLinear().domain([0, 1]).range([margin.top + plotHeight, margin.top])

    // Grid lines
    const gridTicks = [0.2, 0.4, 0.6, 0.8]
    gridTicks.forEach((tick) => {
      svg
        .append('line')
        .attr('x1', margin.left)
        .attr('y1', y(tick))
        .attr('x2', margin.left + plotWidth)
        .attr('y2', y(tick))
        .attr('stroke', 'rgba(255,255,255,0.06)')
        .attr('stroke-width', 1)
      svg
        .append('line')
        .attr('x1', x(tick))
        .attr('y1', margin.top)
        .attr('x2', x(tick))
        .attr('y2', margin.top + plotHeight)
        .attr('stroke', 'rgba(255,255,255,0.06)')
        .attr('stroke-width', 1)
    })

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${margin.top + plotHeight})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.08)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Y axis
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.08)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Axis labels
    svg
      .append('text')
      .attr('x', margin.left + plotWidth / 2)
      .attr('y', size - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('False Positive Rate')

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + plotHeight / 2))
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('True Positive Rate')

    // Diagonal dashed line (random classifier)
    svg
      .append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(1))
      .attr('y2', y(1))
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '6,4')

    // Area under the ROC curve (shaded)
    // Sort by FPR ascending for proper area rendering
    const sortedByFpr = [...ROC_DATA].sort((a, b) => a.fpr - b.fpr)

    const areaGen = d3
      .area<(typeof ROC_DATA)[0]>()
      .x((d) => x(d.fpr))
      .y0(y(0))
      .y1((d) => y(d.tpr))
      .curve(d3.curveMonotoneX)

    svg
      .append('path')
      .datum(sortedByFpr)
      .attr('d', areaGen)
      .attr('fill', '#10b981')
      .attr('opacity', 0.1)

    // ROC curve line
    const lineGen = d3
      .line<(typeof ROC_DATA)[0]>()
      .x((d) => x(d.fpr))
      .y((d) => y(d.tpr))
      .curve(d3.curveMonotoneX)

    svg
      .append('path')
      .datum(sortedByFpr)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2.5)

    // Reference dashed lines from selected point to axes
    const pt = ROC_DATA[selectedIdx]

    svg
      .append('line')
      .attr('class', 'ref-line-x')
      .attr('x1', x(pt.fpr))
      .attr('y1', y(pt.tpr))
      .attr('x2', x(pt.fpr))
      .attr('y2', y(0))
      .attr('stroke', '#34d399')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6)

    svg
      .append('line')
      .attr('class', 'ref-line-y')
      .attr('x1', x(pt.fpr))
      .attr('y1', y(pt.tpr))
      .attr('x2', x(0))
      .attr('y2', y(pt.tpr))
      .attr('stroke', '#34d399')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6)

    // Draggable point
    const dragCircle = svg
      .append('circle')
      .attr('cx', x(pt.fpr))
      .attr('cy', y(pt.tpr))
      .attr('r', 7)
      .attr('fill', '#10b981')
      .attr('stroke', '#050505')
      .attr('stroke-width', 2)
      .attr('cursor', 'grab')

    // D3 drag behavior
    const dragBehavior = d3
      .drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).attr('cursor', 'grabbing')
      })
      .on('drag', function (event) {
        // Convert mouse position to data coordinates
        const mouseX = x.invert(event.x)
        const mouseY = y.invert(event.y)

        // Find the closest ROC data point
        let closestIdx = 0
        let closestDist = Infinity

        ROC_DATA.forEach((d, i) => {
          const dist = (d.fpr - mouseX) ** 2 + (d.tpr - mouseY) ** 2
          if (dist < closestDist) {
            closestDist = dist
            closestIdx = i
          }
        })

        const closest = ROC_DATA[closestIdx]

        // Update circle position
        d3.select(this).attr('cx', x(closest.fpr)).attr('cy', y(closest.tpr))

        // Update reference lines
        svg
          .select('.ref-line-x')
          .attr('x1', x(closest.fpr))
          .attr('y1', y(closest.tpr))
          .attr('x2', x(closest.fpr))
          .attr('y2', y(0))

        svg
          .select('.ref-line-y')
          .attr('x1', x(closest.fpr))
          .attr('y1', y(closest.tpr))
          .attr('x2', x(0))
          .attr('y2', y(closest.tpr))

        // Update React state
        setSelectedIdx(closestIdx)
      })
      .on('end', function () {
        d3.select(this).attr('cursor', 'grab')
      })

    dragCircle.call(dragBehavior)

    // AUC label in the plot area
    svg
      .append('text')
      .attr('x', x(0.6))
      .attr('y', y(0.15))
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '12px')
      .text(`AUC = ${auc.toFixed(2)}`)
  }, [selectedIdx, auc])

  return (
    <div className="space-y-4">
      {/* SVG visualization */}
      <div ref={containerRef} className="flex justify-center">
        <svg ref={svgRef} />
      </div>

      {/* Metrics row */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div>
          <span className="text-gray-500">Threshold: </span>
          <span className="text-white font-mono font-bold tabular-nums">
            {selected.threshold.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">TPR (Sensitivity): </span>
          <span className="text-emerald-400 font-mono font-bold tabular-nums">
            {selected.tpr.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">FPR (1 - Specificity): </span>
          <span className="text-emerald-400 font-mono font-bold tabular-nums">
            {selected.fpr.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Precision: </span>
          <span className="text-emerald-400 font-mono font-bold tabular-nums">
            {precision.toFixed(2)}
          </span>
        </div>
      </div>

      {/* AUC interpretation */}
      <div className="space-y-1">
        <p className="text-sm">
          <span className="text-gray-500">AUC = </span>
          <span className="text-white font-mono font-bold">{auc.toFixed(2)}</span>
        </p>
        <p className="text-xs text-gray-600">
          AUC interpretation: a randomly chosen admitted student has an {aucPct}%
          chance of having a higher predicted probability than a randomly chosen
          rejected student.
        </p>
      </div>
    </div>
  )
}

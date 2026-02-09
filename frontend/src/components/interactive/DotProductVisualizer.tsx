import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Math helpers
   ───────────────────────────────────────────── */
function norm(v: [number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1])
}

function dot(a: [number, number], b: [number, number]): number {
  return a[0] * b[0] + a[1] * b[1]
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function toDeg(rad: number): number {
  return (rad * 180) / Math.PI
}

function clamp(val: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, val))
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function DotProductVisualizer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [vecA, setVecA] = useState<[number, number]>([3, 2])
  const [vecB, setVecB] = useState<[number, number]>([2, -1])
  const [predictionMode, setPredictionMode] = useState(false)

  // Derived values
  const normA = norm(vecA)
  const normB = norm(vecB)
  const dotAB = dot(vecA, vecB)
  const cosTheta = normA > 0 && normB > 0 ? dotAB / (normA * normB) : 0
  const clampedCos = clamp(cosTheta, -1, 1)
  const theta = Math.acos(clampedCos)
  const thetaDeg = toDeg(theta)

  // Prediction mode values
  const z = dotAB
  const prob = sigmoid(z)

  // Stable drag handlers via ref to avoid stale closures
  const vecARef = useRef(vecA)
  const vecBRef = useRef(vecB)
  vecARef.current = vecA
  vecBRef.current = vecB

  const onDragA = useCallback(
    (x: number, y: number) => setVecA([x, y]),
    [],
  )
  const onDragB = useCallback(
    (x: number, y: number) => setVecB([x, y]),
    [],
  )

  /* ─────────────────────────────────────────────
     D3 rendering
     ───────────────────────────────────────────── */
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = container.clientWidth
    const height = 350
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    svg
      .attr('width', totalWidth)
      .attr('height', height)
      .style('background', '#050505')

    const plotW = totalWidth - margin.left - margin.right
    const plotH = height - margin.top - margin.bottom
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales: domain [-5, 5] on both axes, centered
    const halfRange = Math.min(plotW, plotH) / 2
    const cx = plotW / 2
    const cy = plotH / 2

    const xScale = d3
      .scaleLinear()
      .domain([-5, 5])
      .range([cx - halfRange, cx + halfRange])
    const yScale = d3
      .scaleLinear()
      .domain([-5, 5])
      .range([cy + halfRange, cy - halfRange]) // y inverted for SVG

    // ─── Gridlines ───
    for (let i = -5; i <= 5; i++) {
      // Vertical
      g.append('line')
        .attr('x1', xScale(i))
        .attr('y1', yScale(-5))
        .attr('x2', xScale(i))
        .attr('y2', yScale(5))
        .attr('stroke', i === 0 ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.05)')
        .attr('stroke-width', i === 0 ? 1.5 : 1)

      // Horizontal
      g.append('line')
        .attr('x1', xScale(-5))
        .attr('y1', yScale(i))
        .attr('x2', xScale(5))
        .attr('y2', yScale(i))
        .attr('stroke', i === 0 ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.05)')
        .attr('stroke-width', i === 0 ? 1.5 : 1)
    }

    // Axis labels
    const labelOffset = 10
    g.append('text')
      .attr('x', xScale(5) + labelOffset)
      .attr('y', yScale(0) + 4)
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('x')

    g.append('text')
      .attr('x', xScale(0) + 4)
      .attr('y', yScale(5) - labelOffset + 4)
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .text('y')

    // Tick labels on axes (skip 0)
    for (let i = -5; i <= 5; i++) {
      if (i === 0) continue
      // x-axis ticks
      g.append('text')
        .attr('x', xScale(i))
        .attr('y', yScale(0) + 14)
        .attr('text-anchor', 'middle')
        .attr('fill', '#6b7280')
        .attr('font-size', '9px')
        .text(String(i))

      // y-axis ticks
      g.append('text')
        .attr('x', xScale(0) - 10)
        .attr('y', yScale(i) + 3)
        .attr('text-anchor', 'end')
        .attr('fill', '#6b7280')
        .attr('font-size', '9px')
        .text(String(i))
    }

    // Origin dot
    g.append('circle')
      .attr('cx', xScale(0))
      .attr('cy', yScale(0))
      .attr('r', 3)
      .attr('fill', '#ffffff')

    // ─── Arrowhead marker definitions ───
    const defs = svg.append('defs')

    defs
      .append('marker')
      .attr('id', 'arrow-a')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9)
      .attr('refY', 5)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
      .attr('fill', '#34d399')

    defs
      .append('marker')
      .attr('id', 'arrow-b')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9)
      .attr('refY', 5)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
      .attr('fill', '#60a5fa')

    defs
      .append('marker')
      .attr('id', 'arrow-proj')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9)
      .attr('refY', 5)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
      .attr('fill', '#fbbf24')

    // ─── Projection of A onto B ───
    const normBSq = vecB[0] * vecB[0] + vecB[1] * vecB[1]
    let projX = 0
    let projY = 0
    if (normBSq > 0.001) {
      const scalar = dotAB / normBSq
      projX = scalar * vecB[0]
      projY = scalar * vecB[1]

      // Dashed line from A tip to projection point
      g.append('line')
        .attr('x1', xScale(vecA[0]))
        .attr('y1', yScale(vecA[1]))
        .attr('x2', xScale(projX))
        .attr('y2', yScale(projY))
        .attr('stroke', 'rgba(251,191,36,0.4)')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,4')

      // Projection vector (from origin to projection point)
      g.append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(projX))
        .attr('y2', yScale(projY))
        .attr('stroke', '#fbbf24')
        .attr('stroke-width', 2.5)
        .attr('marker-end', 'url(#arrow-proj)')
        .attr('opacity', 0.8)

      // Projection dot
      g.append('circle')
        .attr('cx', xScale(projX))
        .attr('cy', yScale(projY))
        .attr('r', 3.5)
        .attr('fill', '#fbbf24')
    }

    // ─── Angle arc ───
    if (normA > 0.01 && normB > 0.01) {
      const arcRadius = 25
      const angleA = Math.atan2(vecA[1], vecA[0])
      const angleB = Math.atan2(vecB[1], vecB[0])

      // D3 arc uses angles measured clockwise from 12 o'clock,
      // but we need standard math angles. We convert: SVG angle = -math angle
      // because SVG y is inverted.
      const svgAngleA = -angleA
      const svgAngleB = -angleB

      const arc = d3.arc()

      // Determine start/end so we always draw the smaller arc
      let startAngle = svgAngleB
      let endAngle = svgAngleA
      let diff = endAngle - startAngle
      // Normalize to [-PI, PI]
      while (diff > Math.PI) { endAngle -= 2 * Math.PI; diff = endAngle - startAngle }
      while (diff < -Math.PI) { endAngle += 2 * Math.PI; diff = endAngle - startAngle }
      if (diff < 0) {
        const tmp = startAngle
        startAngle = endAngle
        endAngle = tmp
      }

      g.append('path')
        .attr('d', arc({
          innerRadius: 0,
          outerRadius: arcRadius,
          startAngle: startAngle,
          endAngle: endAngle,
        }))
        .attr('transform', `translate(${xScale(0)},${yScale(0)})`)
        .attr('fill', 'rgba(16,185,129,0.12)')
        .attr('stroke', 'rgba(16,185,129,0.4)')
        .attr('stroke-width', 1)

      // Angle label
      const midAngle = -(startAngle + endAngle) / 2 // back to math convention
      const labelR = arcRadius + 14
      g.append('text')
        .attr('x', xScale(0) + labelR * Math.cos(midAngle))
        .attr('y', yScale(0) - labelR * Math.sin(midAngle))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .text(`${thetaDeg.toFixed(1)}\u00B0`)
    }

    // ─── Vector A (emerald) ───
    const labelA = predictionMode ? 'x (features)' : 'A'
    g.append('line')
      .attr('x1', xScale(0))
      .attr('y1', yScale(0))
      .attr('x2', xScale(vecA[0]))
      .attr('y2', yScale(vecA[1]))
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#arrow-a)')

    g.append('text')
      .attr('x', xScale(vecA[0]) + 10)
      .attr('y', yScale(vecA[1]) - 10)
      .attr('fill', '#34d399')
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(labelA)

    // Draggable handle for A
    const handleA = g
      .append('circle')
      .attr('cx', xScale(vecA[0]))
      .attr('cy', yScale(vecA[1]))
      .attr('r', 8)
      .attr('fill', '#34d399')
      .attr('stroke', '#050505')
      .attr('stroke-width', 2)
      .style('cursor', 'grab')

    // ─── Vector B (blue) ───
    const labelB = predictionMode ? '\u03B2 (coefficients)' : 'B'
    g.append('line')
      .attr('x1', xScale(0))
      .attr('y1', yScale(0))
      .attr('x2', xScale(vecB[0]))
      .attr('y2', yScale(vecB[1]))
      .attr('stroke', '#60a5fa')
      .attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#arrow-b)')

    g.append('text')
      .attr('x', xScale(vecB[0]) + 10)
      .attr('y', yScale(vecB[1]) - 10)
      .attr('fill', '#60a5fa')
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(labelB)

    // Draggable handle for B
    const handleB = g
      .append('circle')
      .attr('cx', xScale(vecB[0]))
      .attr('cy', yScale(vecB[1]))
      .attr('r', 8)
      .attr('fill', '#60a5fa')
      .attr('stroke', '#050505')
      .attr('stroke-width', 2)
      .style('cursor', 'grab')

    // ─── Projection label ───
    if (normBSq > 0.001) {
      g.append('text')
        .attr('x', xScale(projX / 2) - 6)
        .attr('y', yScale(projY / 2) + 16)
        .attr('fill', '#fbbf24')
        .attr('font-size', '10px')
        .attr('opacity', 0.8)
        .text('proj')
    }

    // ─── Drag behaviors ───
    const dragA = d3
      .drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing')
      })
      .on('drag', function (event) {
        const [mx, my] = d3.pointer(event, g.node())
        const newX = clamp(xScale.invert(mx), -5, 5)
        const newY = clamp(yScale.invert(my), -5, 5)
        onDragA(
          Math.round(newX * 100) / 100,
          Math.round(newY * 100) / 100,
        )
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'grab')
      })

    const dragB = d3
      .drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing')
      })
      .on('drag', function (event) {
        const [mx, my] = d3.pointer(event, g.node())
        const newX = clamp(xScale.invert(mx), -5, 5)
        const newY = clamp(yScale.invert(my), -5, 5)
        onDragB(
          Math.round(newX * 100) / 100,
          Math.round(newY * 100) / 100,
        )
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'grab')
      })

    handleA.call(dragA)
    handleB.call(dragB)
  }, [vecA, vecB, predictionMode, dotAB, normA, normB, thetaDeg, onDragA, onDragB])

  /* ─────────────────────────────────────────────
     Resize handler
     ───────────────────────────────────────────── */
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver(() => {
      // Trigger re-render by reading container width indirectly.
      // The main useEffect already reads container.clientWidth on each run.
      // We force a re-draw by toggling a dummy state.
      setVecA((prev) => [...prev] as [number, number])
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  /* ─────────────────────────────────────────────
     Render
     ───────────────────────────────────────────── */
  return (
    <div className="space-y-4">
      {/* Toggle button */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPredictionMode((p) => !p)}
          className="relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none"
          style={{
            backgroundColor: predictionMode ? '#10b981' : 'rgba(255,255,255,0.15)',
          }}
          aria-label="Toggle prediction mode"
        >
          <span
            className="inline-block h-4 w-4 rounded-full bg-white transition-transform duration-200"
            style={{
              transform: predictionMode ? 'translateX(1.375rem)' : 'translateX(0.25rem)',
            }}
          />
        </button>
        <span className="text-sm text-gray-400">
          {predictionMode ? 'Prediction Mode' : 'Dot Product Mode'}
        </span>
      </div>

      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Computed values */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
        <div>
          <span className="text-gray-500">|{predictionMode ? 'x' : 'A'}| = </span>
          <span className="font-mono text-white">{normA.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-gray-500">|{predictionMode ? '\u03B2' : 'B'}| = </span>
          <span className="font-mono text-white">{normB.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-gray-500">
            {predictionMode ? 'x' : 'A'} {'\u00B7'} {predictionMode ? '\u03B2' : 'B'} ={' '}
          </span>
          <span className="font-mono font-bold text-emerald-400">
            {dotAB.toFixed(3)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">cos({'\u03B8'}) = </span>
          <span className="font-mono text-white">{cosTheta.toFixed(3)}</span>
        </div>
        <div className="col-span-2 sm:col-span-4">
          <span className="text-gray-500">{'\u03B8'} = </span>
          <span className="font-mono text-white">{thetaDeg.toFixed(1)}{'\u00B0'}</span>
        </div>

        {predictionMode && (
          <>
            <div className="col-span-2">
              <span className="text-gray-500">z = x{'\u1D40'}{'\u03B2'} = </span>
              <span className="font-mono font-bold text-emerald-400">
                {z.toFixed(3)}
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-gray-500">P(admit) = {'\u03C3'}(z) = </span>
              <span className="font-mono font-bold text-emerald-400">
                {prob.toFixed(4)}
              </span>
              <span className="ml-2 text-gray-600 text-xs">
                ({(prob * 100).toFixed(1)}%)
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

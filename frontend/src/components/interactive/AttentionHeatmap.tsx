import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Attention Heatmap
   Visualizes mock attention weights between
   input features: Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
   ───────────────────────────────────────────── */

const FEATURES = ['GPA', 'Province', 'Program', 'Term', 'Competitive', 'Interaction']

const ATTENTION_WEIGHTS = [
  [0.35, 0.10, 0.25, 0.05, 0.15, 0.10], // GPA attends to...
  [0.08, 0.40, 0.12, 0.15, 0.10, 0.15], // Province attends to...
  [0.30, 0.12, 0.30, 0.05, 0.13, 0.10], // Program attends to...
  [0.05, 0.15, 0.05, 0.50, 0.10, 0.15], // Term attends to...
  [0.20, 0.10, 0.20, 0.08, 0.32, 0.10], // Competitive attends to...
  [0.25, 0.08, 0.18, 0.09, 0.10, 0.30], // Interaction attends to...
]

export default function AttentionHeatmap() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [hoveredCell, setHoveredCell] = useState<[number, number] | null>(null)
  const [containerWidth, setContainerWidth] = useState(0)

  // Stable hover callbacks to avoid re-binding on every render
  const hoverHandlers = useRef<{
    enter: (ri: number, ci: number) => void
    leave: () => void
  }>({
    enter: (ri, ci) => setHoveredCell([ri, ci]),
    leave: () => setHoveredCell(null),
  })

  // ─── ResizeObserver to track container width ───
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const measure = () => setContainerWidth(container.clientWidth)
    measure()

    let resizeTimeout: ReturnType<typeof setTimeout>
    const observer = new ResizeObserver(() => {
      clearTimeout(resizeTimeout)
      resizeTimeout = setTimeout(measure, 150)
    })

    observer.observe(container)
    return () => {
      clearTimeout(resizeTimeout)
      observer.disconnect()
    }
  }, [])

  // ─── D3 draw (runs on mount + whenever containerWidth changes) ───
  const draw = useCallback(() => {
    const svgEl = svgRef.current
    if (!svgEl || containerWidth === 0) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = containerWidth
    const n = FEATURES.length

    const margin = { top: 60, right: 20, bottom: 20, left: 80 }
    const cellSize = Math.floor((totalWidth - margin.left - margin.right) / n)
    const gridSize = cellSize * n
    const height = margin.top + gridSize + margin.bottom

    svg.attr('width', totalWidth).attr('height', height)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Color scale: dark to bright emerald
    const colorScale = d3
      .scaleSequential()
      .domain([0, 0.5])
      .interpolator(d3.interpolateRgb('#0a0a0a', '#34d399'))

    // ─── Draw cells ───
    ATTENTION_WEIGHTS.forEach((row, ri) => {
      row.forEach((value, ci) => {
        const cellG = g.append('g').attr('class', `cell-${ri}-${ci}`)

        // Cell rectangle
        cellG
          .append('rect')
          .attr('x', ci * cellSize)
          .attr('y', ri * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', colorScale(value))
          .attr('stroke', '#050505')
          .attr('stroke-width', 1.5)
          .attr('rx', 2)

        // Value text inside cell (only if cells are large enough)
        if (cellSize >= 40) {
          cellG
            .append('text')
            .attr('x', ci * cellSize + cellSize / 2)
            .attr('y', ri * cellSize + cellSize / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', value > 0.25 ? '#050505' : '#9ca3af')
            .attr('font-size', cellSize >= 60 ? '12px' : '10px')
            .attr('font-family', 'monospace')
            .attr('font-weight', '600')
            .attr('pointer-events', 'none')
            .text(value.toFixed(2))
        }

        // Invisible overlay for hover interaction
        cellG
          .append('rect')
          .attr('x', ci * cellSize)
          .attr('y', ri * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', 'transparent')
          .attr('cursor', 'pointer')
          .on('mouseenter', () => hoverHandlers.current.enter(ri, ci))
          .on('mouseleave', () => hoverHandlers.current.leave())
      })
    })

    // ─── Row labels (Query features, left side) ───
    FEATURES.forEach((label, i) => {
      g.append('text')
        .attr('x', -10)
        .attr('y', i * cellSize + cellSize / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#9ca3af')
        .attr('font-size', '12px')
        .text(label)
    })

    // ─── Column labels (Key features, top, rotated) ───
    FEATURES.forEach((label, i) => {
      g.append('text')
        .attr('x', i * cellSize + cellSize / 2)
        .attr('y', -10)
        .attr('text-anchor', 'start')
        .attr('fill', '#9ca3af')
        .attr('font-size', '12px')
        .attr(
          'transform',
          `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`,
        )
        .text(label)
    })

    // ─── Axis group labels ───
    // "Query" label for rows
    g.append('text')
      .attr('x', -margin.left + 4)
      .attr('y', gridSize / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('transform', `rotate(-90, ${-margin.left + 4}, ${gridSize / 2})`)
      .text('Query')

    // "Key" label for columns
    g.append('text')
      .attr('x', gridSize / 2)
      .attr('y', -margin.top + 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .text('Key')
  }, [containerWidth])

  useEffect(() => {
    draw()
  }, [draw])

  // ─── Hover highlight effect (separate from full draw) ───
  useEffect(() => {
    const svgEl = svgRef.current
    if (!svgEl) return

    const svg = d3.select(svgEl)
    const n = FEATURES.length

    // Reset all highlight strokes
    for (let ri = 0; ri < n; ri++) {
      for (let ci = 0; ci < n; ci++) {
        svg
          .select(`.cell-${ri}-${ci} rect:first-child`)
          .attr('stroke', '#050505')
          .attr('stroke-width', 1.5)
      }
    }

    // Apply highlight for hovered row and column
    if (hoveredCell) {
      const [hr, hc] = hoveredCell

      // Highlight entire row
      for (let ci = 0; ci < n; ci++) {
        svg
          .select(`.cell-${hr}-${ci} rect:first-child`)
          .attr('stroke', '#34d399')
          .attr('stroke-width', 2)
      }

      // Highlight entire column
      for (let ri = 0; ri < n; ri++) {
        svg
          .select(`.cell-${ri}-${hc} rect:first-child`)
          .attr('stroke', '#34d399')
          .attr('stroke-width', 2)
      }

      // Extra emphasis on the hovered cell itself
      svg
        .select(`.cell-${hr}-${hc} rect:first-child`)
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 2.5)
    }
  }, [hoveredCell])

  return (
    <div className="space-y-4">
      {/* Planned feature badge */}
      <div className="flex items-start gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2">
        <span className="mt-0.5 inline-block rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-emerald-400">
          Planned Feature
        </span>
        <p className="text-xs text-gray-400">
          This visualization shows mock attention weights. The attention
          mechanism is a planned enhancement.
        </p>
      </div>

      {/* SVG heatmap */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      {/* Hover detail */}
      <div className="h-5 text-sm">
        {hoveredCell ? (
          <p className="text-gray-300">
            <span className="text-emerald-400 font-medium">
              &lsquo;{FEATURES[hoveredCell[0]]}&rsquo;
            </span>
            {' attends to '}
            <span className="text-emerald-400 font-medium">
              &lsquo;{FEATURES[hoveredCell[1]]}&rsquo;
            </span>
            {' with weight '}
            <span className="font-mono font-bold text-white tabular-nums">
              {ATTENTION_WEIGHTS[hoveredCell[0]][hoveredCell[1]].toFixed(2)}
            </span>
          </p>
        ) : (
          <p className="text-gray-600 text-xs">
            Hover over a cell to see the attention weight detail.
          </p>
        )}
      </div>

      {/* Formula */}
      <div className="rounded-lg border border-white/5 bg-white/[0.02] px-4 py-3 text-center">
        <p className="text-sm text-gray-300">
          <M>{"\\text{Attention}(Q, K, V) = \\text{softmax}(QK^T / \\sqrt{d})\\, V"}</M>
        </p>
      </div>

      {/* Explanation */}
      <p className="text-xs text-gray-600">
        Each row shows how much a feature &ldquo;attends to&rdquo; other
        features. High values on the diagonal mean self-attention dominates.
      </p>
    </div>
  )
}

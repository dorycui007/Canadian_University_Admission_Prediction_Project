import { useState, useRef, useEffect, useCallback } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Node data
   ───────────────────────────────────────────── */
interface ChainNode {
  module: string
  desc: string
  detail: string
}

const NODES: ChainNode[] = [
  {
    module: 'vectors',
    desc: '8 operations',
    detail: 'dot, norm, scale, add, subtract, normalize, cosine_similarity, angle',
  },
  {
    module: 'matrices',
    desc: '6 operations',
    detail: 'multiply, transpose, identity, trace, rank, condition_number',
  },
  {
    module: 'projections',
    desc: 'Hat matrix',
    detail: 'compute_hat_matrix, leverage, projection_residuals',
  },
  {
    module: 'qr',
    desc: 'Householder',
    detail: 'qr_householder, back_substitution, solve_qr',
  },
  {
    module: 'svd',
    desc: 'Decomposition',
    detail: 'svd_decompose, low_rank_approx, condition_number_svd',
  },
  {
    module: 'ridge',
    desc: 'Regularized solve',
    detail: 'weighted_ridge_solve, ridge_path, gcv_optimal_lambda',
  },
  {
    module: 'logistic',
    desc: 'IRLS prediction',
    detail: 'fit (IRLS), predict, sigmoid, log_likelihood',
  },
]

/* ─────────────────────────────────────────────
   Constants
   ───────────────────────────────────────────── */
const NODE_WIDTH = 110
const NODE_HEIGHT = 60
const NODE_RX = 8
const SVG_HEIGHT = 160
const MONO_STACK = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function DependencyChainDiagram() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  /* ── Draw function shared by initial render and resize ── */
  const draw = useCallback(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    svg.attr('width', width).attr('height', SVG_HEIGHT)

    /* ── Layout: evenly space nodes across container ── */
    const totalPadding = width - NODES.length * NODE_WIDTH
    const gap = totalPadding / (NODES.length + 1)

    const nodePositions = NODES.map((_, i) => ({
      x: gap + i * (NODE_WIDTH + gap),
      y: SVG_HEIGHT / 2 - NODE_HEIGHT / 2,
    }))

    /* ── Arrowhead marker definition ── */
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'dep-arrowhead')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 10)
      .attr('refY', 5)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 Z')
      .attr('fill', 'rgba(52,211,153,0.5)')

    /* ── Arrows between consecutive nodes ── */
    for (let i = 0; i < NODES.length - 1; i++) {
      const startX = nodePositions[i].x + NODE_WIDTH
      const endX = nodePositions[i + 1].x
      const cy = SVG_HEIGHT / 2

      svg
        .append('path')
        .attr('d', `M ${startX} ${cy} L ${endX} ${cy}`)
        .attr('stroke', 'rgba(52,211,153,0.5)')
        .attr('stroke-width', 1.5)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#dep-arrowhead)')
    }

    /* ── Node groups ── */
    const nodeGroups = svg
      .selectAll<SVGGElement, ChainNode>('g.node')
      .data(NODES)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (_, i) => `translate(${nodePositions[i].x}, ${nodePositions[i].y})`)
      .style('cursor', 'pointer')

    /* ── Rounded rectangles ── */
    nodeGroups
      .append('rect')
      .attr('width', NODE_WIDTH)
      .attr('height', NODE_HEIGHT)
      .attr('rx', NODE_RX)
      .attr('ry', NODE_RX)
      .attr('fill', 'rgba(255,255,255,0.05)')
      .attr('stroke', 'rgba(255,255,255,0.1)')
      .attr('stroke-width', 1.5)
      .attr('class', 'node-rect')

    /* ── Module name (emerald, mono, bold) ── */
    nodeGroups
      .append('text')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', NODE_HEIGHT / 2 - 6)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#34d399')
      .attr('font-size', '12px')
      .attr('font-family', MONO_STACK)
      .attr('font-weight', '700')
      .text((d) => d.module)

    /* ── Short description (gray) ── */
    nodeGroups
      .append('text')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', NODE_HEIGHT / 2 + 12)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text((d) => d.desc)

    /* ── Hover interactions ── */
    nodeGroups
      .on('mouseenter', function (_, d) {
        const idx = NODES.indexOf(d)
        setHoveredIndex(idx)
        d3.select(this)
          .select('.node-rect')
          .transition()
          .duration(150)
          .attr('stroke', '#34d399')
          .attr('stroke-width', 2)
      })
      .on('mouseleave', function () {
        setHoveredIndex(null)
        d3.select(this)
          .select('.node-rect')
          .transition()
          .duration(150)
          .attr('stroke', 'rgba(255,255,255,0.1)')
          .attr('stroke-width', 1.5)
      })
  }, [])

  /* ── Initial draw ── */
  useEffect(() => {
    draw()
  }, [draw])

  /* ── Redraw on container resize ── */
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver(() => {
      draw()
    })

    observer.observe(container)
    return () => observer.disconnect()
  }, [draw])

  /* ── Tooltip position calculation ── */
  const getTooltipPosition = (): { left: number; top: number } | null => {
    if (hoveredIndex === null || !containerRef.current) return null
    const width = containerRef.current.clientWidth
    const totalPadding = width - NODES.length * NODE_WIDTH
    const gap = totalPadding / (NODES.length + 1)
    const nodeX = gap + hoveredIndex * (NODE_WIDTH + gap)
    return {
      left: nodeX + NODE_WIDTH / 2,
      top: SVG_HEIGHT / 2 - NODE_HEIGHT / 2 - 8,
    }
  }

  const tooltipPos = getTooltipPosition()

  return (
    <div ref={containerRef} className="relative w-full">
      <svg ref={svgRef} className="w-full" />

      {/* Tooltip */}
      {hoveredIndex !== null && tooltipPos && (
        <div
          className="absolute z-10 pointer-events-none px-3 py-2 rounded-lg border border-white/10 bg-[#0a0a0a]/95 backdrop-blur-sm shadow-lg"
          style={{
            left: tooltipPos.left,
            top: tooltipPos.top,
            transform: 'translate(-50%, -100%)',
            maxWidth: 260,
          }}
        >
          <p className="text-xs font-mono font-bold text-emerald-400 mb-1">
            {NODES[hoveredIndex].module}
          </p>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            {NODES[hoveredIndex].detail}
          </p>
        </div>
      )}
    </div>
  )
}

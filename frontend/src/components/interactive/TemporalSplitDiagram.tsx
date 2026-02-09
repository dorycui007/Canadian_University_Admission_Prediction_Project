import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Types & constants
   ───────────────────────────────────────────── */
type SplitMode = 'random' | 'temporal'

const YEARS = ['2022-23', '2023-24', '2024-25'] as const

const BLOCK_HEIGHT = 60
const BLOCK_RX = 8
const FOLD_ROW_HEIGHT = 36
const FOLD_GAP = 10

const COLORS = {
  train: {
    fill: 'rgba(52,211,153,0.2)',
    stroke: '#34d399',
    dot: '#34d399',
  },
  test: {
    fill: 'rgba(251,191,36,0.2)',
    stroke: '#fbbf24',
    dot: '#fbbf24',
  },
  text: {
    white: '#ffffff',
    gray: '#9ca3af',
    darkGray: '#6b7280',
  },
} as const

/* ─────────────────────────────────────────────
   Deterministic pseudo-random scatter using a
   simple seed-based LCG to keep circles stable
   across re-renders
   ───────────────────────────────────────────── */
function seededRandom(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function TemporalSplitDiagram() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState<SplitMode>('random')

  /* ── Draw function ── */
  const draw = useCallback(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const blockGap = 16
    const sideMargin = 24
    const blockWidth = (width - sideMargin * 2 - blockGap * (YEARS.length - 1)) / YEARS.length

    /* ── Total SVG height ── */
    const mainSectionTop = 16
    const mainSectionBottom = mainSectionTop + BLOCK_HEIGHT
    const messageY = mainSectionBottom + 28
    const expandingLabelY = messageY + 32
    const foldStartY = expandingLabelY + 24
    const foldSectionHeight = 3 * FOLD_ROW_HEIGHT + 2 * FOLD_GAP
    const totalHeight = foldStartY + foldSectionHeight + 16

    svg.attr('width', width).attr('height', totalHeight)

    /* ─────────────────────────────────────
       Main split diagram (top)
       ───────────────────────────────────── */
    const mainGroup = svg.append('g')

    YEARS.forEach((year, i) => {
      const bx = sideMargin + i * (blockWidth + blockGap)
      const by = mainSectionTop

      if (mode === 'temporal') {
        /* Temporal split: 2022-23 & 2023-24 = Train, 2024-25 = Validate */
        const isTrain = i < 2
        const col = isTrain ? COLORS.train : COLORS.test

        mainGroup
          .append('rect')
          .attr('x', bx)
          .attr('y', by)
          .attr('width', blockWidth)
          .attr('height', BLOCK_HEIGHT)
          .attr('rx', BLOCK_RX)
          .attr('ry', BLOCK_RX)
          .attr('fill', col.fill)
          .attr('stroke', col.stroke)
          .attr('stroke-width', 1.5)

        /* Uniform dots filling the block */
        const rand = seededRandom(42 + i * 100)
        const dotsPerBlock = 18
        for (let d = 0; d < dotsPerBlock; d++) {
          const cx = bx + 10 + rand() * (blockWidth - 20)
          const cy = by + 10 + rand() * (BLOCK_HEIGHT - 20)
          mainGroup
            .append('circle')
            .attr('cx', cx)
            .attr('cy', cy)
            .attr('r', 3)
            .attr('fill', col.dot)
            .attr('opacity', 0.65)
        }

        /* Year label */
        mainGroup
          .append('text')
          .attr('x', bx + blockWidth / 2)
          .attr('y', by + BLOCK_HEIGHT / 2 + 1)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', COLORS.text.white)
          .attr('font-size', '13px')
          .attr('font-weight', '700')
          .text(year)

        /* Role label below block */
        mainGroup
          .append('text')
          .attr('x', bx + blockWidth / 2)
          .attr('y', by + BLOCK_HEIGHT + 16)
          .attr('text-anchor', 'middle')
          .attr('fill', col.stroke)
          .attr('font-size', '11px')
          .attr('font-weight', '600')
          .text(isTrain ? 'Train' : 'Validate')
      } else {
        /* Random split: each block has mixed Train/Test dots */
        mainGroup
          .append('rect')
          .attr('x', bx)
          .attr('y', by)
          .attr('width', blockWidth)
          .attr('height', BLOCK_HEIGHT)
          .attr('rx', BLOCK_RX)
          .attr('ry', BLOCK_RX)
          .attr('fill', 'rgba(255,255,255,0.04)')
          .attr('stroke', 'rgba(255,255,255,0.15)')
          .attr('stroke-width', 1.5)

        const rand = seededRandom(42 + i * 100)
        const dotsPerBlock = 18
        for (let d = 0; d < dotsPerBlock; d++) {
          const cx = bx + 10 + rand() * (blockWidth - 20)
          const cy = by + 10 + rand() * (BLOCK_HEIGHT - 20)
          const isTrain = rand() > 0.35 // ~65% train, ~35% test
          mainGroup
            .append('circle')
            .attr('cx', cx)
            .attr('cy', cy)
            .attr('r', 3)
            .attr('fill', isTrain ? COLORS.train.dot : COLORS.test.dot)
            .attr('opacity', 0.7)
        }

        /* Year label */
        mainGroup
          .append('text')
          .attr('x', bx + blockWidth / 2)
          .attr('y', by + BLOCK_HEIGHT / 2 + 1)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', COLORS.text.white)
          .attr('font-size', '13px')
          .attr('font-weight', '700')
          .text(year)

        /* "Mixed" label below block */
        mainGroup
          .append('text')
          .attr('x', bx + blockWidth / 2)
          .attr('y', by + BLOCK_HEIGHT + 16)
          .attr('text-anchor', 'middle')
          .attr('fill', COLORS.text.darkGray)
          .attr('font-size', '11px')
          .text('Train + Test mixed')
      }
    })

    /* ─────────────────────────────────────
       Warning / success message
       ───────────────────────────────────── */
    if (mode === 'random') {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', messageY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ef4444')
        .attr('font-size', '13px')
        .attr('font-weight', '600')
        .text('\u26A0 Data leakage: future data trains the model')
    } else {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', messageY)
        .attr('text-anchor', 'middle')
        .attr('fill', '#10b981')
        .attr('font-size', '13px')
        .attr('font-weight', '600')
        .text('\u2713 Model only sees past data, validated on unseen future')
    }

    /* ─────────────────────────────────────
       Expanding Window CV section
       ───────────────────────────────────── */
    svg
      .append('text')
      .attr('x', sideMargin)
      .attr('y', expandingLabelY)
      .attr('fill', COLORS.text.gray)
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text('Expanding Window CV')

    /* Fold definitions:
       Each fold is an array of { yearIndex, role } objects.
       role: 'train' | 'test' | 'train-half' | 'test-half'
    */
    interface FoldSegment {
      yearIndex: number
      role: 'train' | 'test' | 'train-half' | 'test-half'
    }

    const folds: FoldSegment[][] = [
      /* Fold 1: Train [2022-23] -> Test [2023-24] */
      [
        { yearIndex: 0, role: 'train' },
        { yearIndex: 1, role: 'test' },
      ],
      /* Fold 2: Train [2022-23, 2023-24] -> Test [2024-25] */
      [
        { yearIndex: 0, role: 'train' },
        { yearIndex: 1, role: 'train' },
        { yearIndex: 2, role: 'test' },
      ],
      /* Fold 3: Train [2022-23, 2023-24, first half 2024-25] -> Test [second half 2024-25] */
      [
        { yearIndex: 0, role: 'train' },
        { yearIndex: 1, role: 'train' },
        { yearIndex: 2, role: 'train-half' },
      ],
    ]

    const foldLabels = ['Fold 1', 'Fold 2', 'Fold 3']

    folds.forEach((fold, foldIdx) => {
      const fy = foldStartY + foldIdx * (FOLD_ROW_HEIGHT + FOLD_GAP)
      const foldGroup = svg.append('g')

      /* Fold label */
      foldGroup
        .append('text')
        .attr('x', sideMargin)
        .attr('y', fy + FOLD_ROW_HEIGHT / 2 + 1)
        .attr('dominant-baseline', 'middle')
        .attr('fill', COLORS.text.darkGray)
        .attr('font-size', '11px')
        .attr('font-weight', '600')
        .text(foldLabels[foldIdx])

      const labelOffset = 52

      fold.forEach((seg) => {
        const bx = labelOffset + sideMargin + seg.yearIndex * (blockWidth + blockGap)
        const segBlockWidth = blockWidth

        if (seg.role === 'train' || seg.role === 'test') {
          const col = seg.role === 'train' ? COLORS.train : COLORS.test

          foldGroup
            .append('rect')
            .attr('x', bx)
            .attr('y', fy)
            .attr('width', segBlockWidth)
            .attr('height', FOLD_ROW_HEIGHT)
            .attr('rx', 6)
            .attr('ry', 6)
            .attr('fill', col.fill)
            .attr('stroke', col.stroke)
            .attr('stroke-width', 1)

          foldGroup
            .append('text')
            .attr('x', bx + segBlockWidth / 2)
            .attr('y', fy + FOLD_ROW_HEIGHT / 2 + 1)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', col.stroke)
            .attr('font-size', '10px')
            .attr('font-weight', '600')
            .text(seg.role === 'train' ? YEARS[seg.yearIndex] : YEARS[seg.yearIndex])
        } else if (seg.role === 'train-half') {
          /* Split the 2024-25 block in half: left = train, right = test */
          const halfWidth = segBlockWidth / 2

          /* Train half (left) */
          foldGroup
            .append('rect')
            .attr('x', bx)
            .attr('y', fy)
            .attr('width', halfWidth - 1)
            .attr('height', FOLD_ROW_HEIGHT)
            .attr('rx', 6)
            .attr('ry', 6)
            .attr('fill', COLORS.train.fill)
            .attr('stroke', COLORS.train.stroke)
            .attr('stroke-width', 1)

          foldGroup
            .append('text')
            .attr('x', bx + halfWidth / 2)
            .attr('y', fy + FOLD_ROW_HEIGHT / 2 + 1)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', COLORS.train.stroke)
            .attr('font-size', '9px')
            .attr('font-weight', '600')
            .text('24-25 H1')

          /* Test half (right) */
          foldGroup
            .append('rect')
            .attr('x', bx + halfWidth + 1)
            .attr('y', fy)
            .attr('width', halfWidth - 1)
            .attr('height', FOLD_ROW_HEIGHT)
            .attr('rx', 6)
            .attr('ry', 6)
            .attr('fill', COLORS.test.fill)
            .attr('stroke', COLORS.test.stroke)
            .attr('stroke-width', 1)

          foldGroup
            .append('text')
            .attr('x', bx + halfWidth + 1 + halfWidth / 2)
            .attr('y', fy + FOLD_ROW_HEIGHT / 2 + 1)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', COLORS.test.stroke)
            .attr('font-size', '9px')
            .attr('font-weight', '600')
            .text('24-25 H2')
        }
      })

      /* Arrow between train and test sections */
      const lastTrainIdx = fold.filter(
        (s) => s.role === 'train' || s.role === 'train-half',
      ).length - 1
      const lastTrainSeg = fold.filter(
        (s) => s.role === 'train' || s.role === 'train-half',
      )[lastTrainIdx]
      const firstTestSeg = fold.find((s) => s.role === 'test')

      if (lastTrainSeg && firstTestSeg) {
        const arrowStartX =
          labelOffset +
          sideMargin +
          lastTrainSeg.yearIndex * (blockWidth + blockGap) +
          blockWidth +
          4
        const arrowEndX =
          labelOffset +
          sideMargin +
          firstTestSeg.yearIndex * (blockWidth + blockGap) -
          4
        const arrowY = fy + FOLD_ROW_HEIGHT / 2

        if (arrowEndX - arrowStartX > 12) {
          /* Arrowhead marker (unique per fold to avoid ID collisions) */
          svg
            .append('defs')
            .append('marker')
            .attr('id', `fold-arrow-${foldIdx}`)
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 10)
            .attr('refY', 5)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
            .attr('fill', COLORS.text.darkGray)

          foldGroup
            .append('line')
            .attr('x1', arrowStartX)
            .attr('y1', arrowY)
            .attr('x2', arrowEndX)
            .attr('y2', arrowY)
            .attr('stroke', COLORS.text.darkGray)
            .attr('stroke-width', 1.5)
            .attr('marker-end', `url(#fold-arrow-${foldIdx})`)
        }
      }
    })

    /* ─────────────────────────────────────
       Legend at the bottom-right of the
       expanding window section
       ───────────────────────────────────── */
    const legendY = foldStartY + foldSectionHeight - 6
    const legendGroup = svg.append('g')

    /* Train swatch */
    legendGroup
      .append('rect')
      .attr('x', width - sideMargin - 170)
      .attr('y', legendY)
      .attr('width', 12)
      .attr('height', 12)
      .attr('rx', 2)
      .attr('fill', COLORS.train.fill)
      .attr('stroke', COLORS.train.stroke)
      .attr('stroke-width', 1)

    legendGroup
      .append('text')
      .attr('x', width - sideMargin - 154)
      .attr('y', legendY + 10)
      .attr('fill', COLORS.text.gray)
      .attr('font-size', '10px')
      .text('Train')

    /* Test swatch */
    legendGroup
      .append('rect')
      .attr('x', width - sideMargin - 90)
      .attr('y', legendY)
      .attr('width', 12)
      .attr('height', 12)
      .attr('rx', 2)
      .attr('fill', COLORS.test.fill)
      .attr('stroke', COLORS.test.stroke)
      .attr('stroke-width', 1)

    legendGroup
      .append('text')
      .attr('x', width - sideMargin - 74)
      .attr('y', legendY + 10)
      .attr('fill', COLORS.text.gray)
      .attr('font-size', '10px')
      .text('Test / Validate')
  }, [mode])

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

  return (
    <div className="space-y-4">
      {/* Toggle buttons */}
      <div className="flex gap-1 p-1 rounded-lg bg-white/5 border border-white/10 w-fit">
        <button
          onClick={() => setMode('random')}
          className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
            mode === 'random'
              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
              : 'text-gray-500 hover:text-gray-300 border border-transparent'
          }`}
        >
          Random Split
        </button>
        <button
          onClick={() => setMode('temporal')}
          className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
            mode === 'temporal'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'text-gray-500 hover:text-gray-300 border border-transparent'
          }`}
        >
          Temporal Split
        </button>
      </div>

      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} className="w-full" />
      </div>

      <p className="text-xs text-gray-600">
        Temporal splitting prevents data leakage by ensuring the model is only
        trained on past admissions cycles. The expanding window cross-validation
        progressively grows the training set while always validating on unseen
        future data.
      </p>
    </div>
  )
}

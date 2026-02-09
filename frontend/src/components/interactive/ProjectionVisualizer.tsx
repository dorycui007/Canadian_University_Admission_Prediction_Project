import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   3D vector type and math helpers
   ───────────────────────────────────────────── */
type Vec3 = [number, number, number]

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function norm3(v: Vec3): number {
  return Math.sqrt(dot3(v, v))
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function clamp(val: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, val))
}

/* ─────────────────────────────────────────────
   Oblique projection: 3D -> 2D
   Cabinet projection with foreshortening = 0.5
   z-axis projects at 30 degrees
   ───────────────────────────────────────────── */
const Z_ANGLE = (Math.PI * 210) / 180 // z-axis goes lower-left
const Z_SCALE = 0.5 // cabinet foreshortening

function project3Dto2D(p: Vec3): [number, number] {
  const px = p[0] + Z_SCALE * p[2] * Math.cos(Z_ANGLE)
  const py = p[1] + Z_SCALE * p[2] * Math.sin(Z_ANGLE)
  return [px, -py] // negate y for SVG coordinate system
}

/* ─────────────────────────────────────────────
   Inverse: given a 2D drag position, recover
   the 3D y vector. We fix one DOF by keeping
   the y[2] component as-is during dragging,
   then solve for y[0] and y[1] from the 2D
   projected coordinates.
   ───────────────────────────────────────────── */
function unproject2DTo3D(sx: number, sy: number, z: number): Vec3 {
  // sx = x + Z_SCALE * z * cos(Z_ANGLE)
  // -sy = y + Z_SCALE * z * sin(Z_ANGLE)   (we negated y in project)
  const x = sx - Z_SCALE * z * Math.cos(Z_ANGLE)
  const y = -sy - Z_SCALE * z * Math.sin(Z_ANGLE)
  return [x, y, z]
}

/* ─────────────────────────────────────────────
   Hat matrix computation for a 3x2 matrix X
   H = X (X^T X)^{-1} X^T
   ───────────────────────────────────────────── */
function computeHatMatrix(X: [Vec3, Vec3]): number[][] {
  const [c1, c2] = X

  // X^T X is 2x2
  const a = dot3(c1, c1)
  const b = dot3(c1, c2)
  const c = dot3(c2, c1)
  const d = dot3(c2, c2)

  // Invert the 2x2 matrix
  const det = a * d - b * c
  if (Math.abs(det) < 1e-12) {
    return [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]
  }

  const inv = [
    [d / det, -b / det],
    [-c / det, a / det],
  ]

  // H = X * inv * X^T  (3x3)
  // X is 3x2 with columns c1, c2
  // X[i][j] = [c1, c2][j][i]
  const H: number[][] = []
  for (let i = 0; i < 3; i++) {
    H[i] = []
    for (let j = 0; j < 3; j++) {
      let val = 0
      for (let k = 0; k < 2; k++) {
        for (let l = 0; l < 2; l++) {
          const Xi_k = k === 0 ? c1[i] : c2[i]
          const Xj_l = l === 0 ? c1[j] : c2[j]
          val += Xi_k * inv[k][l] * Xj_l
        }
      }
      H[i][j] = val
    }
  }

  return H
}

function matVec3(M: number[][], v: Vec3): Vec3 {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ]
}

/* ─────────────────────────────────────────────
   Constants
   ───────────────────────────────────────────── */
const X_COLUMNS: [Vec3, Vec3] = [
  [1, 0, 0.5],
  [0, 1, 0.5],
]

const INITIAL_Y: Vec3 = [1.5, 1.0, 2.5]

const SVG_WIDTH = 500
const SVG_HEIGHT = 400

const SCALE = 50 // pixels per unit

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function ProjectionVisualizer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [y, setY] = useState<Vec3>(INITIAL_Y)

  /* ─── Derived computations ────────────────── */
  const H = useMemo(() => computeHatMatrix(X_COLUMNS), [])

  const yHat = useMemo<Vec3>(() => matVec3(H, y), [H, y])

  const e = useMemo<Vec3>(() => sub3(y, yHat), [y, yHat])

  const normY = norm3(y)
  const normYHat = norm3(yHat)
  const normE = norm3(e)

  const angle = useMemo(() => {
    if (normY < 1e-8 || normYHat < 1e-8) return 0
    const cosA = clamp(dot3(y, yHat) / (normY * normYHat), -1, 1)
    return (Math.acos(cosA) * 180) / Math.PI
  }, [y, yHat, normY, normYHat])

  /* ─── Stable drag callback ────────────────── */
  const yRef = useRef(y)
  yRef.current = y

  const onDrag = useCallback(
    (svgX: number, svgY: number) => {
      const newY = unproject2DTo3D(svgX, svgY, yRef.current[2])
      setY([
        Math.round(newY[0] * 100) / 100,
        Math.round(newY[1] * 100) / 100,
        Math.round(newY[2] * 100) / 100,
      ])
    },
    [],
  )

  /* ─── D3 rendering ────────────────────────── */
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const totalWidth = Math.max(container.clientWidth, 320)
    const height = SVG_HEIGHT

    svg
      .attr('viewBox', `0 0 ${totalWidth} ${height}`)
      .attr('width', totalWidth)
      .attr('height', height)

    const cx = totalWidth / 2
    const cy = height / 2 + 20

    // Helper: 3D world point to SVG pixel
    function toSvg(p: Vec3): [number, number] {
      const [px, py] = project3Dto2D(p)
      return [cx + px * SCALE, cy + py * SCALE]
    }

    const defs = svg.append('defs')

    // Arrow markers
    const markers = [
      { id: 'arr-y', color: '#ffffff' },
      { id: 'arr-yhat', color: '#34d399' },
      { id: 'arr-e', color: '#f87171' },
      { id: 'arr-axis', color: '#6b7280' },
    ]
    markers.forEach(({ id, color }) => {
      defs
        .append('marker')
        .attr('id', id)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 9)
        .attr('refY', 5)
        .attr('markerWidth', 7)
        .attr('markerHeight', 7)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 1 L 10 5 L 0 9 Z')
        .attr('fill', color)
    })

    const g = svg.append('g')

    // ─── Draw the column space plane ───
    // The plane is spanned by c1 and c2.
    // Draw a filled parallelogram: +/-2 * c1 +/-2 * c2
    const planeExtent = 3.0
    const corners: Vec3[] = [
      [
        -planeExtent * X_COLUMNS[0][0] - planeExtent * X_COLUMNS[1][0],
        -planeExtent * X_COLUMNS[0][1] - planeExtent * X_COLUMNS[1][1],
        -planeExtent * X_COLUMNS[0][2] - planeExtent * X_COLUMNS[1][2],
      ],
      [
        planeExtent * X_COLUMNS[0][0] - planeExtent * X_COLUMNS[1][0],
        planeExtent * X_COLUMNS[0][1] - planeExtent * X_COLUMNS[1][1],
        planeExtent * X_COLUMNS[0][2] - planeExtent * X_COLUMNS[1][2],
      ],
      [
        planeExtent * X_COLUMNS[0][0] + planeExtent * X_COLUMNS[1][0],
        planeExtent * X_COLUMNS[0][1] + planeExtent * X_COLUMNS[1][1],
        planeExtent * X_COLUMNS[0][2] + planeExtent * X_COLUMNS[1][2],
      ],
      [
        -planeExtent * X_COLUMNS[0][0] + planeExtent * X_COLUMNS[1][0],
        -planeExtent * X_COLUMNS[0][1] + planeExtent * X_COLUMNS[1][1],
        -planeExtent * X_COLUMNS[0][2] + planeExtent * X_COLUMNS[1][2],
      ],
    ]

    const svgCorners = corners.map(toSvg)
    const planePathData =
      `M ${svgCorners[0][0]} ${svgCorners[0][1]} ` +
      svgCorners
        .slice(1)
        .map(([x, yp]) => `L ${x} ${yp}`)
        .join(' ') +
      ' Z'

    g.append('path')
      .attr('d', planePathData)
      .attr('fill', 'rgba(52,211,153,0.07)')
      .attr('stroke', 'rgba(52,211,153,0.15)')
      .attr('stroke-width', 1)

    // Plane grid lines for visual texture
    const gridSteps = 7
    for (let i = 0; i <= gridSteps; i++) {
      const t = -planeExtent + (2 * planeExtent * i) / gridSteps
      // Lines along c1 direction
      const startC1: Vec3 = [
        t * X_COLUMNS[0][0] - planeExtent * X_COLUMNS[1][0],
        t * X_COLUMNS[0][1] - planeExtent * X_COLUMNS[1][1],
        t * X_COLUMNS[0][2] - planeExtent * X_COLUMNS[1][2],
      ]
      const endC1: Vec3 = [
        t * X_COLUMNS[0][0] + planeExtent * X_COLUMNS[1][0],
        t * X_COLUMNS[0][1] + planeExtent * X_COLUMNS[1][1],
        t * X_COLUMNS[0][2] + planeExtent * X_COLUMNS[1][2],
      ]
      const [sx1, sy1] = toSvg(startC1)
      const [ex1, ey1] = toSvg(endC1)
      g.append('line')
        .attr('x1', sx1)
        .attr('y1', sy1)
        .attr('x2', ex1)
        .attr('y2', ey1)
        .attr('stroke', 'rgba(52,211,153,0.06)')
        .attr('stroke-width', 0.5)

      // Lines along c2 direction
      const startC2: Vec3 = [
        -planeExtent * X_COLUMNS[0][0] + t * X_COLUMNS[1][0],
        -planeExtent * X_COLUMNS[0][1] + t * X_COLUMNS[1][1],
        -planeExtent * X_COLUMNS[0][2] + t * X_COLUMNS[1][2],
      ]
      const endC2: Vec3 = [
        planeExtent * X_COLUMNS[0][0] + t * X_COLUMNS[1][0],
        planeExtent * X_COLUMNS[0][1] + t * X_COLUMNS[1][1],
        planeExtent * X_COLUMNS[0][2] + t * X_COLUMNS[1][2],
      ]
      const [sx2, sy2] = toSvg(startC2)
      const [ex2, ey2] = toSvg(endC2)
      g.append('line')
        .attr('x1', sx2)
        .attr('y1', sy2)
        .attr('x2', ex2)
        .attr('y2', ey2)
        .attr('stroke', 'rgba(52,211,153,0.06)')
        .attr('stroke-width', 0.5)
    }

    // Plane label
    const planeLabelPos: Vec3 = [
      planeExtent * X_COLUMNS[0][0] + planeExtent * X_COLUMNS[1][0],
      planeExtent * X_COLUMNS[0][1] + planeExtent * X_COLUMNS[1][1],
      planeExtent * X_COLUMNS[0][2] + planeExtent * X_COLUMNS[1][2],
    ]
    const [plx, ply] = toSvg(planeLabelPos)
    g.append('text')
      .attr('x', plx + 8)
      .attr('y', ply - 4)
      .attr('fill', 'rgba(52,211,153,0.5)')
      .attr('font-size', '11px')
      .attr('font-style', 'italic')
      .text('col(X)')

    // ─── Axes ───
    const axisLen = 3.5
    const axes = [
      { dir: [axisLen, 0, 0] as Vec3, label: 'x\u2081' },
      { dir: [0, axisLen, 0] as Vec3, label: 'x\u2082' },
      { dir: [0, 0, axisLen] as Vec3, label: 'x\u2083' },
    ]
    const origin: Vec3 = [0, 0, 0]
    const [ox, oy] = toSvg(origin)

    axes.forEach(({ dir, label }) => {
      const [ax, ay] = toSvg(dir)
      g.append('line')
        .attr('x1', ox)
        .attr('y1', oy)
        .attr('x2', ax)
        .attr('y2', ay)
        .attr('stroke', '#4b5563')
        .attr('stroke-width', 1)
        .attr('marker-end', 'url(#arr-axis)')

      g.append('text')
        .attr('x', ax + 8)
        .attr('y', ay + 4)
        .attr('fill', '#6b7280')
        .attr('font-size', '11px')
        .text(label)
    })

    // Origin dot
    g.append('circle')
      .attr('cx', ox)
      .attr('cy', oy)
      .attr('r', 2.5)
      .attr('fill', '#6b7280')

    // ─── Residual vector e (dashed, red) ───
    const [ysx, ysy] = toSvg(y)
    const [yhsx, yhsy] = toSvg(yHat)

    g.append('line')
      .attr('x1', yhsx)
      .attr('y1', yhsy)
      .attr('x2', ysx)
      .attr('y2', ysy)
      .attr('stroke', '#f87171')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .attr('marker-end', 'url(#arr-e)')
      .attr('opacity', 0.8)

    // Right-angle mark at yHat
    if (normE > 0.1) {
      const eDir = sub3(y, yHat)
      const eNorm = norm3(eDir)
      const eUnit: Vec3 = [eDir[0] / eNorm, eDir[1] / eNorm, eDir[2] / eNorm]

      // Pick a direction along the plane (use yHat direction)
      const yhNorm = norm3(yHat)
      if (yhNorm > 0.1) {
        const yhUnit: Vec3 = [
          yHat[0] / yhNorm,
          yHat[1] / yhNorm,
          yHat[2] / yhNorm,
        ]
        const markSize = 0.25
        const p1: Vec3 = [
          yHat[0] + markSize * eUnit[0],
          yHat[1] + markSize * eUnit[1],
          yHat[2] + markSize * eUnit[2],
        ]
        const p2: Vec3 = [
          yHat[0] + markSize * eUnit[0] - markSize * yhUnit[0],
          yHat[1] + markSize * eUnit[1] - markSize * yhUnit[1],
          yHat[2] + markSize * eUnit[2] - markSize * yhUnit[2],
        ]
        const p3: Vec3 = [
          yHat[0] - markSize * yhUnit[0],
          yHat[1] - markSize * yhUnit[1],
          yHat[2] - markSize * yhUnit[2],
        ]
        const [rp1x, rp1y] = toSvg(p1)
        const [rp2x, rp2y] = toSvg(p2)
        const [rp3x, rp3y] = toSvg(p3)

        g.append('path')
          .attr(
            'd',
            `M ${rp1x} ${rp1y} L ${rp2x} ${rp2y} L ${rp3x} ${rp3y}`,
          )
          .attr('fill', 'none')
          .attr('stroke', 'rgba(248,113,113,0.5)')
          .attr('stroke-width', 1.2)
      }
    }

    // Residual label
    const eMid: Vec3 = [
      (y[0] + yHat[0]) / 2,
      (y[1] + yHat[1]) / 2,
      (y[2] + yHat[2]) / 2,
    ]
    const [emx, emy] = toSvg(eMid)
    g.append('text')
      .attr('x', emx + 10)
      .attr('y', emy)
      .attr('fill', '#f87171')
      .attr('font-size', '11px')
      .attr('font-style', 'italic')
      .text('e = y \u2212 \u0177')

    // ─── Projection vector yHat (emerald, from origin) ───
    g.append('line')
      .attr('x1', ox)
      .attr('y1', oy)
      .attr('x2', yhsx)
      .attr('y2', yhsy)
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#arr-yhat)')

    // yHat dot
    g.append('circle')
      .attr('cx', yhsx)
      .attr('cy', yhsy)
      .attr('r', 4)
      .attr('fill', '#34d399')

    // yHat label
    g.append('text')
      .attr('x', yhsx - 8)
      .attr('y', yhsy + 18)
      .attr('fill', '#34d399')
      .attr('font-size', '11px')
      .attr('font-weight', '700')
      .attr('font-style', 'italic')
      .text('\u0177 = Hy')

    // ─── Vector y (white, from origin) ───
    g.append('line')
      .attr('x1', ox)
      .attr('y1', oy)
      .attr('x2', ysx)
      .attr('y2', ysy)
      .attr('stroke', '#e5e7eb')
      .attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#arr-y)')

    // y label
    g.append('text')
      .attr('x', ysx + 12)
      .attr('y', ysy - 12)
      .attr('fill', '#ffffff')
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .attr('font-style', 'italic')
      .text('y')

    // ─── Draggable handle ───
    const handle = g
      .append('circle')
      .attr('cx', ysx)
      .attr('cy', ysy)
      .attr('r', 9)
      .attr('fill', '#ffffff')
      .attr('stroke', '#34d399')
      .attr('stroke-width', 2)
      .style('cursor', 'grab')
      .attr('filter', 'drop-shadow(0 0 4px rgba(52,211,153,0.4))')

    // "drag me" hint
    g.append('text')
      .attr('x', ysx + 16)
      .attr('y', ysy + 4)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .attr('class', 'drag-hint')
      .text('drag me')

    // ─── Drag behavior ───
    const drag = d3
      .drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing')
        // Hide the drag-me hint
        svg.selectAll('.drag-hint').attr('opacity', 0)
      })
      .on('drag', function (event) {
        const [mx, my] = d3.pointer(event, svg.node())
        // Convert SVG pixel coordinates back to world units
        const worldX = (mx - cx) / SCALE
        const worldY = (my - cy) / SCALE
        onDrag(worldX, worldY)
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'grab')
      })

    handle.call(drag)
  }, [y, yHat, e, normE, onDrag])

  /* ─── Resize handler ──────────────────────── */
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver(() => {
      setY((prev) => [...prev] as Vec3)
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  /* ─── Render ──────────────────────────────── */
  return (
    <div className="space-y-4">
      {/* Title */}
      <div>
        <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-1">
          Interactive
        </p>
        <p className="text-white font-semibold">
          Projection of <M>{"y"}</M> onto <M>{"\\text{col}(X)"}</M>
        </p>
        <p className="text-gray-500 text-sm mt-1">
          Drag the white handle to move <M>{"y"}</M> and watch{' '}
          <M>{"\\hat{y}"}</M> and <M>{"e"}</M> update in real time.
        </p>
      </div>

      {/* SVG visualization */}
      <div ref={containerRef} className="w-full">
        <svg
          ref={svgRef}
          className="w-full"
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      {/* Live values */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <div>
          <M>{`\\|y\\| = ${normY.toFixed(2)}`}</M>
        </div>
        <div>
          <span className="text-emerald-400">
            <M>{`\\|\\hat{y}\\| = ${normYHat.toFixed(2)}`}</M>
          </span>
        </div>
        <div>
          <span className="text-red-400">
            <M>{`\\|e\\| = ${normE.toFixed(2)}`}</M>
          </span>
        </div>
        <div>
          <M>{`\\angle(y, \\hat{y}) = ${angle.toFixed(1)}^\\circ`}</M>
        </div>
      </div>

      {/* Key relationships */}
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-gray-500">
        <span>
          <M>{"\\hat{y} = Hy"}</M>
        </span>
        <span>
          <M>{"e = y - \\hat{y}"}</M>
        </span>
        <span>
          <M>{"e \\perp \\text{col}(X)"}</M>
        </span>
        <span>
          <M>{"\\|y\\|^2 = \\|\\hat{y}\\|^2 + \\|e\\|^2"}</M>
        </span>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   Emerald Vector Waves — topographic contour
   line-art background spanning the full viewport.

   Three clusters of parallel flowing lines plus
   sparse solo accents. Opacity tapers at cluster
   edges to create depth, like real contour maps.
   ───────────────────────────────────────────── */

const WIDTH = 1440
const HEIGHT = 900

/* ── Catmull-Rom → cubic-bezier ──────────────── */

function toSmoothPath(pts: [number, number][]): string {
  const f = (n: number) => n.toFixed(1)
  let d = `M${f(pts[0][0])},${f(pts[0][1])}`

  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)]
    const p1 = pts[i]
    const p2 = pts[i + 1]
    const p3 = pts[Math.min(pts.length - 1, i + 2)]

    const cp1x = p1[0] + (p2[0] - p0[0]) / 6
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6

    d += ` C${f(cp1x)},${f(cp1y)} ${f(cp2x)},${f(cp2y)} ${f(p2[0])},${f(p2[1])}`
  }
  return d
}

/* ── Cluster definitions ─────────────────────── */

/* Each cluster is a hand-crafted master curve with
   parallel offsets. Opacity tapers: center = bold,
   edges = faint, mimicking topographic contours. */

const groups: {
  keyPoints: [number, number][]
  offsets: number[]
  opacities: number[]
}[] = [
  {
    // Upper flow — gentle, slow undulation
    keyPoints: [
      [0, 170], [240, 135], [480, 205], [720, 150],
      [960, 200], [1200, 155], [1440, 180],
    ],
    offsets:   [-36, -18, 0, 18, 36],
    opacities: [0.12, 0.24, 0.38, 0.24, 0.12],
  },
  {
    // Middle flow — more dynamic, tighter curves
    keyPoints: [
      [0, 435], [200, 490], [420, 410], [640, 480],
      [860, 420], [1080, 505], [1300, 440], [1440, 465],
    ],
    offsets:   [-44, -22, 0, 22, 44],
    opacities: [0.10, 0.22, 0.35, 0.22, 0.10],
  },
  {
    // Lower flow — wide, calm swells
    keyPoints: [
      [0, 705], [220, 660], [440, 740], [660, 675],
      [880, 735], [1100, 685], [1320, 720], [1440, 695],
    ],
    offsets:   [-28, -14, 0, 14, 28],
    opacities: [0.10, 0.20, 0.30, 0.20, 0.10],
  },
]

/* Solo accent lines — sparse, faint, add rhythm
   between the main clusters */
const soloLines: { points: [number, number][]; opacity: number }[] = [
  {
    points: [[0, 305], [360, 330], [720, 295], [1080, 325], [1440, 310]],
    opacity: 0.10,
  },
  {
    points: [[0, 575], [300, 600], [600, 565], [900, 598], [1200, 570], [1440, 588]],
    opacity: 0.08,
  },
  {
    points: [[0, 840], [480, 860], [960, 835], [1440, 855]],
    opacity: 0.07,
  },
]

/* ── Pre-compute all paths ───────────────────── */

const allPaths: { d: string; opacity: number }[] = []

for (const g of groups) {
  for (let i = 0; i < g.offsets.length; i++) {
    const pts = g.keyPoints.map(
      ([x, y]) => [x, y + g.offsets[i]] as [number, number],
    )
    allPaths.push({ d: toSmoothPath(pts), opacity: g.opacities[i] })
  }
}

for (const s of soloLines) {
  allPaths.push({ d: toSmoothPath(s.points), opacity: s.opacity })
}

/* ── Component ───────────────────────────────── */

export default function EmeraldWaves() {
  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 w-full h-full overflow-hidden z-0 pointer-events-none"
    >
      <svg
        className="block w-full h-full"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="xMidYMid slice"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {allPaths.map((p, i) => (
          <path
            key={i}
            d={p.d}
            stroke="#10b981"
            strokeWidth={1.5}
            strokeOpacity={p.opacity}
            fill="none"
          />
        ))}
      </svg>
    </div>
  )
}

import { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

/* ─────────────────────────────────────────────
   Canonical university names
   ───────────────────────────────────────────── */
const UNIVERSITIES = [
  'University of Toronto',
  'University of Waterloo',
  'McGill University',
  'University of British Columbia',
  'McMaster University',
  'Western University',
  'Queen\'s University',
  'University of Ottawa',
  'University of Alberta',
  'York University',
  'Ryerson University',
  'University of Calgary',
  'Carleton University',
  'University of Guelph',
  'Simon Fraser University',
]

/* ─────────────────────────────────────────────
   Simple token-sort similarity (no dependency)
   ───────────────────────────────────────────── */
function tokenSortSimilarity(a: string, b: string): number {
  const normalize = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).sort().join(' ')

  const na = normalize(a)
  const nb = normalize(b)

  if (na === nb) return 1
  if (!na || !nb) return 0

  // Levenshtein-based similarity
  const maxLen = Math.max(na.length, nb.length)
  if (maxLen === 0) return 1

  const dist = levenshtein(na, nb)
  return 1 - dist / maxLen
}

function levenshtein(a: string, b: string): number {
  const m = a.length
  const n = b.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))

  for (let i = 0; i <= m; i++) dp[i][0] = i
  for (let j = 0; j <= n; j++) dp[0][j] = j

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    }
  }

  return dp[m][n]
}

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function FuzzyMatchDemo() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [query, setQuery] = useState('uoft')
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null)

  const [debouncedQuery, setDebouncedQuery] = useState(query)

  // Debounce input
  useEffect(() => {
    debounceRef.current = setTimeout(() => setDebouncedQuery(query), 150)
    return () => clearTimeout(debounceRef.current ?? undefined)
  }, [query])

  // Compute scores
  const scores = useMemo(() => {
    if (!debouncedQuery.trim()) {
      return UNIVERSITIES.map((u) => ({ name: u, score: 0 }))
    }
    return UNIVERSITIES.map((u) => ({
      name: u,
      score: tokenSortSimilarity(debouncedQuery, u),
    })).sort((a, b) => b.score - a.score)
  }, [debouncedQuery])

  // D3 bar chart
  useEffect(() => {
    const svgEl = svgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const barHeight = 22
    const gap = 3
    const margin = { top: 8, right: 50, bottom: 8, left: 160 }
    const height = scores.length * (barHeight + gap) + margin.top + margin.bottom

    svg.attr('width', width).attr('height', height)

    const x = d3
      .scaleLinear()
      .domain([0, 1])
      .range([margin.left, width - margin.right])

    const threshold = 0.85

    // Threshold line
    svg
      .append('line')
      .attr('x1', x(threshold))
      .attr('y1', margin.top)
      .attr('x2', x(threshold))
      .attr('y2', height - margin.bottom)
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6)

    svg
      .append('text')
      .attr('x', x(threshold) + 4)
      .attr('y', margin.top - 1)
      .attr('fill', '#f59e0b')
      .attr('font-size', '9px')
      .text('85% threshold')

    // Bars
    scores.forEach((d, i) => {
      const y = margin.top + i * (barHeight + gap)
      const barWidth = Math.max(0, x(d.score) - margin.left)

      // Interpolate gray → emerald
      const color =
        d.score >= threshold
          ? '#10b981'
          : d3.interpolateRgb('#4b5563', '#10b981')(d.score / threshold)

      svg
        .append('rect')
        .attr('x', margin.left)
        .attr('y', y)
        .attr('width', barWidth)
        .attr('height', barHeight)
        .attr('rx', 3)
        .attr('fill', color)
        .attr('opacity', d.score > 0 ? 0.7 : 0.15)

      // University name
      svg
        .append('text')
        .attr('x', margin.left - 6)
        .attr('y', y + barHeight / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', d.score >= threshold ? '#fff' : '#9ca3af')
        .attr('font-size', '10px')
        .text(d.name.length > 22 ? d.name.slice(0, 20) + '...' : d.name)

      // Score label
      if (d.score > 0) {
        svg
          .append('text')
          .attr('x', margin.left + barWidth + 4)
          .attr('y', y + barHeight / 2 + 4)
          .attr('fill', d.score >= threshold ? '#10b981' : '#6b7280')
          .attr('font-size', '10px')
          .attr('font-weight', d.score >= threshold ? '600' : '400')
          .text(`${(d.score * 100).toFixed(0)}%`)
      }
    })
  }, [scores])

  const bestMatch = scores[0]

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <label className="text-xs text-gray-500">
          Type a university name
        </label>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., uoft, waterloo, mcgill..."
          className="w-full px-3 py-2 rounded-lg text-sm text-white bg-white/5 border border-white/10 outline-none transition-colors focus:border-emerald-400 placeholder-gray-500"
        />
      </div>

      {bestMatch && bestMatch.score >= 0.85 && (
        <p className="text-xs text-emerald-400">
          Best match: <span className="font-bold">{bestMatch.name}</span> ({(bestMatch.score * 100).toFixed(0)}%)
        </p>
      )}

      <div ref={containerRef} className="w-full overflow-x-auto">
        <svg ref={svgRef} className="w-full" />
      </div>
    </div>
  )
}

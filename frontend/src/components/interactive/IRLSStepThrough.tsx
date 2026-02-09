import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import StepNavigator from '@/components/shared/StepNavigator'
import useReducedMotion from '@/hooks/useReducedMotion'

/* ─────────────────────────────────────────────
   Pre-computed IRLS iteration snapshots
   (educational approximation — not live)
   ───────────────────────────────────────────── */
const FEATURE_NAMES = ['bias', 'top6', 'gr12', 'interact', 'ontario', 'bc', 'alberta', 'quebec']

const ITERATIONS = [
  { coefficients: [0, 0, 0, 0, 0, 0, 0, 0], loss: 0.693 },
  { coefficients: [-0.12, 0.02, 0.005, 0.008, 0.03, 0.01, 0.01, 0.02], loss: 0.52 },
  { coefficients: [-0.25, 0.04, 0.01, 0.015, 0.06, 0.025, 0.025, 0.04], loss: 0.41 },
  { coefficients: [-0.35, 0.06, 0.015, 0.022, 0.08, 0.038, 0.038, 0.06], loss: 0.33 },
  { coefficients: [-0.42, 0.07, 0.018, 0.026, 0.09, 0.044, 0.044, 0.07], loss: 0.28 },
  { coefficients: [-0.47, 0.075, 0.019, 0.028, 0.095, 0.048, 0.048, 0.075], loss: 0.25 },
  { coefficients: [-0.49, 0.078, 0.0195, 0.029, 0.098, 0.049, 0.049, 0.078], loss: 0.235 },
  { coefficients: [-0.50, 0.08, 0.02, 0.03, 0.10, 0.05, 0.05, 0.08], loss: 0.23 },
]

/* ─────────────────────────────────────────────
   Component
   ───────────────────────────────────────────── */
export default function IRLSStepThrough() {
  const barSvgRef = useRef<SVGSVGElement>(null)
  const lossSvgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const reducedMotion = useReducedMotion()

  const [step, setStep] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)

  // Autoplay
  useEffect(() => {
    if (!autoPlay) return
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= ITERATIONS.length - 1) {
          setAutoPlay(false)
          return s
        }
        return s + 1
      })
    }, reducedMotion ? 400 : 800)
    return () => clearInterval(id)
  }, [autoPlay, reducedMotion])

  const handlePrev = useCallback(() => setStep((s) => Math.max(0, s - 1)), [])
  const handleNext = useCallback(
    () => setStep((s) => Math.min(ITERATIONS.length - 1, s + 1)),
    [],
  )

  const iteration = ITERATIONS[step]

  // Coefficient bar chart
  useEffect(() => {
    const svgEl = barSvgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const barHeight = 22
    const gap = 4
    const margin = { top: 8, right: 48, bottom: 8, left: 64 }
    const height =
      FEATURE_NAMES.length * (barHeight + gap) + margin.top + margin.bottom

    svg.attr('width', width).attr('height', height)

    const maxAbs = 0.55
    const x = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([margin.left, width - margin.right])

    const zeroX = x(0)

    // Zero line
    svg
      .append('line')
      .attr('x1', zeroX)
      .attr('y1', margin.top)
      .attr('x2', zeroX)
      .attr('y2', height - margin.bottom)
      .attr('stroke', 'rgba(255,255,255,0.1)')

    FEATURE_NAMES.forEach((name, i) => {
      const coeff = iteration.coefficients[i]
      const y = margin.top + i * (barHeight + gap)
      const barX = coeff >= 0 ? zeroX : x(coeff)
      const barW = Math.abs(x(coeff) - zeroX)
      const color = coeff >= 0 ? '#10b981' : '#ef4444'

      svg
        .append('rect')
        .attr('x', barX)
        .attr('y', y)
        .attr('width', Math.max(1, barW))
        .attr('height', barHeight)
        .attr('rx', 3)
        .attr('fill', color)
        .attr('opacity', 0.7)

      svg
        .append('text')
        .attr('x', margin.left - 4)
        .attr('y', y + barHeight / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .text(name)

      svg
        .append('text')
        .attr('x', coeff >= 0 ? zeroX + barW + 4 : barX - 4)
        .attr('y', y + barHeight / 2 + 4)
        .attr('text-anchor', coeff >= 0 ? 'start' : 'end')
        .attr('fill', color)
        .attr('font-size', '10px')
        .attr('font-weight', '600')
        .text(coeff >= 0 ? `+${coeff.toFixed(3)}` : coeff.toFixed(3))
    })
  }, [step, iteration.coefficients])

  // Loss curve
  useEffect(() => {
    const svgEl = lossSvgRef.current
    const container = containerRef.current
    if (!svgEl || !container) return

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const width = container.clientWidth
    const height = 140
    const margin = { top: 16, right: 24, bottom: 32, left: 44 }

    svg.attr('width', width).attr('height', height)

    const x = d3
      .scaleLinear()
      .domain([0, ITERATIONS.length - 1])
      .range([margin.left, width - margin.right])

    const y = d3
      .scaleLinear()
      .domain([0.2, 0.7])
      .range([height - margin.bottom, margin.top])

    // X axis
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(7).tickFormat((d) => `${d}`))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    svg
      .append('text')
      .attr('x', (margin.left + width - margin.right) / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Iteration')

    // Y axis
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.2f')))
      .call((g) => {
        g.select('.domain').attr('stroke', 'rgba(255,255,255,0.1)')
        g.selectAll('.tick line').attr('stroke', 'rgba(255,255,255,0.05)')
        g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px')
      })

    // Loss line (up to current step)
    const lossData = ITERATIONS.slice(0, step + 1).map((it, i) => ({
      i,
      loss: it.loss,
    }))

    const line = d3
      .line<{ i: number; loss: number }>()
      .x((d) => x(d.i))
      .y((d) => y(d.loss))

    svg
      .append('path')
      .datum(lossData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)

    // Points
    svg
      .selectAll('circle')
      .data(lossData)
      .join('circle')
      .attr('cx', (d) => x(d.i))
      .attr('cy', (d) => y(d.loss))
      .attr('r', (d) => (d.i === step ? 5 : 3))
      .attr('fill', (d) => (d.i === step ? '#10b981' : '#10b981'))
      .attr('stroke', (d) => (d.i === step ? '#050505' : 'none'))
      .attr('stroke-width', 2)
      .attr('opacity', (d) => (d.i === step ? 1 : 0.6))

    // Current loss label
    svg
      .append('text')
      .attr('x', x(step) + 8)
      .attr('y', y(iteration.loss) - 8)
      .attr('fill', '#fff')
      .attr('font-size', '11px')
      .attr('font-weight', '700')
      .text(`loss = ${iteration.loss.toFixed(3)}`)
  }, [step, iteration])

  return (
    <div className="space-y-4">
      <StepNavigator
        current={step}
        total={ITERATIONS.length}
        onPrev={handlePrev}
        onNext={handleNext}
        autoPlay={autoPlay}
        onAutoPlayToggle={() => setAutoPlay((p) => !p)}
      />

      <div ref={containerRef} className="space-y-4 w-full">
        <div>
          <p className="text-xs text-gray-500 mb-1">
            Coefficients at iteration {step}
          </p>
          <svg ref={barSvgRef} className="w-full" />
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Loss curve</p>
          <svg ref={lossSvgRef} className="w-full" />
        </div>
      </div>
    </div>
  )
}

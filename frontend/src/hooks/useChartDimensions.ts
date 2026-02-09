import { useRef, useState, useEffect, useCallback } from 'react'

/**
 * Responsive chart dimension hook using ResizeObserver.
 * Returns a containerRef to attach to the wrapper div, plus the current width.
 */
export default function useChartDimensions(fixedHeight?: number) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  const measure = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const w = Math.round(el.getBoundingClientRect().width)
    setWidth((prev) => (prev === w ? prev : w))
  }, [])

  useEffect(() => {
    measure()
    const el = containerRef.current
    if (!el) return

    let rafId: number
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(rafId)
      rafId = requestAnimationFrame(measure)
    })
    observer.observe(el)

    return () => {
      observer.disconnect()
      cancelAnimationFrame(rafId)
    }
  }, [measure])

  const height = fixedHeight ?? Math.round(width * 0.6)

  return { containerRef, width, height }
}

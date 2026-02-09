import { useRef, useCallback, useState } from 'react'
import type { ReactNode } from 'react'

interface TooltipState {
  visible: boolean
  x: number
  y: number
  content: ReactNode
}

/**
 * Glass-morphism chart tooltip hook.
 * Returns a containerRef (attach to wrapper div with position:relative),
 * show/hide callbacks for D3 events, and the TooltipElement to render.
 */
export function useChartTooltip() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    content: null,
  })

  const show = useCallback((event: MouseEvent, content: ReactNode) => {
    const container = containerRef.current
    if (!container) return

    const rect = container.getBoundingClientRect()
    let tx = event.clientX - rect.left + 12
    let ty = event.clientY - rect.top - 8

    // Flip if near right edge
    if (tx + 180 > rect.width) tx = tx - 192
    // Flip if near bottom
    if (ty + 100 > rect.height) ty = ty - 80

    setTooltip({ visible: true, x: tx, y: ty, content })
  }, [])

  const hide = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }))
  }, [])

  const TooltipElement = (
    <div
      style={{
        position: 'absolute',
        left: tooltip.x,
        top: tooltip.y,
        opacity: tooltip.visible ? 1 : 0,
        transform: tooltip.visible ? 'translateY(0)' : 'translateY(4px)',
        transition: 'opacity 150ms ease, transform 150ms ease',
        pointerEvents: 'none',
        zIndex: 50,
      }}
      className="
        px-3 py-2 rounded-lg text-xs
        bg-white/[0.08] border border-white/[0.12]
        backdrop-blur-xl shadow-2xl
        min-w-[120px] max-w-[220px]
      "
    >
      {tooltip.content}
    </div>
  )

  return { containerRef, show, hide, TooltipElement }
}

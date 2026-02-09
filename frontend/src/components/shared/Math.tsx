import katex from 'katex'
import { useMemo } from 'react'

interface MathProps {
  /** LaTeX expression string */
  children: string
  /** Display mode (block) vs inline */
  display?: boolean
  /** Additional CSS class */
  className?: string
}

/**
 * Renders a LaTeX expression using KaTeX.
 *
 * Usage:
 *   <M>{"\\beta"}</M>                    -- inline
 *   <M display>{"y = X\\beta"}</M>       -- block (centered)
 */
export default function M({ children, display = false, className = '' }: MathProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
        strict: false,
        trust: true,
      }),
    [children, display],
  )

  if (display) {
    return (
      <div
        className={`my-4 overflow-x-auto ${className}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

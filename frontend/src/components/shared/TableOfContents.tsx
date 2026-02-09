import { useState, useEffect, useRef } from 'react'

export interface TocItem {
  id: string
  label: string
  depth?: number
}

interface TableOfContentsProps {
  items: TocItem[]
}

export default function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('')
  const [mobileOpen, setMobileOpen] = useState(false)
  const observerRef = useRef<IntersectionObserver | null>(null)

  useEffect(() => {
    observerRef.current?.disconnect()

    const callback: IntersectionObserverCallback = (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)

      if (visible.length > 0) {
        setActiveId(visible[0].target.id)
      }
    }

    observerRef.current = new IntersectionObserver(callback, {
      rootMargin: '-80px 0px -60% 0px',
      threshold: 0,
    })

    items.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observerRef.current?.observe(el)
    })

    return () => observerRef.current?.disconnect()
  }, [items])

  const handleClick = (id: string) => {
    setMobileOpen(false)
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const activeLabel = items.find((i) => i.id === activeId)?.label ?? 'Contents'

  return (
    <>
      {/* Mobile dropdown */}
      <div className="lg:hidden sticky top-16 z-20 bg-[#050505]/90 backdrop-blur-md border-b border-white/5">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center justify-between w-full px-6 py-3 text-sm text-gray-300"
        >
          <span className="truncate">{activeLabel}</span>
          <span className="text-gray-500 text-xs ml-2">
            {mobileOpen ? '\u25B2' : '\u25BC'}
          </span>
        </button>
        {mobileOpen && (
          <div className="px-6 pb-3 space-y-1">
            {items.map((item) => (
              <button
                key={item.id}
                onClick={() => handleClick(item.id)}
                className={`block w-full text-left text-sm py-1.5 transition-colors ${
                  activeId === item.id
                    ? 'text-emerald-400 font-medium'
                    : 'text-gray-500 hover:text-gray-300'
                } ${item.depth && item.depth > 0 ? 'pl-4' : ''}`}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Desktop sticky sidebar */}
      <aside className="hidden lg:block sticky top-24 self-start w-56 shrink-0">
        <p className="text-xs uppercase tracking-widest text-gray-600 font-medium mb-3">
          On this page
        </p>
        <nav className="space-y-1 border-l border-white/10">
          {items.map((item) => (
            <button
              key={item.id}
              onClick={() => handleClick(item.id)}
              className={`block w-full text-left text-sm py-1 transition-colors ${
                item.depth && item.depth > 0 ? 'pl-6' : 'pl-3'
              } ${
                activeId === item.id
                  ? 'text-emerald-400 font-medium border-l-2 border-emerald-400 -ml-px'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>
    </>
  )
}

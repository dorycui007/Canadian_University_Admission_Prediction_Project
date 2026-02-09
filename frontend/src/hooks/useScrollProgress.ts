import { useRef, useState, useEffect } from 'react'

interface ScrollProgress {
  ref: React.RefObject<HTMLDivElement | null>
  isVisible: boolean
}

export default function useScrollProgress(
  threshold: number | number[] = [0, 0.25, 0.5, 0.75, 1.0],
): ScrollProgress {
  const ref = useRef<HTMLDivElement | null>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          observer.unobserve(el)
        }
      },
      { threshold },
    )

    observer.observe(el)
    return () => observer.disconnect()
  // eslint-disable-next-line react-hooks/exhaustive-deps -- threshold is stable on mount
  }, [])

  return { ref, isVisible }
}

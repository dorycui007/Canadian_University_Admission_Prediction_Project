import type { ReactNode } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
  padding?: string
}

export default function GlassCard({
  children,
  className = '',
  padding = 'p-6',
}: GlassCardProps) {
  return (
    <div
      className={`bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl ${padding} ${className}`}
    >
      {children}
    </div>
  )
}

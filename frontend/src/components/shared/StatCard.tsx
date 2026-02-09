import GlassCard from '@/components/shared/GlassCard'

interface StatCardProps {
  value: string
  label: string
  description?: string
}

export default function StatCard({ value, label, description }: StatCardProps) {
  return (
    <GlassCard>
      <p className="text-3xl font-bold text-white tabular-nums">{value}</p>
      <p className="mt-1 text-sm text-gray-400">{label}</p>
      {description && (
        <p className="mt-2 text-xs text-gray-500">{description}</p>
      )}
    </GlassCard>
  )
}

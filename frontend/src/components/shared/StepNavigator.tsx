import GlassCard from '@/components/shared/GlassCard'

interface StepNavigatorProps {
  current: number
  total: number
  onPrev: () => void
  onNext: () => void
  autoPlay?: boolean
  onAutoPlayToggle?: () => void
}

export default function StepNavigator({
  current,
  total,
  onPrev,
  onNext,
  autoPlay,
  onAutoPlayToggle,
}: StepNavigatorProps) {
  return (
    <GlassCard padding="px-4 py-2">
      <div className="flex items-center gap-3">
        <button
          onClick={onPrev}
          disabled={current <= 0}
          className="px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Prev
        </button>

        <span className="text-sm text-gray-400 tabular-nums min-w-[4rem] text-center">
          {current + 1} / {total}
        </span>

        <button
          onClick={onNext}
          disabled={current >= total - 1}
          className="px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Next
        </button>

        {onAutoPlayToggle && (
          <button
            onClick={onAutoPlayToggle}
            className="ml-auto px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/20 transition-colors"
          >
            {autoPlay ? 'Pause' : 'Play'}
          </button>
        )}
      </div>
    </GlassCard>
  )
}

import type { Activity } from '@/types/api'

interface ActivityEntryProps {
  index: number
  value: Activity
  onChange: (updated: Activity) => void
  onRemove: () => void
}

export default function ActivityEntry({ index, value, onChange, onRemove }: ActivityEntryProps) {
  return (
    <div className="relative bg-white/[0.03] border border-white/10 rounded-xl p-4 space-y-2">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500">Activity {index + 1}</span>
        <button
          type="button"
          onClick={onRemove}
          className="text-gray-500 hover:text-red-400 transition-colors p-1 cursor-pointer"
          aria-label="Remove activity"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <textarea
        value={value.description}
        onChange={(e) => onChange({ description: e.target.value })}
        placeholder="e.g. DECA VP — placed top 10 at provincials, robotics club captain, 200+ volunteer hours at food bank"
        rows={2}
        className="
          w-full px-3 py-2 rounded-lg text-sm text-white
          bg-white/5 border border-white/10
          outline-none transition-colors
          focus:border-emerald-400
          resize-none placeholder:text-gray-600
        "
      />
    </div>
  )
}

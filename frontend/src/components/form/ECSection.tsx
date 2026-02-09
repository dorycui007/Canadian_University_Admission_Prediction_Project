import type { Activity } from '@/types/api'
import ActivityEntry from './ActivityEntry'

interface ECSectionProps {
  activities: Activity[]
  onChange: (activities: Activity[]) => void
}

const MAX_ACTIVITIES = 10

function defaultActivity(): Activity {
  return { description: '' }
}

export default function ECSection({ activities, onChange }: ECSectionProps) {
  function addActivity() {
    if (activities.length < MAX_ACTIVITIES) {
      onChange([...activities, defaultActivity()])
    }
  }

  function updateActivity(index: number, updated: Activity) {
    const next = [...activities]
    next[index] = updated
    onChange(next)
  }

  function removeActivity(index: number) {
    onChange(activities.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Extracurriculars mainly affect competitive programs (Waterloo Engineering,
        Queen's Commerce, Western Ivey, UBC, UofT Engineering).
      </p>

      {activities.map((activity, i) => (
        <ActivityEntry
          key={i}
          index={i}
          value={activity}
          onChange={(updated) => updateActivity(i, updated)}
          onRemove={() => removeActivity(i)}
        />
      ))}

      {activities.length < MAX_ACTIVITIES && (
        <button
          type="button"
          onClick={addActivity}
          className="
            flex items-center gap-2 px-4 py-2 rounded-lg
            text-sm text-gray-400 hover:text-emerald-400
            bg-white/5 border border-white/10 border-dashed
            hover:border-emerald-500/30
            transition-colors cursor-pointer
          "
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          Add Activity ({activities.length}/{MAX_ACTIVITIES})
        </button>
      )}
    </div>
  )
}

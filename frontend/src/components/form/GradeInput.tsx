import { useState } from 'react'
import type { GradeFormat } from '@/types/api'
import { getFormatLabel, getFormatRange } from '@/lib/gradeConverter'

interface GradeInputProps {
  label: string
  value: number | undefined
  onChange: (val: number | undefined) => void
  format: GradeFormat
  onFormatChange: (f: GradeFormat) => void
  required?: boolean
}

const FORMATS: GradeFormat[] = ['percentage', 'gpa4', 'gpa43', 'ib45', 'letter']

export default function GradeInput({
  label,
  value,
  onChange,
  format,
  onFormatChange,
  required = false,
}: GradeInputProps) {
  const [touched, setTouched] = useState(false)
  const range = getFormatRange(format)

  const isOutOfRange =
    value !== undefined && (value < range.min || value > range.max)
  const showError = touched && isOutOfRange

  function handleValueChange(raw: string) {
    if (raw === '') {
      onChange(undefined)
      return
    }
    const parsed = parseFloat(raw)
    if (!Number.isNaN(parsed)) {
      onChange(parsed)
    }
  }

  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-gray-300">
        {label}
        {required && <span className="text-emerald-400 ml-1">*</span>}
      </label>

      <div className="flex gap-2">
        {/* Numeric input */}
        <input
          type="number"
          value={value ?? ''}
          onChange={(e) => handleValueChange(e.target.value)}
          onBlur={() => setTouched(true)}
          min={range.min}
          max={range.max}
          step={range.step}
          placeholder={`${range.min} - ${range.max}`}
          required={required}
          className={`
            flex-1 min-w-0 px-3 py-2.5 rounded-lg text-sm text-white
            bg-white/5 border placeholder-gray-500
            outline-none transition-colors
            ${showError
              ? 'border-red-400/70 focus:border-red-400'
              : 'border-white/10 focus:border-emerald-400'
            }
          `}
        />

        {/* Format selector */}
        <select
          value={format}
          onChange={(e) => onFormatChange(e.target.value as GradeFormat)}
          className="
            px-3 py-2.5 rounded-lg text-sm text-white
            bg-white/5 border border-white/10
            outline-none transition-colors
            focus:border-emerald-400
            cursor-pointer shrink-0
          "
        >
          {FORMATS.map((f) => (
            <option key={f} value={f} className="bg-[#111] text-white">
              {getFormatLabel(f)}
            </option>
          ))}
        </select>
      </div>

      {/* Error message */}
      {showError && (
        <p className="text-xs text-red-400">
          Value must be between {range.min} and {range.max} for{' '}
          {getFormatLabel(format)} format.
        </p>
      )}
    </div>
  )
}

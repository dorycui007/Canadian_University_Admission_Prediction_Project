import { useState, useRef, useEffect } from 'react'

interface ProgramComboboxProps {
  value: string
  onChange: (val: string) => void
}

const PROGRAMS: string[] = [
  'Computer Science',
  'Engineering',
  'Commerce',
  'Business Administration',
  'Biology',
  'Mathematics',
  'Nursing',
  'Psychology',
  'Economics',
  'Chemistry',
  'Physics',
  'Kinesiology',
  'Political Science',
  'English',
  'History',
  'Sociology',
  'Health Sciences',
  'Life Sciences',
  'Software Engineering',
  'Mechanical Engineering',
  'Electrical Engineering',
  'Civil Engineering',
  'Accounting',
  'Finance',
  'Marketing',
  'Communications',
  'Environmental Science',
  'Architecture',
  'Music',
  'Philosophy',
  'Data Science',
  'Medical Sciences',
  'Biomedical Sciences',
  'Computer Engineering',
  'Mechatronics Engineering',
  'Accounting and Financial Management',
]

export default function ProgramCombobox({ value, onChange }: ProgramComboboxProps) {
  const [query, setQuery] = useState(value)
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef<HTMLDivElement>(null)

  // Sync external value changes
  useEffect(() => {
    setQuery(value)
  }, [value])

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const filtered = query.trim() === ''
    ? PROGRAMS
    : PROGRAMS.filter((p) => p.toLowerCase().includes(query.toLowerCase()))

  function handleSelect(program: string) {
    setQuery(program)
    onChange(program)
    setOpen(false)
  }

  function handleInputChange(raw: string) {
    setQuery(raw)
    setOpen(true)
    if (raw === '') {
      onChange('')
    }
  }

  return (
    <div ref={wrapperRef} className="relative">
      <label className="block text-sm font-medium text-gray-300 mb-1.5">
        Program
      </label>
      <input
        type="text"
        value={query}
        onChange={(e) => handleInputChange(e.target.value)}
        onFocus={() => setOpen(true)}
        placeholder="Search programs..."
        className="
          w-full px-3 py-2.5 rounded-lg text-sm text-white
          bg-white/5 border border-white/10
          outline-none transition-colors
          focus:border-emerald-400
          placeholder-gray-500
        "
      />

      {open && filtered.length > 0 && (
        <ul
          className="
            absolute z-50 mt-1 w-full max-h-60 overflow-y-auto
            bg-[#111] border border-white/10 rounded-lg
            shadow-xl shadow-black/40
          "
        >
          {filtered.map((p) => (
            <li
              key={p}
              onClick={() => handleSelect(p)}
              className="
                px-3 py-2 text-sm text-gray-200
                hover:bg-white/10 hover:text-white
                cursor-pointer transition-colors
              "
            >
              {p}
            </li>
          ))}
        </ul>
      )}

      {open && filtered.length === 0 && query.trim() !== '' && (
        <div
          className="
            absolute z-50 mt-1 w-full
            bg-[#111] border border-white/10 rounded-lg
            shadow-xl shadow-black/40
            px-3 py-3 text-sm text-gray-500
          "
        >
          No programs found for "{query}"
        </div>
      )}
    </div>
  )
}

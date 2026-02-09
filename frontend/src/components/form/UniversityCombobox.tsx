import { useState, useRef, useEffect } from 'react'

interface UniversityComboboxProps {
  value: string
  onChange: (val: string) => void
}

interface UniversityEntry {
  canonical: string
  variations: string[]
}

const UNIVERSITIES: UniversityEntry[] = [
  { canonical: 'University of Toronto', variations: ['UofT', 'U of T', 'Toronto'] },
  { canonical: 'University of Waterloo', variations: ['Waterloo', 'UW', 'UWaterloo'] },
  { canonical: 'McMaster University', variations: ['McMaster', 'Mac'] },
  { canonical: "Queen's University", variations: ["Queen's", 'Queens', 'QU'] },
  { canonical: 'Western University', variations: ['Western', 'UWO', 'Western Ontario'] },
  { canonical: 'University of Ottawa', variations: ['Ottawa', 'uOttawa', 'UOttawa'] },
  { canonical: 'York University', variations: ['York', 'YorkU'] },
  { canonical: 'Wilfrid Laurier University', variations: ['Laurier', 'WLU'] },
  { canonical: 'Toronto Metropolitan University', variations: ['TMU', 'Ryerson', 'Toronto Met'] },
  { canonical: 'University of Guelph', variations: ['Guelph', 'UofG'] },
  { canonical: 'McGill University', variations: ['McGill'] },
  { canonical: 'University of British Columbia', variations: ['UBC', 'British Columbia'] },
  { canonical: 'Simon Fraser University', variations: ['SFU', 'Simon Fraser'] },
  { canonical: 'University of Victoria', variations: ['UVic', 'Victoria'] },
  { canonical: 'University of Alberta', variations: ['Alberta', 'UAlberta', 'UofA'] },
  { canonical: 'University of Calgary', variations: ['Calgary', 'UCalgary', 'UofC'] },
  { canonical: 'Dalhousie University', variations: ['Dalhousie', 'Dal'] },
  { canonical: 'University of Manitoba', variations: ['Manitoba', 'UManitoba', 'UofM'] },
  { canonical: 'University of Saskatchewan', variations: ['Saskatchewan', 'USask'] },
  { canonical: 'Concordia University', variations: ['Concordia'] },
  { canonical: 'University of New Brunswick', variations: ['UNB', 'New Brunswick'] },
  { canonical: 'Memorial University', variations: ['Memorial', 'MUN'] },
  { canonical: 'University of Prince Edward Island', variations: ['UPEI', 'PEI'] },
  { canonical: 'Brock University', variations: ['Brock'] },
  { canonical: 'Carleton University', variations: ['Carleton'] },
]

export default function UniversityCombobox({ value, onChange }: UniversityComboboxProps) {
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
    ? UNIVERSITIES
    : UNIVERSITIES.filter((u) => {
        const q = query.toLowerCase()
        if (u.canonical.toLowerCase().includes(q)) return true
        return u.variations.some((v) => v.toLowerCase().includes(q))
      })

  function handleSelect(canonical: string) {
    setQuery(canonical)
    onChange(canonical)
    setOpen(false)
  }

  function handleInputChange(raw: string) {
    setQuery(raw)
    setOpen(true)
    // If the user clears the input, also clear the selected value
    if (raw === '') {
      onChange('')
    }
  }

  return (
    <div ref={wrapperRef} className="relative">
      <label className="block text-sm font-medium text-gray-300 mb-1.5">
        University
      </label>
      <input
        type="text"
        value={query}
        onChange={(e) => handleInputChange(e.target.value)}
        onFocus={() => setOpen(true)}
        placeholder="Search universities..."
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
          {filtered.map((u) => (
            <li
              key={u.canonical}
              onClick={() => handleSelect(u.canonical)}
              className="
                px-3 py-2 text-sm text-gray-200
                hover:bg-white/10 hover:text-white
                cursor-pointer transition-colors
              "
            >
              <span className="font-medium">{u.canonical}</span>
              {u.variations.length > 0 && (
                <span className="ml-2 text-xs text-gray-500">
                  {u.variations.join(', ')}
                </span>
              )}
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
          No universities found for "{query}"
        </div>
      )}
    </div>
  )
}

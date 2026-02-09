interface SectionHeaderProps {
  label?: string
  title: string
  accent?: string
  description?: string
}

export default function SectionHeader({
  label,
  title,
  accent,
  description,
}: SectionHeaderProps) {
  return (
    <div className="space-y-4">
      {label && (
        <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
          {label}
        </p>
      )}

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
        {title}
        {accent && (
          <>
            {' '}
            <span className="accent-serif">{accent}</span>
          </>
        )}
      </h2>

      {description && (
        <p className="text-lg text-gray-400 max-w-2xl">{description}</p>
      )}
    </div>
  )
}

import { Link } from 'react-router-dom'

interface NavLink {
  label: string
  to: string
}

interface PrevNextNavProps {
  prev?: NavLink
  next?: NavLink
}

export default function PrevNextNav({ prev, next }: PrevNextNavProps) {
  return (
    <nav className="flex items-center justify-between border-t border-white/10 pt-8 mt-16">
      {prev ? (
        <Link
          to={prev.to}
          className="group flex items-center gap-2 text-gray-400 hover:text-emerald-400 transition-colors"
        >
          <span
            aria-hidden="true"
            className="transition-transform group-hover:-translate-x-0.5"
          >
            &larr;
          </span>
          <div className="text-left">
            <p className="text-xs uppercase tracking-widest text-gray-600">
              Previous
            </p>
            <p className="text-sm font-medium">{prev.label}</p>
          </div>
        </Link>
      ) : (
        <div />
      )}

      {next ? (
        <Link
          to={next.to}
          className="group flex items-center gap-2 text-gray-400 hover:text-emerald-400 transition-colors text-right"
        >
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-600">
              Next
            </p>
            <p className="text-sm font-medium">{next.label}</p>
          </div>
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            &rarr;
          </span>
        </Link>
      ) : (
        <div />
      )}
    </nav>
  )
}

import { NavLink, Link } from 'react-router-dom'

const navLinks = [
  { to: '/predict', label: 'Predict' },
  { to: '/explore/programs', label: 'Explore' },
  { to: '/about', label: 'About' },
]

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-black/80 backdrop-blur-md border-b border-white/10">
      <div className="max-w-6xl mx-auto px-6 h-full flex items-center justify-between">
        {/* Wordmark */}
        <Link to="/" className="group">
          <span className="font-serif text-xl text-emerald-400">
            Admission Predictor
          </span>
        </Link>

        {/* Nav links + CTA */}
        <div className="flex items-center gap-6">
          {navLinks.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive
                    ? 'text-emerald-400'
                    : 'text-gray-400 hover:text-white'
                }`
              }
            >
              {label}
            </NavLink>
          ))}

          <Link
            to="/predict"
            className="ml-2 px-4 py-2 text-sm font-semibold text-black bg-emerald-400 rounded-full hover:bg-emerald-300 transition-colors"
          >
            Get Your Prediction
          </Link>
        </div>
      </div>
    </nav>
  )
}

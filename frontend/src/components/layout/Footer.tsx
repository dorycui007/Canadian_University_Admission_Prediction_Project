import { Link } from 'react-router-dom'

const footerLinks = [
  { to: '/predict', label: 'Predict' },
  { to: '/explore/programs', label: 'Explore' },
  { to: '/about', label: 'About' },
]

export default function Footer() {
  return (
    <footer className="border-t border-white/10 pt-12 pb-8">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <p className="text-lg font-bold text-white tracking-tight">
              Admission Predictor
            </p>
            <p className="mt-2 text-sm text-gray-500">
              Data-driven admission predictions for Canadian universities.
            </p>
          </div>

          {/* Links */}
          <div>
            <p className="text-sm font-medium text-gray-400 mb-3">Links</p>
            <ul className="space-y-2">
              {footerLinks.map(({ to, label }) => (
                <li key={to}>
                  <Link
                    to={to}
                    className="text-sm text-gray-500 hover:text-white transition-colors"
                  >
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Disclaimer */}
          <div>
            <p className="text-sm font-medium text-gray-400 mb-3">
              Disclaimer
            </p>
            <p className="text-xs text-gray-500 leading-relaxed">
              This tool provides estimates based on historical data and
              statistical models. Results are not guarantees of admission.
              Always consult official university resources for the most
              accurate and up-to-date information.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}

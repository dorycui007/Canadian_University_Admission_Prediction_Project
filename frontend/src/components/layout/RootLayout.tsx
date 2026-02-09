import { Outlet, useLocation } from 'react-router-dom'
import Navbar from '@/components/layout/Navbar'
import Footer from '@/components/layout/Footer'
import EmeraldWaves from '@/components/shared/EmeraldWaves'

/** Routes where the marketing Footer is displayed. */
const MARKETING_ROUTES = [
  '/',
  '/about',
  '/about/how-it-works',
  '/about/math',
  '/about/models',
  '/about/evaluation',
  '/about/api',
]

export default function RootLayout() {
  const { pathname } = useLocation()
  const showFooter = MARKETING_ROUTES.includes(pathname)

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans flex flex-col">
      <EmeraldWaves />
      <Navbar />

      {/* Main content - offset for fixed navbar */}
      <main className="relative z-10 flex-1 pt-16">
        <Outlet />
      </main>

      {showFooter && <Footer />}
    </div>
  )
}

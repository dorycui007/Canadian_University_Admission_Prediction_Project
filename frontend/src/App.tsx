import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import RootLayout from '@/components/layout/RootLayout'
import Landing from '@/pages/Landing'

const Predict = lazy(() => import('@/pages/Predict'))
const Results = lazy(() => import('@/pages/Results'))
const Programs = lazy(() => import('@/pages/explore/Programs'))
const ProgramDetail = lazy(() => import('@/pages/explore/ProgramDetail'))
const Distributions = lazy(() => import('@/pages/explore/Distributions'))
const About = lazy(() => import('@/pages/about/About'))
const HowItWorks = lazy(() => import('@/pages/about/HowItWorks'))
const MathFoundation = lazy(() => import('@/pages/about/MathFoundation'))
const Models = lazy(() => import('@/pages/about/Models'))
const Evaluation = lazy(() => import('@/pages/about/Evaluation'))
const ApiPipeline = lazy(() => import('@/pages/about/ApiPipeline'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="w-6 h-6 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" />
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route element={<RootLayout />}>
          <Route path="/" element={<Landing />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/results" element={<Results />} />
          <Route path="/explore/programs" element={<Programs />} />
          <Route path="/explore/programs/:university/:program" element={<ProgramDetail />} />
          <Route path="/explore/distributions" element={<Distributions />} />
          <Route path="/about" element={<About />} />
          <Route path="/about/how-it-works" element={<HowItWorks />} />
          <Route path="/about/math" element={<MathFoundation />} />
          <Route path="/about/models" element={<Models />} />
          <Route path="/about/evaluation" element={<Evaluation />} />
          <Route path="/about/api" element={<ApiPipeline />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <h1 className="text-6xl font-bold text-white mb-4">404</h1>
      <p className="text-gray-400 text-lg">Page not found</p>
    </div>
  )
}

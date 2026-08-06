import { NavLink, Route, Routes } from 'react-router-dom'
import Home from './pages/Home'
import Metrics from './pages/Metrics'

export default function App() {
  return (
    <div className="min-h-full">
      <header className="bg-white shadow">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Animal Footprint Direction Detection</h1>
          <nav className="space-x-4">
            <NavLink to="/" className={({isActive})=> isActive? 'text-blue-600 font-medium' : 'text-gray-700 hover:text-blue-600'}>Home</NavLink>
            <NavLink to="/metrics" className={({isActive})=> isActive? 'text-blue-600 font-medium' : 'text-gray-700 hover:text-blue-600'}>Metrics</NavLink>
          </nav>
        </div>
      </header>
      <main className="max-w-5xl mx-auto p-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </main>
    </div>
  )
}



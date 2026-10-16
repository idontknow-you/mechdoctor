import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_URL

export default function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((r) => r.json())
      .then((d) => setStatus(`backend ${d.status}`))
      .catch(() => setStatus('backend not reachable'))
  }, [])

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center gap-3">
      <h1 className="text-4xl font-bold">MechDoctor</h1>
      <p className="text-slate-300">{status}</p>
    </div>
  )
}
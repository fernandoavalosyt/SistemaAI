import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [backendHealth, setBackendHealth] = useState(null)
  const [iaHealth, setIaHealth] = useState(null)
  const [error, setError] = useState(null)

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    // Verificar health del backend
    fetch(`${apiUrl}/health`)
      .then(res => res.json())
      .then(data => setBackendHealth(data))
      .catch(() => setError('Backend no disponible'))

    // Verificar health del servicio IA
    fetch('http://localhost:8001/health')
      .then(res => res.json())
      .then(data => setIaHealth(data))
      .catch(() => {/* IA puede no estar disponible aún */})
  }, [apiUrl])

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🛡️</span>
          <h1>Vigilia</h1>
        </div>
        <p className="subtitle">Sistema de Videovigilancia Inteligente</p>
      </header>

      <main className="main">
        <section className="status-grid">
          {/* Backend Status */}
          <div className={`status-card ${backendHealth ? 'healthy' : 'offline'}`}>
            <div className="status-indicator" />
            <h3>⚙️ Backend API</h3>
            <p className="status-text">
              {backendHealth ? `v${backendHealth.version} — ${backendHealth.status}` : error || 'Conectando...'}
            </p>
            <span className="port">:8000</span>
          </div>

          {/* IA Status */}
          <div className={`status-card ${iaHealth ? 'healthy' : 'offline'}`}>
            <div className="status-indicator" />
            <h3>🧠 Modelo IA</h3>
            <p className="status-text">
              {iaHealth ? `v${iaHealth.version} — ${iaHealth.status}` : 'Conectando...'}
            </p>
            <span className="port">:8001</span>
          </div>

          {/* Frontend Status */}
          <div className="status-card healthy">
            <div className="status-indicator" />
            <h3>🎨 Frontend</h3>
            <p className="status-text">v0.1.0 — healthy</p>
            <span className="port">:5173</span>
          </div>
        </section>

        <section className="info">
          <p>Infraestructura Docker activa. Dashboard en desarrollo.</p>
        </section>
      </main>
    </div>
  )
}

export default App

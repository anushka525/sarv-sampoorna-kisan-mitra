import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

function getCurrentRoute() {
  const hash = window.location.hash || '#/';
  const route = hash.replace('#', '');
  return route === '/detector' ? '/detector' : '/';
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function DetectorPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API_BASE}/health`, { signal: controller.signal })
      .then((response) => response.json())
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: 'unavailable' }));

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);

    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const hasPrediction = useMemo(() => Boolean(result?.top_prediction), [result]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);

    if (!file) {
      setError('Choose an image before running prediction.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict?top_k=3`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || 'Prediction failed');
      }

      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : '';
      if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
        setError(`Backend unreachable at ${API_BASE}. Start the Flask server with python app.py.`);
      } else {
        setError(message || 'Something went wrong.');
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <a className="home-link" href="#/">
        Home
      </a>

      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">Crop intelligence</span>
          <h1>Detect plant disease from a leaf image.</h1>
          <p>
            "<i>From leaf to logic: Bridging the gap between a captured image and a saved crop.</i>"
          </p>
          <div className="status-row">
            <span className={`status-pill ${health?.status === 'ok' ? 'status-ok' : 'status-warn'}`}>
              {health?.status === 'ok' ? 'Backend online' : 'Backend unavailable'}
            </span>
            <span className="status-meta">
              {health?.labels_loaded ? 'Labels loaded' : 'Labels pending'}
            </span>
          </div>
        </div>

        <form className="upload-card" onSubmit={handleSubmit}>
          <div className="upload-dropzone">
            <input
              id="leaf-upload"
              type="file"
              accept="image/*"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <label htmlFor="leaf-upload">
              <span className="upload-title">Choose a leaf image</span>
              <span className="upload-subtitle">PNG, JPG, or JPEG</span>
            </label>
          </div>

          {previewUrl ? (
            <div className="preview-frame">
              <img src={previewUrl} alt="Selected leaf preview" />
            </div>
          ) : (
            <div className="preview-empty">Your image preview will appear here.</div>
          )}

          <button className="predict-button" type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict disease'}
          </button>

          {error ? <p className="feedback error">{error}</p> : null}
        </form>
      </section>

      <section className="results-grid">
        <article className="result-card highlight-card">
          <span className="card-label">Top prediction</span>
          {hasPrediction ? (
            <>
              <h2>{result.top_prediction.label}</h2>
              <p>
                Confidence: <strong>{formatConfidence(result.top_prediction.confidence)}</strong>
              </p>
              <p className="muted">Class index: {result.top_prediction.index}</p>
            </>
          ) : (
            <p className="muted">Run a prediction to see the main disease result here.</p>
          )}
        </article>

        <article className="result-card">
          <span className="card-label">Top 3 predictions</span>
          {result?.top_k?.length ? (
            <ul className="prediction-list">
              {result.top_k.map((item) => (
                <li key={`${item.index}-${item.label}`}>
                  <div>
                    <strong>{item.label}</strong>
                    <p>Class {item.index}</p>
                  </div>
                  <span>{formatConfidence(item.confidence)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">The ranked predictions will show here after inference.</p>
          )}
        </article>

        <article className="result-card metrics-card">
          <span className="card-label">Model metrics</span>
          <div className="metric-row">
            <span>Classes</span>
            <strong>{result?.num_classes || health?.num_classes || 34}</strong>
          </div>
          <div className="metric-row">
            <span>Probability sum</span>
            <strong>{result ? result.confidence_sum.toFixed(6) : '—'}</strong>
          </div>
          <div className="metric-row">
            <span>Input shape</span>
            <strong>{health?.input_shape?.join(' x ') || '224 x 224 x 3'}</strong>
          </div>
        </article>
      </section>
    </main>
  );
}

function HomePage() {
  return (
    <main className="home-shell">
      <section className="home-card home-hero">
        <div className="hero-orb hero-orb-one" aria-hidden="true" />
        <div className="hero-orb hero-orb-two" aria-hidden="true" />

        <div className="home-content-grid">
          <div className="home-copy">
            <span className="eyebrow">AI for Farmers</span>
            <h1>Sarv Sampoorna Kisan Mitra</h1>
            <p>
              A modern intelligence platform for disease detection, crop recommendation, and rapid field-ready decisions.
              Built to help farmers respond early and protect yield with confidence.
            </p>

            <div className="home-cta-row">
              <a className="home-cta home-cta-primary" href="#/detector">
                Open Disease Detector
              </a>
              <a
                className="home-cta"
                href="https://agro-disinfectant-engine-1.onrender.com/"
                target="_blank"
                rel="noreferrer"
              >
                Agro Disinfectant Engine
              </a>
              <a
                className="home-cta"
                href="https://crop-prediction-machine-learning-model.onrender.com/"
                target="_blank"
                rel="noreferrer"
              >
                Crop Recommendation Engine
              </a>
            </div>

            <div className="home-metrics" aria-label="Impact metrics">
              <article>
                <strong>24/7</strong>
                <span>System Availability</span>
              </article>
              <article>
                <strong>34</strong>
                <span>Disease Classes</span>
              </article>
              <article>
                <strong>224x224</strong>
                <span>Model Input Precision</span>
              </article>
            </div>

            <div className="home-feed" aria-label="Live capability highlights">
              <div className="home-feed-track">
                <span>Early Disease Alerts</span>
                <span>Image-Based Screening</span>
                <span>Smart Crop Recommendation</span>
                <span>Actionable Farm Guidance</span>
                <span>Early Disease Alerts</span>
                <span>Image-Based Screening</span>
              </div>
            </div>
          </div>

          <aside className="home-visual" aria-label="Platform highlights">
            <div className="floating-badges" aria-hidden="true">
              <span className="floating-badge">AI Vision</span>
              <span className="floating-badge">Field Ready</span>
              <span className="floating-badge">Fast Inference</span>
            </div>

            <div className="signal-card signal-card-main">
              <p>Live Insight</p>
              <h3>Leaf Image -{'>'} Disease Risk</h3>
              <span>AI-powered classification in seconds</span>
            </div>

            <div className="signal-rail">
              <div className="signal-card">
                <p>Advisory</p>
                <h4>Treatment-first suggestions</h4>
              </div>
              <div className="signal-card">
                <p>Recommendation</p>
                <h4>Crop suitability guidance</h4>
              </div>
              <div className="signal-card">
                <p>Prevention</p>
                <h4>Proactive farm hygiene actions</h4>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

function App() {
  const [route, setRoute] = useState(getCurrentRoute());

  useEffect(() => {
    function onHashChange() {
      setRoute(getCurrentRoute());
    }

    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return route === '/detector' ? <DetectorPage /> : <HomePage />;
}

export default App;

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Person = { id: string; name: string; samples: number; alert_enabled: boolean }
type Track = { track_id: number; state: string; name?: string; reason: string; score: number | null; observations: number; missing: boolean }
type Policy = { threshold: number; margin: number; min_observations: number; min_span: number; [key: string]: number }
type Status = { recognizer: string; detector: string; score_kind: string; score_range: [number, number]; higher_is_better: boolean; state: string; mode?: string; source: string | null; tracks: Track[]; processed: number; captured: number; dropped: number; processing_ms?: number; frame_age: number | null; error: string | null; policy: Policy }
type Appearance = { score_kind: string; id: string; name: string; source: string; score: number; observations: number; created: number; acknowledged: boolean }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* preserve HTTP error */ }
    throw new Error(message)
  }
  return response.status === 204 ? undefined as T : response.json()
}

function App() {
  const [tab, setTab] = useState<'monitor' | 'gallery' | 'events'>('monitor')
  const [status, setStatus] = useState<Status | null>(null)
  const [people, setPeople] = useState<Person[]>([])
  const [events, setEvents] = useState<Appearance[]>([])
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const token = useRef('')
  const photoUrl = useRef<string | null>(null)
  const revision = useRef(0)

  useEffect(() => {
    let active = true
    let polling = false
    const controller = new AbortController()
    const load = async () => {
      if (polling) return
      polling = true
      const observedRevision = revision.current
      try {
        const session = await request<{ token: string }>('/session', { signal: controller.signal })
        token.current = session.token
        const [next, gallery, history] = await Promise.all([
          request<Status>('/status', { signal: controller.signal }),
          request<Person[]>('/people', { signal: controller.signal }),
          request<Appearance[]>('/events', { signal: controller.signal }),
        ])
        if (!active || observedRevision !== revision.current) return
        setStatus(next); setPeople(gallery); setEvents(history); setConnected(true); setReady(true)
        if (next.state === 'running') {
          const response = await fetch('/api/preview', { cache: 'no-store', signal: controller.signal })
          if (response.status === 200) {
            const blob = await response.blob()
            if (!active || observedRevision !== revision.current) return
            const url = URL.createObjectURL(blob)
            if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
            photoUrl.current = url
            setPreview(url)
          } else if (response.status === 204) {
            if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
            photoUrl.current = null
            setPreview(null)
          } else {
            throw new Error(`Preview unavailable (${response.status})`)
          }
        } else {
          if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
          photoUrl.current = null
          setPreview(null)
        }
      } catch (e) {
        if (active) { setConnected(false); setError((e as Error).message) }
      } finally { polling = false }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 600)
    return () => {
      active = false; controller.abort(); window.clearInterval(timer)
      if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
    }
  }, [])

  async function mutate(path: string, body?: BodyInit, method = 'POST') {
    return request(path, { method, body, headers: {
      Authorization: `Bearer ${token.current}`,
      ...(typeof body === 'string' ? { 'Content-Type': 'application/json' } : {}),
    } })
  }

  async function action(work: () => Promise<unknown>, message = '') {
    revision.current += 1
    if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
    photoUrl.current = null
    setPreview(null)
    setStatus(previous => previous ? { ...previous, tracks: [] } : previous)
    setBusy(true); setError(''); setNotice('')
    try { await work(); setNotice(message) }
    catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  function enroll(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    data.set('authorized', 'true')
    data.set('alert_enabled', data.has('alert_enabled') ? 'true' : 'false')
    void action(async () => { await mutate('/people', data); form.reset() }, 'Enrollment saved. Source photographs were not retained.')
  }

  function uploadVideo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    void action(() => mutate('/video', data), 'Replay requested. The uploaded copy is removed when replay ends.')
  }

  function savePolicy(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!status) return
    const data = new FormData(event.currentTarget)
    const policy = { ...status.policy, threshold: Number(data.get('threshold')), margin: Number(data.get('margin')) }
    void action(() => mutate('/policy', JSON.stringify(policy), 'PUT'), 'Decision settings saved. Start a new session to use them.')
  }

  function exportEvents() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url; link.download = 'odin-events.json'; link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const disabled = busy || !ready
  const running = status?.state === 'running' || status?.state === 'starting'
  const pending = events.filter(e => !e.acknowledged).length
  return (
    <div className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="odin home"><span className="brand-mark">o</span> odin<span className="brand-dot">.</span></a>
        <p className="sidebar-caption">FACE OBSERVATION WORKSPACE</p>
        <nav aria-label="Main navigation">
          <button className={tab === 'monitor' ? 'nav-item selected' : 'nav-item'} onClick={() => setTab('monitor')}><span>01</span> Monitor</button>
          <button className={tab === 'gallery' ? 'nav-item selected' : 'nav-item'} onClick={() => setTab('gallery')}><span>02</span> Authorized gallery <b>{people.length}</b></button>
          <button className={tab === 'events' ? 'nav-item selected' : 'nav-item'} onClick={() => setTab('events')}><span>03</span> Appearance events <b>{pending}</b></button>
        </nav>
        <div className="sidebar-bottom"><span className="eyebrow">ACTIVE APPROACH</span><strong>{status?.recognizer ?? 'Loading recognizer'}</strong><p>{status?.detector ?? 'Loading detector'}</p><span className="local-badge">Local processing</span></div>
      </aside>
      <main>
        <header className="topbar"><span>Research prototype <span className="separator">/</span> Single source</span><span className={`connection ${connected ? 'online' : ''}`}><i />{connected ? 'Operator connected' : 'Connecting / unavailable'}</span></header>
        <div className="content">
          <div className="page-heading"><div><span className="eyebrow">ODIN / FACE RECOGNITION</span><h1>{tab === 'monitor' ? 'Observe. Recognize. Review.' : tab === 'gallery' ? 'Authorized gallery' : 'Appearance events'}</h1><p>{tab === 'monitor' ? 'Follow face observations and inspect the evidence behind a match.' : tab === 'gallery' ? 'Enroll authorized participants with several clear, varied reference images.' : 'Review confirmed watchlist appearances. An appearance is not a judgment about behavior.'}</p></div><span className="version-tag">Local prototype</span></div>
          {error && <div className="message error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError('')}>×</button></div>}
          {notice && <div className="message" role="status">{notice}</div>}
          {tab === 'monitor' && <>
            <div className="monitor-grid">
              <section className="panel camera-panel"><div className="panel-heading"><h2>Source preview</h2><span className={`state-pill ${running && connected ? 'active' : ''}`}>{connected ? status?.state ?? 'loading' : 'unavailable'}{status?.mode === 'replay' ? ' · recorded replay' : ''}</span></div>
                <div className="camera-view">{preview ? <img src={preview} alt="Current source with detected face boxes and recognition labels" /> : <div className="empty-camera"><div className="viewfinder">⌗</div><h3>{status?.state === 'ended' ? 'Replay complete' : 'Ready when you are'}</h3><p>Start a local camera or select a recording below.</p></div>}{!connected && preview && <span className="stale-overlay">Connection lost · stale preview</span>}</div>
                <div className="camera-footer"><span>{status?.mode === 'replay' ? 'RECORDED' : 'CAMERA'} <b>{status?.source ?? 'No source selected'}</b></span><span>Frame age {status?.frame_age == null ? '—' : `${status.frame_age.toFixed(1)} s`}</span></div>
              </section>
              <section className="panel controls"><div className="panel-heading"><h2>Session controls</h2></div><label htmlFor="camera">Local camera</label><form onSubmit={e => { e.preventDefault(); const camera = Number(new FormData(e.currentTarget).get('camera')); void action(() => mutate('/start', JSON.stringify({ camera }))) }}><div className="inline-fields"><select id="camera" name="camera"><option value="0">Camera 0</option><option value="1">Camera 1</option><option value="2">Camera 2</option></select><button className="primary" disabled={disabled}>Start</button></div></form><div className="or-divider">or replay a recording</div><form onSubmit={uploadVideo}><label className="file-label">Video file<input name="file" type="file" accept="video/*,.avi,.mkv" required /></label><button className="secondary full" disabled={disabled}>Open recording</button></form><button className="stop-button full" disabled={disabled || !running} onClick={() => { void action(() => mutate('/stop')) }}>Stop session</button><p className="small-note">One source at a time. Replay is paced to the file’s frame rate; slow processing may drop frames.</p>{status?.error && <p className="source-error" role="alert">{status.error}</p>}</section>
            </div>
            <div className="stats-row"><div><span>PROCESSED FRAMES</span><strong>{status?.processed ?? '—'}</strong></div><div><span>DROPPED FRAMES</span><strong>{status?.dropped ?? '—'}</strong></div><div><span>LAST PROCESSING</span><strong>{status?.processing_ms?.toFixed(1) ?? '—'}<small> ms</small></strong></div><div><span>ACTIVE TRACKS</span><strong>{status?.tracks.length ?? 0}</strong></div></div>
            <section className="panel"><div className="panel-heading"><h2>Track evidence</h2><span className="muted">{status?.score_kind.replaceAll('_', ' ')} · {status?.higher_is_better ? 'higher' : 'lower'} is a closer match</span></div><div className="table-wrap"><table><thead><tr><th>Track</th><th>Identity</th><th>Decision</th><th>Score</th><th>Observations</th><th>Evidence</th></tr></thead><tbody>{status?.tracks.map(t => <tr key={t.track_id}><td className="mono">#{t.track_id}</td><td>{t.name ?? '—'}</td><td><span className={`state-pill ${t.state === 'confirmed' ? 'active' : ''}`}>{t.state}{t.missing ? ' · missing' : ''}</span></td><td className="mono">{t.score?.toFixed(3) ?? '—'}</td><td>{t.observations}</td><td className="muted">{t.reason.replaceAll('_', ' ')}</td></tr>)}</tbody></table>{!status?.tracks.length && <p className="empty-row">Track evidence appears when a face is detected.</p>}</div></section>
            <details className="panel settings"><summary>Decision settings <span>Uncalibrated baseline</span></summary><p>Matching scores are not probabilities. Select limits using independent validation recordings. Saving stops the current session.</p>{status && <form key={`${status.policy.threshold}-${status.policy.margin}`} onSubmit={savePolicy}><label>{status.higher_is_better ? 'Minimum matching similarity' : 'Maximum matching distance'}<input name="threshold" type="number" min={status.score_range[0]} max={status.score_range[1]} step="0.005" defaultValue={status.policy.threshold} required /></label><label>Minimum identity margin<input name="margin" type="number" min="0" max={status.score_range[1] - status.score_range[0]} step="0.005" defaultValue={status.policy.margin} required /></label><button className="secondary" disabled={disabled}>Save settings</button></form>}</details>
          </>}
          {tab === 'gallery' && <div className="gallery-grid"><section className="panel enrollment"><div className="panel-heading"><h2>Enroll a participant</h2></div><form onSubmit={enroll}><label>Pseudonym<input name="name" placeholder="e.g. Participant 01" maxLength={80} required /></label><label>Reference images<input name="images" type="file" multiple accept="image/png,image/jpeg,image/webp" required /></label><p className="small-note">Use several distinct images, each showing exactly one clear face. Up to 32 images, 8 MB each.</p><label className="checkbox"><input name="authorized" type="checkbox" required />I am authorized to enroll this participant and use these images.</label><label className="checkbox"><input name="alert_enabled" type="checkbox" />Enable appearance alerts for this participant.</label><button className="primary full" disabled={disabled}>{busy ? 'Processing…' : 'Create enrollment'}</button></form><div className="privacy-note"><strong>Keep only what is needed</strong><p>Face descriptors are encrypted locally. Original enrollment images are discarded. Unknown face crops are not saved.</p></div></section><section className="panel"><div className="panel-heading"><h2>Enrolled participants</h2><span className="muted">{people.length} identities</span></div>{people.map(p => <article className="person-row" key={p.id}><div className="avatar">{p.name.slice(0, 1).toUpperCase()}</div><div className="person-details"><strong>{p.name}</strong><span>{p.samples} reference {p.samples === 1 ? 'sample' : 'samples'}</span></div><label className="checkbox compact"><input type="checkbox" checked={!!p.alert_enabled} disabled={disabled} onChange={e => { void action(() => mutate(`/people/${p.id}/alerts`, JSON.stringify({ enabled: e.target.checked }), 'PATCH')) }} />Alerts</label><button className="text-danger" disabled={disabled} onClick={() => { if (window.confirm(`Delete ${p.name}, their templates and associated events?`)) void action(() => mutate(`/people/${p.id}`, undefined, 'DELETE'), 'Identity and associated data deleted.') }}>Delete</button></article>)}{!people.length && <div className="empty-list"><h3>Your gallery is empty</h3><p>Enroll an authorized participant to begin identity matching.</p></div>}</section></div>}
          {tab === 'events' && <section className="panel"><div className="panel-heading"><h2>Recent appearances</h2><button className="secondary" disabled={!events.length} onClick={exportEvents}>Export JSON</button></div><p className="section-note">Events are retained for seven days. Deleting an identity also deletes its associated events. Exports you download need separate deletion.</p>{events.map(e => <article className="event-row" key={e.id}><div className={`event-dot ${e.acknowledged ? 'reviewed' : ''}`} /><div className="event-details"><strong>{e.name}<span className="event-label">Confirmed appearance</span></strong><p>{new Date(e.created * 1000).toLocaleString()} · {e.source}</p><span className="muted">{e.score_kind.replaceAll('_', ' ')} {e.score.toFixed(3)} · {e.observations} supporting observations · uncalibrated score</span></div><button className="secondary" disabled={disabled || !!e.acknowledged} onClick={() => { void action(() => mutate(`/events/${e.id}/ack`)) }}>{e.acknowledged ? 'Acknowledged' : 'Acknowledge'}</button></article>)}{!events.length && <div className="empty-list"><h3>No appearance events yet</h3><p>Fresh, confirmed matches generate events for participants with alerts enabled.</p></div>}</section>}
          <footer className="page-footer"><span>Controlled research use · authorized participants</span><span>Unknown ≠ insufficient evidence</span></footer>
        </div>
      </main>
    </div>
  )
}
export default App

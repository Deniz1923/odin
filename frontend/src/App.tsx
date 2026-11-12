import { useCallback, useState } from 'react'
import { mutate } from './api'
import { Events } from './pages/Events'
import { Gallery } from './pages/Gallery'
import { Monitor } from './pages/Monitor'
import type { Policy } from './types'
import { useOperatorData } from './useOperatorData'
import './App.css'

type Tab = 'monitor' | 'gallery' | 'events'

const HEADINGS: Record<Tab, { title: string; text: string }> = {
  monitor: { title: 'Observe. Recognize. Review.', text: 'Follow face observations and inspect the evidence behind a match.' },
  gallery: { title: 'Authorized gallery', text: 'Enroll authorized participants with several clear, varied reference images.' },
  events: { title: 'Appearance events', text: 'Review confirmed watchlist appearances. An appearance is not a judgment about behavior.' },
}

function App() {
  const [tab, setTab] = useState<Tab>('monitor')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const reportError = useCallback((message: string) => setError(message), [])
  const { status, people, events, preview, connected, ready, invalidate, refreshLists } = useOperatorData(reportError)

  async function action(work: () => Promise<unknown>, message = '') {
    invalidate()
    setBusy(true); setError(''); setNotice('')
    try { await work(); setNotice(message) }
    catch (e) { setError((e as Error).message) }
    finally { setBusy(false); refreshLists() }
  }

  const disabled = busy || !ready
  const pending = events.filter(e => !e.acknowledged).length
  const heading = HEADINGS[tab]
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
        <div className="sidebar-bottom">
          <span className="eyebrow">ACTIVE APPROACH</span>
          <strong>{status?.recognizer ?? 'Loading recognizer'}</strong>
          <p>{status?.detector ?? 'Loading detector'}</p>
          <span className="local-badge">Local processing</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>Research prototype <span className="separator">/</span> Single source</span>
          <span className={`connection ${connected ? 'online' : ''}`}><i />{connected ? 'Operator connected' : 'Connecting / unavailable'}</span>
        </header>
        <div className="content">
          <div className="page-heading">
            <div><span className="eyebrow">ODIN / FACE RECOGNITION</span><h1>{heading.title}</h1><p>{heading.text}</p></div>
            <span className="version-tag">Local prototype</span>
          </div>
          {error && <div className="message error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError('')}>×</button></div>}
          {notice && <div className="message" role="status">{notice}</div>}
          {tab === 'monitor' && (
            <Monitor
              status={status}
              people={people}
              preview={preview}
              connected={connected}
              disabled={disabled}
              onStart={camera => { void action(() => mutate('/start', JSON.stringify({ camera }))) }}
              onUpload={data => { void action(() => mutate('/video', data), 'Replay requested. The uploaded copy is removed when replay ends.') }}
              onStop={() => { void action(() => mutate('/stop')) }}
              onSavePolicy={(policy: Policy) => { void action(() => mutate('/policy', JSON.stringify(policy), 'PUT'), 'Decision settings saved. Start a new session to use them.') }}
            />
          )}
          {tab === 'gallery' && (
            <Gallery
              people={people}
              disabled={disabled}
              busy={busy}
              onEnroll={(data, form) => { void action(async () => { await mutate('/people', data); form.reset() }, 'Enrollment saved. Source photographs were not retained.') }}
              onToggleAlert={(person, enabled) => { void action(() => mutate(`/people/${person.id}/alerts`, JSON.stringify({ enabled }), 'PATCH')) }}
              onDelete={person => {
                if (window.confirm(`Delete ${person.name}, their templates and associated events?`)) {
                  void action(() => mutate(`/people/${person.id}`, undefined, 'DELETE'), 'Identity and associated data deleted.')
                }
              }}
            />
          )}
          {tab === 'events' && (
            <Events
              events={events}
              retentionDays={status?.retention_days ?? null}
              disabled={disabled}
              onAcknowledge={event => { void action(() => mutate(`/events/${event.id}/ack`)) }}
            />
          )}
          <footer className="page-footer"><span>Controlled research use · authorized participants</span><span>Unknown ≠ insufficient evidence</span></footer>
        </div>
      </main>
    </div>
  )
}

export default App

import type { Status } from '../types'

type Props = { status: Status | null; preview: string | null; connected: boolean; running: boolean }

export function SourcePreview({ status, preview, connected, running }: Props) {
  return (
    <section className="panel camera-panel">
      <div className="panel-heading">
        <h2>Source preview</h2>
        <span className={`state-pill ${running && connected ? 'active' : ''}`}>
          {connected ? status?.state ?? 'loading' : 'unavailable'}{status?.mode === 'replay' ? ' · recorded replay' : ''}
        </span>
      </div>
      <div className="camera-view">
        {preview
          ? <img src={preview} alt="Current source with detected face boxes and recognition labels" />
          : (
            <div className="empty-camera">
              <div className="viewfinder">⌗</div>
              <h3>{status?.state === 'ended' ? 'Replay complete' : 'Ready when you are'}</h3>
              <p>Start a local camera or select a recording below.</p>
            </div>
          )}
        {!connected && preview && <span className="stale-overlay">Connection lost · stale preview</span>}
      </div>
      <div className="camera-footer">
        <span>{status?.mode === 'replay' ? 'RECORDED' : 'CAMERA'} <b>{status?.source ?? 'No source selected'}</b></span>
        <span>Frame age {status?.frame_age == null ? '—' : `${status.frame_age.toFixed(1)} s`}</span>
      </div>
    </section>
  )
}

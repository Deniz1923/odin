import type { Appearance } from '../types'

type Props = { events: Appearance[]; retentionDays: number | null; disabled: boolean; onAcknowledge: (event: Appearance) => void }

function exportEvents(events: Appearance[]) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = 'odin-events.json'
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function EventRow({ event, disabled, onAcknowledge }: { event: Appearance; disabled: boolean; onAcknowledge: Props['onAcknowledge'] }) {
  return (
    <article className="event-row">
      <div className={`event-dot ${event.acknowledged ? 'reviewed' : ''}`} />
      <div className="event-details">
        <strong>{event.name}<span className="event-label">Confirmed appearance</span></strong>
        <p>{new Date(event.created * 1000).toLocaleString()} · {event.source}</p>
        <span className="muted">{event.score_kind.replaceAll('_', ' ')} {event.score.toFixed(3)} · {event.observations} supporting observations · uncalibrated score</span>
      </div>
      <button className="secondary" disabled={disabled || !!event.acknowledged} onClick={() => onAcknowledge(event)}>
        {event.acknowledged ? 'Acknowledged' : 'Acknowledge'}
      </button>
    </article>
  )
}

export function Events({ events, retentionDays, disabled, onAcknowledge }: Props) {
  const retention = retentionDays == null ? 'for the configured retention period' : `for ${retentionDays} ${retentionDays === 1 ? 'day' : 'days'}`
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Recent appearances</h2>
        <button className="secondary" disabled={!events.length} onClick={() => exportEvents(events)}>Export JSON</button>
      </div>
      <p className="section-note">Events are retained {retention}. Deleting an identity also deletes its associated events. Exports you download need separate deletion.</p>
      {events.map(e => <EventRow key={e.id} event={e} disabled={disabled} onAcknowledge={onAcknowledge} />)}
      {!events.length && (
        <div className="empty-list">
          <h3>No appearance events yet</h3>
          <p>Fresh, confirmed matches generate events for participants with alerts enabled.</p>
        </div>
      )}
    </section>
  )
}

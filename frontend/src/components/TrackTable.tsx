import type { Status } from '../types'

type Props = { status: Status | null; selected: number | null; onSelect: (trackId: number | null) => void }

export function TrackTable({ status, selected, onSelect }: Props) {
  const tracks = status?.tracks ?? []
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Track evidence</h2>
        <span className="muted">{status?.score_kind.replaceAll('_', ' ')} · {status?.higher_is_better ? 'higher' : 'lower'} is a closer match</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Track</th><th>Identity</th><th>Decision</th><th>Score</th><th>Observations</th><th>Evidence</th></tr></thead>
          <tbody>
            {tracks.map(t => (
              <tr key={t.track_id} className={t.track_id === selected ? 'selected-row' : undefined}>
                <td>
                  <button className="row-select mono" aria-pressed={t.track_id === selected} title="Inspect method evidence" onClick={() => onSelect(t.track_id === selected ? null : t.track_id)}>
                    #{t.track_id}
                  </button>
                </td>
                <td>{t.name ?? '—'}</td>
                <td><span className={`state-pill ${t.state === 'confirmed' ? 'active' : ''}`}>{t.state}{t.missing ? ' · missing' : ''}</span></td>
                <td className="mono">{t.score?.toFixed(3) ?? '—'}</td>
                <td>{t.observations}</td>
                <td className="muted">{t.reason.replaceAll('_', ' ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!tracks.length && <p className="empty-row">Track evidence appears when a face is detected.</p>}
      </div>
    </section>
  )
}

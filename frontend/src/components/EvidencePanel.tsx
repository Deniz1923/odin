import type { ConfiguredMethod, MethodEvidence, Status } from '../types'

type Props = { status: Status | null; trackId: number | null; names: Map<string, string>; onClose: () => void }

const RULES: Record<string, string> = {
  single_method: 'Single method: the active recognizer’s decision, confirmed over time, is the track decision. No fusion is configured.',
}

function comparison(entry: MethodEvidence) {
  if (entry.score == null) return 'no score'
  const passes = entry.higher_is_better ? entry.score >= entry.threshold : entry.score <= entry.threshold
  const operator = entry.higher_is_better ? (passes ? '≥' : '<') : (passes ? '≤' : '>')
  return `${entry.score.toFixed(3)} ${operator} ${entry.threshold.toFixed(3)}`
}

function marginText(entry: MethodEvidence) {
  if (entry.margin == null) return entry.score == null ? '—' : 'no runner-up'
  const passes = entry.margin > 1e-12 && entry.margin >= entry.required_margin
  return `${entry.margin.toFixed(3)} ${passes ? '≥' : '<'} ${entry.required_margin.toFixed(3)}`
}

const OUTCOME: Record<MethodEvidence['state'], string> = {
  candidate: 'accepts',
  unknown: 'rejects: beyond threshold',
  insufficient: 'abstains: ambiguous or no gallery',
}

/** Methods disagree when they reach different outcomes or accept different identities. */
function disagreement(entries: MethodEvidence[]) {
  return new Set(entries.map(entry => `${entry.state}:${entry.person_id ?? ''}`)).size > 1
}

function MethodRow({ method, entry, names }: { method: ConfiguredMethod; entry?: MethodEvidence; names: Map<string, string> }) {
  if (!entry) {
    return <tr><td>{method.name}</td><td colSpan={4} className="muted">Not run for this observation</td></tr>
  }
  return (
    <tr>
      <td>{method.name}<span className="cell-note">{method.score_kind.replaceAll('_', ' ')}</span></td>
      <td>{entry.person_id ? names.get(entry.person_id) ?? entry.person_id.slice(0, 8) : '—'}</td>
      <td className="mono">{comparison(entry)}</td>
      <td className="mono">{marginText(entry)}</td>
      <td><span className={`state-pill ${entry.state === 'candidate' ? 'active' : ''}`}>{OUTCOME[entry.state]}</span></td>
    </tr>
  )
}

export function EvidencePanel({ status, trackId, names, onClose }: Props) {
  if (trackId == null) return null
  const track = status?.tracks.find(t => t.track_id === trackId)
  const methods = status?.methods ?? []
  return (
    <section className="panel evidence-panel" aria-label={`Evidence for track ${trackId}`}>
      <div className="panel-heading">
        <h2>Method evidence · track #{trackId}</h2>
        <button className="secondary" onClick={onClose}>Close</button>
      </div>
      {!track
        ? <p className="empty-row">Track #{trackId} has ended or was cleared. Select another track.</p>
        : (
          <>
            <div className="evidence-summary">
              <div><span>TRACK DECISION</span><strong><span className={`state-pill ${track.state === 'confirmed' ? 'active' : ''}`}>{track.state}</span></strong></div>
              <div><span>IDENTITY</span><strong>{track.name ?? '—'}</strong></div>
              <div><span>SUPPORTING OBSERVATIONS</span><strong>{track.observations}<small> needed {status?.policy.min_observations}</small></strong></div>
              <div><span>EVIDENCE AGE</span><strong>{track.evidence_age == null ? '—' : `${track.evidence_age.toFixed(2)} s`}</strong></div>
            </div>
            {disagreement(track.methods) && <p className="message error">Methods disagree on this observation.</p>}
            <div className="table-wrap">
              <table>
                <thead><tr><th>Method</th><th>Accepted identity</th><th>Score vs threshold</th><th>Margin vs required</th><th>Outcome</th></tr></thead>
                <tbody>
                  {methods.map(method => (
                    <MethodRow key={method.name} method={method} names={names} entry={track.methods.find(entry => entry.method === method.name)} />
                  ))}
                </tbody>
              </table>
            </div>
            <p className="section-note evidence-rule">
              {RULES[status?.decision_rule ?? ''] ?? status?.decision_rule} Reason: {track.reason.replaceAll('_', ' ')}{track.missing ? ' · face not detected in the latest frame' : ''}.
              Scores are raw and uncalibrated, not probabilities.
            </p>
          </>
        )}
    </section>
  )
}

import type { Status } from '../types'

/** Processing stages in pipeline order; keys match `stage_ms_median` from the API. */
const STAGES = [
  { key: 'detect_ms', label: 'Detection' },
  { key: 'extract_ms', label: 'Face preparation + descriptor' },
  { key: 'match_track_ms', label: 'Matching + tracking' },
  { key: 'events_ms', label: 'Event storage' },
  { key: 'preview_ms', label: 'Annotation + JPEG preview' },
] as const

const ms = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)
const fps = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)

export function RuntimePanel({ status }: { status: Status | null }) {
  const runtime = status?.runtime
  const stages = runtime?.stage_ms_median ?? {}
  const total = STAGES.reduce((sum, stage) => sum + (stages[stage.key] ?? 0), 0)
  return (
    <>
      <div className="stats-row">
        <div><span>PROCESSED FRAMES</span><strong>{status?.processed ?? '—'}</strong></div>
        <div><span>DROPPED FRAMES</span><strong>{status?.dropped ?? '—'}</strong></div>
        <div><span>LAST PROCESSING</span><strong>{status?.processing_ms?.toFixed(1) ?? '—'}<small> ms</small></strong></div>
        <div><span>ACTIVE TRACKS</span><strong>{status?.tracks.length ?? 0}</strong></div>
      </div>
      <section className="panel runtime-panel">
        <div className="panel-heading">
          <h2>Runtime</h2>
          <span className="muted">
            {runtime ? `Rolling medians over the last ${runtime.window_frames} processed frames · operator feedback, not a benchmark` : 'Appears once frames are processed'}
          </span>
        </div>
        <div className="runtime-metrics">
          <div><span>PROCESSED RATE</span><strong>{fps(runtime?.processed_fps)}<small> fps</small></strong></div>
          <div><span>INPUT RATE</span><strong>{fps(runtime?.input_fps)}<small> fps</small></strong></div>
          <div><span>PROCESSING MEDIAN / P95</span><strong>{ms(runtime?.processing_ms_median)}<small> / {ms(runtime?.processing_ms_p95)} ms</small></strong></div>
          <div><span>WAIT BEFORE PROCESSING</span><strong>{ms(stages.queue_ms)}<small> ms</small></strong></div>
        </div>
        {runtime && total > 0 && (
          <div className="stage-breakdown">
            <div className="stage-bar" role="img" aria-label="Median processing time by stage">
              {STAGES.map((stage, index) => (
                <span key={stage.key} className={`stage-${index}`} style={{ width: `${((stages[stage.key] ?? 0) / total) * 100}%` }} />
              ))}
            </div>
            <ul className="stage-legend">
              {STAGES.map((stage, index) => (
                <li key={stage.key}><i className={`stage-${index}`} />{stage.label}<b>{ms(stages[stage.key])} ms</b></li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </>
  )
}

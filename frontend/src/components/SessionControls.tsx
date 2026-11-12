import type { FormEvent } from 'react'

/** Matches the API's accepted camera index range (`CameraRequest` in api/main.py). */
const MAX_CAMERA_INDEX = 7

type Props = {
  disabled: boolean
  running: boolean
  error: string | null
  onStart: (camera: number) => void
  onUpload: (data: FormData) => void
  onStop: () => void
}

export function SessionControls({ disabled, running, error, onStart, onUpload, onStop }: Props) {
  function start(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onStart(Number(new FormData(event.currentTarget).get('camera')))
  }

  function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onUpload(new FormData(event.currentTarget))
  }

  return (
    <section className="panel controls">
      <div className="panel-heading"><h2>Session controls</h2></div>
      <label htmlFor="camera">Local camera index</label>
      <form onSubmit={start}>
        <div className="inline-fields">
          <input id="camera" name="camera" type="number" min={0} max={MAX_CAMERA_INDEX} step={1} defaultValue={0} required />
          <button className="primary" disabled={disabled}>Start</button>
        </div>
      </form>
      <div className="or-divider">or replay a recording</div>
      <form onSubmit={upload}>
        <label className="file-label">Video file<input name="file" type="file" accept="video/*,.avi,.mkv" required /></label>
        <button className="secondary full" disabled={disabled}>Open recording</button>
      </form>
      <button className="stop-button full" disabled={disabled || !running} onClick={onStop}>Stop session</button>
      <p className="small-note">One source at a time. Replay is paced to the file’s frame rate; slow processing may drop frames.</p>
      {error && <p className="source-error" role="alert">{error}</p>}
    </section>
  )
}

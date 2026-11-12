import type { FormEvent } from 'react'
import type { Policy, Status } from '../types'

type Props = { status: Status | null; disabled: boolean; onSave: (policy: Policy) => void }

export function PolicyForm({ status, disabled, onSave }: Props) {
  function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!status) return
    const data = new FormData(event.currentTarget)
    onSave({ ...status.policy, threshold: Number(data.get('threshold')), margin: Number(data.get('margin')) })
  }

  return (
    <details className="panel settings">
      <summary>Decision settings <span>Uncalibrated baseline</span></summary>
      <p>Matching scores are not probabilities. Select limits using independent validation recordings. Saving stops the current session.</p>
      {status && (
        <form key={`${status.policy.threshold}-${status.policy.margin}`} onSubmit={save}>
          <label>
            {status.higher_is_better ? 'Minimum matching similarity' : 'Maximum matching distance'}
            <input name="threshold" type="number" min={status.score_range[0]} max={status.score_range[1]} step="0.005" defaultValue={status.policy.threshold} required />
          </label>
          <label>
            Minimum identity margin
            <input name="margin" type="number" min="0" max={status.score_range[1] - status.score_range[0]} step="0.005" defaultValue={status.policy.margin} required />
          </label>
          <button className="secondary" disabled={disabled}>Save settings</button>
        </form>
      )}
    </details>
  )
}

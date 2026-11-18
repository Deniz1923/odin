import type { FormEvent } from 'react'
import type { Person } from '../types'

type Props = {
  people: Person[]
  disabled: boolean
  busy: boolean
  onEnroll: (data: FormData, form: HTMLFormElement) => void
  onToggleAlert: (person: Person, enabled: boolean) => void
  onDelete: (person: Person) => void
}

function PersonRow({ person, disabled, onToggleAlert, onDelete }: { person: Person; disabled: boolean } & Pick<Props, 'onToggleAlert' | 'onDelete'>) {
  return (
    <article className="person-row">
      <div className="avatar">{person.name.slice(0, 1).toUpperCase()}</div>
      <div className="person-details">
        <strong>{person.name}</strong>
        <span>{person.samples} reference {person.samples === 1 ? 'sample' : 'samples'}</span>
      </div>
      <label className="checkbox compact">
        <input type="checkbox" checked={!!person.alert_enabled} disabled={disabled} onChange={e => onToggleAlert(person, e.target.checked)} />Alerts
      </label>
      <button className="text-danger" disabled={disabled} onClick={() => onDelete(person)}>Delete</button>
    </article>
  )
}

export function Gallery({ people, disabled, busy, onEnroll, onToggleAlert, onDelete }: Props) {
  function enroll(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    data.set('authorized', 'true')
    data.set('alert_enabled', data.has('alert_enabled') ? 'true' : 'false')
    onEnroll(data, form)
  }

  return (
    <div className="gallery-grid">
      <section className="panel enrollment">
        <div className="panel-heading"><h2>Enroll a participant</h2></div>
        <form onSubmit={enroll}>
          <label>Pseudonym<input name="name" placeholder="e.g. Participant 01" maxLength={80} required /></label>
          <label>Reference images<input name="images" type="file" multiple accept="image/png,image/jpeg,image/webp" required /></label>
          <p className="small-note">Use several distinct images, each showing exactly one clear face. Up to 32 images, 8 MB each.</p>
          <label className="checkbox"><input name="authorized" type="checkbox" required />I am authorized to enroll this participant and use these images.</label>
          <label className="checkbox"><input name="alert_enabled" type="checkbox" />Enable appearance alerts for this participant.</label>
          <button className="primary full" disabled={disabled}>{busy ? 'Processing…' : 'Create enrollment'}</button>
        </form>
        <div className="privacy-note">
          <strong>Keep only what is needed</strong>
          <p>Face descriptors are encrypted locally. Original enrollment images are discarded. Unknown face crops are not saved.</p>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading"><h2>Enrolled participants</h2><span className="muted">{people.length} identities</span></div>
        {people.map(p => <PersonRow key={p.id} person={p} disabled={disabled} onToggleAlert={onToggleAlert} onDelete={onDelete} />)}
        {!people.length && (
          <div className="empty-list">
            <h3>Your gallery is empty</h3>
            <p>Enroll an authorized participant to begin identity matching.</p>
          </div>
        )}
      </section>
    </div>
  )
}

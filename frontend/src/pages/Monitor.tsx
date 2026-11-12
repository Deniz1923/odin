import { useMemo, useState } from 'react'
import { EvidencePanel } from '../components/EvidencePanel'
import { PolicyForm } from '../components/PolicyForm'
import { RuntimePanel } from '../components/RuntimePanel'
import { SessionControls } from '../components/SessionControls'
import { SourcePreview } from '../components/SourcePreview'
import { TrackTable } from '../components/TrackTable'
import type { Person, Policy, Status } from '../types'

type Props = {
  status: Status | null
  people: Person[]
  preview: string | null
  connected: boolean
  disabled: boolean
  onStart: (camera: number) => void
  onUpload: (data: FormData) => void
  onStop: () => void
  onSavePolicy: (policy: Policy) => void
}

export function Monitor({ status, people, preview, connected, disabled, onStart, onUpload, onStop, onSavePolicy }: Props) {
  const [selected, setSelected] = useState<number | null>(null)
  const names = useMemo(() => new Map(people.map(p => [p.id, p.name])), [people])
  const running = status?.state === 'running' || status?.state === 'starting'
  return (
    <>
      <div className="monitor-grid">
        <SourcePreview status={status} preview={preview} connected={connected} running={running} />
        <SessionControls disabled={disabled} running={running} error={status?.error ?? null} onStart={onStart} onUpload={onUpload} onStop={onStop} />
      </div>
      <RuntimePanel status={status} />
      <div className={selected == null ? 'evidence-grid' : 'evidence-grid open'}>
        <TrackTable status={status} selected={selected} onSelect={setSelected} />
        <EvidencePanel status={status} trackId={selected} names={names} onClose={() => setSelected(null)} />
      </div>
      <PolicyForm status={status} disabled={disabled} onSave={onSavePolicy} />
    </>
  )
}

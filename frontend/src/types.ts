export type Person = { id: string; name: string; samples: number; alert_enabled: boolean }

/** One recognizer's result for a track's latest usable observation. */
export type MethodEvidence = {
  method: string
  score_kind: string
  higher_is_better: boolean
  state: 'candidate' | 'unknown' | 'insufficient'
  person_id: string | null
  score: number | null
  margin: number | null
  threshold: number
  required_margin: number
}

export type Track = {
  track_id: number
  state: string
  person_id: string | null
  name?: string | null
  reason: string
  score: number | null
  margin: number | null
  observations: number
  evidence_age: number | null
  missing: boolean
  methods: MethodEvidence[]
}

export type Policy = { threshold: number; margin: number; min_observations: number; min_span: number; [key: string]: number }

/** A recognizer configured in the running service, with its active acceptance settings. */
export type ConfiguredMethod = {
  name: string
  score_kind: string
  score_range: [number, number]
  higher_is_better: boolean
  threshold: number
  margin: number
}

export type Runtime = {
  window_frames: number
  processed_fps: number | null
  input_fps: number | null
  processing_ms_median: number
  processing_ms_p95: number
  stage_ms_median: Record<string, number>
}

export type Status = {
  recognizer: string
  detector: string
  methods: ConfiguredMethod[]
  decision_rule: string
  retention_days: number
  score_kind: string
  score_range: [number, number]
  higher_is_better: boolean
  state: string
  mode?: string
  source: string | null
  tracks: Track[]
  processed: number
  captured: number
  dropped: number
  processing_ms?: number
  runtime?: Runtime | null
  frame_age: number | null
  error: string | null
  policy: Policy
}

export type Appearance = {
  id: string
  name: string
  source: string
  score: number
  score_kind: string
  observations: number
  created: number
  acknowledged: boolean
}

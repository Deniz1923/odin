import { useCallback, useEffect, useRef, useState } from 'react'
import { ensureSession, request } from './api'
import type { Appearance, Person, Status } from './types'

const STATUS_INTERVAL_MS = 600
const LIST_INTERVAL_MS = 3000

/**
 * Poll the operator API: status and preview every 600 ms, gallery and events every 3 s
 * or immediately after `invalidate()`. Responses started before an invalidation are dropped.
 */
export function useOperatorData(onError: (message: string) => void) {
  const [status, setStatus] = useState<Status | null>(null)
  const [people, setPeople] = useState<Person[]>([])
  const [events, setEvents] = useState<Appearance[]>([])
  const [preview, setPreview] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [ready, setReady] = useState(false)
  const photoUrl = useRef<string | null>(null)
  const revision = useRef(0)
  const listsDue = useRef(0)
  const reportError = useRef(onError)

  useEffect(() => { reportError.current = onError }, [onError])

  const replacePreview = useCallback((url: string | null) => {
    if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
    photoUrl.current = url
    setPreview(url)
  }, [])

  useEffect(() => {
    let active = true
    let polling = false
    const controller = new AbortController()
    const { signal } = controller

    const load = async () => {
      if (polling) return
      polling = true
      const observedRevision = revision.current
      try {
        await ensureSession(signal)
        const listsWanted = Date.now() >= listsDue.current
        const [next, lists] = await Promise.all([
          request<Status>('/status', { signal }),
          listsWanted
            ? Promise.all([request<Person[]>('/people', { signal }), request<Appearance[]>('/events', { signal })])
            : Promise.resolve(null),
        ])
        if (!active || observedRevision !== revision.current) return
        setStatus(next)
        if (lists) {
          setPeople(lists[0])
          setEvents(lists[1])
          listsDue.current = Date.now() + LIST_INTERVAL_MS
        }
        setConnected(true)
        setReady(true)
        if (next.state !== 'running') {
          replacePreview(null)
          return
        }
        const response = await fetch('/api/preview', { cache: 'no-store', signal })
        if (response.status === 200) {
          const blob = await response.blob()
          if (!active || observedRevision !== revision.current) return
          replacePreview(URL.createObjectURL(blob))
        } else if (response.status === 204) {
          replacePreview(null)
        } else {
          throw new Error(`Preview unavailable (${response.status})`)
        }
      } catch (e) {
        if (active) {
          setConnected(false)
          reportError.current((e as Error).message)
        }
      } finally {
        polling = false
      }
    }

    void load()
    const timer = window.setInterval(() => { void load() }, STATUS_INTERVAL_MS)
    return () => {
      active = false
      controller.abort()
      window.clearInterval(timer)
      if (photoUrl.current) URL.revokeObjectURL(photoUrl.current)
    }
  }, [replacePreview])

  /** Discard in-flight snapshots and the preview; refetch gallery/events on the next poll. */
  const invalidate = useCallback(() => {
    revision.current += 1
    listsDue.current = 0
    replacePreview(null)
    setStatus(previous => previous ? { ...previous, tracks: [] } : previous)
  }, [replacePreview])

  /** Request fresh gallery/events on the next poll without discarding the current view. */
  const refreshLists = useCallback(() => { listsDue.current = 0 }, [])

  return { status, people, events, preview, connected, ready, invalidate, refreshLists }
}

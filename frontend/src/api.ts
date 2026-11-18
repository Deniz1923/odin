/** HTTP helpers for the local operator API. Mutations carry the per-process session token. */

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* preserve HTTP error */ }
    throw new ApiError(response.status, message)
  }
  return response.status === 204 ? undefined as T : response.json()
}

let token = ''

/** Fetch the session token once; it only changes when the API process restarts. */
export async function ensureSession(signal?: AbortSignal) {
  if (!token) token = (await request<{ token: string }>('/session', { signal })).token
}

export async function mutate<T = unknown>(path: string, body?: BodyInit, method = 'POST'): Promise<T> {
  const send = () => request<T>(path, {
    method,
    body,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(typeof body === 'string' ? { 'Content-Type': 'application/json' } : {}),
    },
  })
  await ensureSession()
  try {
    return await send()
  } catch (error) {
    // A restarted API issues a new token: refresh it once, then report any further rejection.
    if (!(error instanceof ApiError) || error.status !== 403) throw error
    token = ''
    await ensureSession()
    return send()
  }
}

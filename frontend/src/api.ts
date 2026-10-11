// Hand-written to match backend/src/checkbox_detection/api/schemas.py.
export type Box = {
  bbox: [number, number, number, number]
  is_checked: boolean
  page: number
  score: number
  label?: string | null
}

export type DetectResponse = { boxes: Box[] }

/** A failed request, carrying the problem+json `title` and `detail`. */
export class ApiError extends Error {
  title: string
  detail: string

  constructor(title: string, detail: string) {
    super(title)
    this.title = title
    this.detail = detail
  }
}

export async function detect(file: File, signal?: AbortSignal): Promise<DetectResponse> {
  const body = new FormData()
  body.append('file', file)
  let response: Response
  try {
    response = await fetch('/detect?labels=true', { method: 'POST', body, signal })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError("Can't reach the detection service", 'Check your connection and try again.')
  }
  if (!response.ok) {
    const problem = await response.json().catch(() => null)
    throw new ApiError(
      problem?.title ?? `Request failed (${response.status})`,
      problem?.detail ?? 'The service returned an unexpected error. Try again.',
    )
  }
  return response.json()
}

/**
 * Below this locator score a box is flagged as low confidence. YOLOX keeps boxes from 0.2 and real
 * boxes already score near 0.3 (see YoloxParams.score_threshold); OpenCV always scores 1.
 */
export const LOW_CONFIDENCE = 0.3

export const isUncertain = (box: Box) => box.score < LOW_CONFIDENCE

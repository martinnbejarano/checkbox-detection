export type Box = {
  bbox: [number, number, number, number]
  is_checked: boolean
  page: number
  score: number
  label?: string | null
}

/** One line of /detect/stream: a page's boxes, sent as soon as that page is done. */
export type PageResult = { page: number; boxes: Box[] }

export class ApiError extends Error {
  title: string
  detail: string

  constructor(title: string, detail: string) {
    super(title)
    this.title = title
    this.detail = detail
  }
}

export const ACCEPT = 'image/png,image/jpeg,image/tiff,image/webp,application/pdf,.pdf,.tif,.tiff'

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024

const accepted = (file: File) =>
  ACCEPT.split(',').some((type) => (type.startsWith('.') ? file.name.toLowerCase().endsWith(type) : file.type === type))

/** Detects page by page: `onPage` gets each page's boxes as soon as the service has read it. */
export async function detect(file: File, onPage: (page: PageResult) => void, signal?: AbortSignal): Promise<void> {
  if (!accepted(file)) throw new ApiError('Unsupported file type', 'Upload a PNG, JPEG, TIFF, WEBP or PDF file.')

  if (file.size > MAX_UPLOAD_BYTES) throw new ApiError('File too large', 'Upload a file of 50 MB or less.')

  const body = new FormData()
  body.append('file', file)
  let response: Response

  try {
    response = await fetch('/detect/stream?labels=true', { method: 'POST', body, signal })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError("Can't reach the detection service", 'Check your connection and try again.')
  }

  if (!response.ok || !response.body) {
    const problem = await response.json().catch(() => null)
    throw new ApiError(
      problem?.title ?? `Request failed (${response.status})`,
      problem?.detail ?? 'The service returned an unexpected error. Try again.',
    )
  }

  // NDJSON: one page per line. A failure after the first page arrives as an {"error"} line.
  let buffer = ''
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value
      const lines = buffer.split('\n')
      buffer = lines.pop()!
      for (const line of lines.filter(Boolean)) {
        const message = JSON.parse(line)
        if ('error' in message) throw new ApiError('Detection stopped', message.error)
        onPage(message)
      }
    }
  } catch (error) {
    if (signal?.aborted || error instanceof ApiError) throw error
    throw new ApiError('Detection stopped', 'The connection to the service was lost. Try again.')
  }
}

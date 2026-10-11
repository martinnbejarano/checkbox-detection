export type Box = {
  bbox: [number, number, number, number]
  is_checked: boolean
  page: number
  score: number
  label?: string | null
}

export type DetectResponse = { boxes: Box[] }

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

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024

const accepted = (file: File) =>
  ACCEPT.split(',').some((type) => (type.startsWith('.') ? file.name.toLowerCase().endsWith(type) : file.type === type))

export async function detect(file: File, signal?: AbortSignal): Promise<DetectResponse> {
  if (!accepted(file)) throw new ApiError('Unsupported file type', 'Upload a PNG, JPEG, TIFF, WEBP or PDF file.')

  if (file.size > MAX_UPLOAD_BYTES) throw new ApiError('File too large', 'Upload a file of 20 MB or less.')

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

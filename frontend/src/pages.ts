import type { PDFDocumentProxy } from 'pdfjs-dist'

const API_DPI = 300
const RENDER_DPI = 200

export type Page = { url: string; width: number; height: number; index: number }

export function isPdf(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
}

export async function openPdf(file: File): Promise<PDFDocumentProxy> {
  const [{ getDocument, GlobalWorkerOptions }, worker] = await Promise.all([
    import('pdfjs-dist'),
    import('pdfjs-dist/build/pdf.worker.min.mjs?url'),
  ])
  GlobalWorkerOptions.workerSrc = worker.default
  return getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise
}

export async function renderPdfPage(pdf: PDFDocumentProxy, index: number): Promise<Page> {
  const page = await pdf.getPage(index + 1)
  const api = page.getViewport({ scale: API_DPI / 72 })
  const viewport = page.getViewport({ scale: RENDER_DPI / 72 })
  const canvas = document.createElement('canvas')
  canvas.width = Math.ceil(viewport.width)
  canvas.height = Math.ceil(viewport.height)
  await page.render({ canvas, viewport }).promise
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve))
  if (!blob) throw new Error('Could not render the page')
  return { url: URL.createObjectURL(blob), width: Math.round(api.width), height: Math.round(api.height), index }
}

/** Images are shown as-is; the browser applies EXIF rotation like the backend does. */
export async function loadImage(file: File): Promise<Page> {
  const url = URL.createObjectURL(file)
  const image = new Image()
  image.src = url
  try {
    await image.decode()
  } catch (error) {
    URL.revokeObjectURL(url)
    throw error
  }
  return { url, width: image.naturalWidth, height: image.naturalHeight, index: 0 }
}

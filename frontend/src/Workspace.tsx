import { useEffect, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { ApiError, detect, type Box, type DetectResponse } from './api'
import { Connector } from './Connector'
import { Detections, DetectionsSkeleton } from './Detections'
import { DocumentView } from './DocumentView'
import { ChevronIcon, UploadIcon } from './icons'
import { isPdf, loadImage, openPdf, renderPdfPage, type Page } from './pages'

const LOW_CONFIDENCE = 0.3

export type Item = { id: number; box: Box; uncertain: boolean }

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

type Props = {
  file: File
  onUpload: () => void
}

/** One uploaded file. App remounts it per file, so every piece of state starts fresh. */
export function Workspace({ file, onUpload }: Props) {
  const [result, setResult] = useState<DetectResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [pageCount, setPageCount] = useState(1)
  const [pageIndex, setPageIndex] = useState(0)
  const [page, setPage] = useState<Page | null>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [active, setActive] = useState<number | null>(null)
  const stage = useRef<HTMLElement>(null)
  const viewer = useRef<HTMLDivElement>(null)
  const shownUrl = useRef<string | undefined>(undefined)

  useEffect(() => {
    const controller = new AbortController()
    detect(file, controller.signal).then(setResult, (e: unknown) => {
      if (controller.signal.aborted) return
      setError(e instanceof ApiError ? e : new ApiError('Something went wrong', "The response couldn't be read. Try again."))
    })
    return () => controller.abort()
  }, [file])

  useEffect(() => {
    if (!isPdf(file)) return
    let cancelled = false
    let doc: PDFDocumentProxy | undefined
    openPdf(file).then(
      (opened) => {
        doc = opened
        if (cancelled) return void opened.loadingTask.destroy()
        setPdf(opened)
        setPageCount(opened.numPages)
      },
      () => !cancelled && setPreviewFailed(true),
    )
    return () => {
      cancelled = true
      void doc?.loadingTask.destroy()
    }
  }, [file])

  // The page on screen: the image itself, or the current PDF page rendered by pdf.js. The old page
  // stays up until the new one is ready, so its URL is revoked only once it is replaced.
  useEffect(() => {
    const load = pdf ? renderPdfPage(pdf, pageIndex) : isPdf(file) ? null : loadImage(file)
    if (!load) return
    let cancelled = false
    load.then(
      (loaded) => {
        if (cancelled) return URL.revokeObjectURL(loaded.url)
        if (shownUrl.current) URL.revokeObjectURL(shownUrl.current)
        shownUrl.current = loaded.url
        setPage(loaded)
      },
      () => !cancelled && setPreviewFailed(true),
    )
    return () => {
      cancelled = true
    }
  }, [file, pdf, pageIndex])

  useEffect(
    () => () => {
      if (shownUrl.current) URL.revokeObjectURL(shownUrl.current)
    },
    [],
  )

  const items: Item[] = (result?.boxes ?? [])
    .map((box, id) => ({ id, box, uncertain: box.score < LOW_CONFIDENCE }))
    .filter((item) => item.box.page === pageIndex)

  const goToPage = (index: number) => {
    setActive(null)
    setPageIndex(index)
  }

  const select = (id: number) => {
    const box = stage.current?.querySelector(`[data-box="${id}"]`)
    box?.scrollIntoView({ block: 'center', inline: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' })
  }

  // Hovering a row whose box is out of view brings the box in; a visible box never moves.
  const hoverRow = (id: number | null) => {
    setActive(id)
    const box = id === null ? null : stage.current?.querySelector(`[data-box="${id}"]`)
    const pane = viewer.current?.getBoundingClientRect()
    if (!box || !pane) return
    const b = box.getBoundingClientRect()
    if (b.top < pane.top || b.bottom > pane.bottom || b.left < pane.left || b.right > pane.right) select(id!)
  }

  const hoverBox = (id: number | null) => {
    setActive(id)
    if (id !== null) stage.current?.querySelector(`[data-row="${id}"]`)?.scrollIntoView({ block: 'nearest' })
  }

  const status = error
    ? `${error.title}. ${error.detail}`
    : result
      ? `Found ${result.boxes.length} checkboxes`
      : 'Detecting checkboxes'

  return (
    <main className="workspace" ref={stage}>
      <h1 className="sr-only">Checkbox Detection</h1>
      <section className="viewer" aria-label="Document">
        <div className="viewer-bar">
          <div className="file" title={file.name}>
            <span className="file-name">{file.name}</span>
            <span className="file-meta">{formatSize(file.size)}</span>
          </div>
          <ul className="legend" aria-label="Legend">
            <li data-state="checked">Checked</li>
            <li data-state="unchecked">Unchecked</li>
            <li data-state="uncertain">Low confidence</li>
          </ul>
          {pageCount > 1 && (
            <nav className="pager" aria-label="Pages">
              <button
                type="button"
                className="icon-button"
                aria-label="Previous page"
                disabled={pageIndex === 0}
                onClick={() => goToPage(pageIndex - 1)}
              >
                <ChevronIcon style={{ transform: 'scaleX(-1)' }} />
              </button>
              <span>
                Page {pageIndex + 1} <span className="muted">of {pageCount}</span>
              </span>
              <button
                type="button"
                className="icon-button"
                aria-label="Next page"
                disabled={pageIndex === pageCount - 1}
                onClick={() => goToPage(pageIndex + 1)}
              >
                <ChevronIcon />
              </button>
            </nav>
          )}
          <button type="button" className="button" onClick={onUpload}>
            <UploadIcon />
            Upload another
          </button>
        </div>
        <div className="viewer-canvas" ref={viewer}>
          {page ? (
            <DocumentView
              page={page}
              // Until the new page has rendered, the old one shows without boxes rather than with the wrong ones.
              items={page.index === pageIndex ? items : []}
              active={active}
              scanning={!result && !error}
              onHover={hoverBox}
            />
          ) : previewFailed ? (
            <p className="viewer-note">
              Your browser can't preview this file.{!error && ' The detections are still listed on the right.'}
            </p>
          ) : (
            <div className="sheet placeholder" aria-hidden="true" />
          )}
        </div>
      </section>

      <aside className="panel" aria-label="Detections">
        <div className="panel-head">
          <h2>Detections</h2>
          {result && <span className="count">{items.length}</span>}
          {pageCount > 1 && <span className="muted">on page {pageIndex + 1}</span>}
        </div>
        <div className="panel-body">
          {error ? (
            <div className="panel-note error" role="alert">
              <p className="panel-note-title">{error.title}</p>
              <p>{error.detail}</p>
              <button type="button" className="button" onClick={onUpload}>
                Choose another file
              </button>
            </div>
          ) : result ? (
            <Detections items={items} active={active} onHover={hoverRow} onSelect={select} />
          ) : (
            <DetectionsSkeleton />
          )}
        </div>
      </aside>

      <Connector active={active} stage={stage} viewer={viewer} />
      <p className="sr-only" aria-live="polite">
        {status}
      </p>
    </main>
  )
}

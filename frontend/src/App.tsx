import { useEffect, useRef, useState, type DragEvent } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { ApiError, detect, type DetectResponse } from './api'
import { Connector } from './Connector'
import { Detections, DetectionsSkeleton } from './Detections'
import { DocumentView } from './DocumentView'
import { ChevronIcon, UploadIcon } from './icons'
import { isPdf, loadImage, openPdf, renderPdfPage, type Page } from './pages'

const ACCEPT = 'image/png,image/jpeg,image/tiff,image/webp,application/pdf,.pdf,.tif,.tiff'

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DetectResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [pageCount, setPageCount] = useState(1)
  const [pageIndex, setPageIndex] = useState(0)
  const [page, setPage] = useState<Page | null>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [active, setActive] = useState<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const stage = useRef<HTMLElement>(null)
  const viewer = useRef<HTMLDivElement>(null)

  const choose = (next: File | undefined) => {
    if (!next) return
    setFile(next)
    setResult(null)
    setError(null)
    setPdf(null)
    setPageCount(1)
    setPageIndex(0)
    setPage(null)
    setPreviewFailed(false)
    setActive(null)
  }

  // A new file aborts the previous request.
  useEffect(() => {
    if (!file) return
    const controller = new AbortController()
    detect(file, controller.signal).then(setResult, (e: unknown) => {
      if (controller.signal.aborted) return
      setError(e instanceof ApiError ? e : new ApiError('Something went wrong', String(e)))
    })
    return () => controller.abort()
  }, [file])

  useEffect(() => {
    if (!file || !isPdf(file)) return
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

  // The page on screen: the image itself, or the current PDF page rendered by pdf.js.
  useEffect(() => {
    if (!file) return
    const load = pdf ? renderPdfPage(pdf, pageIndex) : isPdf(file) ? null : loadImage(file)
    if (!load) return
    let cancelled = false
    let url: string | undefined
    load.then(
      (loaded) => {
        url = loaded.url
        if (cancelled) URL.revokeObjectURL(loaded.url)
        else setPage(loaded)
      },
      () => !cancelled && setPreviewFailed(true),
    )
    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [file, pdf, pageIndex])

  const ids: number[] = []
  result?.boxes.forEach((box, id) => box.page === pageIndex && ids.push(id))
  const boxes = ids.map((id) => result!.boxes[id])

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

  const onDragOver = (e: DragEvent) => {
    e.preventDefault()
    setDragging(true)
  }
  const onDragLeave = (e: DragEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    choose(e.dataTransfer.files[0])
  }

  const status = error
    ? `${error.title}. ${error.detail}`
    : result
      ? `Found ${result.boxes.length} checkboxes`
      : file
        ? 'Detecting checkboxes'
        : ''

  return (
    <div className="app" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      {file ? (
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
              <button type="button" className="button" onClick={() => input.current?.click()}>
                <UploadIcon />
                Upload another
              </button>
            </div>
            <div className="viewer-canvas" ref={viewer}>
              {page ? (
                <DocumentView
                  page={page}
                  boxes={boxes}
                  ids={ids}
                  active={active}
                  scanning={!result && !error}
                  onHover={hoverBox}
                />
              ) : previewFailed ? (
                <p className="viewer-note">
                  Your browser can't preview this file. The detections are still listed on the right.
                </p>
              ) : (
                <div className="sheet placeholder" aria-hidden="true" />
              )}
            </div>
          </section>

          <aside className="panel" aria-label="Detections">
            <div className="panel-head">
              <h2>Detections</h2>
              {result && <span className="count">{boxes.length}</span>}
              {pageCount > 1 && <span className="muted">on page {pageIndex + 1}</span>}
            </div>
            <div className="panel-body">
              {error ? (
                <div className="panel-note error" role="alert">
                  <p className="panel-note-title">{error.title}</p>
                  <p>{error.detail}</p>
                  <button type="button" className="button" onClick={() => input.current?.click()}>
                    Choose another file
                  </button>
                </div>
              ) : result ? (
                <Detections boxes={boxes} ids={ids} active={active} onHover={hoverRow} onSelect={select} />
              ) : (
                <DetectionsSkeleton />
              )}
            </div>
          </aside>

          <Connector active={active} stage={stage} viewer={viewer} />
        </main>
      ) : (
        <main className="empty">
          <h1>Find every checkbox on a form</h1>
          <p>Each checkbox is located, marked as checked or unchecked, and matched with the label next to it.</p>
          <button
            type="button"
            className="dropzone"
            aria-label="Choose a form to upload, or drag and drop it here"
            data-dragging={dragging || undefined}
            onClick={() => input.current?.click()}
          >
            <span className="dropzone-art" aria-hidden="true">
              <span data-state="checked" />
              <span data-state="unchecked" />
              <span data-state="uncertain" />
            </span>
            <span className="dropzone-title">{dragging ? 'Drop to detect checkboxes' : 'Drag and drop a form here'}</span>
            <span className="dropzone-hint">PNG, JPEG, TIFF, WEBP or PDF · up to 20 MB</span>
            <span className="button primary">
              <UploadIcon />
              Browse files
            </span>
          </button>
        </main>
      )}

      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          choose(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      {dragging && file && (
        <div className="drop-overlay" aria-hidden="true">
          <p>Drop to detect checkboxes</p>
        </div>
      )}
      <p className="sr-only" aria-live="polite">
        {status}
      </p>
    </div>
  )
}

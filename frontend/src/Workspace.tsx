import { useEffect, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { ApiError, detect, type Box } from './api'
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
  const [boxes, setBoxes] = useState<Box[]>([])
  // Pages arrive in order, so the first `readPages` pages have all their boxes.
  const [readPages, setReadPages] = useState(0)
  const [done, setDone] = useState(false)
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
    const onPage = ({ page, boxes }: { page: number; boxes: Box[] }) => {
      setBoxes((prev) => [...prev, ...boxes])
      setReadPages(page + 1)
    }
    detect(file, onPage, controller.signal).then(() => setDone(true), (e: unknown) => {
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

  const pageRead = pageIndex < readPages
  const items: Item[] = boxes
    .map((box, id) => ({ id, box, uncertain: box.score < LOW_CONFIDENCE }))
    .filter((item) => item.box.page === pageIndex)
  const checkedCount = items.filter((item) => item.box.is_checked).length
  const queryCount = items.filter((item) => item.uncertain).length
  const focused = items.find((item) => item.id === active)

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
    : done
      ? `Found ${boxes.length} checkboxes`
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
          {previewFailed && !page ? (
            <p className="viewer-note">
              Your browser can't preview this file.{!error && ' The detections are still listed on the right.'}
            </p>
          ) : (
            // Rendered before the page exists too, so the sheet is one element from the first frame
            // (the upload's view transition morphs into it and must not see it replaced).
            <DocumentView
              page={page}
              // Until the new page has rendered, the old one shows without boxes rather than with the wrong ones.
              items={page?.index === pageIndex ? items : []}
              active={active}
              scanning={!pageRead && !error}
              onHover={hoverBox}
            />
          )}
        </div>
      </section>

      <aside className="panel" aria-label="Detections">
        <div className="panel-head">
          <h2>Detections</h2>
          {pageRead && <span className="count">{items.length}</span>}
          {pageCount > 1 && (
            <span className="muted">
              on page {pageIndex + 1}
              {!done && !error && ` · ${readPages} of ${pageCount} read`}
            </span>
          )}
        </div>
        {pageRead && items.length > 0 && (
          <p className="tally">
            <span>{checkedCount} checked</span>
            <span>{items.length - checkedCount} unchecked</span>
            {queryCount > 0 && <span className="tally-query">{queryCount} to check by eye</span>}
          </p>
        )}
        <div className="panel-body">
          {/* Pages read before a failure keep their detections; the error shows on the rest. */}
          {pageRead ? (
            <Detections items={items} active={active} onHover={hoverRow} onSelect={select} />
          ) : error ? (
            <div className="panel-note error" role="alert">
              <p className="panel-note-title">{error.title}</p>
              <p>{error.detail}</p>
              <button type="button" className="button" onClick={onUpload}>
                Choose another file
              </button>
            </div>
          ) : (
            <DetectionsSkeleton />
          )}
        </div>
        {pageRead && items.length > 0 && (
          <p className="slip">
            {focused ? (
              <>
                <span className="slip-no">No. {items.indexOf(focused) + 1}</span>
                <span>
                  x {focused.box.bbox[0]} · y {focused.box.bbox[1]}
                </span>
                <span>
                  {focused.box.bbox[2] - focused.box.bbox[0]} × {focused.box.bbox[3] - focused.box.bbox[1]} px
                </span>
                <span>Confidence {focused.box.score.toFixed(2)}</span>
              </>
            ) : (
              <span className="muted">Select a detection to read its position and confidence.</span>
            )}
          </p>
        )}
      </aside>

      <Connector active={active} stage={stage} viewer={viewer} />
      <p className="sr-only" aria-live="polite">
        {status}
      </p>
    </main>
  )
}

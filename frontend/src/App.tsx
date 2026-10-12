import { useRef, useState, type DragEvent } from 'react'
import { flushSync } from 'react-dom'
import { ACCEPT } from './api'
import { UploadIcon } from './icons'
import { Reader } from './Reader'
import { Workspace } from './Workspace'

export default function App() {
  const [upload, setUpload] = useState<{ file: File; key: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const choose = (file: File | undefined) => {
    if (!file) return
    const open = () => setUpload((prev) => ({ file, key: (prev?.key ?? 0) + 1 }))
    // The first upload morphs the example sheet into the document's sheet (view-transition-name: sheet).
    if (upload || !document.startViewTransition || matchMedia('(prefers-reduced-motion: reduce)').matches) return open()
    document.startViewTransition(() => flushSync(open))
  }
  const browse = () => input.current?.click()

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

  return (
    <div className="app" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      {upload ? (
        <Workspace key={upload.key} file={upload.file} onUpload={browse} />
      ) : (
        <main className="empty">
          <div className="empty-copy">
            <h1>Checkbox detection for appraisal forms</h1>
            <p>Upload a page. Each box gets a position on the page, a checked or unchecked state, and the label next to it.</p>
            <button
              type="button"
              className="dropzone"
              aria-label="Choose a form to upload, or drag and drop it here"
              data-dragging={dragging || undefined}
              onClick={browse}
            >
              <span className="crop" data-corner="tl" aria-hidden="true" />
              <span className="crop" data-corner="tr" aria-hidden="true" />
              <span className="crop" data-corner="bl" aria-hidden="true" />
              <span className="crop" data-corner="br" aria-hidden="true" />
              <span className="dropzone-title">{dragging ? 'Drop to detect checkboxes' : 'Drag and drop a form here'}</span>
              <span className="dropzone-hint">PNG, JPEG, TIFF, WEBP or PDF · up to 20 MB</span>
              <span className="button primary">
                <UploadIcon />
                Browse files
              </span>
            </button>
          </div>
          <Reader />
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
      {dragging && upload && (
        <div className="drop-overlay" aria-hidden="true">
          <p>Drop to detect checkboxes</p>
        </div>
      )}
    </div>
  )
}

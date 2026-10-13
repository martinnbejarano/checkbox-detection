import { useRef, useState, type DragEvent } from 'react'
import { flushSync } from 'react-dom'
import { ACCEPT } from './api'
import { UploadIcon } from './icons'
import { Reader } from './reader/Reader'
import { Workspace } from './Workspace'

// Copies of samples/ served from public/ (plus a small .webp preview each), for reviewers without a form at hand.
const SAMPLES = [
  { name: 'sample-000', form: '1004' },
  { name: 'sample-001', form: '1004' },
  { name: 'sample-002', form: '1004MC' },
  { name: 'sample-003', form: '1004C' },
]

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
  const [loadingSample, setLoadingSample] = useState<string | null>(null)
  const [sampleFailed, setSampleFailed] = useState(false)
  const trySample = async (name: string) => {
    setLoadingSample(name)
    setSampleFailed(false)
    try {
      const response = await fetch(`/samples/${name}.png`)
      if (!response.ok) throw new Error(response.statusText)
      choose(new File([await response.blob()], `${name}.png`, { type: 'image/png' }))
    } catch {
      setSampleFailed(true)
    } finally {
      setLoadingSample(null)
    }
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
              <span className="dropzone-hint">PNG, JPEG, TIFF, WEBP or PDF · up to 50 MB</span>
              <span className="button primary">
                <UploadIcon />
                Browse files
              </span>
            </button>
            <section className="samples" aria-labelledby="samples-title">
              <h2 id="samples-title">No form at hand? Try a sample</h2>
              <ul>
                {SAMPLES.map(({ name, form }) => (
                  <li key={name}>
                    <button
                      type="button"
                      className="sample"
                      aria-busy={loadingSample === name || undefined}
                      disabled={loadingSample !== null}
                      onClick={() => trySample(name)}
                    >
                      <span className="sample-sheet">
                        <img src={`/samples/${name}.webp`} alt="" width={176} height={228} />
                        {loadingSample === name && <span className="scan" />}
                      </span>
                      <span className="sample-form">{form}</span>
                    </button>
                  </li>
                ))}
              </ul>
              {sampleFailed && (
                <p className="samples-error" role="alert">
                  The sample couldn't be loaded. Try again.
                </p>
              )}
            </section>
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

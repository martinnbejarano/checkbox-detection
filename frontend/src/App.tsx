import { useRef, useState, type DragEvent } from 'react'
import { ACCEPT } from './api'
import { UploadIcon } from './icons'
import { Workspace } from './Workspace'

export default function App() {
  const [upload, setUpload] = useState<{ file: File; key: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const choose = (file: File | undefined) => {
    if (file) setUpload((prev) => ({ file, key: (prev?.key ?? 0) + 1 }))
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
          <h1>Find every checkbox on a form</h1>
          <p>Each checkbox is located, marked as checked or unchecked, and matched with the label next to it.</p>
          <button
            type="button"
            className="dropzone"
            aria-label="Choose a form to upload, or drag and drop it here"
            data-dragging={dragging || undefined}
            onClick={browse}
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
      {dragging && upload && (
        <div className="drop-overlay" aria-hidden="true">
          <p>Drop to detect checkboxes</p>
        </div>
      )}
    </div>
  )
}

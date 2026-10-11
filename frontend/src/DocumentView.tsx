import { isUncertain, type Box } from './api'
import type { Page } from './pages'

type Props = {
  page: Page
  boxes: Box[] // this page's boxes; `ids` gives each one's index in the full response
  ids: number[]
  active: number | null
  scanning: boolean
  onHover: (id: number | null) => void
}

/** The page with its boxes, positioned in percent of the bbox space so they scale with the image. */
export function DocumentView({ page, boxes, ids, active, scanning, onHover }: Props) {
  const pct = (value: number, total: number) => `${(value / total) * 100}%`
  return (
    <div className="sheet" data-active={active !== null || undefined}>
      <img src={page.url} alt="Uploaded document" draggable={false} />
      {boxes.map((box, i) => {
        const [x1, y1, x2, y2] = box.bbox
        const state = isUncertain(box) ? 'uncertain' : box.is_checked ? 'checked' : 'unchecked'
        return (
          <div
            key={ids[i]}
            className="box"
            data-box={ids[i]}
            data-state={state}
            data-active={ids[i] === active || undefined}
            style={{
              left: pct(x1, page.width),
              top: pct(y1, page.height),
              width: pct(x2 - x1, page.width),
              height: pct(y2 - y1, page.height),
            }}
            onMouseEnter={() => onHover(ids[i])}
            onMouseLeave={() => onHover(null)}
            aria-hidden="true"
          />
        )
      })}
      {scanning && <div className="scan" aria-hidden="true" />}
    </div>
  )
}

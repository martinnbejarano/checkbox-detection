import type { Page } from './pages'
import type { Item } from './Workspace'

type Props = {
  page: Page
  items: Item[]
  active: number | null
  scanning: boolean
  onHover: (id: number | null) => void
}

/** The page with its boxes, positioned in percent of the bbox space so they scale with the image. */
export function DocumentView({ page, items, active, scanning, onHover }: Props) {
  const pct = (value: number, total: number) => `${(value / total) * 100}%`
  return (
    <div className="sheet" data-active={active !== null || undefined}>
      <img src={page.url} alt="Uploaded document" draggable={false} />
      {items.map(({ id, box, uncertain }) => {
        const [x1, y1, x2, y2] = box.bbox
        return (
          <div
            key={id}
            className="box"
            data-box={id}
            data-state={box.is_checked ? 'checked' : 'unchecked'}
            data-uncertain={uncertain || undefined}
            data-active={id === active || undefined}
            style={{
              left: pct(x1, page.width),
              top: pct(y1, page.height),
              width: pct(x2 - x1, page.width),
              height: pct(y2 - y1, page.height),
            }}
            onMouseEnter={() => onHover(id)}
            onMouseLeave={() => onHover(null)}
            aria-hidden="true"
          />
        )
      })}
      {scanning && <div className="scan" aria-hidden="true" />}
    </div>
  )
}

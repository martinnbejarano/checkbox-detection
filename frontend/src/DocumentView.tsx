import type { CSSProperties } from 'react'
import type { Page } from './pages'
import type { Item } from './Workspace'

type Props = {
  page: Page | null
  items: Item[]
  active: number | null
  scanning: boolean
  onHover: (id: number | null) => void
}

// Ruler ticks every TICK px of the bbox space, numbered every LABEL px.
const TICK = 100
const LABEL = 500

const pct = (value: number, total: number) => `${(value / total) * 100}%`

/**
 * The page with its boxes, positioned in percent of the bbox space so they scale with the image.
 * Without a page yet it shows the blank sheet the page will appear on.
 */
export function DocumentView({ page, items, active, scanning, onHover }: Props) {
  const current = items.find((item) => item.id === active)?.box.bbox
  const ruler = (total: number, from?: number, to?: number) => (
    <>
      {Array.from({ length: Math.floor(total / LABEL) + 1 }, (_, i) => (
        <span key={i} className="ruler-label" style={{ '--at': pct(i * LABEL, total) } as CSSProperties}>
          {i * LABEL}
        </span>
      ))}
      {from !== undefined && to !== undefined && (
        <span
          className="ruler-mark"
          style={{ '--at': pct(from, total), '--size': pct(to - from, total) } as CSSProperties}
        />
      )}
    </>
  )

  return (
    <div
      className="sheet-frame"
      style={page ? ({ '--tick-x': pct(TICK, page.width), '--tick-y': pct(TICK, page.height) } as CSSProperties) : undefined}
    >
      {page && (
        <div className="ruler ruler-x" aria-hidden="true">
          {ruler(page.width, current?.[0], current?.[2])}
        </div>
      )}
      {page && (
        <div className="ruler ruler-y" aria-hidden="true">
          {ruler(page.height, current?.[1], current?.[3])}
        </div>
      )}
      <div className={page ? 'sheet' : 'sheet placeholder'} data-active={active !== null || undefined}>
        {page && <img src={page.url} alt="Uploaded document" draggable={false} />}
        {['tl', 'tr', 'bl', 'br'].map((corner) => (
          <span key={corner} className="crop" data-corner={corner} aria-hidden="true" />
        ))}
        {page &&
          items.map(({ id, box, uncertain }, i) => {
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
                  // Marks land in reading order, like a corrector working down the page.
                  animationDelay: `${Math.min(i * 12, 900)}ms`,
                }}
                onMouseEnter={() => onHover(id)}
                onMouseLeave={() => onHover(null)}
                aria-hidden="true"
              />
            )
          })}
        {scanning && <div className="scan" aria-hidden="true" />}
      </div>
    </div>
  )
}

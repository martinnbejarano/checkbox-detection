import { useLayoutEffect, useState, type RefObject } from 'react'

type Props = {
  active: number | null
  stage: RefObject<HTMLElement | null> 
  viewer: RefObject<HTMLElement | null> 
}

type Line = { d: string; box: [number, number]; row: [number, number]; offscreen: boolean }

// Keeps the line's end off the pane's very edge when the box is scrolled out of view.
const EDGE = 12
// The active box's focus ring (.box[data-active] box-shadow), so the line starts outside it.
const RING = 5
// How far above the box the line turns toward the list. Leaving from the top keeps the
// line off the label, which sits to the right of the box on the same form row.
const LIFT = 2

/**
 * Draws a leader from the active box to its row: up from the box's top edge, right to the pane
 * divider, along the divider to the row's height, then into the row. Re-measured on scroll and on resize.
 */
export function Connector({ active, stage, viewer }: Props) {
  const [line, setLine] = useState<Line | null>(null)

  useLayoutEffect(() => {
    const root = stage.current
    if (active === null || !root) return
    const measure = () => {
      const box = root.querySelector(`[data-box="${active}"]`)
      const row = root.querySelector(`[data-row="${active}"]`)
      const pane = viewer.current
      if (!box || !row || !pane) return setLine(null)
      const s = root.getBoundingClientRect()
      const b = box.getBoundingClientRect()
      const r = row.getBoundingClientRect()
      const p = pane.getBoundingClientRect()
      const clamp = (y: number) => Math.min(Math.max(y, p.top + EDGE), p.bottom - EDGE) - s.top
      const top = b.top - RING
      const by = clamp(top)
      const lane = clamp(top - LIFT)
      const ry = clamp(r.top + r.height / 2)
      const bx = Math.min(b.left + b.width / 2, p.right) - s.left
      const gx = p.right - s.left
      const rx = r.left - s.left
      setLine({
        d: `M${bx} ${by} V${lane} H${gx} V${ry} H${rx}`,
        box: [bx, by],
        row: [rx, ry],
        offscreen: top < p.top || top > p.bottom,
      })
    }
    measure()
    // Resizes cover the window, the image finishing its load and the list reflowing.
    const observer = new ResizeObserver(measure)
    for (const el of [root, root.querySelector(`[data-box="${active}"]`), root.querySelector(`[data-row="${active}"]`)]) {
      if (el) observer.observe(el)
    }
    root.addEventListener('scroll', measure, true)
    return () => {
      observer.disconnect()
      root.removeEventListener('scroll', measure, true)
    }
  }, [active, stage, viewer])

  if (active === null || !line) return null
  return (
    <svg className="connector" aria-hidden="true">
      <path className="casing" d={line.d} />
      <path d={line.d} data-offscreen={line.offscreen || undefined} />
      <circle cx={line.box[0]} cy={line.box[1]} r="3" />
      <circle cx={line.row[0]} cy={line.row[1]} r="3" />
    </svg>
  )
}

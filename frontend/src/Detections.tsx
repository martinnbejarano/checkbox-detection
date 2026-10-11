import { isUncertain, type Box } from './api'
import { CheckedIcon, UncheckedIcon, WarningIcon } from './icons'

type Props = {
  boxes: Box[]
  ids: number[]
  active: number | null
  onHover: (id: number | null) => void
  onSelect: (id: number) => void
}

export function Detections({ boxes, ids, active, onHover, onSelect }: Props) {
  if (boxes.length === 0) {
    return (
      <div className="panel-note">
        <p className="panel-note-title">No checkboxes found</p>
        <p>Nothing on this page looked like a checkbox. Try a sharper scan or another page.</p>
      </div>
    )
  }
  return (
    <ol className="rows">
      {boxes.map((box, i) => {
        const id = ids[i]
        return (
          <li key={id}>
            <button
              type="button"
              className="row"
              data-row={id}
              data-active={id === active || undefined}
              onMouseEnter={() => onHover(id)}
              onMouseLeave={() => onHover(null)}
              onFocus={() => onHover(id)}
              onBlur={() => onHover(null)}
              onClick={() => onSelect(id)}
            >
              <span className="row-index">{i + 1}</span>
              <span className={box.is_checked ? 'row-icon checked' : 'row-icon unchecked'}>
                {box.is_checked ? <CheckedIcon /> : <UncheckedIcon />}
              </span>
              <span className="row-label">
                {box.label ? box.label : <span className="no-label">Label not detected</span>}
              </span>
              {isUncertain(box) && (
                <span className="uncertain" title={`Locator confidence ${box.score.toFixed(2)}`}>
                  <WarningIcon />
                  Low confidence
                </span>
              )}
              <span className="row-state">{box.is_checked ? 'Checked' : 'Unchecked'}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

export function DetectionsSkeleton() {
  return (
    <ol className="rows" aria-hidden="true">
      {Array.from({ length: 9 }, (_, i) => (
        <li key={i} className="row skeleton" style={{ animationDelay: `${i * 60}ms` }}>
          <span />
          <span />
        </li>
      ))}
    </ol>
  )
}

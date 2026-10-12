import { useEffect, useRef } from 'react'
import { startReader } from './story'

const STEPS = ['Scan', 'Align', 'Extract', 'Read out']

/** The empty state's piece: the service's process told in ink dots, on a loop. It takes no input. */
export function Reader() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const steps = useRef<HTMLOListElement>(null)

  useEffect(() => {
    if (!canvas.current || !steps.current) return
    return startReader(canvas.current, [...steps.current.querySelectorAll('li')])
  }, [])

  return (
    <figure
      className="reader"
      role="img"
      aria-label="A looping animation in ink dots: a photographed form arrives tilted, straightens, its checkboxes light up in blue row by row, and nine of them lift off with their labels and settle into a list that reads each one as checked or unchecked."
    >
      <canvas ref={canvas} aria-hidden="true" />
      {/* Status, not controls: the current step of the loop. */}
      <ol className="steps" ref={steps} aria-hidden="true">
        {STEPS.map((step, i) => (
          <li key={step}>
            <span className="steps-n">{i + 1}</span>
            {step}
            <i />
          </li>
        ))}
      </ol>
    </figure>
  )
}

/**
 * The empty state's piece: ~5k ink dots tell the service's process on a loop. A photographed form
 * arrives tilted (Scan), straightens (Align), its checkboxes light up and lift off with their labels
 * (Extract), then settle into the same list the workspace shows (Read out). Canvas 2D, no input.
 *
 * Everything is a pure function of the clock, so a dot never jumps: each frame recomputes every
 * position from the time. The clock only runs while the canvas is on screen and the tab is visible.
 */

type Point = {
  x: number
  y: number
  kind: number // 0 page ink, 1 box outline, 2 box hatch
  a: number
  box: number
  pair: number
  label: boolean
  hx: number
  hy: number
  checked: boolean
  u: number
  v: number
  lx: number
  ly: number
  sx: number
  sy: number
  sz: number
  sa: number
  jx: number
  jy: number
  w1: number
  w2: number
  dA: number
  dS: number
  dD: number
}

type Box = { x: number; y: number; s: number; checked: boolean; label: string; row: number; col: number }
type Pose = number[] // [rotZ, rotX, rotY, scale, tx, ty, tz]

// ---------- The form: a 1004 Subject + Neighborhood fragment, in page units (width 1, height 1.3, y down).
const W = 1
const H = 1.3
const X0 = -0.43
const X1 = 0.43
const CH = 0.0076

// The first nine boxes in reading order are read out; the rest stay on the page.
const PAIRS = 9

function buildForm() {
  const rules: { x0: number; x1: number; y: number; weight: number }[] = []
  const texts: { x: number; y: number; w: number; h: number; a: number; box: number }[] = []
  const boxes: Box[] = []
  const rule = (x0: number, x1: number, y: number, weight = 1) => rules.push({ x0, x1, y, weight })
  const type = (str: string, x: number, y: number, { cw = CH, h = 0.0085, a = 0.8 } = {}) => {
    for (const word of str.split(' ')) {
      const w = word.length * cw
      texts.push({ x, y: y - h, w, h, a, box: -1 })
      x += w + cw * 1.1
    }
    return x
  }
  const textW = (s: string, cw = CH) => s.length * cw * 1.07

  let y = -H / 2 + 0.075
  type('Uniform Residential Appraisal Report', X0, y, { cw: 0.0135, h: 0.02, a: 0.95 })
  type('File #', X1 - textW('File #') - 0.12, y, { a: 0.7 })
  rule(X1 - 0.11, X1, y)
  y += 0.026
  type('The purpose of this summary appraisal report is to provide the lender with an accurate opinion', X0, y, {
    a: 0.45,
    h: 0.008,
    cw: 0.0062,
  })
  y += 0.02
  rule(X0, X1, y, 2.2)

  const OPT_COL = X0 + 0.27
  const BOX = 0.024
  const ROW = 0.05
  const rows: [string, string, string[]?, number?][] = [
    ['section', 'Subject'],
    ['fields', 'Property Address', ['City', 'State', 'Zip Code']],
    ['fields', 'Borrower', ['Owner of Public Record', 'County']],
    ['fields', 'Legal Description', ['Tax Year', 'R.E. Taxes $']],
    ['opts', 'Occupant', ['Owner', 'Tenant', 'Vacant'], 0],
    ['opts', 'Property Rights', ['Fee Simple', 'Leasehold', 'Other (describe)'], 0],
    ['opts', 'Assignment Type', ['Purchase Transaction', 'Refinance Transaction', 'Other'], 1],
    ['opts', 'Offered for sale?', ['Yes', 'No'], 1],
    ['opts', 'HOA $', ['per year', 'per month'], 0],
    ['opts', 'Contract analyzed?', ['I did', 'did not'], 0],
    ['opts', 'Financial assistance?', ['Yes', 'No'], 1],
    ['section', 'Neighborhood'],
    ['opts', 'Location', ['Urban', 'Suburban', 'Rural'], 1],
    ['opts', 'Built-Up', ['Over 75%', '25-75%', 'Under 25%'], 0],
    ['opts', 'Growth', ['Rapid', 'Stable', 'Slow'], 1],
    ['opts', 'Property Values', ['Increasing', 'Stable', 'Declining'], 1],
    ['opts', 'Demand/Supply', ['Shortage', 'In Balance', 'Over Supply'], 1],
  ]
  let optRow = 0
  for (const [kind, label, items = [], checked] of rows) {
    if (kind === 'section') {
      y += 0.048
      type(label, X0, y, { cw: 0.0105, h: 0.014, a: 0.95 })
      y += 0.012
      rule(X0, X1, y, 1.4)
      continue
    }
    y += ROW
    const base = y - 0.014
    type(label, X0, base, { a: 0.85 })
    if (kind === 'fields') {
      const span = (X1 - OPT_COL) / items.length
      items.forEach((f, i) => type(f, OPT_COL + i * span, base, { a: 0.55 }))
    } else {
      let x = OPT_COL
      items.forEach((o, i) => {
        boxes.push({ x, y: base - BOX * 0.85, s: BOX, checked: i === checked, label: o, row: optRow, col: i })
        const t0 = texts.length
        x = type(o, x + BOX + 0.011, base, { a: 0.55 }) + 0.03
        for (let k = t0; k < texts.length; k++) texts[k].box = boxes.length - 1
      })
      optRow++
    }
    rule(X0, X1, y, 0.8)
  }
  y += 0.05
  type('Neighborhood Description', X0, y, { a: 0.85 })
  for (let i = 0; i < 4 && y < H / 2 - 0.08; i++) {
    y += 0.04
    rule(X0, X1, y, 0.6)
  }
  return { rules, texts, boxes }
}

// ---------- Points (~5k): rules, type, boxes. Pair labels get denser type so they hold a shape in flight.
function buildPoints(form: ReturnType<typeof buildForm>) {
  const { rules, texts, boxes } = form
  const N_TARGET = 5000
  const PER_BOX = 24
  const HATCH = 18
  const rand = mulberry(7)
  const gauss = () => {
    let u = 0
    for (let i = 0; i < 4; i++) u += rand()
    return (u - 2) * 0.87
  }
  const tw = (t: (typeof texts)[number]) => t.w * t.h * (t.box >= 0 && t.box < PAIRS ? 4 : 1)
  const rest = N_TARGET - boxes.length * (PER_BOX + HATCH)
  const ruleN = Math.round(rest * 0.45)
  const textN = rest - ruleN
  const ruleLen = rules.reduce((s, r) => s + (r.x1 - r.x0) * r.weight, 0)
  const textArea = texts.reduce((s, t) => s + tw(t), 0)

  const P: Point[] = []
  const add = (x: number, y: number, o: Partial<Point>) =>
    P.push({
      x,
      y,
      kind: 0,
      a: 0.8,
      box: -1,
      pair: -1,
      label: false,
      hx: x,
      hy: y,
      checked: false,
      u: 0,
      v: 0,
      lx: 0,
      ly: 0,
      sx: 0,
      sy: 0,
      sz: 0,
      sa: 0,
      jx: 0,
      jy: 0,
      w1: 0,
      w2: 0,
      dA: 0,
      dS: 0,
      dD: 0,
      ...o,
    })
  for (const r of rules) {
    const n = Math.max(2, Math.round((ruleN * (r.x1 - r.x0) * r.weight) / ruleLen))
    for (let i = 0; i < n; i++) {
      const thick = r.weight > 1.2 ? (rand() - 0.5) * 0.004 * r.weight : (rand() - 0.5) * 0.0008
      add(r.x0 + ((i + rand() * 0.6) / n) * (r.x1 - r.x0), r.y + thick, { a: 0.78 })
    }
  }
  // Pair labels are kept per label (not per word) so they can be laid out as one line of text later.
  const labelSpan = boxes.map(() => ({ x0: Infinity, x1: -Infinity }))
  for (const t of texts) {
    if (t.box < 0) continue
    const s = labelSpan[t.box]
    s.x0 = Math.min(s.x0, t.x)
    s.x1 = Math.max(s.x1, t.x + t.w)
  }
  for (const t of texts) {
    const n = Math.max(2, Math.round((textN * tw(t)) / textArea))
    const isPair = t.box >= 0 && t.box < PAIRS
    for (let i = 0; i < n; i++) {
      const x = t.x + rand() * t.w
      const y = t.y + rand() * t.h
      const s = isPair ? labelSpan[t.box] : null
      add(x, y, {
        a: t.a,
        pair: isPair ? t.box : -1,
        label: isPair,
        box: t.box,
        u: s ? (x - s.x0) / (s.x1 - s.x0) : 0,
        v: (y - t.y) / t.h,
      })
    }
  }
  boxes.forEach((b, bi) => {
    const per = (k: number, n: number) => {
      const d = (k / n) * 4
      const side = Math.floor(d)
      const f = d - side
      return [
        [b.x + f * b.s, b.y],
        [b.x + b.s, b.y + f * b.s],
        [b.x + b.s - f * b.s, b.y + b.s],
        [b.x, b.y + b.s - f * b.s],
      ][side]
    }
    const pair = bi < PAIRS ? bi : -1
    for (let i = 0; i < PER_BOX; i++) {
      const [x, y] = per(i, PER_BOX)
      add(x, y, { kind: 1, a: 0.95, box: bi, pair, checked: b.checked })
    }
    const lines = 5
    for (let i = 0; i < HATCH; i++) {
      const l = i % lines
      const u = (Math.floor(i / lines) + 0.5) / Math.ceil(HATCH / lines)
      const c = ((l + 0.5) / lines) * 2 * b.s
      const lo = Math.max(0, c - b.s)
      const hi = Math.min(b.s, c)
      const hx = lo + u * (hi - lo) * 0.9 + (hi - lo) * 0.05
      const inset = 0.15 * b.s
      const [x, y] = per(i + 0.5, HATCH)
      add(x, y, {
        kind: 2,
        a: 0.95,
        box: bi,
        pair,
        checked: b.checked,
        hx: b.x + inset + hx * 0.7,
        hy: b.y + inset + (c - hx) * 0.7,
      })
    }
  })

  // Per-point scatter (the cloud they come from and go back to), jitter and delays.
  for (const p of P) {
    p.sx = p.x * 1.35 + 0.3 + gauss() * 0.24
    p.sy = p.y * 1.1 - 0.05 + gauss() * 0.2
    p.sz = -0.35 + gauss() * 0.5
    p.sa = 0.25 + rand() * 0.35
    p.jx = gauss() * 0.004
    p.jy = gauss() * 0.004
    p.w1 = 0.5 + rand() * 1.3
    p.w2 = rand() * 6.28
    p.dA = (p.y + 0.65) * 520 + rand() * 420 // stream in top to bottom, like a scan
    p.dS = rand() * 220 // slide-out spread within a pair
    p.dD = rand() * 420 + (p.x + 0.5) * 180 // dissolve left to right
  }
  return P
}

// ---------- Timeline (ms). Segments overlap so it never reads as slides.
const T = 14300
const STEP_AT = [0, 2500, 4400, 7400, T]
const OUT = 12250 // the list starts to dissolve
const STILL = 10500 // reduced motion: the read-out, held

// Page poses. A phone photo: turned, keystoned, a little off-centre.
const RAD = Math.PI / 180
const SCAN: Pose = [9 * RAD, 15 * RAD, -13 * RAD, 0.86, 0.03, 0.01, 0]
const FLAT: Pose = [0, 0, 0, 0.98, 0, 0, 0]

/**
 * Where the page goes once it is read and where the list sits, in world units. Wide: the page shrinks
 * to the left of the list. Narrow: it shrinks above it, so the list keeps a readable size.
 */
const layouts = {
  wide: { back: [0, 0, 6 * RAD, 0.36, -0.46, 0.03, -0.08], x0: -0.17, x1: 0.68, y0: -0.37, fit: 1.05 },
  narrow: { back: [0, 0, 4 * RAD, 0.26, 0, -0.5, -0.06], x0: -0.45, x1: 0.45, y0: -0.2, fit: 0.98 },
}
const NARROW = 560 // px of canvas width

const LROW = 0.09
const ICON = 0.0275
const FS = { label: 0.035, state: 0.03125, idx: 0.0287, head: 0.0475 }
const FONT = '"Archivo Variable", -apple-system, sans-serif'
const INK = [26, 27, 24]
const CHECKED = '#16a34a'
const UNCHECKED = '#dc2626'
// Ink to the box's state color in 9 steps, as a box lights up.
const palette = (to: number[]) =>
  Array.from({ length: 9 }, (_, i) => {
    const k = i / 8
    return `rgb(${INK.map((c, j) => Math.round(c + (to[j] - c) * k)).join(',')})`
  })
const PAL_CHECKED = palette([22, 163, 74])
const PAL_UNCHECKED = palette([220, 38, 38])

const mixPose = (A: Pose, B: Pose, k: number) => A.map((v, i) => v + (B[i] - v) * k)
function poseMatrix([rz, rx, ry, s, tx, ty, tz]: Pose) {
  const cz = Math.cos(rz)
  const sz = Math.sin(rz)
  const cx = Math.cos(rx)
  const sx = Math.sin(rx)
  const cy = Math.cos(ry)
  const sy = Math.sin(ry)
  const ap = (x: number, y: number) => {
    x *= s
    y *= s
    const x1 = x * cz - y * sz
    const y1 = x * sz + y * cz
    const y2 = y1 * cx
    const z2 = y1 * sx
    return [x1 * cy + z2 * sy, y2, -x1 * sy + z2 * cy]
  }
  return [...ap(1, 0), ...ap(0, 1), tx, ty, tz]
}
const ease = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const easeOut = (x: number) => 1 - Math.pow(1 - x, 3)
const seg = (t: number, start: number, dur: number, e = ease) => {
  const k = (t - start) / dur
  return k <= 0 ? 0 : k >= 1 ? 1 : e(k)
}
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
function mulberry(a: number) {
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Starts the piece on `canvas`, marking the current step's <li> in `steps`. Returns a stop function. */
export function startReader(canvas: HTMLCanvasElement, steps: HTMLLIElement[]): () => void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return () => {}
  const form = buildForm()
  const { boxes } = form
  const P = buildPoints(form)
  const N = P.length
  const bars = steps.map((li) => li.querySelector('i'))

  const litT = boxes.map((b) => 4500 + b.row * 150 + b.col * 55)
  const liftT = Array.from({ length: PAIRS }, (_, i) => 5450 + i * 90)
  const slideT = Array.from({ length: PAIRS }, (_, i) => 6650 + i * 110)
  const textT = Array.from({ length: PAIRS }, (_, i) => 8150 + i * 90)

  let L = layouts.wide
  const labelW = boxes.slice(0, PAIRS).map((b) => b.label.length * 0.0185)
  const rowY = (i: number) => L.y0 + (i + 0.5) * LROW
  const iconX = () => L.x0 + 0.0825
  const labX = () => L.x0 + 0.147
  const computeTargets = () => {
    for (const p of P) {
      if (p.pair < 0) continue
      const ry = rowY(p.pair)
      if (p.label) {
        p.lx = labX() + p.u * labelW[p.pair]
        p.ly = ry + (p.v - 0.55) * 0.022
        continue
      }
      const b = boxes[p.box]
      const qx = p.kind === 2 && p.checked ? p.hx : p.x
      const qy = p.kind === 2 && p.checked ? p.hy : p.y
      p.lx = iconX() + ((qx - b.x) / b.s) * ICON
      p.ly = ry - ICON / 2 + ((qy - b.y) / b.s) * ICON
    }
  }
  computeTargets()

  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  let w = 0
  let h = 0
  let dpr = 1
  let stepNow = -1

  const draw = (t: number, ts: number) => {
    const align = seg(t, 2400, 1900)
    const recede = seg(t, 5900, 1600)
    const out = seg(t, OUT, 900)
    const m = poseMatrix(mixPose(mixPose(SCAN, FLAT, align), L.back, recede))
    const Y = -3 * RAD + Math.sin(ts * 0.21) * 2.5 * RAD
    const X = 2 * RAD + Math.sin(ts * 0.17 + 1) * 1.2 * RAD
    const cy_ = Math.cos(Y)
    const sy_ = Math.sin(Y)
    const cx_ = Math.cos(X)
    const sx_ = Math.sin(X)
    const S = Math.min(w / L.fit, h / 1.38) * 0.93
    const D = 3.4
    const CX = w / 2
    const CY = h / 2
    const o = [0, 0, 0]
    const project = (x: number, y: number, z: number) => {
      const x1 = x * cy_ - z * sy_
      const z1 = x * sy_ + z * cy_
      const y2 = y * cx_ - z1 * sx_
      const z2 = y * sx_ + z1 * cx_
      const s = D / (D - z2)
      o[0] = CX + x1 * s * S
      o[1] = CY + y2 * s * S
      o[2] = s
      return o
    }
    const onPage = (x: number, y: number, z = 0) =>
      project(x * m[0] + y * m[3] + m[6], x * m[1] + y * m[4] + m[7], x * m[2] + y * m[5] + m[8] + z)

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    // The sheet: greyish like a photo while it is a scan, white once it is aligned.
    const sheetA = seg(t, 600, 1700) * (1 - 0.4 * recede) * (1 - out)
    if (sheetA > 0.01) {
      const q = [
        [-W / 2, -H / 2],
        [W / 2, -H / 2],
        [W / 2, H / 2],
        [-W / 2, H / 2],
      ].map(([x, y]) => onPage(x, y, -0.003).slice())
      const g = Math.round(238 + 17 * align)
      ctx.save()
      ctx.globalAlpha = sheetA
      ctx.shadowColor = `rgba(26,27,24,${0.22 * (1 - 0.6 * recede)})`
      ctx.shadowBlur = 30 * (1 - 0.5 * recede)
      ctx.shadowOffsetY = 14 * (1 - 0.5 * recede)
      ctx.beginPath()
      q.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      ctx.closePath()
      ctx.fillStyle = `rgb(${g - 3},${g - 2},${g - 6})`
      ctx.fill()
      ctx.restore()
    }
    // Crop marks snap in to the corners once the page is upright.
    const snap = seg(t, 3950, 450, easeOut)
    const cropA = snap * (1 - 0.5 * recede) * (1 - out)
    if (cropA > 0.01) {
      ctx.globalAlpha = cropA
      ctx.strokeStyle = '#61655d'
      ctx.lineWidth = 1.25
      ctx.beginPath()
      const gap = 0.03 + 0.08 * (1 - snap)
      const CL = 0.055
      for (const [sx, sy] of [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ]) {
        const cx = (sx * W) / 2 + sx * gap
        const cy = (sy * H) / 2 + sy * gap
        onPage(cx, cy - sy * CL)
        ctx.moveTo(o[0], o[1])
        onPage(cx, cy)
        ctx.lineTo(o[0], o[1])
        onPage(cx - sx * CL, cy)
        ctx.lineTo(o[0], o[1])
      }
      ctx.stroke()
    }

    // Row read-out alpha (text replaces the dots).
    const rowA = textT.map((s) => seg(t, s, 520) * (1 - seg(t, OUT, 420)))

    // Dots.
    const jit = 1 - align
    for (let i = 0; i < N; i++) {
      const p = P[i]
      let px = p.x
      let py = p.y
      let col = 0
      if (p.kind) {
        col = seg(t, litT[p.box], 320)
        if (p.kind === 2 && p.checked) {
          const k = seg(t, litT[p.box] + 140, 450)
          px += (p.hx - px) * k
          py += (p.hy - py) * k
        }
      }
      const lx = px
      const ly = py
      if (jit > 0) {
        px += (p.jx + Math.sin(ts * p.w1 * 3 + p.w2) * 0.0015) * jit
        py += (p.jy + Math.cos(ts * p.w1 * 3 + p.w2) * 0.0015) * jit
      }
      let x = px * m[0] + py * m[3] + m[6]
      let y = px * m[1] + py * m[4] + m[7]
      let z = px * m[2] + py * m[5] + m[8]
      let al = p.a
      const kA = seg(t, p.dA, 1500)
      x = p.sx + (x - p.sx) * kA
      y = p.sy + (y - p.sy) * kA
      z = p.sz + (z - p.sz) * kA
      al = p.sa + (al - p.sa) * kA
      // Dots left on the page shrink with it, so the read page stays a quiet thumbnail.
      let shrink = 1
      if (p.pair >= 0) {
        // Lift toward the viewer, then slide out into the list.
        const kL = seg(t, liftT[p.pair] + p.dS * 0.3, 900)
        x += (lx * 0.98 + 0.16 - x) * kL
        y += (ly * 0.98 - 0.03 - y) * kL
        z += (0.85 - z) * kL
        if (p.label) al += (0.9 - al) * kL
        const kS = seg(t, slideT[p.pair] + p.dS, 1300)
        x += (p.lx - x) * kS
        y += (p.ly - y) * kS
        z -= z * kS
        al *= 1 - rowA[p.pair]
      } else {
        al *= 1 - (p.kind ? 0.68 : 0.76) * recede
        shrink = 1 - 0.45 * recede
      }
      const kD = seg(t, OUT + 100 + p.dD, 1150)
      if (kD > 0) {
        x += (p.sx - x) * kD
        y += (p.sy - y) * kD
        z += (p.sz - z) * kD
        al += (p.sa - al) * kD
        col *= 1 - kD
      }
      const loose = Math.min(1, 1 - kA + kD)
      if (loose > 0) {
        const ph = ts * p.w1 + p.w2
        const a = loose * 0.022
        x += Math.sin(ph) * a
        y += Math.cos(ph * 1.3) * a
        z += Math.sin(ph * 0.7) * a * 3
      }
      if (al < 0.01) continue
      project(x, y, z)
      const size = (col > 0.5 ? 1.55 : 1.35) * o[2] ** 1.6 * shrink
      ctx.globalAlpha = al * clamp(0.55 + (o[2] - 0.8) * 1.5, 0.35, 1)
      ctx.fillStyle = (p.checked ? PAL_CHECKED : PAL_UNCHECKED)[Math.round(col * 8)]
      ctx.fillRect(o[0] - size / 2, o[1] - size / 2, size, size)
    }

    // The read-out: index, mark, label, state, hairline. Drawn as real text once the dots arrive.
    const px = (u: number, s: number) => u * S * s
    const headA = rowA[0]
    ctx.textBaseline = 'middle'
    if (headA > 0.01) {
      ctx.globalAlpha = headA
      project(L.x0 + 0.01, L.y0 - 0.055, 0)
      ctx.fillStyle = '#1a1b18'
      ctx.fontStretch = 'semi-expanded'
      ctx.font = `750 ${px(FS.head, o[2])}px ${FONT}`
      ctx.textAlign = 'left'
      ctx.fillText('Detections', o[0], o[1])
      const hw = ctx.measureText('Detections ').width
      ctx.fontStretch = 'normal'
      ctx.font = `400 ${px(FS.head, o[2])}px ${FONT}`
      ctx.fillStyle = '#4a4e46'
      ctx.fillText(String(PAIRS), o[0] + hw, o[1])
      ctx.strokeStyle = '#b3b8ad'
      ctx.lineWidth = 1
      ctx.beginPath()
      project(L.x0, L.y0, 0)
      ctx.moveTo(o[0], o[1])
      project(L.x1, L.y0, 0)
      ctx.lineTo(o[0], o[1])
      ctx.stroke()
    }
    for (let i = 0; i < PAIRS; i++) {
      const a = rowA[i]
      if (a < 0.01) continue
      const b = boxes[i]
      const ry = rowY(i)
      ctx.globalAlpha = a
      project(L.x0 + 0.03, ry, 0)
      ctx.font = `500 ${px(FS.idx, o[2])}px ${FONT}`
      ctx.fillStyle = '#61655d'
      ctx.textAlign = 'center'
      ctx.fillText(String(i + 1), o[0], o[1])
      project(labX(), ry, 0)
      ctx.font = `400 ${px(FS.label, o[2])}px ${FONT}`
      ctx.fillStyle = '#1a1b18'
      ctx.textAlign = 'left'
      ctx.fillText(b.label, o[0], o[1])
      project(L.x1 - 0.025, ry, 0)
      ctx.font = `400 ${px(FS.state, o[2])}px ${FONT}`
      ctx.fillStyle = '#4a4e46'
      ctx.textAlign = 'right'
      ctx.fillText(b.checked ? 'Checked' : 'Unchecked', o[0], o[1])
      // Mark icon, same geometry as the app's 16-unit icons (rect 2.5..13.5).
      const u = ICON / 11
      const ox = iconX() - 2.5 * u
      const oy = ry - ICON / 2 - 2.5 * u
      const path = (pts: number[][]) =>
        pts.forEach(([x, y2], k) => {
          project(ox + x * u, oy + y2 * u, 0)
          if (k) ctx.lineTo(o[0], o[1])
          else ctx.moveTo(o[0], o[1])
        })
      ctx.strokeStyle = b.checked ? CHECKED : UNCHECKED
      ctx.lineWidth = 1.5
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.beginPath()
      path([
        [2.5, 2.5],
        [13.5, 2.5],
        [13.5, 13.5],
        [2.5, 13.5],
      ])
      ctx.closePath()
      ctx.stroke()
      if (b.checked) {
        ctx.lineWidth = 1.25
        ctx.beginPath()
        path([
          [2.75, 8.25],
          [8.25, 2.75],
        ])
        path([
          [2.75, 12.75],
          [12.75, 2.75],
        ])
        path([
          [7.25, 13.25],
          [13.25, 7.25],
        ])
        ctx.stroke()
      }
      if (i < PAIRS - 1) {
        ctx.strokeStyle = '#d2d6ce'
        ctx.lineWidth = 1
        ctx.beginPath()
        project(L.x0, ry + LROW / 2, 0)
        ctx.moveTo(o[0], o[1])
        project(L.x1, ry + LROW / 2, 0)
        ctx.lineTo(o[0], o[1])
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1

    // Status under the piece.
    let s = 0
    while (s < 3 && t >= STEP_AT[s + 1]) s++
    if (s !== stepNow) {
      steps.forEach((li, k) => li.toggleAttribute('data-now', k === s))
      stepNow = s
    }
    const bar = bars[s]
    if (bar) bar.style.transform = `scaleX(${clamp((t - STEP_AT[s]) / (STEP_AT[s + 1] - STEP_AT[s]), 0, 1).toFixed(3)})`
  }

  // ---------- Clock: advances only while visible, so it never jumps after a pause.
  let clock = 0
  let raf = 0
  let last = 0
  let onScreen = true
  const frame = (now: number) => {
    raf = requestAnimationFrame(frame)
    clock += Math.min(50, now - last)
    last = now
    draw(clock % T, clock / 1000)
  }
  const sync = () => {
    if (reduce) return
    const run = onScreen && !document.hidden
    if (run && !raf) {
      last = performance.now()
      raf = requestAnimationFrame(frame)
    }
    if (!run && raf) {
      cancelAnimationFrame(raf)
      raf = 0
    }
  }
  const resize = () => {
    const r = canvas.getBoundingClientRect()
    dpr = Math.min(devicePixelRatio || 1, 2)
    w = r.width
    h = r.height
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    const next = w < NARROW ? layouts.narrow : layouts.wide
    if (next !== L) {
      L = next
      computeTargets()
    }
    if (reduce) draw(STILL, 0)
  }

  document.addEventListener('visibilitychange', sync)
  const visibility = new IntersectionObserver(([e]) => {
    onScreen = e.isIntersecting
    sync()
  })
  visibility.observe(canvas)
  const size = new ResizeObserver(resize)
  size.observe(canvas)
  resize()

  // Measure the real label widths so the dots gather exactly where the text will sit.
  let stopped = false
  document.fonts.load(`400 14px ${FONT}`).then(() => {
    if (stopped) return
    ctx.font = `400 14px ${FONT}`
    boxes.slice(0, PAIRS).forEach((b, i) => {
      labelW[i] = (ctx.measureText(b.label).width / 14) * FS.label
    })
    computeTargets()
    if (reduce) draw(STILL, 0)
  })
  sync()

  return () => {
    stopped = true
    cancelAnimationFrame(raf)
    raf = 0
    document.removeEventListener('visibilitychange', sync)
    visibility.disconnect()
    size.disconnect()
  }
}

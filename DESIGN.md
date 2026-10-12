---
name: Checkbox Detection
description: Checkbox detection for appraisal forms, each box with its position, state and label, marked like a galley proof.
colors:
  paper: "#eef0ec"
  desk: "#d9ddd5"
  sheet: "#ffffff"
  rule: "#d2d6ce"
  rule-strong: "#b3b8ad"
  ink: "#1a1b18"
  ink-2: "#4a4e46"
  ink-3: "#61655d"
  ink-hover: "#33352f"
  pencil: "#0f67ad"
  pencil-soft: "rgb(15 103 173 / 0.09)"
  query: "#cc4410"
  query-text: "#a5360b"
  query-soft: "#fbeae1"
typography:
  display:
    fontFamily: "Archivo Variable, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "clamp(38px, 4.6vw, 62px)"
    fontWeight: 800
    lineHeight: 0.98
    letterSpacing: "-0.03em"
    fontVariation: "'wdth' 120"
  headline:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "19px"
    fontWeight: 750
    letterSpacing: "-0.01em"
    fontVariation: "'wdth' 116"
  title:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    letterSpacing: "-0.01em"
    fontVariation: "'wdth' 112"
  body-lead:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.45
  body:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum'"
  label:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    fontFeature: "'tnum'"
  scale:
    fontFamily: "Archivo Variable, sans-serif"
    fontSize: "9.5px"
    fontWeight: 400
    lineHeight: 1
rounded:
  mark: "1px"
  sm: "3px"
  pill: "999px"
  circle: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  gutter: "18px"
  lg: "28px"
components:
  button:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    height: "32px"
    padding: "0 12px"
  button-hover:
    backgroundColor: "{colors.paper}"
  button-active:
    backgroundColor: "{colors.desk}"
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.sheet}"
    rounded: "{rounded.sm}"
    height: "40px"
    padding: "0 18px"
  button-primary-hover:
    backgroundColor: "{colors.ink-hover}"
  icon-button:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.sm}"
    size: "28px"
  detection-row:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    height: "36px"
    padding: "5px 10px 5px 4px"
  detection-row-active:
    backgroundColor: "{colors.pencil-soft}"
  note-number-active:
    backgroundColor: "{colors.pencil}"
    textColor: "{colors.sheet}"
    rounded: "{rounded.circle}"
    size: "22px"
  query-ring:
    textColor: "{colors.query-text}"
    rounded: "{rounded.pill}"
    padding: "0 6px"
  panel-note:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.sm}"
    padding: "16px"
  panel-note-error:
    backgroundColor: "{colors.query-soft}"
    textColor: "{colors.query-text}"
---

# Design System: Checkbox Detection

## Overview

**Creative North Star: "The Galley Proof"**

The service reads a form the way a corrector reads a galley proof. Every detected box gets a pencil mark on the page and a numbered note in the margin; anything the corrector doubts gets a dashed vermilion border and a "Low confidence" chip in the margin. The interface is proof paper and ink on a slightly darker desk, with exactly two pencils: blue for marks, vermilion for queries. Everything else is ink.

The document is the hero. The viewer is a desk holding a white sheet framed by printer's crop marks and pixel rulers along its edges; the right-hand column is the proof margin, a ruled list of numbered notes ending in a measurement slip. Hovering a note circles its box on the page, projects its extent onto both rulers, and draws a leader line from box to note. Density is that of a working proof: 14px body, 36px rows, hairline rules, no cards inside cards.

Archivo is set wide and heavy for headings (width axis 108 to 120%) and plain for reading. Numbers are tabular everywhere, because coordinates and counts are the evidence. The world refuses the gray SaaS dashboard with colored status chips: state is carried by mark form (hatched, open, dashed), not by colored badges.

**Key Characteristics:**
- Neutral green-gray proof paper, ink text, one blue pencil, one vermilion pencil.
- Checked = hatched fill, unchecked = open square, low confidence = dashed vermilion border plus a "Low confidence" chip in the list. Readable without hue.
- Crop marks, pixel rulers, reading rule and leader line as the world's native furniture.
- Wide, heavy Archivo headings; tabular numerals throughout.
- Marks draw in left to right like pencil strokes; motion is short and eased out.

## Colors

A near-monochrome proof: green-tinted neutrals and ink carry everything, and two pencil colors carry meaning.

### Primary
- **Corrector's Blue Pencil** (pencil): every detection mark on the page, the legend swatches, row icons, the active note's filled number, the circle around the active box, ruler projections, the leader line, focus outlines, selection, caret and accent-color. Its soft tint (pencil-soft) is the hover/active row wash. The hatch for checked boxes is a 135deg repeating gradient of the same blue at 0.62 and 0.16 alpha (1.5px / 4px).

### Secondary
- **Vermilion Query Pencil** (query): the dashed border of low-confidence boxes and the ring around the "Low confidence" chip. Its darker text variant (query-text) sets the chip, the "to check by eye" tally and error titles for 4.5:1; query-soft is the error note's ground.

### Neutral
- **Proof Paper** (paper): page ground, viewer bar, the margin panel, secondary button hover.
- **Desk** (desk): the viewer behind the sheet, active/pressed states, icon-button hover.
- **Sheet** (sheet): the document page, the drop target, notes, secondary buttons. Pure white so the scanned form reads as printed.
- **Hairline** (rule) and **Strong Rule** (rule-strong): row separators, and structural dividers, ruler ticks, crop marks on the loaded page, scrollbars, disabled icons.
- **Ink** (ink), **Ink 2** (ink-2), **Ink 3** (ink-3): text in three steps (primary, secondary copy and counts, metadata and hints). Ink also fills the primary button and draws its 1px button borders.

### Named Rules
**The Two Pencils Rule.** Blue marks detections and focus; vermilion marks queries and errors. No third hue enters the system, and neither pencil is used decoratively.

**The Form Not Hue Rule.** Checked vs unchecked is told by hatch vs empty, low confidence by dashed stroke plus the "Low confidence" chip. A colorblind reader loses nothing.

## Typography

**Display Font:** Archivo Variable (with -apple-system, Segoe UI, Roboto, sans-serif)
**Body Font:** the same family, at normal width.

**Character:** One grotesque used two ways: stretched wide and heavy for anything that names a thing (headline, file name, primary button), plain and tabular for everything that reads or measures.

### Hierarchy
- **Display** (800, clamp(38px, 4.6vw, 62px), 0.98, width 120%, -0.03em, max 13ch, balanced): the empty state's single headline.
- **Headline** (750, 19px, width 116%): the margin's "Detections" title; the count beside it uses the same size at 400 in ink-2.
- **Title** (700, 18px, width 112%): the drop target's instruction. File names and form field names use 650 to 700 at width 108%.
- **Body** (400, 14px, 1.45, tabular): rows, buttons (600), notes. Lead copy steps up to 16px, max 40ch.
- **Label** (400, 12.5px): legend, tally, row state, slip, hints, captions (12px).
- **Scale** (9.5px, 1): ruler numbers, set along the ruler (vertical-rl on the y-axis).

### Named Rules
**The Wide For Names Rule.** Width above 100% is reserved for titles, names and the primary action; reading text and numbers stay at normal width.

**The Tabular Evidence Rule.** Numerals are tabular everywhere (set on :root) so coordinates, counts and indices align.

## Layout

Desktop is a two-column workspace filling the viewport (100dvh): the viewer (flexible) and the proof margin (340 to 420px), split by a strong rule. Each column scrolls independently under a fixed 52px bar. The sheet sits centered on the desk, max 980px plus a 26px ruler band on top and left, in crop marks.

The empty state is a two-column composition (1fr / 1.15fr, max 1240px, gap clamp(32px, 6vw, 96px)): headline, lead and drop target on the left, the looping process piece (Scan · Align · Extract · Read out) on the right.

Spacing works on a loose 4px base with 18px as the recurring horizontal gutter (bar, panel head, tally, slip). Rows are 36px min with 9px column gaps.

Breakpoints: at 860px and below everything stacks, the page scrolls as a whole, the legend wraps under the file name, the slip becomes sticky at the bottom, and the leader line is removed. At 560px and below rulers shrink to 20px, page marks thin to 1px with a flat 0.3 blue fill for checked (the hatch is too fine at that scale). Low confidence is the dashed border alone at every size.

## Elevation & Depth

Flat, with one exception: paper on a desk. The sheet, the drop target and the page drawn in the process piece carry the sheet shadow; nothing else lifts. Depth otherwise comes from tonal steps (desk behind sheet, paper around both) and hairline rules.

### Shadow Vocabulary
- **Sheet** (`box-shadow: 0 1px 1px rgb(26 27 24 / 0.08), 0 10px 28px -10px rgb(26 27 24 / 0.28)`): any surface that is a piece of paper: the document page, the blank drop sheet, the page in the process piece.

### Named Rules
**The Only Paper Lifts Rule.** A shadow means "this is a sheet". Panels, buttons, rows and notes stay flat.

## Shapes

Square and printed. Controls and notes use a barely-there 3px corner; page marks 1px. Circles are reserved for the corrector's gestures: the ring around the active box (50%, 1.5px pencil, inset -8px), the filled active note number, and the pill around "Low confidence". Crop marks are 16px L-shaped 1.5px strokes set 10px outside the sheet's corners; on drag they turn blue and close in 6px. Dashed lines mean "provisional or detached": uncertain boxes, the slip's top rule, and the leader line when its box is scrolled out of view.

## Components

### Buttons
- **Shape:** gently squared (3px), 1px ink border.
- **Secondary:** white sheet, ink text 600, 32px tall, 12px padding; hover paper, active desk. 150ms color transitions.
- **Primary:** ink fill, white text, 40px tall, 18px padding, width 108%; hover ink-hover. Used once per view (Browse files).
- **Icon button:** 28px transparent square in ink-2; hover desk; disabled rule-strong. Used for the pager chevrons.
- **Focus:** 2px blue pencil outline, 2px offset, everywhere.

### Detection Rows (the proof margin)
- Grid of index, mark icon, label, optional "Low confidence" chip, state; 36px min, hairline separator.
- Hover and active wash in pencil-soft; the active row's number becomes a filled blue circle, matching the ring on its box.
- Missing labels read "Label not detected" in ink-3 italic.
- Loading shows nine pulsing skeleton bars in rule color.

### Low Confidence Chip
- "Low confidence", 11px 700 query-text, one line, inside a 1.5px vermilion pill. Pinned to its own column so states line up. It appears only in the list; on the page low confidence is the dashed border alone. Never use the "Qy" abbreviation.

### Notes (empty, error)
- White sheet, 1px strong rule, 3px corner, 16px padding, 650 ink title. Error variant: vermilion border, query-soft ground, query-text title, with a secondary button to retry.

### Drop Target
- A blank white sheet with the sheet shadow inside four crop marks: title, format hint in ink-3, primary button. Hover darkens the crop marks to ink; dragging turns them and the title blue and closes them in. Over a loaded document, a full-window overlay (paper at 0.9 with a 1.5px blue frame) takes the drop.

### Viewer Bar and Legend
- 52px paper bar: file name (650, wide) and size, a legend whose swatches repeat the exact box styles, the pager, and the upload button. The legend lives in the bar, never over the document.

### Signature: Marks, Rulers, Leader and Slip
- **Marks:** 2px blue squares over each detection, hatched when checked; dashed vermilion when uncertain. They draw in with a left-to-right clip over 420ms. While one is active the others fade to 0.3.
- **Rulers:** pixel scales in the bbox coordinate space along the top and left of the sheet; the active box's extent is projected as a translucent blue band with a 3px pencil edge, sliding at 200ms.
- **Leader line:** an orthogonal 1.5px blue path with a translucent white casing, running up from the box, along the pane divider and into the row, with 3px dots at both ends. Desktop only.
- **Reading rule:** while detecting, a blue rule with a soft gradient sweeps down the sheet (2.4s loop).
- **Slip:** the margin's foot, under a dashed rule: "No. N" in bold blue, then position, size and confidence in label size.

### Motion
One ease-out curve (cubic-bezier(0.22, 1, 0.36, 1)) for everything: 150ms for color, 200 to 260ms for rulers, crops and the focus circle, 360 to 420ms for marks and margin notes drawing in. Under prefers-reduced-motion all draws, slides and pulses stop and the reading rule becomes a static 6% blue wash.

The empty state's piece (Reader.tsx + story.ts) tells the service's process in ~5k ink dots on a 14.3s loop, with no input: Scan (a photographed form streams in tilted and keystoned), Align (it straightens; crop marks snap to its corners), Extract (boxes light up pencil blue in reading order, checked ones hatch, nine box+label pairs lift toward the viewer) and Read out (the pairs settle into the workspace's Detections row grammar as real Archivo text; the read page shrinks to a quiet thumbnail). Every dot is a pure function of the clock, so nothing teleports; the clock runs only while the canvas is on screen and the tab visible. Under 560px of canvas width the thumbnail sits above the list instead of beside it. The step names under it (Scan · Align · Extract · Read out) are status with a filling hairline, never controls. Reduced motion shows the Read out frame still. Align is illustrative: the service applies EXIF rotation but does not deskew. On the first upload the piece morphs into the document's sheet through a same-document view transition (view-transition-name: sheet, 650ms cubic-bezier(0.16, 1, 0.3, 1); the old content fades in 220ms so its stretch never shows). The workspace keeps one .sheet element from the blank placeholder to the loaded page: replacing it mid-transition cancels the morph. No morph where view transitions are unsupported or motion is reduced.

## Do's and Don'ts

### Do:
- **Do** tie every result to its place on the page: a row, its box, its ruler projection and its slip move together.
- **Do** encode state with mark form (hatch, open, dashed plus "Low confidence") and keep a legend that repeats the exact styles.
- **Do** use the blue pencil for detections and focus only, and vermilion only for queries and errors.
- **Do** keep numerals tabular and headings wide (108 to 120%) and heavy.
- **Do** keep surfaces flat; only paper gets the sheet shadow.
- **Do** respect prefers-reduced-motion with static equivalents, not just shorter animations.

### Don't:
- **Don't** introduce colored status chips, badges or a third accent hue; that is the gray SaaS dashboard this world refuses.
- **Don't** place UI over the document where it could hide a box.
- **Don't** shadow panels, rows or buttons.
- **Don't** set reading text or numbers in the wide widths.

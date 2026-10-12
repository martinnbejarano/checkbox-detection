import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

function Icon({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

// The two proof marks repeat the boxes drawn on the page: a pencil square, hatched when checked.
export const CheckedIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2.5" y="2.5" width="11" height="11" rx="1" />
    <path d="M2.75 8.25l5.5-5.5M2.75 12.75l10-10M7.25 13.25l6-6" strokeWidth="1.25" />
  </Icon>
)

export const UncheckedIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2.5" y="2.5" width="11" height="11" rx="1" />
  </Icon>
)

export const UploadIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 10.5V2.75M4.75 6L8 2.75 11.25 6M2.75 10.5v1.75c0 .55.45 1 1 1h8.5c.55 0 1-.45 1-1V10.5" />
  </Icon>
)

export const ChevronIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6 3.5L10.5 8 6 12.5" />
  </Icon>
)

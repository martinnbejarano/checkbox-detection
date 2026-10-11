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

export const CheckedIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="2" width="12" height="12" rx="3" fill="currentColor" stroke="none" />
    <path d="M5 8.2l2 2 4-4.4" stroke="#fff" strokeWidth="1.75" />
  </Icon>
)

export const UncheckedIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2.75" y="2.75" width="10.5" height="10.5" rx="2.5" />
  </Icon>
)

export const WarningIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 2.5l6 10.5H2z" />
    <path d="M8 6.75v2.5M8 11.3v.01" />
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

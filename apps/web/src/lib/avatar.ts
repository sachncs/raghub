export function gradientFromString(input: string): readonly [string, string] {
  let hash = 0
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i)
    hash |= 0
  }
  const hueA = Math.abs(hash) % 360
  const hueB = (hueA + 40) % 360
  return [`oklch(0.62 0.18 ${hueA})`, `oklch(0.7 0.16 ${hueB})`]
}

export function initialsFromString(input: string): string {
  if (!input) return "?"
  const parts = input.trim().split(/[\s.@_-]+/).filter(Boolean)
  if (parts.length === 0) return input.slice(0, 1).toUpperCase()
  if (parts.length === 1) {
    const single = parts[0] ?? ""
    return single.slice(0, 2).toUpperCase()
  }
  const first = parts[0]?.[0] ?? ""
  const last = parts[parts.length - 1]?.[0] ?? ""
  return `${first}${last}`.toUpperCase()
}
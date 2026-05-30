// Typed client for the zen-jobdash control-plane API. Mirrors the Rust DTOs in
// crates/zen-jobdash/src/{views,fleet}.rs and the control responses in bin/server.rs.

export interface KindProgress {
  kind: string
  total: number
  done: number
  failed: number
  poison: number
  in_flight: number
}

export interface FailureCell {
  error_class: string
  codec: string
  kind: string
  count: number
}

export interface TierCost {
  tier: string
  cost_per_1000_jobs: number | null
}

export interface CostView {
  total_spent_usd: number
  burn_usd_per_hr: number
  jobs_done: number
  cost_per_1000_jobs: number | null
  per_tier: TierCost[]
}

export interface TierStorage {
  tier: string
  blobs: number
  bytes: number
}

export interface RunSummary {
  total: number
  done: number
  remaining: number
  poison: number
  fleet_jobs_per_min: number
  eta_secs: number | null
  spent_usd: number
  burn_usd_per_hr: number
  projected_total_usd: number | null
}

export interface CatalogRow {
  key: string
  codec: string
  kind: string
  metric: string
  config: string
  images: number
  q_min: number
  q_max: number
  total: number
  done: number
}

export interface ResultRow {
  kind: string
  codec: string
  image_path: string
  q: number
  output_sha: string
  worker: string
}

export interface PeekResult {
  sha?: string
  size?: number
  text?: string
  error?: string
}

export interface WorkerStat {
  worker: string
  provider: string
  tier: string
  rate_usd_per_hr: number
  uptime_secs: number
  jobs_done: number
  jobs_per_min: number
  spent_usd: number
}

export interface FleetBox {
  id: number
  name: string
  status: string
  server_type: string
  datacenter: string
  ipv4: string | null
  group: string | null
}

export interface FleetView {
  actuation: boolean
  label: string
  boxes: FleetBox[]
  /** Names of running boxes with no matching worker heartbeat (idle reap targets, goal F). */
  idle?: string[]
  note?: string
  error?: string
}

export interface KillResult {
  selector: string
  killed: FleetBox[]
  errors: string[]
  note: string | null
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url, { credentials: "same-origin" })
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`)
  return r.json() as Promise<T>
}

async function postControl(body: unknown): Promise<unknown> {
  const r = await fetch("/api/control", {
    method: "POST",
    headers: { "content-type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`/api/control: ${r.status} ${r.statusText}`)
  return r.json()
}

export const api = {
  progress: () => getJSON<KindProgress[]>("/api/progress"),
  summary: () => getJSON<RunSummary>("/api/summary"),
  failures: () => getJSON<FailureCell[]>("/api/failures"),
  cost: () => getJSON<CostView>("/api/cost"),
  storage: () => getJSON<TierStorage[]>("/api/storage"),
  workers: () => getJSON<WorkerStat[]>("/api/workers"),
  catalog: () => getJSON<CatalogRow[]>("/api/catalog"),
  results: () => getJSON<ResultRow[]>("/api/results"),
  peek: (sha: string) => getJSON<PeekResult>(`/api/peek/${sha}`),
  fleet: () => getJSON<FleetView>("/api/fleet"),

  gcDryRun: () => postControl({ action: "gc_dry_run" }),
  stopSpend: (cap_usd: number) => postControl({ action: "stop_spend", cap_usd }),
  killFleet: () => postControl({ action: "kill_fleet" }) as Promise<{ actuated: boolean; result?: KillResult; note?: string; selector?: string }>,
  killTier: (tier: string) => postControl({ action: "kill_tier", tier }),
  killRun: (run: string) => postControl({ action: "kill_run", run }),
  pause: () => postControl({ action: "pause", run: "global" }),
  drain: () => postControl({ action: "drain", run: "global" }),
  resume: () => postControl({ action: "resume", run: "global" }),
  reapIdle: () => postControl({ action: "reap_idle" }),
}

export function fmtUsd(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—"
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  const units = ["KB", "MB", "GB", "TB", "PB"]
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}

export function fmtInt(n: number): string {
  return n.toLocaleString()
}

export function fmtDuration(secs: number): string {
  if (secs < 60) return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
  return `${Math.floor(secs / 86400)}d ${Math.floor((secs % 86400) / 3600)}h`
}

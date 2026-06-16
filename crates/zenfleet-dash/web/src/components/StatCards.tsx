import { Activity, CircleDollarSign, Flame, Server } from "lucide-react"

import { Card, CardContent } from "@/components/ui/card"
import { fmtDuration, fmtInt, fmtUsd, type CostView, type FleetView, type RunSummary } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Props {
  cost?: CostView
  fleet?: FleetView
  summary?: RunSummary
  speculative?: number
}

function Stat({
  icon,
  label,
  value,
  sub,
  tone,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  tone?: "burn" | "normal"
}) {
  return (
    <Card className="gap-2 py-4">
      <CardContent className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-muted-foreground text-xs font-medium tracking-wide uppercase">{label}</div>
          <div className={cn("mt-1 text-2xl font-semibold tabular-nums", tone === "burn" && Number(value.replace(/[^0-9.]/g, "")) > 0 && "text-amber-400")}>
            {value}
          </div>
          {sub && <div className="text-muted-foreground mt-0.5 text-xs">{sub}</div>}
        </div>
        <div className="text-muted-foreground/70">{icon}</div>
      </CardContent>
    </Card>
  )
}

export function StatCards({ cost, fleet, summary, speculative }: Props) {
  const liveBoxes = fleet?.boxes.length ?? 0
  const running = fleet?.boxes.filter((b) => b.status === "running").length ?? 0
  const eta = summary?.eta_secs != null ? `~${fmtDuration(summary.eta_secs)} ETA` : "idle (no ETA)"
  const specNote = speculative ? ` · ${speculative} speculative` : ""
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <Stat
        icon={<CircleDollarSign className="size-5" />}
        label="Total spent"
        value={fmtUsd(cost?.total_spent_usd)}
        sub={
          summary?.projected_total_usd != null
            ? `proj. ${fmtUsd(summary.projected_total_usd)} total`
            : cost
              ? `${fmtUsd(cost.cost_per_1000_jobs)} / 1k jobs`
              : undefined
        }
      />
      <Stat
        icon={<Flame className="size-5" />}
        label="Burn rate"
        value={cost ? `${fmtUsd(cost.burn_usd_per_hr)}/hr` : "—"}
        tone="burn"
        sub={cost && cost.burn_usd_per_hr > 0 ? eta : "no paid burn"}
      />
      <Stat
        icon={<Activity className="size-5" />}
        label="Jobs done"
        value={summary ? fmtInt(summary.done) : cost ? fmtInt(cost.jobs_done) : "—"}
        sub={
          summary
            ? `${fmtInt(summary.remaining)} remaining of ${fmtInt(summary.total)}` +
              (summary.fleet_jobs_per_min > 0 ? ` · ${summary.fleet_jobs_per_min.toFixed(1)}/min` : "") +
              specNote
            : undefined
        }
      />
      <Stat
        icon={<Server className="size-5" />}
        label="Live fleet"
        value={fleet ? String(liveBoxes) : "—"}
        sub={fleet ? `${running} running · label "${fleet.label}"` : undefined}
      />
    </div>
  )
}

import { Activity, CircleDollarSign, Flame, Server } from "lucide-react"

import { Card, CardContent } from "@/components/ui/card"
import { fmtInt, fmtUsd, type CostView, type FleetView } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Props {
  cost?: CostView
  fleet?: FleetView
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

export function StatCards({ cost, fleet }: Props) {
  const liveBoxes = fleet?.boxes.length ?? 0
  const running = fleet?.boxes.filter((b) => b.status === "running").length ?? 0
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <Stat
        icon={<CircleDollarSign className="size-5" />}
        label="Total spent"
        value={fmtUsd(cost?.total_spent_usd)}
        sub={cost ? `${fmtUsd(cost.cost_per_1000_jobs)} / 1k jobs` : undefined}
      />
      <Stat
        icon={<Flame className="size-5" />}
        label="Burn rate"
        value={cost ? `${fmtUsd(cost.burn_usd_per_hr)}/hr` : "—"}
        tone="burn"
        sub={cost && cost.burn_usd_per_hr > 0 ? "paid boxes active" : "no paid burn"}
      />
      <Stat
        icon={<Activity className="size-5" />}
        label="Jobs done"
        value={cost ? fmtInt(cost.jobs_done) : "—"}
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

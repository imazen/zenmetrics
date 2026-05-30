import { useState } from "react"
import { HardDriveDownload, Loader2, ShieldHalf } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { KillFleetButton } from "@/components/KillFleetButton"
import { api, fmtBytes, type FleetView } from "@/lib/api"

interface GcDryRun {
  kept: number
  evict_cheap: number
  evict_under_pressure: number
  refuse_surface: number
  freed_cheap_bytes: number
  freed_under_pressure_bytes: number
  refused_bytes: number
}
interface StopSpendResp {
  decision: { over_budget: boolean; tear_down: string[]; keep_free: string[] }
  teardown: { actuated: boolean; note?: string; result?: { killed: { name: string }[]; errors: string[] } }
}

export function ControlPane({ fleet, onChange }: { fleet?: FleetView; onChange?: () => void }) {
  const [gc, setGc] = useState<GcDryRun | null>(null)
  const [gcBusy, setGcBusy] = useState(false)
  const [cap, setCap] = useState("10")
  const [spend, setSpend] = useState<StopSpendResp | null>(null)
  const [spendBusy, setSpendBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function runGc() {
    setGcBusy(true)
    setErr(null)
    try {
      setGc((await api.gcDryRun()) as GcDryRun)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setGcBusy(false)
    }
  }
  async function runStopSpend() {
    setSpendBusy(true)
    setErr(null)
    try {
      setSpend((await api.stopSpend(parseFloat(cap) || 0)) as StopSpendResp)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setSpendBusy(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {/* GC dry-run */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardDriveDownload className="size-4" /> Garbage collection
          </CardTitle>
          <CardDescription>Preview what GC would free before deleting anything (goal C/G).</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button variant="outline" onClick={runGc} disabled={gcBusy} className="w-full">
            {gcBusy && <Loader2 className="animate-spin" />} Run dry-run
          </Button>
          {gc && (
            <dl className="space-y-1 text-sm">
              <Row k="Kept" v={`${gc.kept} blobs`} />
              <Row k="Free now (cheap)" v={`${gc.evict_cheap} · ${fmtBytes(gc.freed_cheap_bytes)}`} />
              <Row k="Under pressure" v={`${gc.evict_under_pressure} · ${fmtBytes(gc.freed_under_pressure_bytes)}`} />
              <Row k="Refused (surfaced)" v={`${gc.refuse_surface} · ${fmtBytes(gc.refused_bytes)}`} tone="warn" />
            </dl>
          )}
        </CardContent>
      </Card>

      {/* Stop spend */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldHalf className="size-4" /> Stop spend
          </CardTitle>
          <CardDescription>Tear down paid boxes if cumulative spend exceeds a cap (goal F).</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="cap">Budget cap (USD)</Label>
            <Input id="cap" inputMode="decimal" value={cap} onChange={(e) => setCap(e.target.value)} />
          </div>
          <Button variant="outline" onClick={runStopSpend} disabled={spendBusy} className="w-full">
            {spendBusy && <Loader2 className="animate-spin" />} Evaluate
          </Button>
          {spend && (
            <dl className="space-y-1 text-sm">
              <Row
                k="Over budget"
                v={spend.decision.over_budget ? "YES" : "no"}
                tone={spend.decision.over_budget ? "warn" : undefined}
              />
              <Row k="Tear down (paid)" v={spend.decision.tear_down.length ? spend.decision.tear_down.join(", ") : "—"} />
              <Row k="Keep (free tier)" v={spend.decision.keep_free.length ? spend.decision.keep_free.join(", ") : "—"} />
              {spend.teardown.actuated && (
                <Row k="Actuated" v={`killed ${spend.teardown.result?.killed.length ?? 0} box(es)`} tone="warn" />
              )}
              {spend.teardown.note && <Row k="Note" v={spend.teardown.note} />}
            </dl>
          )}
        </CardContent>
      </Card>

      {/* Kill fleet */}
      <Card>
        <CardHeader>
          <CardTitle>Emergency stop</CardTitle>
          <CardDescription>Delete every fleet box now. Requires confirmation.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-muted-foreground text-sm">
            {fleet ? `${fleet.boxes.length} live box(es) on label "${fleet.label}".` : "Loading fleet…"}
          </p>
          <KillFleetButton
            boxCount={fleet?.boxes.length ?? 0}
            canActuate={fleet?.actuation ?? false}
            label={fleet?.label ?? "group"}
            onDone={onChange}
          />
          {!fleet?.actuation && (
            <p className="text-amber-400 text-xs">No Hetzner token — kill records intent only.</p>
          )}
        </CardContent>
      </Card>

      {err && <p className="text-destructive lg:col-span-3 text-sm">{err}</p>}
    </div>
  )
}

function Row({ k, v, tone }: { k: string; v: string; tone?: "warn" }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={tone === "warn" ? "text-amber-400 font-medium" : "font-medium"}>{v}</dd>
    </div>
  )
}

import { Cpu } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtDuration, fmtInt, fmtUsd, type WorkerStat } from "@/lib/api"

export function WorkersPane({ workers }: { workers?: WorkerStat[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="size-4" /> Workers
        </CardTitle>
        <CardDescription>
          Live per-worker heartbeat: provider, tier, rate, uptime, and throughput (jobs/min).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!workers?.length ? (
          <p className="text-muted-foreground text-sm">No worker heartbeats reported.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Worker</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead className="text-right">$/hr</TableHead>
                <TableHead className="text-right">Uptime</TableHead>
                <TableHead className="text-right">Jobs</TableHead>
                <TableHead className="text-right">Jobs/min</TableHead>
                <TableHead className="text-right">Spent</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {workers.map((w) => (
                <TableRow key={w.worker}>
                  <TableCell className="font-mono text-xs">{w.worker}</TableCell>
                  <TableCell className="text-xs">{w.provider}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{w.tier}</Badge>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {w.rate_usd_per_hr > 0 ? (
                      fmtUsd(w.rate_usd_per_hr)
                    ) : (
                      <Badge variant="success">free</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{fmtDuration(w.uptime_secs)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtInt(w.jobs_done)}</TableCell>
                  <TableCell className="text-right tabular-nums">{w.jobs_per_min.toFixed(1)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtUsd(w.spent_usd)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

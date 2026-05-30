import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtBytes, fmtInt, type CostView, type TierStorage } from "@/lib/api"

const TIER_HINT: Record<string, string> = {
  CheapRegenerable: "freely evictable",
  ExpensiveRegenerable: "evict under pressure",
  NotRegenerable: "never auto-evict",
}

export function StoragePane({ rows, cost }: { rows?: TierStorage[]; cost?: CostView }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Storage per tier</CardTitle>
          <CardDescription>Blob bytes grouped by regenerability (drives GC policy).</CardDescription>
        </CardHeader>
        <CardContent>
          {!rows?.length ? (
            <p className="text-muted-foreground text-sm">Blob index empty.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Tier</TableHead>
                  <TableHead className="text-right">Blobs</TableHead>
                  <TableHead className="text-right">Bytes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.tier}>
                    <TableCell>
                      <div className="font-medium">{r.tier}</div>
                      <div className="text-muted-foreground text-xs">{TIER_HINT[r.tier] ?? ""}</div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{fmtInt(r.blobs)}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmtBytes(r.bytes)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cost per 1000 jobs by tier</CardTitle>
          <CardDescription>The measured cheapest-tier number (lower is better).</CardDescription>
        </CardHeader>
        <CardContent>
          {!cost?.per_tier.length ? (
            <p className="text-muted-foreground text-sm">No worker cost data yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Tier</TableHead>
                  <TableHead className="text-right">$ / 1k jobs</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cost.per_tier.map((t) => (
                  <TableRow key={t.tier}>
                    <TableCell className="font-medium">{t.tier}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {t.cost_per_1000_jobs === null ? "—" : `$${t.cost_per_1000_jobs.toFixed(4)}`}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtInt, type KindProgress } from "@/lib/api"

export function ProgressPane({ rows }: { rows?: KindProgress[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Progress per job kind</CardTitle>
        <CardDescription>Status breakdown across every job kind in the ledger.</CardDescription>
      </CardHeader>
      <CardContent>
        {!rows?.length ? (
          <p className="text-muted-foreground text-sm">No jobs in the ledger yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Kind</TableHead>
                <TableHead className="w-[34%]">Done</TableHead>
                <TableHead className="text-right">Total</TableHead>
                <TableHead className="text-right">In flight</TableHead>
                <TableHead className="text-right">Failed</TableHead>
                <TableHead className="text-right">Poison</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => {
                const pct = r.total ? Math.round((r.done / r.total) * 100) : 0
                return (
                  <TableRow key={r.kind}>
                    <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="bg-muted h-2 w-full max-w-40 overflow-hidden rounded-full">
                          <div className="h-full rounded-full bg-emerald-500/80" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-muted-foreground w-9 text-right text-xs tabular-nums">{pct}%</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{fmtInt(r.total)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {r.in_flight ? <Badge variant="secondary">{fmtInt(r.in_flight)}</Badge> : "0"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {r.failed ? <Badge variant="warning">{fmtInt(r.failed)}</Badge> : "0"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {r.poison ? <Badge variant="destructive">{fmtInt(r.poison)}</Badge> : "0"}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtInt, type FailureCell } from "@/lib/api"

export function FailuresPane({ rows }: { rows?: FailureCell[] }) {
  const total = rows?.reduce((a, r) => a + r.count, 0) ?? 0
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Failures
          {total > 0 && <Badge variant="destructive">{fmtInt(total)}</Badge>}
        </CardTitle>
        <CardDescription>Exactly what failed — by error class, codec, and kind (goal B).</CardDescription>
      </CardHeader>
      <CardContent>
        {!rows?.length ? (
          <p className="text-muted-foreground text-sm">No failures recorded. 🎉</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Error class</TableHead>
                <TableHead>Codec</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead className="text-right">Count</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={`${r.error_class}-${r.codec}-${r.kind}-${i}`}>
                  <TableCell>
                    <Badge variant="warning">{r.error_class}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{r.codec || "—"}</TableCell>
                  <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                  <TableCell className="text-right font-medium tabular-nums">{fmtInt(r.count)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

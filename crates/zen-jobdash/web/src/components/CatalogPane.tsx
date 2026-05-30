import { useState } from "react"
import { Library, Search } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtInt, type CatalogRow } from "@/lib/api"

export function CatalogPane({ rows }: { rows?: CatalogRow[] }) {
  const [q, setQ] = useState("")
  const needle = q.trim().toLowerCase()
  const filtered = (rows ?? []).filter((r) =>
    !needle ||
    [r.codec, r.kind, r.metric, r.config, r.key].some((f) => f.toLowerCase().includes(needle))
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Library className="size-4" /> Result catalog
        </CardTitle>
        <CardDescription>
          Every result set by semantic identity (codec · kind · config · q-range), derived from the
          ledger. Consult coverage here before enqueuing so no work is duplicated (goal I).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative max-w-sm">
          <Search className="text-muted-foreground absolute top-2.5 left-2.5 size-4" />
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="find by description (codec, metric, config…)"
            className="pl-8"
          />
        </div>
        {!rows?.length ? (
          <p className="text-muted-foreground text-sm">No result sets in the ledger yet.</p>
        ) : !filtered.length ? (
          <p className="text-muted-foreground text-sm">No result sets match “{q}”.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Codec</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Config</TableHead>
                <TableHead className="text-right">Images</TableHead>
                <TableHead className="text-right">q-range</TableHead>
                <TableHead className="w-[22%]">Coverage</TableHead>
                <TableHead>Key</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((r) => {
                const pct = r.total ? Math.round((r.done / r.total) * 100) : 0
                return (
                  <TableRow key={r.key + r.kind}>
                    <TableCell className="font-mono text-xs">{r.codec || "—"}</TableCell>
                    <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                    <TableCell className="text-muted-foreground max-w-40 truncate font-mono text-xs" title={r.config}>
                      {r.config || "{}"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{fmtInt(r.images)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {r.q_min === r.q_max ? r.q_min : `${r.q_min}–${r.q_max}`}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="bg-muted h-2 w-full max-w-28 overflow-hidden rounded-full">
                          <div className="h-full rounded-full bg-emerald-500/80" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-muted-foreground text-xs tabular-nums">
                          {fmtInt(r.done)}/{fmtInt(r.total)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono" title={r.key}>
                        {r.key.slice(0, 10)}
                      </Badge>
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

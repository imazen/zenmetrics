import { useState } from "react"
import { Eye, FlaskConical, Loader2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, fmtInt, type PeekResult, type ResultRow } from "@/lib/api"

export function ResultsPane({ rows }: { rows?: ResultRow[] }) {
  const [open, setOpen] = useState(false)
  const [peek, setPeek] = useState<PeekResult | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  async function view(sha: string) {
    setBusy(sha)
    setPeek(null)
    setOpen(true)
    try {
      setPeek(await api.peek(sha))
    } catch (e) {
      setPeek({ error: e instanceof Error ? e.message : String(e) })
    } finally {
      setBusy(null)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FlaskConical className="size-4" /> Results
        </CardTitle>
        <CardDescription>
          Completed jobs with an output blob — peek the score in-browser by its content hash (goal B).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!rows?.length ? (
          <p className="text-muted-foreground text-sm">No completed results yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Kind</TableHead>
                <TableHead>Codec</TableHead>
                <TableHead>Image</TableHead>
                <TableHead className="text-right">q</TableHead>
                <TableHead>Output</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.output_sha + r.kind + r.image_path}>
                  <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                  <TableCell className="font-mono text-xs">{r.codec || "—"}</TableCell>
                  <TableCell className="text-muted-foreground max-w-40 truncate text-xs" title={r.image_path}>
                    {r.image_path}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{r.q}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="font-mono" title={r.output_sha}>
                      {r.output_sha.slice(0, 10)}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="sm" onClick={() => view(r.output_sha)} disabled={busy === r.output_sha}>
                      {busy === r.output_sha ? <Loader2 className="animate-spin" /> : <Eye />} peek
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Result blob</DialogTitle>
            <DialogDescription className="font-mono text-xs">
              {peek?.sha ? `${peek.sha.slice(0, 16)}… · ${fmtInt(peek.size ?? 0)} bytes` : "fetching…"}
            </DialogDescription>
          </DialogHeader>
          {!peek ? (
            <div className="text-muted-foreground flex items-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin" /> fetching from R2…
            </div>
          ) : peek.error ? (
            <p className="text-destructive text-sm">{peek.error}</p>
          ) : (
            <pre className="bg-muted/40 max-h-80 overflow-auto rounded-md p-3 text-xs whitespace-pre-wrap break-all">
              {peek.text}
            </pre>
          )}
        </DialogContent>
      </Dialog>
    </Card>
  )
}

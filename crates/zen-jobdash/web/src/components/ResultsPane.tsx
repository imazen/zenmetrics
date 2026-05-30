import { useState } from "react"
import { Eye, FlaskConical, Image as ImageIcon, Loader2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, fmtInt, type PeekResult, type ResultRow } from "@/lib/api"

// Encode + diffmap outputs are images; metric outputs are scores (text).
const isImageRow = (r: ResultRow) => r.kind.startsWith("encode") || r.kind.startsWith("diffmap")

export function ResultsPane({ rows }: { rows?: ResultRow[] }) {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<{ row: ResultRow; peek?: PeekResult } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  async function show(row: ResultRow) {
    setView({ row })
    setOpen(true)
    if (!isImageRow(row)) {
      setBusy(row.output_sha)
      try {
        setView({ row, peek: await api.peek(row.output_sha) })
      } catch (e) {
        setView({ row, peek: { error: e instanceof Error ? e.message : String(e) } })
      } finally {
        setBusy(null)
      }
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FlaskConical className="size-4" /> Results
        </CardTitle>
        <CardDescription>
          Completed jobs with an output blob — image encodes/diffmaps render inline, metric scores peek
          by content hash (goal B).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!rows?.length ? (
          <p className="text-muted-foreground text-sm">No completed results yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">Preview</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Codec</TableHead>
                <TableHead>Image</TableHead>
                <TableHead className="text-right">q</TableHead>
                <TableHead>Output</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.output_sha + r.kind + r.image_path} className="cursor-pointer" onClick={() => show(r)}>
                  <TableCell>
                    {isImageRow(r) ? (
                      <img
                        src={api.blobUrl(r.output_sha)}
                        alt=""
                        loading="lazy"
                        className="size-10 rounded border object-cover"
                        onError={(e) => (e.currentTarget.style.display = "none")}
                      />
                    ) : (
                      <div className="bg-muted text-muted-foreground grid size-10 place-items-center rounded">
                        {busy === r.output_sha ? <Loader2 className="size-4 animate-spin" /> : <Eye className="size-4" />}
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                  <TableCell className="font-mono text-xs">{r.codec || "—"}</TableCell>
                  <TableCell className="text-muted-foreground max-w-40 truncate text-xs" title={r.image_path}>
                    {r.image_path.split("/").pop()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{r.q}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="font-mono" title={r.output_sha}>
                      {r.output_sha.slice(0, 10)}
                    </Badge>
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
            <DialogTitle className="flex items-center gap-2">
              {view && isImageRow(view.row) ? <ImageIcon className="size-4" /> : <Eye className="size-4" />}
              {view?.row.kind}
            </DialogTitle>
            <DialogDescription className="font-mono text-xs">
              {view?.row.output_sha.slice(0, 16)}…
              {view?.peek?.size != null && ` · ${fmtInt(view.peek.size)} bytes`}
            </DialogDescription>
          </DialogHeader>
          {view && isImageRow(view.row) ? (
            <img
              src={api.blobUrl(view.row.output_sha)}
              alt={view.row.kind}
              className="max-h-[60vh] w-full rounded border bg-[repeating-conic-gradient(#0003_0_25%,transparent_0_50%)] bg-[length:16px_16px] object-contain"
            />
          ) : !view?.peek ? (
            <div className="text-muted-foreground flex items-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin" /> fetching from R2…
            </div>
          ) : view.peek.error ? (
            <p className="text-destructive text-sm">{view.peek.error}</p>
          ) : (
            <pre className="bg-muted/40 max-h-80 overflow-auto rounded-md p-3 text-xs break-all whitespace-pre-wrap">
              {view.peek.text}
            </pre>
          )}
        </DialogContent>
      </Dialog>
    </Card>
  )
}

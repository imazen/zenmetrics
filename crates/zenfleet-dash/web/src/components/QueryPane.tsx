import { useState } from "react"
import { Loader2, Search } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, type QueryRow } from "@/lib/api"

function statusBadge(s: string) {
  if (s === "Done") return <Badge variant="success">{s}</Badge>
  if (s === "Failed") return <Badge variant="warning">{s}</Badge>
  if (s === "Poison") return <Badge variant="destructive">{s}</Badge>
  return <Badge variant="secondary">{s}</Badge>
}

export function QueryPane() {
  const [kind, setKind] = useState("")
  const [codec, setCodec] = useState("")
  const [status, setStatus] = useState("")
  const [image, setImage] = useState("")
  const [rows, setRows] = useState<QueryRow[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function run() {
    setBusy(true)
    setErr(null)
    try {
      setRows(await api.query({ kind, codec, status, image, limit: "500" }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const field = (label: string, value: string, set: (v: string) => void, placeholder: string) => (
    <div className="space-y-1.5">
      <Label className="text-xs">{label}</Label>
      <Input value={value} onChange={(e) => set(e.target.value)} placeholder={placeholder} className="h-8" />
    </div>
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="size-4" /> Ad-hoc query
        </CardTitle>
        <CardDescription>
          Structured filter over the Parquet ledger (substring match, newest-first, capped at 500).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {field("kind", kind, setKind, "metric:cvvdp")}
          {field("codec", codec, setCodec, "zenjpeg")}
          {field("status", status, setStatus, "done / failed / poison")}
          {field("image", image, setImage, "path substring")}
        </div>
        <Button onClick={run} disabled={busy} size="sm">
          {busy ? <Loader2 className="animate-spin" /> : <Search />} Run query
        </Button>
        {err && <p className="text-destructive text-sm">{err}</p>}
        {rows && (
          <>
            <p className="text-muted-foreground text-xs">{rows.length} row(s)</p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Kind</TableHead>
                  <TableHead>Codec</TableHead>
                  <TableHead>Image</TableHead>
                  <TableHead className="text-right">q</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Worker</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r, i) => (
                  <TableRow key={`${r.output_sha ?? ""}-${r.kind}-${i}`}>
                    <TableCell className="font-mono text-xs">{r.kind}</TableCell>
                    <TableCell className="font-mono text-xs">{r.codec || "—"}</TableCell>
                    <TableCell className="text-muted-foreground max-w-32 truncate text-xs" title={r.image_path}>
                      {r.image_path.split("/").pop()}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{r.q}</TableCell>
                    <TableCell>
                      {statusBadge(r.status)}
                      {r.error_class && <span className="text-muted-foreground ml-1 text-xs">{r.error_class}</span>}
                    </TableCell>
                    <TableCell className="text-xs">{r.worker}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        )}
      </CardContent>
    </Card>
  )
}

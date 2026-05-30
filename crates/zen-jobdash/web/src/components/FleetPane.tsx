import { useState } from "react"
import { Recycle, ServerOff } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { KillFleetButton } from "@/components/KillFleetButton"
import { api, type FleetView } from "@/lib/api"

function statusBadge(status: string) {
  if (status === "running") return <Badge variant="success">running</Badge>
  if (status === "off" || status === "stopping" || status === "deleting")
    return <Badge variant="warning">{status}</Badge>
  return <Badge variant="secondary">{status}</Badge>
}

export function FleetPane({ fleet, onChange }: { fleet?: FleetView; onChange?: () => void }) {
  const boxes = fleet?.boxes ?? []
  const idle = fleet?.idle ?? []
  const [reaping, setReaping] = useState(false)
  async function reap() {
    setReaping(true)
    try {
      await api.reapIdle()
      onChange?.()
    } finally {
      setReaping(false)
    }
  }
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="space-y-1.5">
          <CardTitle>Live fleet</CardTitle>
          <CardDescription>
            Boxes Hetzner reports for label <code className="font-mono">{fleet?.label ?? "group"}</code>.{" "}
            {idle.length > 0 && <span className="text-amber-400">{idle.length} idle (no worker heartbeat).</span>}{" "}
            {fleet && !fleet.actuation && (
              <span className="text-amber-400">{fleet.note ?? "kill won't actuate (no token)"}</span>
            )}
            {fleet?.error && <span className="text-destructive">{fleet.error}</span>}
          </CardDescription>
        </div>
        <div className="flex gap-2">
          {idle.length > 0 && (
            <Button variant="outline" size="sm" onClick={reap} disabled={reaping}>
              <Recycle /> Reap idle ({idle.length})
            </Button>
          )}
          <KillFleetButton
            boxCount={boxes.length}
            canActuate={fleet?.actuation ?? false}
            label={fleet?.label ?? "group"}
            size="sm"
            onDone={onChange}
          />
        </div>
      </CardHeader>
      <CardContent>
        {boxes.length === 0 ? (
          <div className="text-muted-foreground flex items-center gap-2 text-sm">
            <ServerOff className="size-4" /> No live fleet boxes.
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Datacenter</TableHead>
                <TableHead>IPv4</TableHead>
                <TableHead>Group</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {boxes.map((b) => (
                <TableRow key={b.id}>
                  <TableCell className="font-mono text-xs">{b.name}</TableCell>
                  <TableCell>{statusBadge(b.status)}</TableCell>
                  <TableCell className="font-mono text-xs">{b.server_type}</TableCell>
                  <TableCell className="text-muted-foreground text-xs">{b.datacenter}</TableCell>
                  <TableCell className="font-mono text-xs">{b.ipv4 ?? "—"}</TableCell>
                  <TableCell className="text-xs">{b.group ?? "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

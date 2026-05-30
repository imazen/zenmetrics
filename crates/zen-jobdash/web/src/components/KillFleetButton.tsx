import { useState } from "react"
import { Skull, TriangleAlert } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { api, type KillResult } from "@/lib/api"

interface Props {
  /** Boxes about to be killed, for the confirmation summary. */
  boxCount: number
  /** Whether the dashboard can actuate (Hetzner token present). */
  canActuate: boolean
  label: string
  size?: "sm" | "default"
  onDone?: () => void
}

export function KillFleetButton({ boxCount, canActuate, label, size = "default", onDone }: Props) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<KillResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function doKill() {
    setBusy(true)
    setError(null)
    try {
      const resp = await api.killFleet()
      if (resp.result) setResult(resp.result)
      else setError(resp.note ?? "no boxes touched")
      onDone?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) {
          setResult(null)
          setError(null)
        }
      }}
    >
      <DialogTrigger asChild>
        <Button variant="destructive" size={size} disabled={!canActuate && boxCount === 0}>
          <Skull /> Kill fleet
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TriangleAlert className="size-5 text-destructive" /> Kill the entire fleet?
          </DialogTitle>
          <DialogDescription>
            This permanently deletes every Hetzner box carrying the <code className="font-mono">{label}</code> label
            ({boxCount} live {boxCount === 1 ? "box" : "boxes"}). Unlabeled boxes (e.g. your persistent dev box) are
            never touched. This cannot be undone.
            {!canActuate && (
              <span className="mt-2 block text-amber-400">
                No Hetzner token configured — the intent will be recorded but no boxes will be deleted.
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        {result && (
          <div className="rounded-md border bg-muted/40 p-3 text-sm">
            <div className="font-medium">Killed {result.killed.length} box(es)</div>
            {result.killed.length > 0 && (
              <ul className="text-muted-foreground mt-1 font-mono text-xs">
                {result.killed.map((b) => (
                  <li key={b.id}>
                    {b.name} ({b.server_type})
                  </li>
                ))}
              </ul>
            )}
            {result.note && <div className="text-muted-foreground mt-1 text-xs">{result.note}</div>}
            {result.errors.length > 0 && (
              <div className="text-destructive mt-1 text-xs">{result.errors.join("; ")}</div>
            )}
          </div>
        )}
        {error && <div className="text-destructive text-sm">{error}</div>}

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">{result ? "Close" : "Cancel"}</Button>
          </DialogClose>
          {!result && (
            <Button variant="destructive" onClick={doKill} disabled={busy}>
              {busy ? "Killing…" : canActuate ? "Yes, kill the fleet" : "Record intent"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

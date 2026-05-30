import { RefreshCw } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CatalogPane } from "@/components/CatalogPane"
import { ControlPane } from "@/components/ControlPane"
import { FailuresPane } from "@/components/FailuresPane"
import { FleetPane } from "@/components/FleetPane"
import { ProgressPane } from "@/components/ProgressPane"
import { StatCards } from "@/components/StatCards"
import { StoragePane } from "@/components/StoragePane"
import { WorkersPane } from "@/components/WorkersPane"
import { api } from "@/lib/api"
import { usePoll } from "@/lib/usePoll"

export default function App() {
  const cost = usePoll(api.cost)
  const progress = usePoll(api.progress)
  const failures = usePoll(api.failures)
  const storage = usePoll(api.storage)
  const workers = usePoll(api.workers)
  const catalog = usePoll(api.catalog, 30000)
  const fleet = usePoll(api.fleet, 20000)

  const refreshAll = () => {
    cost.refresh()
    progress.refresh()
    failures.refresh()
    storage.refresh()
    workers.refresh()
    catalog.refresh()
    fleet.refresh()
  }

  const age = Math.min(cost.ageSecs, progress.ageSecs, fleet.ageSecs)
  const anyError = cost.error || progress.error || failures.error || storage.error || fleet.error
  const failCount = failures.data?.reduce((a, r) => a + r.count, 0) ?? 0

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-primary text-primary-foreground grid size-9 place-items-center rounded-lg font-bold">
            z
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">zen-jobdash</h1>
            <p className="text-muted-foreground text-xs">
              control plane · content-addressed image-codec sweep fleet
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-muted-foreground flex items-center gap-1.5 text-xs">
            <span className={`size-2 rounded-full ${anyError ? "bg-destructive" : "bg-emerald-500"}`} />
            {anyError ? "connection error" : `updated ${age}s ago`}
          </span>
          <Button variant="outline" size="sm" onClick={refreshAll}>
            <RefreshCw /> Refresh
          </Button>
        </div>
      </header>

      <StatCards cost={cost.data} fleet={fleet.data} />

      <Tabs defaultValue="overview" className="mt-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="fleet">
            Fleet
            {(fleet.data?.boxes.length ?? 0) > 0 && (
              <Badge variant="secondary" className="ml-1">
                {fleet.data?.boxes.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="catalog">
            Catalog
            {(catalog.data?.length ?? 0) > 0 && (
              <Badge variant="secondary" className="ml-1">
                {catalog.data?.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="failures">
            Failures
            {failCount > 0 && (
              <Badge variant="destructive" className="ml-1">
                {failCount}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="control">Control</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <ProgressPane rows={progress.data} />
          <StoragePane rows={storage.data} cost={cost.data} />
        </TabsContent>

        <TabsContent value="fleet" className="space-y-4">
          <FleetPane fleet={fleet.data} onChange={() => setTimeout(fleet.refresh, 1500)} />
          <WorkersPane workers={workers.data} />
        </TabsContent>

        <TabsContent value="catalog" className="space-y-4">
          <CatalogPane rows={catalog.data} />
        </TabsContent>

        <TabsContent value="failures" className="space-y-4">
          <FailuresPane rows={failures.data} />
        </TabsContent>

        <TabsContent value="control" className="space-y-4">
          <ControlPane fleet={fleet.data} onChange={() => setTimeout(fleet.refresh, 1500)} />
        </TabsContent>
      </Tabs>

      <footer className="text-muted-foreground mt-10 text-center text-xs">
        zen job system · ledger-driven · auth via HTTP Basic
      </footer>
    </div>
  )
}

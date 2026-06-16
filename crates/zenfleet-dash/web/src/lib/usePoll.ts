import { useCallback, useEffect, useRef, useState } from "react"

export interface Poll<T> {
  data: T | undefined
  error: string | undefined
  loading: boolean
  /** Seconds since the last successful load (for a "live" indicator). */
  ageSecs: number
  refresh: () => void
}

/** Poll an async source on an interval, exposing data/error/age and a manual refresh. */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs = 15000): Poll<T> {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<string>()
  const [loading, setLoading] = useState(true)
  const [lastOk, setLastOk] = useState<number>(() => Date.now())
  const [ageSecs, setAgeSecs] = useState(0)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const refresh = useCallback(() => {
    let cancelled = false
    fetcherRef
      .current()
      .then((d) => {
        if (cancelled) return
        setData(d)
        setError(undefined)
        setLastOk(Date.now())
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, intervalMs)
    return () => clearInterval(id)
  }, [refresh, intervalMs])

  useEffect(() => {
    const id = setInterval(() => setAgeSecs(Math.round((Date.now() - lastOk) / 1000)), 1000)
    return () => clearInterval(id)
  }, [lastOk])

  return { data, error, loading, ageSecs, refresh }
}

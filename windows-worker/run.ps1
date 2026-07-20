# run.ps1 -- the Zen worker loop for a Windows PC (the Windows port of fleet-entrypoint.sh pool_mode).
#
# Works the SAME R2 job pool the cloud boxes use: shuffled round-robin over the runlist, one
# zenfleet-worker.exe pass per run, coordinating via the R2 claim ledger. Runs at LOW priority so the
# kids' games always win the CPU, and loops until Task Scheduler stops it (when the PC is no longer
# idle). ASCII-only (PowerShell 5.1 reads BOM-less files as ANSI).
$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = Join-Path $here 'worker.log'
function Log($m) { "{0} {1}" -f (Get-Date -Format o), $m | Tee-Object -FilePath $log -Append | Out-Host }

# --- config from worker.env (KEY=VALUE): scoped AWS/R2 creds + pool settings ---
$envFile = Join-Path $here 'worker.env'
if (Test-Path $envFile) {
  foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
      [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
  }
}
$bucket   = if ($env:ZEN_BUCKET) { $env:ZEN_BUCKET } else { 'zentrain' }
$endpoint = $env:ZEN_R2_ENDPOINT
$runlist  = $env:ZEN_POOL_RUNLIST
$worker   = if ($env:ZEN_WORKER) { $env:ZEN_WORKER } else { "winpool-$env:COMPUTERNAME" }
if (-not $endpoint -or -not $runlist) { Log "FATAL: worker.env must set ZEN_R2_ENDPOINT and ZEN_POOL_RUNLIST"; exit 2 }

$wk   = Join-Path $here 'zenfleet-worker.exe'
$exec = Join-Path $here 'zenfleet-exec.exe'        # the built-in shim -> zenmetrics.exe jobexec
$env:Path = "$here;$env:Path"                       # so the worker finds s5cmd.exe / aws.exe bundled here
foreach ($b in $wk,$exec,(Join-Path $here 'zenmetrics.exe'),(Join-Path $here 's5cmd.exe')) {
  if (-not (Test-Path $b)) { Log "FATAL: missing $b"; exit 3 }
}

# --- CPU niceness: cap threads + low priority so gaming is unaffected ---
if (-not $env:ZEN_THREADS) { $env:ZEN_THREADS = [string][math]::Max(1, [int]($env:NUMBER_OF_PROCESSORS / 2)) }
$env:RAYON_NUM_THREADS = $env:ZEN_THREADS
$env:OMP_NUM_THREADS   = $env:ZEN_THREADS
$env:ZEN_PERSISTENT_EXEC = '1'                       # keep one warm jobexec child per pass
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}

# --- fetch + shuffle the runlist (spread the fleet across runs, not all on run #1) ---
$rl = Join-Path $env:TEMP 'zen_runlist.tsv'
& s5cmd.exe --endpoint-url $endpoint cp $runlist $rl 2>$null
if (-not (Test-Path $rl)) { Log "FATAL: could not fetch runlist $runlist"; exit 4 }
$runs = Get-Content $rl | Where-Object { $_ -match '\S' } | Sort-Object { Get-Random }
Log ("POOL start: {0} runs, worker={1}, threads={2}" -f $runs.Count, $worker, $env:ZEN_THREADS)

function Expand-Gz([string]$in, [string]$out) {
  $i = [IO.File]::OpenRead($in); $g = New-Object IO.Compression.GzipStream($i, [IO.Compression.CompressionMode]::Decompress)
  $o = [IO.File]::Create($out); $g.CopyTo($o); $o.Close(); $g.Close(); $i.Close()
}

# --- pool loop: cycle the runlist until the scheduler stops us (PC no longer idle) ---
$cyc = 0
while ($true) {
  $cyc++; $didAny = $false
  foreach ($line in $runs) {
    $p = $line -split "`t"
    if ($p.Count -lt 2) { continue }
    $run = $p[0]; $src = $p[1]; $mode = if ($p.Count -ge 3) { $p[2] } else { 'tar' }

    # manifest (cached per run)
    $mf = Join-Path $env:TEMP "m_$run.json"
    if (-not (Test-Path $mf)) {
      $gz = "$mf.gz"
      & s5cmd.exe --endpoint-url $endpoint cp "s3://$bucket/jobs/$run/manifest.json.gz" $gz 2>$null
      if (Test-Path $gz) { Expand-Gz $gz $mf; Remove-Item $gz -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $mf)) { continue }

    # per-mode source env: 'enc' = direct-object (zenjpeg, s5cmd); else byte-range from the tar (aws)
    if ($mode -eq 'enc') {
      $env:ZEN_ENCODES_PREFIX = $src; $env:ZEN_ENCODES_BUCKET = $bucket
      $env:ZEN_VARIANTS_TAR_URI = $null; $env:ZEN_VARIANT_INDEX_URI = $null
    } else {
      $env:ZEN_VARIANTS_TAR_URI = $src
      $env:ZEN_VARIANT_INDEX_URI = "s3://$bucket/jobs/$run/variant_index.tsv"
      $env:ZEN_ENCODES_PREFIX = $null; $env:ZEN_ENCODES_BUCKET = $null
    }

    $before = Test-Path $log  # cheap "did work happen" proxy is the worker's own stdout below
    $out = & $wk --manifest $mf `
      --ledger-out "s3://$bucket/jobs/$run/ledger/win-$worker-$cyc.parquet" `
      --blobs-r2-bucket $bucket --blobs-r2-prefix "jobs/$run/blobs" `
      --claims-r2-bucket $bucket --claims-prefix "jobs/$run/claims" `
      --r2-endpoint $endpoint --exec $exec --worker $worker --provider 'windows-idle' 2>&1
    $done = 0
    if ($out -match 'done=(\d+)') { $done = [int]$matches[1] }
    if ($done -gt 0) { $didAny = $true }
    Log ("cyc=$cyc run=$run mode=$mode done=$done")
  }
  if (-not $didAny) { Log "POOL: whole pool drained (no work in a full cycle) -- exiting"; break }
  Start-Sleep -Milliseconds 300
}
Log "POOL exit"

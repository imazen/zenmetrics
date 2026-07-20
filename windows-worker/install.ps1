# install.ps1 -- register the Zen worker as an idle-only, logout-surviving SYSTEM scheduled task.
#
# Self-elevates via UAC (one click). Copies this folder's payload to C:\ProgramData\zenworker and
# registers a task that runs ONLY when the PC is idle, as SYSTEM (so it runs whether anyone is logged
# on or not) and STOPS the instant someone touches the machine. Re-run any time to update. ASCII-only
# (Windows PowerShell 5.1 reads BOM-less files as ANSI).
$ErrorActionPreference = 'Stop'

# --- self-elevate: task registration as SYSTEM needs admin ---
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "Requesting administrator rights (UAC)..."
  Start-Process powershell.exe -Verb RunAs -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $PSCommandPath)
  exit
}

$taskName   = 'ZenWorker'
$srcDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$installDir = Join-Path $env:ProgramData 'zenworker'          # C:\ProgramData\zenworker

# --- copy payload (binaries + run.ps1 + worker.env) into ProgramData ---
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
foreach ($f in 'zenmetrics.exe','zenfleet-worker.exe','s5cmd.exe','aws.exe','run.ps1','worker.env') {
  $p = Join-Path $srcDir $f
  if (Test-Path $p) { Copy-Item $p (Join-Path $installDir $f) -Force }
  elseif ($f -in 'zenmetrics.exe','zenfleet-worker.exe','run.ps1') { throw "missing required payload file: $f (build/copy it into $srcDir first)" }
}
$runner = Join-Path $installDir 'run.ps1'
Write-Host "Payload installed to $installDir"

# --- idle-only SYSTEM task ---
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
             -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $runner)
$trigger = New-ScheduledTaskTrigger -AtStartup
# RunOnlyIfIdle + (default) StopOnIdleEnd + RestartOnIdle == run only while nobody's using the PC.
# ExecutionTimeLimit 0 == no time cap (it runs as long as the box stays idle). StartWhenAvailable so a
# missed boot-time start still fires. Keeps going on battery (laptops) but yields the moment it's used.
$settings = New-ScheduledTaskSettingsSet `
             -RunOnlyIfIdle `
             -IdleDuration (New-TimeSpan -Minutes 10) `
             -IdleWaitTimeout (New-TimeSpan -Hours 24) `
             -RestartOnIdle `
             -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -ExecutionTimeLimit ([TimeSpan]::Zero) `
             -StartWhenAvailable `
             -MultipleInstances IgnoreNew
# SYSTEM principal == runs whether a user is logged on or not, no stored password.
$principal = New-ScheduledTaskPrincipal -UserId 'NT AUTHORITY\SYSTEM' -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
  -Description 'Zen idle-only distributed feature-extraction worker. Runs only when the PC is idle, whether logged on or not; stops the moment the PC is used.' | Out-Null

Write-Host ""
Write-Host "Installed scheduled task '$taskName':"
Write-Host "  - runs as SYSTEM (survives logout and reboot)"
Write-Host "  - starts ~10 min after the PC goes idle, stops instantly when someone uses it"
Write-Host "  - low priority, capped threads (set in worker.env) so games always win"
Write-Host ""
Write-Host "Verify:  Get-ScheduledTask ZenWorker | Get-ScheduledTaskInfo"
Write-Host "Logs:    $installDir\worker.log"

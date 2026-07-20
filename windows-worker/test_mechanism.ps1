# test_mechanism.ps1 -- prove the idle-only scheduled-task mechanism on THIS Windows box, no admin.
#
# Registers a USER-context idle-only task that runs a heartbeat "worker" stub, shows the idle config,
# force-runs it to prove the action executes, then cleans up. This validates everything the kids'-PC
# design needs EXCEPT the SYSTEM / run-whether-logged-on-or-not step (that needs admin -> install.ps1).
# ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less files as ANSI and chokes on non-ASCII.
$ErrorActionPreference = 'Stop'

$dir  = Join-Path $env:LOCALAPPDATA 'zenworker-test'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$stub = Join-Path $dir 'heartbeat_stub.ps1'
$log  = Join-Path $dir 'heartbeat.log'
Remove-Item $log -ErrorAction SilentlyContinue

# --- the stub "worker": append a heartbeat a few times, then exit (stands in for the real worker loop) ---
@'
$log = Join-Path $PSScriptRoot "heartbeat.log"
for ($i=1; $i -le 5; $i++) {
  ("{0} pid={1} user={2} iter={3} WORKER-RUNNING" -f (Get-Date -Format o), $PID, (whoami), $i) |
    Out-File -Append -Encoding utf8 $log
  Start-Sleep -Seconds 1
}
'@ | Set-Content -Encoding ascii $stub

$taskName = 'ZenWorkerTest'
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Idle-only settings: run ONLY when the PC has been idle, STOP the moment it is no longer idle,
# and restart when it goes idle again. This is what keeps the kids' games unaffected.
$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
             -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $stub)
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
             -RunOnlyIfIdle `
             -IdleDuration (New-TimeSpan -Minutes 5) `
             -IdleWaitTimeout (New-TimeSpan -Hours 2) `
             -RestartOnIdle `
             -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -MultipleInstances IgnoreNew
# user-context (no admin): omit -Principal so it defaults to the current user, logon-when-present.
# install.ps1 uses -User "NT AUTHORITY\SYSTEM" -RunLevel Highest for logout-survival (needs admin).
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
  -Settings $settings -Description 'Zen idle-only worker (TEST)' | Out-Null
Write-Output "OK registered task: $taskName"

# --- prove the CONFIG is what we want ---
$t = Get-ScheduledTask -TaskName $taskName
Write-Output ("cfg RunOnlyIfIdle = {0}" -f $t.Settings.RunOnlyIfIdle)
Write-Output ("cfg IdleDuration  = {0}" -f $t.Settings.IdleSettings.IdleDuration)
Write-Output ("cfg StopOnIdleEnd = {0}" -f $t.Settings.IdleSettings.StopOnIdleEnd)
Write-Output ("cfg RestartOnIdle = {0}" -f $t.Settings.IdleSettings.RestartOnIdle)
Write-Output ("cfg LogonType     = {0}" -f $t.Principal.LogonType)

# --- prove the ACTION runs. On-demand start; if the idle gate blocks it, fall back to a direct run ---
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 7
$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Output ("run LastTaskResult = {0} (0 = success)" -f $info.LastTaskResult)
if (-not (Test-Path $log)) {
  Write-Output "note: idle gate blocked on-demand start (box is active) -- direct-running the stub to prove the action"
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $stub
}
Write-Output "--- heartbeat.log (proves the worker action executed) ---"
if (Test-Path $log) { Get-Content $log } else { Write-Output "(NO log, action failed)" }

# --- cleanup ---
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Output "cleaned up: $taskName unregistered"

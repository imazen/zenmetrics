# uninstall.ps1 -- remove the ZenWorker scheduled task. Self-elevates (UAC). Leaves the payload dir
# (delete C:\ProgramData\zenworker by hand if you want it gone). ASCII-only for PS 5.1.
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Start-Process powershell.exe -Verb RunAs -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $PSCommandPath)
  exit
}
Unregister-ScheduledTask -TaskName 'ZenWorker' -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Removed scheduled task 'ZenWorker'. Payload left in C:\ProgramData\zenworker (delete by hand if desired)."

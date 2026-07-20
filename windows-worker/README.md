# zen idle-only Windows worker

Turn spare Windows PCs (kids' gaming rigs, an office desktop) into fleet workers for the zensim-720
feature backfill. Each PC runs the **same** `zenfleet-worker` the cloud boxes run, pulling jobs from the
**same** R2 pool — it just becomes another worker on the ledger. It runs **only when the PC is idle**
and **whether anyone is logged in or not**, and yields the CPU the instant someone touches the machine.

## How it works

```
Task Scheduler (idle-only, SYSTEM)  ->  run.ps1 (pool loop, low priority)  ->  zenfleet-worker.exe
   runs when idle, whether logged        shuffled round-robin over the           claims + scores cells,
   on or not; stops when PC is used       R2 runlist, one pass per run            writes blobs + ledger to R2
```

- **"Whether logged out or not"** = a **Windows scheduled task running as `NT AUTHORITY\SYSTEM`**. A
  normal program (or Docker Desktop) is tied to your login session and dies at logout; a SYSTEM task
  runs with no desktop, starts at boot, and survives logout + reboot + Windows Update.
- **"Only when idle"** = the task's `RunOnlyIfIdle` + `StopOnIdleEnd` + `RestartOnIdle` settings. It
  starts ~10 min after the PC goes idle and stops the moment there's mouse/keyboard input.
- **Neighborly** = capped threads (`ZEN_THREADS`) + `BelowNormal`/`Idle` process priority, so games and
  apps always win the CPU.

## Build the binaries (once, on a machine with the workspace)

`build_win.bat` cross-uses the MSVC toolchain against the WSL-hosted workspace (source over
`\\wsl.localhost\...`, target dir kept local for speed). It produces, in
`C:\zenbuild-target\x86_64-pc-windows-msvc\release\`:

- `zenmetrics.exe` — the scorer (`jobexec`)
- `zenfleet-worker.exe` — the claim/ledger loop
- `zenfleet-exec.exe` — the exec shim (`--exec` target; forwards to `zenmetrics jobexec`)

Prereqs: VS 2022 Build Tools (vcvars64), CMake, NASM (for aws-lc-sys), and a Rust toolchain. Building on
GitHub's Windows CI runner is an equally valid way to produce these.

## Assemble the payload

Put these in one folder (this is what you copy to each PC):

```
zenmetrics.exe  zenfleet-worker.exe  zenfleet-exec.exe  s5cmd.exe
run.ps1  install.ps1  uninstall.ps1  worker.env
```

`s5cmd.exe`: grab the Windows build from https://github.com/peak/s5cmd/releases .
`worker.env`: copy `worker.env.example`, fill in creds + settings (below).

## Credentials

Never put the R2 account root key on a kid's PC. Mint a **scoped, short-TTL temp credential** (Cloudflare
`POST /accounts/{id}/r2/temp-access-credentials`, `object-read-write`, prefixes `jobs/`,
`canonical/2026-06-27/`, `refs/`) and paste the `accessKeyId` / `secretAccessKey` / `sessionToken` into
`worker.env`. If a PC is compromised, the blast radius is those prefixes, not the account. Refresh the
cred periodically from a trusted machine (a small scheduled task can re-mint and push it out).

## Deploy (per PC)

1. Copy the payload folder to the PC (anywhere).
2. Right-click `install.ps1` → *Run with PowerShell*, or run it from any prompt — it **self-elevates**
   (one UAC click). It copies the payload to `C:\ProgramData\zenworker` and registers the `ZenWorker`
   idle-only SYSTEM task.

That's it. The worker starts within ~10 min of the PC going idle.

## Verify / logs / remove

```powershell
Get-ScheduledTask ZenWorker | Get-ScheduledTaskInfo      # LastRunResult, LastRunTime
Get-Content C:\ProgramData\zenworker\worker.log -Tail 20 # what it's been doing
.\uninstall.ps1                                          # remove the task (self-elevates)
```

The PC shows up as worker `winpool-<COMPUTERNAME>` on the R2 ledger, alongside the cloud boxes.

## Current limitation: byte-range codecs need aws.exe

The zenjpeg run (`enc` mode, direct-object) works with `s5cmd.exe` alone. The other five codecs store
variants in tars fetched by **byte-range GET**, which `zenfleet-worker` currently shells out to the
**aws CLI** for (`aws s3api get-object --range`). To work those runs a PC also needs `aws.exe` on PATH
(AWS CLI v2 for Windows).

**Recommended fix:** move the byte-range GET to the in-process `object_store` client (already a
dependency, used for the claim/ledger). That drops the external `aws` dependency for *every* platform,
not just Windows, and shrinks the payload. Tracked as the next step.

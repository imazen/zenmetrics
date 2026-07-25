# setup-reboot-switch.ps1 -- secure remote "reboot into the zen node" switch for a dual-boot box.
#
# Run ONCE in an ELEVATED PowerShell on the Windows side of a box that also has the Ubuntu zen
# node installed (via ubuntu-node/pxe). It lets the fleet reboot the box remotely so it PXE-boots
# and GRUB (per the tower's worker flag) lands it in Ubuntu.
#
# Creates TWO SSH accounts, both authorized to the same dev-box key:
#   - `zenswitch`: reboot-ONLY. Non-admin, FORCED COMMAND ("shutdown /r /t 0", no shell). Used by
#     fleet-pxe to-ubuntu/to-windows for safe unattended OS-flips -- a leak can only reboot the box.
#   - `zenadmin`:  FULL ADMIN. In the Administrators group, key-only SSH, real shell -- for remote
#     administration (install/update/fix the box without the console).
# SECURITY NOTE: zenadmin grants whoever holds the key full control of this machine over SSH. That is
# intended (the fleet operator administers these shared boxes) but it IS a real exposure -- treat the
# dev-box private key accordingly. Remove zenadmin (`Remove-LocalUser zenadmin`) to revoke admin access;
# zenswitch alone still gives the reboot/OS-flip capability.
#
#   powershell -ExecutionPolicy Bypass -File .\setup-reboot-switch.ps1

$ErrorActionPreference = 'Stop'
$User    = 'zenswitch'   # reboot-ONLY (forced command, non-admin) -- used by fleet-pxe to-ubuntu/to-windows
$Admin   = 'zenadmin'    # FULL admin (key-only SSH, real shell) -- for remote administration of the box
# the dev-box key (same one baked into the Ubuntu node). Replace if yours differs.
$PubKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGI7zVd/Fd0aIgq9ykown4UKZIPc0cp/NTTG1hOtTnW7 lilith@lilith'

# 28-char password guaranteed to satisfy Windows local complexity (>=1 upper, lower, digit, special) --
# validated by executing this exact logic in Windows PowerShell 5.1. Used for both service accounts.
function New-ZenPassword {
  $U='ABCDEFGHIJKLMNOPQRSTUVWXYZ'; $Lo='abcdefghijklmnopqrstuvwxyz'; $Di='0123456789'; $Sp='!@#%^&*-_=+'
  $all  = $U + $Lo + $Di + $Sp
  $req  = @($U, $Lo, $Di, $Sp | ForEach-Object { $_[(Get-Random -Maximum $_.Length)] })
  $fill = 1..24 | ForEach-Object { $all[(Get-Random -Maximum $all.Length)] }
  (($req + $fill) | Get-Random -Count 28) -join ''
}

Write-Host "== 1/6 OpenSSH Server =="
# PREFER the Windows optional-feature OpenSSH -- Windows Update keeps it patched, which matters (an
# unpatched sshd is a security liability). The catch: `Add-WindowsCapability -Online` can hang for many
# minutes with NO output when WU is slow or policy-blocked (it stalled here once at "1/5"). So try WU
# BOUNDED (~3 min, with a heartbeat so it can't look frozen); only if that doesn't finish do we fall back
# to a hash-pinned standalone package from the tower so setup still completes. The standalone path is NOT
# auto-updated, so the script warns loudly and tells you how to get back onto the WU-managed one.
$SshZipUrl = if ($env:ZEN_SSH_ZIP_URL) { $env:ZEN_SSH_ZIP_URL } else { 'http://192.168.50.170:3080/OpenSSH-Win64.zip' }
if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
  Write-Host "   installing via Windows Update (preferred -- WU keeps it patched); up to 3 min..."
  $job = Start-Job { Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 -ErrorAction SilentlyContinue }
  $deadline = (Get-Date).AddMinutes(3)
  while ($job.State -eq 'Running' -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5; Write-Host "   ...still installing via WU" }
  if ($job.State -ne 'Running') { Receive-Job $job -ErrorAction SilentlyContinue | Out-Null }
  Stop-Job $job -ErrorAction SilentlyContinue; Remove-Job $job -Force -ErrorAction SilentlyContinue
}
if (Get-Service sshd -ErrorAction SilentlyContinue) {
  Write-Host "   OpenSSH Server present (Windows-Update managed -> auto-patched)"
} else {
  Write-Warning "Windows Update install didn't finish -> falling back to a STANDALONE package."
  Write-Warning "Standalone OpenSSH is NOT auto-updated. Once Windows Update works, re-run this script"
  Write-Warning "(or add the 'OpenSSH Server' optional feature) to switch to a WU-patched sshd."
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $zip = "$env:TEMP\OpenSSH-Win64.zip"
  Write-Host "   downloading standalone OpenSSH from $SshZipUrl (~5MB)..."
  Invoke-WebRequest $SshZipUrl -OutFile $zip -UseBasicParsing
  # Integrity: fetched over plain HTTP on the LAN, so DON'T trust the transport -- verify the bytes against
  # a pinned SHA256 (Win32-OpenSSH 10.0.0.0p2-Preview OpenSSH-Win64.zip) before extracting/installing it as
  # a service. A tampered/MITM'd package fails here and is never run. Bump zip + hash together to update.
  $EXPECTED_SHA256 = '23F50F3458C4C5D0B12217C6A5DDFDE0137210A30FA870E98B29827F7B43ABA5'
  $got = (Get-FileHash $zip -Algorithm SHA256).Hash
  if ($got -ne $EXPECTED_SHA256) { throw "OpenSSH zip SHA256 mismatch (got $got, expected $EXPECTED_SHA256) -- refusing to install (possible tampering)" }
  Write-Host "   sha256 verified; extracting..."
  $tmp = "$env:TEMP\zen-openssh"; if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  Expand-Archive $zip $tmp -Force
  $src = (Get-ChildItem $tmp -Recurse -Filter sshd.exe | Select-Object -First 1).Directory.FullName
  $dst = Join-Path $env:ProgramFiles 'OpenSSH'
  if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
  Move-Item $src $dst
  Write-Host "   registering sshd service (install-sshd.ps1)..."
  & powershell -ExecutionPolicy Bypass -File (Join-Path $dst 'install-sshd.ps1') | Out-Null
}
Set-Service sshd -StartupType Automatic
Start-Service sshd
# firewall rule (usually added by the capability; ensure it exists)
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

Write-Host "== 2/6 dedicated reboot-only user '$User' (non-admin) =="
$sec = ConvertTo-SecureString (New-ZenPassword) -AsPlainText -Force
if (Get-LocalUser -Name $User -ErrorAction SilentlyContinue) {
  Set-LocalUser -Name $User -Password $sec
} else {
  New-LocalUser -Name $User -Password $sec -PasswordNeverExpires -AccountNeverExpires `
    -Description 'zen fleet reboot-only SSH switch' | Out-Null
}
# stays in 'Users' only -- NEVER Administrators. (Users have the 'Shut down the system' right by
# default on Win10/11 workstations, which is all this needs.)

Write-Host "== 3/6 authorized_keys with a FORCED reboot-only command =="
$home2 = "C:\Users\$User"
$ssh   = "$home2\.ssh"
New-Item -ItemType Directory -Force -Path $ssh | Out-Null
$forced = 'command="shutdown /r /t 0",no-pty,no-agent-forwarding,no-port-forwarding,no-user-rc,no-x11-forwarding'
Set-Content -Path "$ssh\authorized_keys" -Value "$forced $PubKey" -Encoding ascii

Write-Host "== 4/6 lock down .ssh + authorized_keys ACL (sshd StrictModes refuses loose perms) =="
# Lock BOTH the .ssh dir and the file -- sshd's StrictModes rejects the key if either is writable by
# other users. /inheritance:r drops inherited 'Users' access; then only the account + SYSTEM + admins.
icacls "$ssh" /inheritance:r /grant "${User}:F" "SYSTEM:F" "Administrators:F" | Out-Null
icacls "$ssh\authorized_keys" /inheritance:r /grant "${User}:R" "SYSTEM:F" "Administrators:F" | Out-Null

Write-Host "== 5/6 default shell = cmd (so the forced command runs) =="
if (-not (Test-Path "HKLM:\SOFTWARE\OpenSSH")) { New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null }
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "$env:SystemRoot\System32\cmd.exe" -PropertyType String -Force | Out-Null

Write-Host "== 6/6 full-admin user '$Admin' (key-only SSH, for remote administration) =="
$asec = ConvertTo-SecureString (New-ZenPassword) -AsPlainText -Force
if (Get-LocalUser -Name $Admin -ErrorAction SilentlyContinue) {
  Set-LocalUser -Name $Admin -Password $asec
} else {
  New-LocalUser -Name $Admin -Password $asec -PasswordNeverExpires -AccountNeverExpires `
    -Description 'zen fleet admin (key-only SSH)' | Out-Null
}
# Add to local Administrators by SID (S-1-5-32-544) so it works on non-English Windows too.
Add-LocalGroupMember -Group (Get-LocalGroup -SID 'S-1-5-32-544').Name -Member $Admin -ErrorAction SilentlyContinue
# ADMIN users authenticate via ProgramData\ssh\administrators_authorized_keys (per-user .ssh is ignored
# for admin group members). Append our key idempotently (don't clobber other admin keys); lock the ACL to
# Administrators + SYSTEM only, which sshd StrictModes requires for that file.
$akeys = "$env:ProgramData\ssh\administrators_authorized_keys"
$have  = if (Test-Path $akeys) { Get-Content $akeys -ErrorAction SilentlyContinue } else { @() }
if ($have -notcontains $PubKey) { Add-Content -Path $akeys -Value $PubKey -Encoding ascii }
icacls "$akeys" /inheritance:r /grant "Administrators:F" "SYSTEM:F" | Out-Null

Restart-Service sshd

Write-Host ""
Write-Host "DONE. Two SSH accounts are live for this box:"
Write-Host "  reboot-only:  ssh $User@<box>    (connecting reboots it; used by fleet-pxe to-ubuntu/to-windows)"
Write-Host "  full admin:   ssh $Admin@<box>   (real admin shell for remote administration)"

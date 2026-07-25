# setup-reboot-switch.ps1 -- secure remote "reboot into the zen node" switch for a dual-boot box.
#
# Run ONCE in an ELEVATED PowerShell on the Windows side of a box that also has the Ubuntu zen
# node installed (via ubuntu-node/pxe). It lets the fleet reboot the box remotely so it PXE-boots
# and GRUB (per the tower's worker flag) lands it in Ubuntu.
#
# SECURITY: the SSH key we authorize can do NOTHING but reboot this machine --
#   - a DEDICATED NON-ADMIN user `zenswitch` (never added to Administrators)
#   - a FORCED COMMAND in authorized_keys ("shutdown /r /t 0", no-pty) so no shell is possible
# Worst case if the key leaks: someone can reboot this box. The tower -- which you control --
# decides which OS it boots into, so a leaked key can't even force it into Ubuntu.
#
#   powershell -ExecutionPolicy Bypass -File .\setup-reboot-switch.ps1

$ErrorActionPreference = 'Stop'
$User   = 'zenswitch'
# the dev-box key (same one baked into the Ubuntu node). Replace if yours differs.
$PubKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGI7zVd/Fd0aIgq9ykown4UKZIPc0cp/NTTG1hOtTnW7 lilith@lilith'

Write-Host "== 1/5 OpenSSH Server =="
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

Write-Host "== 2/5 dedicated NON-admin user '$User' =="
# Guarantee complexity: >=1 upper, lower, digit, AND special. A local password policy may require a
# special char, and with $ErrorActionPreference='Stop' a New-LocalUser complexity rejection halts the
# WHOLE script -- the exact failure that left the box half-done (OpenSSH installed, zenswitch missing).
$U='ABCDEFGHIJKLMNOPQRSTUVWXYZ'; $Lo='abcdefghijklmnopqrstuvwxyz'; $Di='0123456789'; $Sp='!@#%^&*-_=+'
$all  = $U + $Lo + $Di + $Sp
$req  = @($U, $Lo, $Di, $Sp | ForEach-Object { $_[(Get-Random -Maximum $_.Length)] })
$fill = 1..24 | ForEach-Object { $all[(Get-Random -Maximum $all.Length)] }
$rand = (($req + $fill) | Get-Random -Count 28) -join ''
$sec  = ConvertTo-SecureString $rand -AsPlainText -Force
if (Get-LocalUser -Name $User -ErrorAction SilentlyContinue) {
  Set-LocalUser -Name $User -Password $sec
} else {
  New-LocalUser -Name $User -Password $sec -PasswordNeverExpires -AccountNeverExpires `
    -Description 'zen fleet reboot-only switch (non-admin, forced-command SSH)' | Out-Null
}
# stays in 'Users' only -- NEVER Administrators. (Users have the 'Shut down the system' right by
# default on Win10/11 workstations, which is all this needs.)

Write-Host "== 3/5 authorized_keys with a FORCED reboot-only command =="
$home2 = "C:\Users\$User"
$ssh   = "$home2\.ssh"
New-Item -ItemType Directory -Force -Path $ssh | Out-Null
$forced = 'command="shutdown /r /t 0",no-pty,no-agent-forwarding,no-port-forwarding,no-user-rc,no-x11-forwarding'
Set-Content -Path "$ssh\authorized_keys" -Value "$forced $PubKey" -Encoding ascii

Write-Host "== 4/5 lock down .ssh + authorized_keys ACL (sshd StrictModes refuses loose perms) =="
# Lock BOTH the .ssh dir and the file -- sshd's StrictModes rejects the key if either is writable by
# other users. /inheritance:r drops inherited 'Users' access; then only the account + SYSTEM + admins.
icacls "$ssh" /inheritance:r /grant "${User}:F" "SYSTEM:F" "Administrators:F" | Out-Null
icacls "$ssh\authorized_keys" /inheritance:r /grant "${User}:R" "SYSTEM:F" "Administrators:F" | Out-Null

Write-Host "== 5/5 default shell = cmd (so the forced command runs) =="
if (-not (Test-Path "HKLM:\SOFTWARE\OpenSSH")) { New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null }
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "$env:SystemRoot\System32\cmd.exe" -PropertyType String -Force | Out-Null
Restart-Service sshd

Write-Host ""
Write-Host "DONE. Test from the dev box:  ssh $User@<this-box-ip>   (it will reboot; nothing else)."
Write-Host "The fleet uses:  ./fleet-pxe to-ubuntu <mac>   (sets worker flag + reboots here)."

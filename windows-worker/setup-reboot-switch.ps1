# setup-reboot-switch.ps1 — secure remote "reboot into the zen node" switch for a dual-boot box.
#
# Run ONCE in an ELEVATED PowerShell on the Windows side of a box that also has the Ubuntu zen
# node installed (via ubuntu-node/pxe). It lets the fleet reboot the box remotely so it PXE-boots
# and GRUB (per the tower's worker flag) lands it in Ubuntu.
#
# SECURITY: the SSH key we authorize can do NOTHING but reboot this machine —
#   - a DEDICATED NON-ADMIN user `zenswitch` (never added to Administrators)
#   - a FORCED COMMAND in authorized_keys ("shutdown /r /t 0", no-pty) so no shell is possible
# Worst case if the key leaks: someone can reboot this box. The tower — which you control —
# decides which OS it boots into, so a leaked key can't even force it into Ubuntu.
#
#   powershell -ExecutionPolicy Bypass -File .\setup-reboot-switch.ps1

$ErrorActionPreference = 'Stop'
$User   = 'zenswitch'
# the dev-box key (same one baked into the Ubuntu node). Replace if yours differs.
$PubKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGI7zVd/Fd0aIgq9ykown4UKZIPc0cp/NTTG1hOtTnW7 lilith@lilith'

Write-Host "== 1/5 OpenSSH Server =="
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 -ErrorAction SilentlyContinue | Out-Null
Set-Service sshd -StartupType Automatic
Start-Service sshd
# firewall rule (usually added by the capability; ensure it exists)
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

Write-Host "== 2/5 dedicated NON-admin user '$User' =="
$rand = -join ((1..28) | ForEach-Object { [char]((65..90)+(97..122)+(48..57) | Get-Random) })
$sec  = ConvertTo-SecureString $rand -AsPlainText -Force
if (Get-LocalUser -Name $User -ErrorAction SilentlyContinue) {
  Set-LocalUser -Name $User -Password $sec
} else {
  New-LocalUser -Name $User -Password $sec -PasswordNeverExpires -AccountNeverExpires `
    -Description 'zen fleet reboot-only switch (non-admin, forced-command SSH)' | Out-Null
}
# stays in 'Users' only — NEVER Administrators. (Users have the 'Shut down the system' right by
# default on Win10/11 workstations, which is all this needs.)

Write-Host "== 3/5 authorized_keys with a FORCED reboot-only command =="
$home2 = "C:\Users\$User"
$ssh   = "$home2\.ssh"
New-Item -ItemType Directory -Force -Path $ssh | Out-Null
$forced = 'command="shutdown /r /t 0",no-pty,no-agent-forwarding,no-port-forwarding,no-user-rc,no-x11-forwarding'
Set-Content -Path "$ssh\authorized_keys" -Value "$forced $PubKey" -Encoding ascii

Write-Host "== 4/5 lock down authorized_keys ACL (sshd refuses loose perms) =="
icacls "$ssh\authorized_keys" /inheritance:r | Out-Null
icacls "$ssh\authorized_keys" /grant "${User}:R" "SYSTEM:F" "Administrators:F" | Out-Null

Write-Host "== 5/5 default shell = cmd (so the forced command runs) =="
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "$env:SystemRoot\System32\cmd.exe" -PropertyType String -Force | Out-Null
Restart-Service sshd

Write-Host ""
Write-Host "DONE. Test from the dev box:  ssh $User@<this-box-ip>   (it will reboot; nothing else)."
Write-Host "The fleet uses:  ./fleet-pxe to-ubuntu <mac>   (sets worker flag + reboots here)."

@echo off
REM build_win.bat -- build the native x86_64 Windows worker binary from the WSL-hosted workspace.
REM Uses the MSVC toolchain (vcvars64) so aws-lc-sys/ring compile natively; source over \\wsl.localhost,
REM target dir kept LOCAL (C:\zenbuild-target) so build output writes fast instead of over 9P.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CARGO_TARGET_DIR=C:\zenbuild-target"
set "CARGO_NET_GIT_FETCH_WITH_CLI=true"
set "SRC=\\wsl.localhost\Ubuntu-22.04\home\lilith\work\zen\zenmetrics\Cargo.toml"
echo === build start %DATE% %TIME% ===
"V:\packages\.cargo\bin\cargo.exe" build --release --target x86_64-pc-windows-msvc ^
  --manifest-path "%SRC%" ^
  -p zenmetrics-cli --no-default-features --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics
echo === build end rc=%ERRORLEVEL% %DATE% %TIME% ===

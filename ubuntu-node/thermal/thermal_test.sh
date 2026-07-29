#!/bin/bash
# Thermal/acoustic characterisation run for a fresh CPU cooler.
# Phases: idle baseline -> sustained all-core load (steady state) -> cooldown.
# Emits one TSV row every 2 s: temps, every fan header, avg clock, package watts, throttle count.
set -u
IDLE=${IDLE:-45}; LOAD=${LOAD:-420}; COOL=${COOL:-120}
OUT=${OUT:-$HOME/thermal-$(date +%Y%m%d-%H%M%S).tsv}

hw(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "$1" ] && echo "$h" && return; done; }
CORE=$(hw coretemp); NCT=$(hw nct6687)
[ -n "$CORE" ] || { echo "no coretemp hwmon"; exit 1; }

# Package temp = the coretemp label "Package id 0"; core temps = the rest.
PKG=""; for f in $CORE/temp*_label; do
  [ "$(cat $f)" = "Package id 0" ] && PKG="${f%_label}_input"
done
FANS=$(ls $NCT/fan*_input 2>/dev/null | sort -V)
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj

throttles(){ cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null | paste -sd+ | bc; }
mhz(){ awk '/cpu MHz/{s+=$4;n++}END{if(n)printf "%.0f",s/n}' /proc/cpuinfo; }
# energy_uj is root-only on modern kernels (the 2020 RAPL side-channel fix), hence sudo.
uj(){ sudo -n cat $RAPL 2>/dev/null || echo 0; }

{ printf 'elapsed_s\tphase\tpkg_c\tcore_max_c'
  for f in $FANS; do printf '\t%s' "$(basename ${f%_input})"; done
  printf '\tavg_mhz\tpkg_w\tthrottle_n\n'; } > "$OUT"

t0=$SECONDS; pe=$(uj); pt=$SECONDS
sample(){
  local ph=$1 el=$((SECONDS-t0)) pkg cmax now ne dt w
  pkg=$(( $(cat $PKG 2>/dev/null || echo 0) / 1000 ))
  cmax=0; for f in $CORE/temp*_input; do
    [ "$f" = "$PKG" ] && continue
    v=$(( $(cat $f 2>/dev/null || echo 0) / 1000 )); [ $v -gt $cmax ] && cmax=$v
  done
  ne=$(uj); now=$SECONDS; dt=$((now-pt))
  if [ "$dt" -gt 0 ] && [ "$ne" != 0 ]; then
    w=$(echo "scale=1;($ne-$pe)/1000000/$dt" | bc 2>/dev/null); pe=$ne; pt=$now
  else w=""; fi
  printf '%s\t%s\t%s\t%s' "$el" "$ph" "$pkg" "$cmax" >> "$OUT"
  for f in $FANS; do printf '\t%s' "$(cat $f 2>/dev/null || echo -)" >> "$OUT"; done
  printf '\t%s\t%s\t%s\n' "$(mhz)" "${w:--}" "$(throttles)" >> "$OUT"
}

echo "== phase 1/3: idle baseline ${IDLE}s"
end=$((SECONDS+IDLE)); while [ $SECONDS -lt $end ]; do sample idle; sleep 2; done

echo "== phase 2/3: ALL-CORE LOAD ${LOAD}s (stress-ng matrixprod, AVX-heavy)"
stress-ng --cpu 0 --cpu-method matrixprod --metrics-brief >/dev/null 2>&1 &
SP=$!
end=$((SECONDS+LOAD)); while [ $SECONDS -lt $end ]; do sample load; sleep 2; done
kill $SP 2>/dev/null; pkill -f stress-ng 2>/dev/null; sleep 1

echo "== phase 3/3: cooldown ${COOL}s"
end=$((SECONDS+COOL)); while [ $SECONDS -lt $end ]; do sample cool; sleep 2; done

echo "== done -> $OUT"

#!/bin/bash
# Same load as run1, but with the CPU fan PINNED to a fixed PWM, to measure what a quieter
# fan curve would actually cost in temperature. Measured, not extrapolated.
# Restores automatic fan control on every exit path.
set -u
PIN=${PIN:-64}; LOAD=${LOAD:-300}
OUT=${OUT:-$HOME/thermal-pinned-$PIN.tsv}
hw(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "$1" ] && echo "$h" && return; done; }
CORE=$(hw coretemp); NCT=$(hw nct6687)
PKG=""; for f in $CORE/temp*_label; do [ "$(cat $f)" = "Package id 0" ] && PKG="${f%_label}_input"; done
PE="$NCT/pwm1_enable"; PW="$NCT/pwm1"; RPM="$NCT/fan1_input"
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj

ORIG_E=$(cat "$PE"); ORIG_P=$(cat "$PW")
cleanup(){ pkill -f stress-ng 2>/dev/null
  echo "$ORIG_P" > "$PW" 2>/dev/null; echo "$ORIG_E" > "$PE" 2>/dev/null
  echo "-- restored auto (enable=$(cat $PE))"; }
trap cleanup EXIT INT TERM

echo 1 > "$PE"; echo "$PIN" > "$PW"; sleep 8
echo "pinned pwm=$PIN -> $(cat $RPM) RPM; starting ${LOAD}s load"
printf 'elapsed_s\tpkg_c\tfan1_rpm\tpkg_w\tthrottle_n\n' > "$OUT"
stress-ng --cpu 0 --cpu-method matrixprod >/dev/null 2>&1 &
t0=$SECONDS; pe=$(sudo -n cat $RAPL 2>/dev/null || echo 0); pt=$SECONDS
end=$((SECONDS+LOAD))
while [ $SECONDS -lt $end ]; do
  ne=$(sudo -n cat $RAPL 2>/dev/null || echo 0); now=$SECONDS; dt=$((now-pt)); w="-"
  [ "$dt" -gt 0 ] && { w=$(echo "scale=1;($ne-$pe)/1000000/$dt" | bc); pe=$ne; pt=$now; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$((SECONDS-t0))" \
    "$(( $(cat $PKG)/1000 ))" "$(cat $RPM)" "$w" \
    "$(cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null | paste -sd+ | bc)" >> "$OUT"
  sleep 3
done
echo "-- done -> $OUT"

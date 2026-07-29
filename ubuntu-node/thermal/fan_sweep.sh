#!/bin/bash
# Measure the CPU fan's ACTUAL RPM range by commanding PWM directly, so the fan-curve
# recommendation is a measurement rather than a guess at the fan's spec.
# Restores the original BIOS/auto control on every exit path (including Ctrl-C / error).
set -u
H=$(for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = nct6687 ] && echo "$h"; done | head -1)
[ -n "$H" ] || { echo "nct6687 hwmon not present"; exit 1; }
FAN=${FAN:-1}
PE="$H/pwm${FAN}_enable"; PW="$H/pwm${FAN}"; RPM="$H/fan${FAN}_input"
[ -w "$PW" ] || { echo "FAIL: $PW is not writable — driver has no pwm write support"; exit 1; }

ORIG_E=$(cat "$PE" 2>/dev/null || echo "")
ORIG_P=$(cat "$PW" 2>/dev/null || echo "")
restore(){
  echo "-- restoring automatic fan control (enable=$ORIG_E pwm=$ORIG_P)"
  [ -n "$ORIG_P" ] && echo "$ORIG_P" > "$PW" 2>/dev/null
  [ -n "$ORIG_E" ] && echo "$ORIG_E" > "$PE" 2>/dev/null
  sleep 5
  echo "-- fan now: $(cat $RPM) RPM (enable=$(cat $PE 2>/dev/null))"
}
trap restore EXIT INT TERM

echo "fan${FAN}: baseline $(cat $RPM) RPM (enable=$ORIG_E pwm=$ORIG_P)"
echo 1 > "$PE" 2>/dev/null || { echo "cannot set manual mode"; exit 1; }
printf '%s\t%s\t%s\n' pwm pct rpm
for p in 64 96 128 160 192 224 255; do
  echo "$p" > "$PW" || { echo "write $p failed"; break; }
  sleep 9                      # fans need several seconds to settle at a new duty cycle
  printf '%s\t%s\t%s\n' "$p" "$((p*100/255))" "$(cat $RPM)"
done

#!/usr/bin/env bash
# Per-call HDR-VDP timing via batch: (T(1+K rows) - T(1 row)) / K; the extra rows' PNG decode is included.
cd ~/tmp/iqa-cx-lead/hdr
B=~/tmp/iqa-cx-lead/bin/zenmetrics-nn; O=~/tmp/iqa-cx-lead/perf/i265_hdr_1024.tsv
printf 'ref_path\tdist_path\n' > p1.tsv; printf '%s\t%s\n' $PWD/ref_1024x1024.png $PWD/dist_1024x1024.png >> p1.tsv
cp p1.tsv p4.tsv; for i in 1 2 3; do tail -1 p1.tsv >> p4.tsv; done
echo -e "metric\tshape\tthreads\trepeat\textra_reps\ttask_clock_ms_per_call\twall_ms_per_call\tscore" > $O
for m in hdrvdp hdrvdp3; do
 extra=""; [ $m = hdrvdp3 ] && extra="--hdrvdp3-ppd 30"
 for spec in "1 6" "8 0-7"; do set -- $spec; th=$1; cpus=$2
  for rep in 0 1 2; do
   r=(); for f in p1 p4; do
     s=$(date +%s.%N); tc=$(RAYON_NUM_THREADS=$th perf stat -x, -e task-clock taskset -c $cpus $B batch --hdr --metric $m $extra --pairs $f.tsv --output out_$f.tsv --jobs 1 2>&1 >/dev/null | awk -F, '/task-clock/{print $1}'); e=$(date +%s.%N)
     r+=("$tc" "$(echo "($e-$s)*1000" | bc -l)"); done
   tc=$(echo "(${r[2]}-${r[0]})/3" | bc -l); wl=$(echo "(${r[3]}-${r[1]})/3" | bc -l)
   echo -e "$m\t1024x1024\t$th\t$rep\t3\t$tc\t$wl\t$(tail -1 out_p1.tsv | awk '{print $NF}')" >> $O
 done; done; done
cat $O

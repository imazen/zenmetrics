#!/usr/bin/env bash
# Parse every heaptrack file in benchmarks/heaptrack/ and emit a TSV with
# the headline stats + the top allocator function (skipping libcore /
# std::rt frames) for each cell.
#
# Output columns (tab-separated):
#   metric  mode  size_label  total_runtime_s  n_alloc  n_temp_alloc
#   peak_heap  peak_rss  leaked  top1_calls  top1_peak  top1_caller
#   top2_calls  top2_peak  top2_caller  top3_calls  top3_peak  top3_caller
#
# Usage: parse_heaptracks.sh > stats.tsv

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Filter a backtrace block for the first "interesting" frame.
# Skip:
#   - rustc-libcore (alloc::raw_vec::*, library/core/*, library/alloc/*)
#   - heaptrack hooks
#   - std::rt / std::sys / main / __rust_begin_short_backtrace
# Pick the first cvvdp::/iwssim::/zensim::/fast_ssim2::/dssim_core::/
#   butteraugli::/cpu_profile:: frame.
pick_caller_frame() {
    # stdin: backtrace lines for one allocation group
    awk '
        /alloc::raw_vec/ { next }
        /library\/(alloc|core|std)\// { next }
        /^[[:space:]]*at \/rustc\// { next }
        /heaptrack/ { next }
        /__rust_begin_short_backtrace/ { next }
        /^[[:space:]]*in / { next }
        /^[[:space:]]*main$/ { next }
        /^[[:space:]]*main / { next }
        /lang_start/ { next }
        /^[[:space:]]*at .*\/rt\.rs/ { next }
        /^[[:space:]]*$/ { next }
        # Function-name line — typically "    cvvdp::module::fn::HASH" or similar
        /^[[:space:]]+[a-z][a-zA-Z0-9_:]+::/ {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            # Drop the trailing hash like ::h0123abcd...
            sub(/::h[0-9a-f]+$/, "", line)
            # If still long, take the first 80 chars.
            if (length(line) > 80) line = substr(line, 1, 80) "..."
            print line
            exit
        }
    '
}

parse_one() {
    local file="$1"
    local base
    base=$(basename "$file" .zst)
    local label="${base##*_}"
    local rest="${base%_*}"
    local metric="${rest%%_*}"
    local mode="${rest#*_}"

    local tmp top_tmp
    tmp=$(mktemp)
    top_tmp=$(mktemp)
    heaptrack_print "$file" 2>/dev/null > "$tmp" || { rm -f "$tmp" "$top_tmp"; return 1; }

    local runtime alloc_n temp_n peak_heap peak_rss leaked
    runtime=$(grep "^total runtime:" "$tmp" | sed -E 's/.*: ([0-9.]+)s.*/\1/')
    alloc_n=$(grep "^calls to allocation functions:" "$tmp" | sed -E 's/.*: ([0-9]+) .*/\1/')
    temp_n=$(grep "^temporary memory allocations:" "$tmp" | sed -E 's/.*: ([0-9]+) .*/\1/')
    peak_heap=$(grep "^peak heap memory consumption:" "$tmp" | sed -E 's/.*: (.*)$/\1/')
    peak_rss=$(grep "^peak RSS" "$tmp" | sed -E 's/.*: (.*)$/\1/')
    leaked=$(grep "^total memory leaked:" "$tmp" | sed -E 's/.*: (.*)$/\1/')

    # Walk "MOST CALLS TO ALLOCATION FUNCTIONS" groups. Each group:
    #   "N calls with PEAK from:"  followed by a backtrace block ending
    #   at `main` or next blank.
    # Extract the top 3 distinct picked-caller-frame entries with their
    # call counts + peak sizes.
    awk '
        /^MOST CALLS TO ALLOCATION FUNCTIONS$/ { in_sec=1; next }
        /^MOST TEMPORARY/ { exit }
        in_sec && /^[0-9]+ calls with .* peak consumption from:/ {
            # New group.
            sub(/^[0-9]+ calls with /, "", group_header=$0)
            n = $1
            # Capture "P_unit"
            match($0, /with ([^ ]+) peak/, arr)
            peak = arr[1]
            calls = n
            in_grp = 1
            # Print a marker so the downstream pick script can pair this
            # group header with the chosen frame.
            print "__GROUP__\t" calls "\t" peak
            next
        }
        in_sec && in_grp { print }
    ' "$tmp" > "$top_tmp"

    # Now parse top_tmp: for each __GROUP__\tcalls\tpeak line, the
    # following lines (until next __GROUP__) form the backtrace. Run
    # pick_caller_frame on them and emit (calls, peak, caller).
    local groups
    groups=$(awk '
        BEGIN { calls=""; peak=""; bt="" }
        /^__GROUP__/ {
            if (calls != "") { print calls "\t" peak "\t<<<"; print bt; print ">>>" }
            split($0, a, "\t")
            calls=a[2]; peak=a[3]; bt=""
            next
        }
        { bt = bt $0 "\n" }
        END {
            if (calls != "") { print calls "\t" peak "\t<<<"; print bt; print ">>>" }
        }
    ' "$top_tmp" | awk -v RS="\n>>>\n" '
        # one record per group; first line is "calls\tpeak\t<<<"
        NF > 0 {
            n = split($0, lines, "\n")
            header = lines[1]
            sub(/\t<<<$/, "", header)
            # Build bt = lines[2..n] joined with \n.
            bt = ""
            for (i = 2; i <= n; i++) bt = bt lines[i] "\n"
            print header
            print bt
            print "==="
        }
    ')

    # Now group has alternating header / bt blocks delimited by "===".
    # Pipe each bt through pick_caller_frame, dedupe by caller, keep top 3.
    local top1_c top1_p top1_f top2_c top2_p top2_f top3_c top3_p top3_f
    local seen_callers="" picked=0
    # Use a tmp file because we need stable state across pipe.
    local picks
    picks=$(echo "$groups" | awk -v RS="===\n" '
        NF > 0 {
            # First newline-delimited line is the header (calls\tpeak)
            n = split($0, parts, "\n")
            print parts[1]      # calls\tpeak
            # The remainder is the bt
            for (i = 2; i <= n; i++) print parts[i]
            print "###END"
        }
    ' | awk '
        # When we see ###END, run pick on the buffered bt
        BEGIN { collecting=0; header=""; bt="" }
        /^###END$/ {
            # emit a single record: header + bt
            print "RECORD"
            print header
            print bt
            print "ENDRECORD"
            collecting=0; header=""; bt=""
            next
        }
        !collecting && /^[0-9]+\t/ {
            header = $0
            collecting = 1
            next
        }
        collecting { bt = bt $0 "\n" }
    ')

    # Final extraction
    local out
    out=$(echo "$picks" | awk '
        /^RECORD$/ { in_r=1; header=""; bt=""; next }
        /^ENDRECORD$/ { in_r=0; print header "\t" bt "@@SEP@@"; next }
        in_r && header == "" { header = $0; next }
        in_r { bt = bt $0 "\n" }
    ' | tr '\n' '\1')

    # Now `out` is "header1\tbt1@@SEP@@\1header2\tbt2@@SEP@@\1..." We
    # iterate (split on @@SEP@@) and pick first interesting frame per
    # group, dedupe.
    local results=""
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local header bt
        header="${line%%	*}"  # first tab-delimited field
        bt="${line#*	}"
        bt="${bt%%@@SEP@@*}"
        local frame
        frame=$(printf '%s\n' "$bt" | tr '\1' '\n' | pick_caller_frame)
        [[ -z "$frame" ]] && continue
        # Dedupe by frame name
        case ";$seen_callers;" in
            *";${frame};"*) continue;;
        esac
        seen_callers="${seen_callers};${frame}"
        local calls peak
        calls="${header%%	*}"
        peak="${header##*	}"
        if [[ $picked -lt 3 ]]; then
            results+="${calls}\t${peak}\t${frame}\t"
            picked=$((picked + 1))
        fi
    done < <(printf '%s\n' "$out" | tr '\1' '\n')

    # Pad to 3 entries
    while [[ $picked -lt 3 ]]; do
        results+="\t\t\t"
        picked=$((picked + 1))
    done

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' \
        "$metric" "$mode" "$label" \
        "${runtime:-NA}" "${alloc_n:-NA}" "${temp_n:-NA}" \
        "${peak_heap:-NA}" "${peak_rss:-NA}" "${leaked:-NA}"
    printf '%b\n' "$results"

    rm -f "$tmp" "$top_tmp"
}

echo -e "metric\tmode\tsize_label\ttotal_runtime_s\tn_alloc\tn_temp_alloc\tpeak_heap\tpeak_rss\tleaked\ttop1_calls\ttop1_peak\ttop1_caller\ttop2_calls\ttop2_peak\ttop2_caller\ttop3_calls\ttop3_peak\ttop3_caller"
for f in "$DIR"/*.zst; do
    [[ -f "$f" ]] || continue
    parse_one "$f"
done

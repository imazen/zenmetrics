#!/usr/bin/env bash
# Test executor for the spot-reclaim demo: sleeps (so SIGTERM can land mid-job), then echoes the
# job JSON on stdin to stdout as the "score" blob. SLOW_SECS controls the sleep (default 8).
sleep "${SLOW_SECS:-8}"
cat

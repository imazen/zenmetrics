# Copy this file as justfile into a frozen portable bundle.
# Python and just are the only separately installed runtime tools.
run output:
    python3 mac_service.py install . '{{output}}' --run-heavy tools/run-heavy --lock "$HOME/tmp/mac-heavy.lock"

status output:
    python3 mac_service.py status . '{{output}}'
    python3 durable.py status manifest.json '{{output}}'

stop output:
    python3 mac_service.py stop . '{{output}}'

resume output:
    python3 mac_service.py start . '{{output}}'

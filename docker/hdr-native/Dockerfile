# Native extractors are built/tested before this image; no runtime installs.
ARG BASE=ghcr.io/imazen/zenfleet-worker@sha256:a890b058b84e081bcdcd92d6c00972c8c1d9490ced1f3ca3c55f146d5ecee424
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 time procps && rm -rf /var/lib/apt/lists/*
COPY --chmod=0755 hdr_features_exec.py /usr/local/bin/hdr-features-exec
COPY --chmod=0755 hdr944_extract /opt/hdr-native/bin/hdr944_extract
COPY --chmod=0755 upiq_pu_score /opt/hdr-native/bin/upiq_pu_score
ENV ZEN_EXEC=/usr/local/bin/hdr-features-exec ZEN_HDR_BIN_DIR=/opt/hdr-native/bin
ENV ZEN_PERSISTENT_EXEC=1 ZEN_PERSISTENT_KINDS=bake
ENV ZEN_HDR_CAPABILITIES=hdr-native-rev7-features-v1,hdr-upiq380-rev7-features-v1

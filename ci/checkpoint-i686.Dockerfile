FROM ghcr.io/cross-rs/i686-unknown-linux-gnu@sha256:2647400f884dd399a3ce6f6d918fec0776d43e73f471f52e310cbe664ac1671f
ENV TMPDIR=/root/tmp
RUN mkdir -p "$TMPDIR" && apt-get update && apt-get install --no-install-recommends -y python3

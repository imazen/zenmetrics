FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openssh-server mosh tmux locales curl ca-certificates gawk less nano htop \
 && rm -rf /var/lib/apt/lists/* \
 && locale-gen en_US.UTF-8 \
 && mkdir -p /run/sshd \
 && ssh-keygen -A
RUN curl -fsSL https://herdr.dev/install.sh | HERDR_INSTALL_DIR=/usr/local/bin sh
COPY sshd_config /etc/ssh/sshd_config
COPY tmux.conf /etc/tmux.conf
COPY --chmod=0755 hostshell tmux-host /usr/local/bin/
RUN echo /usr/local/bin/tmux-host >> /etc/shells \
 && usermod -s /usr/local/bin/tmux-host root
EXPOSE 2223/tcp 60000-61000/udp
CMD ["/usr/sbin/sshd", "-D", "-e"]

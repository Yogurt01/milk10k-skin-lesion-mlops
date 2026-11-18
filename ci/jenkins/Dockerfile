# Jenkins controller for the MILK10k pipeline (P1-CI-01; decisions CI-D1, CI-D9..CI-D11). See ci/README.md.
# Base: current Jenkins LTS on JDK 21, pinned by tag and digest (checked 2026-11-19: 2.580.2, Java 21 supported).
FROM jenkins/jenkins:2.580.2-lts-jdk21@sha256:74401de6a6f92e1afe1b02d1c8e3d0c33f263bfb5275dd8813f9150355f2367b

# uv (pinned by tag and digest): the pipeline builds the serving venv with `make setup`.
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/

USER root
# Docker CLI and Compose plugin from Docker's apt repository, the same versions as the host daemon (29.9.3 / v5.7.1);
# the repository key is checked against Docker's published fingerprint before use. `make`: the pipeline reuses the
# repository's Makefile (the base image has none). git and curl come with the base image. No Docker engine.
ARG DOCKER_CLI_VERSION=5:29.9.3-1~debian.13~trixie
ARG COMPOSE_PLUGIN_VERSION=5.7.1-1~debian.13~trixie
ARG DOCKER_KEY_FINGERPRINT=9DC858229FC7DD38854AE2D88D81803C0EBFCD88
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gnupg make; \
    install -m 0755 -d /etc/apt/keyrings; \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc; \
    fpr="$(gpg --show-keys --with-colons /etc/apt/keyrings/docker.asc | awk -F: '/^fpr/ {print $10; exit}')"; \
    test "$fpr" = "$DOCKER_KEY_FINGERPRINT"; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable" \
      > /etc/apt/sources.list.d/docker.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      "docker-ce-cli=${DOCKER_CLI_VERSION}" "docker-compose-plugin=${COMPOSE_PLUGIN_VERSION}"; \
    apt-get purge -y --auto-remove gnupg; \
    rm -rf /var/lib/apt/lists/*

USER jenkins
# Plugins at exact versions, dependencies included (ci/jenkins/plugins.txt); --latest false keeps them as listed.
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt --latest false

# Configuration as code: system settings, admin user, credentials and the pipeline job (Job DSL).
COPY --chown=jenkins:jenkins casc.yaml jobs.groovy /usr/share/jenkins/casc/
ENV CASC_JENKINS_CONFIG=/usr/share/jenkins/casc/casc.yaml

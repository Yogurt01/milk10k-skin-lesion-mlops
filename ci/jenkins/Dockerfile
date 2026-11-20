# Jenkins controller for the MILK10k pipeline (P1-CI-01; decisions CI-D1, CI-D9..CI-D11). See ci/README.md.
# Also kubectl and Helm for the deploy to the local Kubernetes cluster.
# Base: current Jenkins LTS on JDK 21, pinned by tag and digest (checked 2026-11-19: 2.580.2, Java 21 supported).
FROM jenkins/jenkins:2.580.2-lts-jdk21@sha256:74401de6a6f92e1afe1b02d1c8e3d0c33f263bfb5275dd8813f9150355f2367b

# uv (pinned by tag and digest): the pipeline builds the serving venv with `make setup`.
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/

USER root
# Docker CLI, Buildx and Compose plugins from Docker's apt repository, the same versions as the host (29.9.3 / 0.39.0 /
# v5.7.1; Buildx so `docker build` uses BuildKit and its cache, not the deprecated legacy builder);
# the repository key is checked against Docker's published fingerprint before use. `make`: the pipeline reuses the
# repository's Makefile (the base image has none). git and curl come with the base image. No Docker engine.
ARG DOCKER_CLI_VERSION=5:29.9.3-1~debian.13~trixie
ARG COMPOSE_PLUGIN_VERSION=5.7.1-1~debian.13~trixie
ARG BUILDX_PLUGIN_VERSION=0.39.0-1~debian.13~trixie
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
      "docker-ce-cli=${DOCKER_CLI_VERSION}" "docker-compose-plugin=${COMPOSE_PLUGIN_VERSION}" \
      "docker-buildx-plugin=${BUILDX_PLUGIN_VERSION}"; \
    apt-get purge -y --auto-remove gnupg; \
    rm -rf /var/lib/apt/lists/*

# kubectl and Helm for the Kubernetes stage (static binaries; version and published sha256 pinned, checked here).
# kubectl: latest patch of the cluster's minor version (cluster v1.37.0). Helm: the same version as on the host.
ARG KUBECTL_VERSION=v1.37.2
ARG KUBECTL_SHA256=c04487523e7aa7d1493f7808174c899db93d7a9cf57531420cd76b9bd415f8f4
ARG HELM_VERSION=v4.3.2
ARG HELM_SHA256=d1c0278d46beab79582d7e50f5b3dc206944ba8e577cde618513fd07daa53983
RUN set -eux; \
    tmp="$(mktemp -d)"; \
    curl -fsSLo "$tmp/kubectl" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"; \
    echo "${KUBECTL_SHA256}  $tmp/kubectl" | sha256sum -c -; \
    install -m 0755 "$tmp/kubectl" /usr/local/bin/kubectl; \
    curl -fsSLo "$tmp/helm.tar.gz" "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz"; \
    echo "${HELM_SHA256}  $tmp/helm.tar.gz" | sha256sum -c -; \
    tar -xzf "$tmp/helm.tar.gz" -C "$tmp" linux-amd64/helm; \
    install -m 0755 "$tmp/linux-amd64/helm" /usr/local/bin/helm; \
    rm -rf "$tmp"

USER jenkins
# Plugins at exact versions, dependencies included (ci/jenkins/plugins.txt); --latest false keeps them as listed.
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt --latest false

# Configuration as code: system settings, admin user, credentials and the pipeline job (Job DSL).
COPY --chown=jenkins:jenkins casc.yaml jobs.groovy /usr/share/jenkins/casc/
ENV CASC_JENKINS_CONFIG=/usr/share/jenkins/casc/casc.yaml

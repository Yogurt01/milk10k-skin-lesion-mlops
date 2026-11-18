# ci/: Jenkins for the MILK10k pipeline

A local Jenkins controller in its own Compose project (`milk10k-ci`), configured entirely as code, which runs the
repository's [`Jenkinsfile`](../Jenkinsfile) from `main`. It is started on demand and is not part of `make up`.
Research model, not a medical device.

| Path | What |
|---|---|
| `compose.yaml` | The `jenkins` service: image, user, mounts, secrets, limits, UI on 127.0.0.1:8081 |
| `jenkins/Dockerfile` | `jenkins/jenkins:2.580.2-lts-jdk21` (pinned by digest) + Docker CLI 29.9.3 + Compose plugin 5.7.1 (Docker's apt repository, key fingerprint checked) + `make` + uv 0.12.5; nothing else |
| `jenkins/plugins.txt` | 67 plugins at exact versions: 12 chosen (each with its reason) + their dependencies |
| `jenkins/casc.yaml` | Configuration as Code: 1 executor, no inbound agent port, admin account, no anonymous access, the Grafana credential, the job |
| `jenkins/jobs.groovy` | Job DSL: the pipeline job `milk10k` (repository over public HTTPS, branch `main`, pollSCM `H/5 * * * *`) |
| `make_secrets.py` | `make ci-secrets`: creates `ci/secrets/` (gitignored, mode 600) |
| `.env.example` | Template for `ci/.env` (gitignored): dataset path, port, memory limit |

## Set up and start (a fresh Jenkins from nothing)

1. Docker running, your user in group `docker`; the repository's `.env` (`make env` creates it).
2. The model package in `artifacts/model_package/final-P1a-kg-onnx` (`make fetch-model`).
3. `cp ci/.env.example ci/.env` and set `CI_DATASET_DIR` to the prepared MILK10k dataset folder
   (`test_metadata.csv`, `test/derm/`, `test/clinical/`).
4. `make ci-up`. This target:
   - writes the secrets: a random admin password created once, and the Grafana password copied from `.env`;
   - creates `~/milk10k-ci/jenkins_home` and `~/milk10k-ci/model` (`CI_HOME=~/milk10k-ci` by default);
   - hard-links the model package into `~/milk10k-ci/model/final-P1a-kg-onnx` (`cp -al`: no extra disk; it needs
     the same filesystem) and verifies it with `scripts/fetch_model.sh --check`;
   - builds the image, starts Jenkins and waits until it is healthy.
5. Open http://127.0.0.1:8081 and log in as `admin` with the password in `ci/secrets/jenkins_admin_password`. Read
   the file yourself; it is never printed.

The job `milk10k` exists right after the start. There is no setup wizard and nothing to click.

**First build:** a Pipeline job that has never run builds on its first poll, within 5 minutes of the start
(workflow-job's poll logs "no previous build to compare to" and builds). After that, it builds only when `main` gets
a new commit.

To start from scratch, run `make ci-down`, then delete `~/milk10k-ci/jenkins_home` (build history and the uv cache)
and `make ci-up` again. `make ci-down` alone keeps everything under `~/milk10k-ci`.

## Design (decisions CI-D1..CI-D15 in the project's decision log)

- **Docker socket: root-equivalent on the host.**
  - Jenkins runs `docker` against the host daemon through `/var/run/docker.sock`. Anyone who can change what a build
    runs (the Jenkinsfile, the Makefile, tests on `main`) can start a privileged container that mounts `/`.
  - A `:ro` socket mount is **not** a security boundary. Tested 2026-11-19: with the socket mounted read-only,
    `docker volume create` still succeeded. The socket is therefore mounted plainly.
  - Mitigations:
    - Jenkins is not privileged, runs as the non-root `jenkins` user (UID 1000) plus the socket's group (GID read
      with `stat` by `make ci-up`), with no Linux capabilities and `no-new-privileges`.
    - The UI is on 127.0.0.1 only, with no inbound agent port and no anonymous access.
    - The job builds `main` of this repository only, never pull requests or forks.
- **JENKINS_HOME at the same absolute path inside and outside.** The host daemon resolves bind mounts of anything
  that Compose starts from a workspace, so workspace paths must exist on the host. Tested: a file written in a
  workspace inside the container appears at the same host path, and a Compose bind mount from there works.
- **Model package and dataset are mounted read-only at their host paths.** The dataset is only for
  `make_sample.py --data-dir`. Its variable is `CI_DATASET_DIR`, deliberately not `MILK10K_DATA_DIR`, which would
  start the 479-lesion parity test.
- **Tests run inside this container with uv** (`make setup`, `make test-report`). The uv cache and the uv-managed
  Python 3.12 live under `JENKINS_HOME/caches/`, so packages are downloaded once. The image has no `python3` of its
  own; scripts that need one (e.g. `fetch_model.sh`) run after `make setup` with `serving/.venv/bin` on `PATH`.
- **Reaching the API** (for the deploy checks, M7b-2):
  - The deploy stage attaches this container to the stack's network after `docker compose up`
    (`docker network connect milk10k_default "$HOSTNAME"`, skipped if it is already attached), calls
    `http://api:8000`, and detaches in `post { always }`.
  - Rejected:
    - Declaring the network `external` in this file: Jenkins cannot start before the stack exists.
    - Declaring it with the same name but not external: whichever project starts first owns it. The stack then
      warns on every `up`, and `down` of either project leaves the network behind.
- **Resources:**
  - Memory limit 1.5 GiB with no extra swap (`memswap_limit` = limit), Java heap 768 MiB, 1 executor.
  - Measured idle: about 540-565 MiB (`docker stats`).
- **Known limitations:**
  - A failed deploy is rolled back by hand: `API_IMAGE_TAG=<last good tag> docker compose up -d --wait`.
  - Jenkins does not restart by itself after a reboot or a crash (`restart: "no"`, started on demand), and it only
    polls while it runs.

## Updating versions

- **Jenkins:** change the tag **and** digest in `jenkins/Dockerfile`; check Java support on
  https://www.jenkins.io/doc/book/platform-information/support-policy-java/.
- **Plugins:** resolve again with `jenkins-plugin-cli --plugins <the 12 chosen names> --list` in the new image,
  replace the versions in `plugins.txt`, then `make ci-up`.

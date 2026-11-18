# ci/: Jenkins for the MILK10k pipeline

A local Jenkins controller in its own Compose project (`milk10k-ci`), configured entirely as code, which runs the
repository's [`Jenkinsfile`](../Jenkinsfile) from `main`: **test → build → deploy → smoke → parity**. It is started
on demand and is not part of `make up`. Research model, not a medical device.

| Path | What |
|---|---|
| `compose.yaml` | The `jenkins` service: image, user, mounts, secrets, limits, UI on 127.0.0.1:8081 |
| `jenkins/Dockerfile` | `jenkins/jenkins:2.580.2-lts-jdk21` (pinned by digest) plus, from Docker's apt repository (key fingerprint checked), Docker CLI 29.9.3, Buildx 0.39.0 and Compose 5.7.1; plus `make` and uv 0.12.5. Nothing else (no python3, kubectl or Helm) |
| `jenkins/plugins.txt` | 69 plugins at exact versions: 13 chosen (each with its reason) and their dependencies |
| `jenkins/casc.yaml` | Configuration as Code: 1 executor, no inbound agent port, admin account, no anonymous access, the Grafana credential, the job |
| `jenkins/jobs.groovy` | Job DSL: the pipeline job `milk10k` (repository over public HTTPS, branch `main`, pollSCM `H/5 * * * *`) |
| `make_secrets.py` | `make ci-secrets`: creates `ci/secrets/` (gitignored, mode 600) |
| `.env.example` | Template for `ci/.env` (gitignored): dataset path, port, memory limit |

## Set up and start (also: from an empty JENKINS_HOME)

1. Docker is running and your user is in group `docker`. The repository's `.env` exists (`make env` creates it).
2. The model package is in `artifacts/model_package/final-P1a-kg-onnx` (`make fetch-model`).
3. `cp ci/.env.example ci/.env` and set `CI_DATASET_DIR` to the prepared MILK10k dataset folder. It must contain
   `test_metadata.csv`, `test/derm/` and `test/clinical/`; e.g. `…/dataset/raw`.
4. `make ci-up`. This target:
   - writes the secrets: a random admin password once, and the Grafana password copied from `.env`;
   - creates `~/milk10k-ci/jenkins_home` and `~/milk10k-ci/model` (`CI_HOME=~/milk10k-ci` by default);
   - hard-links the model package into `~/milk10k-ci/model/final-P1a-kg-onnx` (`cp -al`: no extra disk, same
     filesystem needed) and verifies it with `scripts/fetch_model.sh --check`;
   - builds the image, starts Jenkins and waits until it is healthy.
5. Open http://127.0.0.1:8081 and log in as `admin` with the password in `ci/secrets/jenkins_admin_password`. Read
   the file yourself; it is never printed.

The job `milk10k` exists right after the start. There is no setup wizard and nothing to click.

**To start from an empty JENKINS_HOME:**
- `make ci-down`, delete `~/milk10k-ci/jenkins_home`, then `make ci-up`.
- That removes the build history and the uv cache: the next build downloads Python and the packages again (about 70 s
  instead of about 30 s for the tests).
- `make ci-down` alone keeps everything under `~/milk10k-ci`.

## Seeing a build

- **Job page:** http://127.0.0.1:8081/job/milk10k/ lists the builds and their causes ("Started by an SCM change" for
  polling).
- **Each build:**
  - *Pipeline Overview* (plugin `pipeline-graph-view`): the stages, with the parallel Unit tests ‖ Lint ‖ Hygiene.
  - *Test Result*: the JUnit reports of the unit tests and the parity test.
  - *Console Output*: timestamped.
  - *Build Artifacts*: `reports/*.xml` (JUnit, coverage) and `reports/*.log` (smoke test, deployed containers, the
    API's last 300 log lines).
- **Sample lesion images** are dataset images: they are deleted after every build and are never archived.

## The pipeline

| Stage | Command | Fails when |
|---|---|---|
| Setup | `make setup` (serving venv; `UV_VENV_CLEAR=1`, so fresh every build) | install fails |
| Test (parallel) | `make test-report` ‖ `make lint` ‖ `make hygiene` | a test fails, coverage < 90 %, ruff finds something, data/secrets in git |
| Build | `docker build -t milk10k-api:<build>-<sha7> serving` (no registry) | build error |
| Deploy | `fetch_model.sh --check` on the CI model copy, `docker compose up -d --no-build --wait` (Grafana password from the credential), attach to the stack network, wait for `/ready` | package changed, a service not healthy, `/ready` not 200 within 120 s |
| Smoke | `make smoke` (`scripts/smoke_test.py` against `http://api:8000`) | any smoke check |
| Parity | `make sample` (6 lesions from `CI_DATASET_DIR`), `REQUIRE_MODEL_PACKAGE=1 make test-container` (tolerance 0.002; `MILK10K_DATA_DIR` unset; the 479-lesion test deselected) | a lesion > 0.002, missing sample files |
| post (always) | detach from the stack network, keep the API's log, delete the sample files, remove `milk10k-api:<build>-<sha7>` tags older than the newest 3, archive reports | — |

**Python:** `make setup` builds the serving venv with uv-managed CPython 3.12. The Jenkins image has no `python3`;
the Jenkinsfile puts `serving/.venv/bin` first on `PATH`, so every `python3` in the pipeline (`fetch_model.sh`, smoke
test, hygiene) is that venv's. The versions differ by design (decision CI-D17):

| Where | Python |
|---|---|
| CI tests | CPython 3.12.14 (uv-managed) |
| API image | 3.12.16 |
| Host serving venv | 3.12.3 |

Parity runs over HTTP against the deployed image, so it tests the Python that actually serves.

## Design and security (decisions CI-D1..CI-D18 in the project's decision log)

- **Docker socket: root-equivalent on the host.**
  - Jenkins runs `docker` against the host daemon through `/var/run/docker.sock`. Anyone who can change what a build
    runs (the Jenkinsfile, the Makefile, tests on `main`) can start a privileged container that mounts `/`.
  - A `:ro` socket mount is **not** a security boundary. Tested 2026-11-19: with the socket mounted read-only,
    `docker volume create` still succeeded. It is therefore mounted plainly.
  - Mitigations:
    - Jenkins is not privileged and runs as the non-root `jenkins` user (UID 1000) plus the socket's group (GID read
      with `stat` by `make ci-up`), with no Linux capabilities and `no-new-privileges`.
    - The UI is on 127.0.0.1 only, with no inbound agent port and no anonymous access.
    - The job builds `main` of this repository only, never pull requests or forks.
  - The pipeline's Docker commands touch only `milk10k*` resources. Never add `docker system prune` or an unfiltered
    prune here: it would act on the whole host.
- **JENKINS_HOME at the same absolute path inside and outside.** The host daemon resolves bind mounts of anything
  Compose starts from a workspace, so workspace paths must exist on the host. Tested: a file written in a workspace
  inside the container appears at the same host path, and a Compose bind mount from there works.
- **Read-only mounts:**
  - The model package is mounted read-only at its host path.
  - The dataset is too, and is used only by `make_sample.py --data-dir`. Its variable is `CI_DATASET_DIR`,
    deliberately not `MILK10K_DATA_DIR`, which would start the 479-lesion parity test.
- **Network** (CI-D13): the deploy stage attaches this container to `milk10k_default` after
  `docker compose up` and calls `http://api:8000`; the pipeline's `post { always }` detaches it, also after a
  failure.
  - Rejected:
    - The network declared `external`: Jenkins cannot start before the stack exists.
    - The same name declared non-external: whichever project starts first owns it, and either `down` leaves it
      behind.
- **Resources:** memory limit 2 GiB with no extra swap (`memswap_limit` = limit), Java heap 768 MiB, 1 executor.
  The measured figures are in the project's checklist (P1-CON-05).

## Known pitfalls and limitations

- **Polling stays silent after a build that failed before checkout.**
  - A Pipeline job polls only the repositories its *last* build checked out.
  - A job that has never built builds on its first poll ("no previous build to compare to").
  - But if that build fails while loading the Jenkinsfile, e.g. the branch has no Jenkinsfile yet, it records no
    checkout. Every later poll then logs "Took 0 ms / No changes", and new commits are never built.
  - Fix:
    1. Make sure the branch has a Jenkinsfile.
    2. Delete the failed build (build page → *Delete build*), or start one build by hand.
    3. The next poll, or that build, records the checkout, and polling works from then on.
  - Check the job's *Git Polling Log*: "Took 0 ms" means nothing was compared.
- **No automatic rollback** (CI-D7). After a failed deploy, run from the repository:
  `API_IMAGE_TAG=<last good build>-<sha7> docker compose up -d --no-build --wait`. The pipeline keeps the newest 3
  image tags (`docker image ls milk10k-api`).
- **The deploy replaces the local stack.** Compose project `milk10k`, the same ports and volumes as `make up`, so
  the deployed stack is the one the README's URLs show.
- **Restarts:** Jenkins does not restart after a reboot or a crash (`restart: "no"`, started on demand), and it only
  polls while it runs.
- **Log warning on every start:** `User$AllUsers#scanAll: entry for admin … duplicates one seen earlier`. Cause,
  from the Jenkins and plugin code:
  1. JCasC (`ConfigurationAsCode.init`, after `SYSTEM_CONFIG_LOADED`) creates or updates the `admin` user from
     `securityRealm.local.users` in `casc.yaml` and saves it.
  2. Core's `User.AllUsers.scanAll` (after `JOB_CONFIG_ADAPTED`, later in startup) then reads the same
     `users/admin_<hash>` folder.
  3. It finds `admin` already in its map and logs the warning, replacing the entry with the copy just read from disk.
  - Effect checked: one user folder; login with the password from the secret file works after every start.
- **The image's anonymous volume at `/var/jenkins_home`** (the base image's `VOLUME`; `HOME` of the `jenkins` user)
  holds caches outside JENKINS_HOME:
  - the plugin-manager cache from the image build;
  - fontconfig;
  - `copy_reference_file.log`;
  - onnxruntime's `.cache/Microsoft/DeveloperTools/.onnxruntime/` files from the test runs.
  - `make ci-down` removes it (`down -v`).

## Updating versions

- **Jenkins:** change the tag **and** digest in `jenkins/Dockerfile`; check Java support on
  https://www.jenkins.io/doc/book/platform-information/support-policy-java/.
- **Plugins:** resolve again with `jenkins-plugin-cli --plugins <the chosen names> --list` in the new image, replace
  the versions in `plugins.txt`, then `make ci-up`.

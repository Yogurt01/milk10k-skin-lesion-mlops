# Installing the prerequisites on Ubuntu

> **Research model, not a medical device.** Its output is not a diagnosis and must not be used for clinical decisions.

What a machine needs before `make up` (see the [README](../README.md#install-and-run-on-ubuntu)). Written for
Ubuntu 26.04 LTS. The Docker step reads the release name from `/etc/os-release`, so it also picks the right repository
on other Ubuntu releases; only 26.04 was checked.

Commands that start with `sudo` change the system and ask for your password. Run each block once, as the user who will
run the project.

## Reference machine

The machine every number in this repository was measured on (output of the commands, 2026-11-19):

| Command | Value |
|---|---|
| `lsb_release -a` | Ubuntu 26.04.1 LTS (`resolute`) |
| `uname -r` | 7.0.0-43-generic |
| `free -h` | 14 GiB RAM visible to the system (a 16 GB laptop), 4.0 GiB swap |
| `nproc` | 16 (12th Gen Intel Core i5-12500H) |
| `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 580.178.04 (not used by the stack: the API runs on the CPU) |
| `docker version --format 'client {{.Client.Version}} server {{.Server.Version}}'` | client 29.9.3, server 29.9.3 |
| `docker compose version` | v5.7.1 |
| `docker buildx version` | v0.39.0 |
| `df -h /` | 468 GB disk, 247 GB free |

**Disk and memory you need:** the model package is 1,365 MiB, the API image about 640 MB, and the monitoring images
and build cache a few GB more. The stack (API plus monitoring) uses about 2.4 GiB of RAM; Jenkins, when started, is
limited to 2 GiB. On a 16 GB laptop close large applications before starting both.

## 1. Base tools

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git make python3
```

`python3` is the system Python; it is only used by small scripts (`make env`, the smoke test, the hygiene check). The
API and the tests use Python 3.12 from uv (step 3).

Check:

```bash
git --version      # git version 2.53.0 on the reference machine
make --version | head -1
curl --version | head -1
python3 --version
```

## 2. Docker Engine with the Compose and Buildx plugins

From Docker's own apt repository, which has current Docker Engine, Compose and Buildx versions. The steps follow
<https://docs.docker.com/engine/install/ubuntu/>.

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
gpg --show-keys /etc/apt/keyrings/docker.asc   # must show 9DC858229FC7DD38854AE2D88D81803C0EBFCD88
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Docker without `sudo`

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in (or reboot) so the new group applies, then check as your normal user:

```bash
id -nG                       # the list contains docker
docker run --rm hello-world  # prints "Hello from Docker!"
docker compose version
docker buildx version
```

**Security note:** membership of the `docker` group is equivalent to root on this machine: anyone in it can start a
container that mounts the whole file system. Add only users you would also give `sudo`.

## 3. uv and Python 3.12

[uv](https://docs.astral.sh/uv/) creates the virtual environments and downloads Python 3.12 itself, so the system
Python version does not matter (Ubuntu 26.04 ships 3.14).

```bash
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh
```

Open a new terminal (the installer adds `~/.local/bin` to `PATH`), then:

```bash
uv --version              # uv 0.12.5
uv python install 3.12
uv python find 3.12
```

## How this guide was checked

On the reference machine, 2026-11-19, as the normal user:
- Run and passing: every check command above (`--version`, `id -nG`, `docker run --rm hello-world`,
  `uv python install 3.12`, `uv python find 3.12`), `gpg --show-keys` on a downloaded copy of Docker's key (fingerprint
  as shown), the `Suites:` and `Architectures:` expressions (`resolute`, `amd64`), and the uv installer into a
  temporary folder.
- Not re-run: the `sudo` commands, because everything was already installed on that machine. They are run in full in
  the fresh-account check of this guide.

## 4. Next

Clone the repository and start the stack: [README, Install and run](../README.md#install-and-run-on-ubuntu).

Tools for the Kubernetes and cloud parts (minikube, Helm, Terraform, gcloud) get their own sections when those parts
are added.

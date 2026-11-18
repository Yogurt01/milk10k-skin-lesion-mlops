// MILK10k pipeline: test → build → deploy → smoke → parity (P1-CI-03..08, P1-CI-10). Research model, not a medical
// device. Runs on the Jenkins controller (ci/README.md). Trigger: pollSCM H/5 on main, set in the job definition
// (ci/jenkins/jobs.groovy, P1-DEC-15). Docker commands go to the host daemon and touch milk10k resources only.
// A failed deploy is not rolled back automatically (CI-D7): see ci/README.md for the manual rollback.
pipeline {
    agent any

    options {
        timeout(time: 60, unit: 'MINUTES')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    environment {
        // A fresh serving venv every build; packages come from the uv cache under JENKINS_HOME (ci/compose.yaml).
        UV_VENV_CLEAR = '1'
        // The one Python of this pipeline: the serving venv built by `make setup` (uv-managed CPython 3.12; the
        // Jenkins image has no python3). Every `python3` below (fetch_model.sh, smoke test, hygiene) is this one.
        PATH = "${env.WORKSPACE}/serving/.venv/bin:${env.PATH}"
        // The deployed API as seen from Jenkins once it is attached to the stack network (CI-D13).
        API_BASE_URL = 'http://api:8000'
        STACK_NETWORK = 'milk10k_default'
        // Containers of the deployed stack by label: `docker compose ps` would need the Grafana password to parse
        // compose.yaml, so read-only checks use plain docker with Compose's labels.
        STACK_LABEL = 'com.docker.compose.project=milk10k'
    }

    stages {
        stage('Setup') {
            options { timeout(time: 10, unit: 'MINUTES') }
            steps {
                sh '''
                    rm -rf reports training/sample/images training/sample/sample_metadata.csv
                    mkdir -p reports
                    env -u MILK10K_DATA_DIR make setup
                    echo "python3 for this build: $(command -v python3) ($(python3 --version))"
                '''
            }
        }

        stage('Test') {
            options { timeout(time: 15, unit: 'MINUTES') }
            parallel {
                stage('Unit tests') {
                    steps {
                        // Fast tests only (no model package, no dataset); fails below 90 % coverage (COV_MIN).
                        sh 'env -u MILK10K_DATA_DIR make test-report'
                    }
                    post {
                        always { junit testResults: 'reports/junit.xml' }
                    }
                }
                stage('Lint') {
                    steps { sh 'make lint' }
                }
                stage('Hygiene') {
                    // P1-CI-08: no data, images, weights or secrets in the index or among untracked files.
                    steps { sh 'make hygiene' }
                }
            }
        }

        stage('Build') {
            options { timeout(time: 15, unit: 'MINUTES') }
            steps {
                script {
                    def sha = sh(returnStdout: true, script: 'git rev-parse --short=7 HEAD').trim()
                    env.API_IMAGE_TAG = "${env.BUILD_NUMBER}-${sha}"
                }
                // No registry (CI-D3): the image stays local as milk10k-api:<build>-<sha7>.
                sh '''
                    start=$(date +%s)
                    docker build -t "milk10k-api:${API_IMAGE_TAG}" serving
                    echo "build: milk10k-api:${API_IMAGE_TAG} in $(( $(date +%s) - start )) s"
                    docker image inspect "milk10k-api:${API_IMAGE_TAG}" --format 'image {{.Id}} size {{.Size}} bytes'
                '''
            }
        }

        stage('Deploy') {
            options { timeout(time: 10, unit: 'MINUTES') }
            steps {
                // The model package: the read-only CI copy, verified, never downloaded (CI-D4).
                sh 'scripts/fetch_model.sh --check "$MODEL_HOST_DIR"'
                // Compose project milk10k from this workspace; the Grafana password from the Jenkins credential, the
                // same at every deploy (CI-D18); ports stay on 127.0.0.1 (compose.yaml).
                withCredentials([string(credentialsId: 'grafana-admin-password', variable: 'GRAFANA_ADMIN_PASSWORD')]) {
                    sh 'docker compose up -d --no-build --wait --wait-timeout 180'
                }
                // Attach Jenkins to the stack network (skipped if already attached), then wait for /ready.
                sh '''
                    self=$(cat /etc/hostname)
                    if docker inspect "$self" --format '{{json .NetworkSettings.Networks}}' | grep -q "\\"$STACK_NETWORK\\""; then
                        echo "already attached to $STACK_NETWORK"
                    else
                        docker network connect "$STACK_NETWORK" "$self" && echo "attached to $STACK_NETWORK"
                    fi
                    timeout 120 sh -c 'until curl -fsS "$API_BASE_URL/ready"; do sleep 2; done'; echo
                    api=$(docker ps -q --filter label="$STACK_LABEL" --filter label=com.docker.compose.service=api)
                    docker inspect "$api" --format 'api container image: {{.Config.Image}}'
                    docker ps --filter label="$STACK_LABEL" --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}' | tee reports/deploy-ps.log
                '''
            }
        }

        stage('Smoke') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                sh '''#!/bin/bash -e
                    set -o pipefail
                    make smoke 2>&1 | tee reports/smoke.log
                '''
            }
        }

        stage('Parity') {
            options { timeout(time: 15, unit: 'MINUTES') }
            steps {
                // The 6 sample lesions from the read-only dataset mount (P1-BSH-03; files gitignored, deleted below,
                // never archived), then the HTTP parity test against the deployed image with the tolerance 0.002;
                // MILK10K_DATA_DIR unset and the 479-lesion test deselected (in `make test-container`, API-D10).
                sh 'make sample SAMPLE_DATA_DIR="$CI_DATASET_DIR"'
                sh 'REQUIRE_MODEL_PACKAGE=1 PYTEST_ADDOPTS=-rP make test-container'
            }
            post {
                always { junit testResults: 'reports/junit-parity.xml' }
            }
        }
    }

    post {
        always {
            sh '''
                self=$(cat /etc/hostname)
                if docker inspect "$self" --format '{{json .NetworkSettings.Networks}}' | grep -q "\\"$STACK_NETWORK\\""; then
                    docker network disconnect "$STACK_NETWORK" "$self" && echo "detached from $STACK_NETWORK"
                else
                    echo "not attached to $STACK_NETWORK"
                fi
                api=$(docker ps -q --filter label="$STACK_LABEL" --filter label=com.docker.compose.service=api)
                if [ -n "$api" ]; then docker logs --tail 300 "$api" > reports/api.log 2>&1 || true; fi
                # Dataset licence: the sample lesion images never stay in the workspace and are never archived.
                rm -rf training/sample/images training/sample/sample_metadata.csv
                # Keep the 3 newest pipeline images; only milk10k-api:<build>-<sha7> tags, an image in use is kept.
                docker image ls milk10k-api --format '{{.Tag}}' | grep -E '^[0-9]+-[0-9a-f]{7}$' | sort -t- -k1,1nr \\
                    | tail -n +4 | while read -r tag; do
                        docker image rm "milk10k-api:$tag" || echo "kept milk10k-api:$tag (in use)"
                    done
            '''
            archiveArtifacts artifacts: 'reports/*.xml, reports/*.log', allowEmptyArchive: true
        }
    }
}

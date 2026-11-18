// MILK10k pipeline (P1-CI-03). M7b-1: checkout + test only; build, deploy, smoke and parity come in M7b-2.
// Runs on the Jenkins controller (ci/README.md). Trigger: pollSCM H/5 on main, set in the job definition
// (ci/jenkins/jobs.groovy, P1-DEC-15). Research model, not a medical device.
pipeline {
    agent any

    options {
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    environment {
        // A fresh serving venv every build; packages come from the uv cache under JENKINS_HOME (ci/compose.yaml).
        UV_VENV_CLEAR = '1'
    }

    stages {
        stage('Test') {
            steps {
                // Fast tests only (no model package, no dataset); fails below 90 % coverage (Makefile COV_MIN).
                // MILK10K_DATA_DIR stays unset, so no dataset-dependent test can run here.
                sh '''
                    rm -rf reports
                    env -u MILK10K_DATA_DIR make setup
                    env -u MILK10K_DATA_DIR make test-report
                '''
            }
            post {
                always {
                    junit testResults: 'reports/junit.xml'
                    archiveArtifacts artifacts: 'reports/coverage.xml', allowEmptyArchive: true
                }
            }
        }
    }
}

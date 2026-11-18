// The one pipeline job (Job DSL, run by JCasC at every start; CI-D1, CI-D11, P1-DEC-15).
// Public HTTPS repository, branch main only, no pull-request or fork discovery (a plain Pipeline job, not
// multibranch). The poll trigger is set here, not in the Jenkinsfile, so it exists before the first build.
pipelineJob('milk10k') {
    description('MILK10k pipeline: the Jenkinsfile of the main branch. Defined in ci/jenkins/jobs.groovy.')
    properties {
        disableConcurrentBuilds()
        pipelineTriggers {
            triggers {
                pollSCM {
                    scmpoll_spec('H/5 * * * *')
                }
            }
        }
    }
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('https://github.com/Yogurt01/milk10k-skin-lesion-mlops.git')
                    }
                    branch('*/main')
                }
            }
            scriptPath('Jenkinsfile')
            lightweight(true)
        }
    }
}

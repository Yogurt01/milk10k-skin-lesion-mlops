{{- define "milk10k-api.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "milk10k-api.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "milk10k-api.image" -}}
{{- $tag := required "image.tag is required (a pinned tag, never latest)" .Values.image.tag -}}
{{- if eq $tag "latest" -}}
{{- fail "image.tag must not be latest" -}}
{{- end -}}
{{- $repository := required "image.repository is required" .Values.image.repository -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}

{{/* Image of the fetch-model init container (model.source gcs), always pinned by digest. */}}
{{- define "milk10k-api.fetchImage" -}}
{{- $i := .Values.model.gcs.image -}}
{{- $digest := required "model.gcs.image.digest is required (pin the image by digest)" $i.digest -}}
{{- printf "%s:%s@%s" (required "model.gcs.image.repository is required" $i.repository) (required "model.gcs.image.tag is required" $i.tag) $digest -}}
{{- end -}}

{{- define "milk10k-api.serviceAccountName" -}}
{{- .Values.serviceAccount.name | default .Release.Name -}}
{{- end -}}

{{- define "milk10k-api.securityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end -}}

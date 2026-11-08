"""P1-DEC-04: export one member (s42 fold0) to ONNX; parity vs PyTorch, CPU latency, RAM.

Usage (from the repository root; the notebook venv has torch, milk10k, onnx and onnxruntime):
  MILK10K_DATA_DIR=... notebooks/.venv/bin/python benchmarks/serving/export.py
"""

import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import onnxruntime as ort  # noqa: E402
from common import OUT, batches, config, inf, load_all, sample_meta, save, timed, torch  # noqa: E402

INPUTS = ["derm", "clin", "clin_mask", "site", "sex", "fst", "age", "age_missing"]
R = {"onnxruntime": ort.__version__, "torch": torch.__version__}
cfg = config("cpu")
device = torch.device("cpu")
s, k, m, a = load_all(cfg, device, subset={(42, 0)})[0]
meta = sample_meta(cfg)
root = cfg["paths"]["sample_dir"]
clean, ok, _ = inf.sanitize_inputs(meta, root)
df = clean[ok].reset_index(drop=True)
batch = batches(df, root, a)[0]  # 6 lesions: 3 intact + 3 missing clinical (clin_mask False)
args = tuple(batch[n] for n in INPUTS)


class Probs(torch.nn.Module):
    """Member forward + the same softmax/sigmoid as inference.predict_views (fp32, no autocast on CPU)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, derm, clin, clin_mask, site, sex, fst, age, age_missing):
        logits_cls, logit_bin = self.net(derm, clin, clin_mask, site, sex, fst, age, age_missing)
        return torch.softmax(logits_cls.float(), dim=1), torch.sigmoid(logit_bin.float())


wrapped = Probs(m).eval()
with torch.inference_mode():
    ref_cls, ref_bin = wrapped(*args)

path = OUT / "member_s42_f0.onnx"
batch_dim = torch.export.Dim("batch", min=1, max=64)
t0 = time.perf_counter()
try:
    torch.onnx.export(
        wrapped,
        args,
        str(path),
        input_names=INPUTS,
        output_names=["probs_cls", "prob_bin"],
        dynamo=True,
        dynamic_shapes={n: {0: batch_dim} for n in INPUTS},
        external_data=False,
    )
    R["exporter"] = "torch.onnx.export(dynamo=True)"
except Exception as e:  # fall back to the legacy TorchScript-based exporter
    R["dynamo_export_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    torch.onnx.export(
        wrapped,
        args,
        str(path),
        input_names=INPUTS,
        output_names=["probs_cls", "prob_bin"],
        dynamo=False,
        opset_version=17,
        dynamic_axes={n: {0: "batch"} for n in INPUTS},
    )
    R["exporter"] = "torch.onnx.export(dynamo=False, opset 17)"
R["export_s"] = time.perf_counter() - t0
data_files = list(OUT.glob("member_s42_f0.onnx*"))
R["onnx_mb"] = sum(p.stat().st_size for p in data_files) / 2**20

feeds = {n: batch[n].numpy() for n in INPUTS}
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
o_cls, o_bin = sess.run(None, feeds)
R["parity_vs_pytorch_6_lesions"] = {
    "max_abs_diff_cls": float(np.abs(o_cls - ref_cls.numpy()).max()),
    "max_abs_diff_bin": float(np.abs(o_bin - ref_bin.numpy()).max()),
    "top1_identical": bool((o_cls.argmax(1) == ref_cls.numpy().argmax(1)).all()),
}

# Latency, 1 lesion, ORT vs PyTorch at the same thread counts
one_feeds = {n: v[:1] for n, v in feeds.items()}
one_args = tuple(t[:1] for t in args)
lat = {}
for th in (12, 4, 2):
    so = ort.SessionOptions()
    so.intra_op_num_threads = th
    so.inter_op_num_threads = 1
    s_th = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    s_th.run(None, one_feeds)
    torch.set_num_threads(th)
    with torch.inference_mode():
        wrapped(*one_args)
        lat[f"threads_{th}"] = {
            "onnxruntime_1_member": timed(lambda s=s_th: s.run(None, one_feeds), 20),
            "pytorch_1_member": timed(lambda: wrapped(*one_args), 20),
        }
R["latency_1_lesion"] = lat

# RAM of 15 ORT sessions in a process WITHOUT torch (what a torch-free serving image would hold)
probe = f"""
import json, numpy as np, onnxruntime as ort
def rss():
    return [int(l.split()[1]) / 1024 for l in open('/proc/self/status') if l.startswith('VmRSS:')][0]
r = {{'rss_mb_after_import_ort': rss()}}
so = ort.SessionOptions(); so.intra_op_num_threads = 4
ss = [ort.InferenceSession({str(path)!r}, so, providers=['CPUExecutionProvider']) for _ in range(15)]
r['rss_mb_15_sessions'] = rss()
feeds = {{k: np.load({str(OUT)!r} + '/feed_' + k + '.npy') for k in {INPUTS!r}}}
for s in ss: s.run(None, feeds)
r['rss_mb_15_sessions_after_run'] = rss()
print(json.dumps(r))
"""
for n, v in one_feeds.items():
    np.save(OUT / f"feed_{n}.npy", v)
out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
R["ort_only_process"] = json.loads(out.stdout.strip().splitlines()[-1])
print(save("export.json", R))

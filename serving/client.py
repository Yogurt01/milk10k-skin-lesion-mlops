"""Example client: send one sample lesion to a running API and print the result. Standard library only.

Usage (from the repository root; the sample lesions are built by `training/sample/make_sample.py`):

    python serving/client.py ISIC_0145429
    python serving/client.py ISIC_0145429 --base-url http://localhost:8000 --sample-dir training/sample

The lesion is looked up in `<sample-dir>/sample_metadata.csv`; its dermoscopic image, its clinical image (if the file
exists) and its metadata are sent as `POST /predict` multipart form data. Empty optional metadata is omitted, as the
API treats it as missing. Research model, not a medical device.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

METADATA = ("age_approx", "sex", "anatomical_site_general", "fitzpatrick_type")


def load_lesion(sample_dir: Path, isic_id: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """(form fields, files) for one lesion of `<sample_dir>/sample_metadata.csv`."""
    with open(sample_dir / "sample_metadata.csv", newline="") as fh:
        rows = {row["isic_id"]: row for row in csv.DictReader(fh)}
    if isic_id not in rows:
        raise SystemExit(f"{isic_id} is not in {sample_dir / 'sample_metadata.csv'} (known: {', '.join(rows)})")
    row = rows[isic_id]
    form = {key: row[key].strip() for key in METADATA if row[key].strip()}
    files = {"derm_image": ("derm.jpg", (sample_dir / row["derm_image_path"]).read_bytes())}
    clinical = row["clinical_image_path"].strip()
    if clinical and (sample_dir / clinical).is_file():
        files["clinical_image"] = ("clinical.jpg", (sample_dir / clinical).read_bytes())
    return form, files


def encode_multipart(form: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in form.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data) in files.items():
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        )
        parts.append(head.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def predict(base_url: str, form: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[int, dict, str]:
    """(status, JSON body, request id) of one POST /predict."""
    body, content_type = encode_multipart(form, files)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/predict", data=body, method="POST", headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response), response.headers.get("X-Request-ID", "-")
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), error.headers.get("X-Request-ID", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lesion", help="isic_id of a lesion in sample_metadata.csv, e.g. ISIC_0145429")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--sample-dir", type=Path, default=Path("training/sample"))
    args = parser.parse_args(argv)
    form, files = load_lesion(args.sample_dir, args.lesion)
    status, body, request_id = predict(args.base_url, form, files)
    if status != 200:
        print(f"HTTP {status} (request id {request_id}): {json.dumps(body)}", file=sys.stderr)
        return 1
    print(f"lesion:          {args.lesion} (request id {request_id})")
    print(f"diagnosis:       {body['diagnosis']} (p = {body['probabilities'][body['diagnosis']]:.3f})")
    print(f"P(malignant):    {body['p_malignant']:.3f}")
    print(f"clinical image:  {'used' if body['clinical_image_used'] else 'not used'}")
    if body["metadata_missing"]:
        print(f"missing:         {', '.join(body['metadata_missing'])}")
    print(f"model:           {body['model']} {body['model_version']}")
    print(f"disclaimer:      {body['disclaimer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

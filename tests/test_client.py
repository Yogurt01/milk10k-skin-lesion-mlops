"""P1-API-08: the example client (serving/client.py) against a real server with the fake predictor."""

import csv
import importlib.util

import pytest

from milk10k_serving import DISCLAIMER
from milk10k_serving.api import create_app
from milk10k_serving.settings import Settings

from .conftest import ROOT
from .fakes import FakeEnsemble, image_bytes, make_predictor
from .live_server import live_server

pytestmark = pytest.mark.api

_spec = importlib.util.spec_from_file_location("serving_client", ROOT / "serving" / "client.py")
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


@pytest.fixture
def sample_dir(tmp_path):
    """A sample_metadata.csv like make_sample.py writes, with synthetic images (never lesion images)."""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "d.jpg").write_bytes(image_bytes(seed=1))
    (tmp_path / "images" / "c.jpg").write_bytes(image_bytes(seed=2))
    rows = [
        ["ISIC_A", "images/d.jpg", "images/c.jpg", "60.0", "male", "head/neck", "4"],
        ["ISIC_B", "images/d.jpg", "", "", "female", "", "3"],
    ]
    with open(tmp_path / "sample_metadata.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["isic_id", "derm_image_path", "clinical_image_path", *client.METADATA])
        writer.writerows(rows)
    return tmp_path


def test_client_prints_the_diagnosis_and_the_disclaimer(sample_dir, capsys):
    ensemble = FakeEnsemble(top="BCC")
    with live_server(create_app(Settings(), loader=lambda: make_predictor(ensemble))) as base:
        assert client.main(["ISIC_A", "--base-url", base, "--sample-dir", str(sample_dir)]) == 0
        out = capsys.readouterr().out
        assert "diagnosis:       BCC" in out and f"disclaimer:      {DISCLAIMER}" in out
        assert "clinical image:  used" in out and "missing:" not in out
        assert client.main(["ISIC_B", "--base-url", base, "--sample-dir", str(sample_dir)]) == 0
        out = capsys.readouterr().out
        assert "clinical image:  not used" in out and "missing:         age_approx, anatomical_site_general" in out
    feeds = ensemble.calls[1]
    assert feeds["age_missing"][0] == 1 and feeds["site"][0] == 0


def test_client_reports_an_error_status(sample_dir, capsys):
    loader_never_finishes = make_predictor()
    app = create_app(Settings(), loader=lambda: loader_never_finishes)
    with live_server(app) as base:
        app.state.model.status = "loading"  # simulate a model that is not ready
        assert client.main(["ISIC_A", "--base-url", base, "--sample-dir", str(sample_dir)]) == 1
    assert "HTTP 503" in capsys.readouterr().err

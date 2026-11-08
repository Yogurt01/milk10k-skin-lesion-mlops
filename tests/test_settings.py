import re

import pytest

from milk10k_serving.settings import ENV_VARS, Settings, load_settings

from .conftest import ROOT

pytestmark = pytest.mark.unit


def test_defaults_when_environment_is_empty():
    assert load_settings({}) == Settings()


def test_environment_overrides_a_setting(monkeypatch):
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "2")
    monkeypatch.setenv("MODEL_VERIFY_SHA256", "false")
    settings = load_settings()
    assert settings.ort_threads == 2
    assert settings.verify_sha256 is False


def test_empty_value_keeps_default():
    assert load_settings({"LOG_LEVEL": "  "}).log_level == "INFO"


@pytest.mark.parametrize(
    ("name", "value"),
    [("ORT_INTRA_OP_THREADS", "0"), ("MODEL_VERIFY_SHA256", "maybe"), ("LOG_LEVEL", "LOUD"), ("MAX_UPLOAD_MB", "ten")],
)
def test_invalid_value_names_the_variable(name, value):
    with pytest.raises(ValueError, match=name):
        load_settings({name: value})


def test_env_example_lists_every_setting():
    listed = set(re.findall(r"^([A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), flags=re.M))
    assert {name for name, _ in ENV_VARS.values()} <= listed

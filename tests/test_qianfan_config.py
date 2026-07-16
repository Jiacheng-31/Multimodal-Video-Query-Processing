import pytest

from clipplan.api.qianfan import DEFAULT_BASE_URL, DEFAULT_MODEL, QianfanConfig


def test_qianfan_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QIANFAN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="QIANFAN_API_KEY"):
        QianfanConfig.from_env()


def test_qianfan_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "test-key")
    config = QianfanConfig.from_env()
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.headers()["Authorization"] == "Bearer test-key"

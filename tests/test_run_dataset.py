"""Checks for the dataset launcher without downloading a benchmark or calling a model."""

import bz2
import io
import json
import tarfile
from decimal import Decimal

import pytest

from scripts.fetch_small_sources import files_to_fetch
from scripts.run_dataset import convert_hotpot_archive, download_large, forecast


def test_small_source_selection_does_not_fetch_other_datasets():
    assert {path for path, _ in files_to_fetch(["qasper"])} == {
        "qasper/qasper/qasper-train.parquet",
        "qasper/qasper/qasper-validation.parquet",
        "qasper/qasper/qasper-test.parquet",
    }
    assert len(files_to_fetch(["clerc"])) == 2


def test_hotpot_conversion_preserves_intro_sentence_ids_and_gold_title(tmp_path):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps([{"supporting_facts": [["Alpha", 1]]}]))
    article = {"title": "alpha", "text": [["Title"], ["First complete intro sentence with enough text to pass.",
                                                 " Second sentence carries the evidence."]]}
    payload = bz2.compress((json.dumps(article) + "\n").encode())
    archive = tmp_path / "wiki.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        member = tarfile.TarInfo("pages/part.bz2")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    output = tmp_path / "wiki.jsonl"
    convert_hotpot_archive(archive, dev, output, lambda _: None)
    row = json.loads(output.read_text())
    assert row["title"] == "Alpha"
    assert row["sentences"][1] == " Second sentence carries the evidence."


def test_large_download_resumes_at_exact_byte_offset(tmp_path, monkeypatch):
    import scripts.run_dataset as launcher

    data = b"complete-pinned-source"
    monkeypatch.setitem(launcher.LARGE, "clerc", ("clerc/collection/source.gz", "https://example.test/source", len(data), None))
    partial = tmp_path / "clerc/collection/source.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(data[:7])

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 7-{len(data) - 1}/{len(data)}"}

    def urlopen(request, timeout):
        assert request.get_header("Range") == "bytes=7-"
        return Response(data[7:])

    monkeypatch.setattr(launcher.urllib.request, "urlopen", urlopen)
    target = download_large(tmp_path, "clerc", lambda _: None)
    assert target.read_bytes() == data
    assert json.loads((tmp_path / "clerc/large_source_manifest.json").read_text())["bytes"] == len(data)


def test_rank_only_forecast_excludes_openrouter():
    result = forecast("qasper", "10", {"rpm": 60, "input_usd_per_million": "0.042",
                                       "output_usd_per_million": "0"}, Decimal("0"))
    assert result["max_jev_requests_before_cache"] == 3000
    assert Decimal(result["conservative_api_upper_usd"]) == Decimal("4.032")
    assert result["minimum_hours_at_configured_rpm"] == pytest.approx(0.83, abs=0.01)

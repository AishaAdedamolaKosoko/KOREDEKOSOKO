from pathlib import Path

from main import classify_upload


def test_classify_excel_upload():
    filename = "sample.xlsx"
    result = classify_upload(filename, b"PK\x03\x04")
    assert result["kind"] == "spreadsheet"
    assert result["format"] == "excel"


def test_classify_video_upload():
    filename = "sample.mp4"
    result = classify_upload(filename, b"ftypmp42")
    assert result["kind"] == "video"
    assert result["format"] == "video"


def test_classify_simulation_upload():
    filename = "model.inp"
    payload = b"*HEADING\n*NODE\n*ELEMENT, TYPE=C3D8R"
    result = classify_upload(filename, payload)
    assert result["kind"] == "simulation"
    assert result["format"] in {"abaqus", "ansys"}

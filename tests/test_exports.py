from pathlib import Path

import gp3tools as gp3


def test_bids_export(tmp_path: Path):
    master = gp3.load_example_master().head(100)
    result = gp3.export_gazepoint_to_bids(master, tmp_path, subject_col="subject", task="demo")
    assert (tmp_path / "dataset_description.json").exists()
    assert result["files"]


def test_workflow_smoke(tmp_path: Path):
    master = gp3.load_example_master().head(500)
    result = gp3.run_gazepoint_workflow(data=master, output_dir=tmp_path)
    assert "master" in result
    assert (tmp_path / "gp3tools_report.html").exists()

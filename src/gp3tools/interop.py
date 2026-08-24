"""Interoperability and export adapters."""

from __future__ import annotations

import json
from pathlib import Path

from ._utils import ensure_dataframe, infer_column


def _rename_if(df, mapping):
    return df.rename(columns={k: v for k, v in mapping.items() if k and k in df and k != v})


def prepare_gazepoint_eyetrackingr_data(
    data, participant_col=None, trial_col=None, time_col=None, x_col=None, y_col=None, **kwargs
):
    df = ensure_dataframe(data)
    return _rename_if(
        df,
        {
            participant_col or infer_column(df, "subject"): "Participant",
            trial_col or infer_column(df, "trial"): "Trial",
            time_col or infer_column(df, "time"): "Time",
            x_col or infer_column(df, "x"): "X",
            y_col or infer_column(df, "y"): "Y",
        },
    )


def prepare_gazepoint_pupillometryr_data(
    data, participant_col=None, time_col=None, pupil_col=None, **kwargs
):
    df = ensure_dataframe(data)
    return _rename_if(
        df,
        {
            participant_col or infer_column(df, "subject"): "Participant",
            time_col or infer_column(df, "time"): "Time",
            pupil_col or infer_column(df, "pupil"): "Pupil",
        },
    )


def prepare_gazepoint_gazer_data(data, **kwargs):
    return ensure_dataframe(data)


def prepare_gazepoint_eyetools_data(data, **kwargs):
    return ensure_dataframe(data)


def prepare_gazepoint_gpbiometrics_bridge(data, participant_col=None, time_col=None, **kwargs):
    df = ensure_dataframe(data)
    return _rename_if(
        df,
        {
            participant_col or infer_column(df, "subject"): "participant",
            time_col or infer_column(df, "time"): "time",
        },
    )


def run_gazepoint_gpbiometrics_workflow(data, **kwargs):
    return {
        "bridge_data": prepare_gazepoint_gpbiometrics_bridge(data, **kwargs),
        "status": "prepared_for_gpbiometrics",
    }


def run_gazepoint_gazer_crosscheck(data, **kwargs):
    return {"input": prepare_gazepoint_gazer_data(data, **kwargs), "status": "python_adapter_ready"}


def run_gazepoint_eyetools_fixation_detection(data, **kwargs):
    from .events import detect_gazepoint_fixations_velocity

    return detect_gazepoint_fixations_velocity(
        data,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in {"x_col", "y_col", "time_col", "velocity_threshold", "min_duration_ms", "group_cols"}
        },
    )


def prepare_gazepoint_hddm_export(
    data, response_col=None, rt_col=None, subject_col=None, condition_col=None, **kwargs
):
    df = ensure_dataframe(data)
    mapping = {
        response_col: "response",
        rt_col: "rt",
        subject_col or infer_column(df, "subject"): "subj_idx",
        condition_col or infer_column(df, "condition"): "condition",
    }
    return _rename_if(df, mapping)


def create_gazepoint_hddm_fit_script(path=None, model_name="HDDM", **kwargs) -> str:
    script = """import hddm\nimport pandas as pd\n\ndata = pd.read_csv("hddm_data.csv")\nmodel = hddm.HDDM(data)\nmodel.find_starting_values()\nmodel.sample(5000, burn=1000)\nmodel.save("hddm_model")\n"""
    if path:
        Path(path).write_text(script, encoding="utf-8")
    return script


def _export_matrix(data, path):
    df = ensure_dataframe(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return Path(path)


def export_gazepoint_mne_cluster_input(data, path="mne_cluster_input.csv", **kwargs):
    return _export_matrix(data, path)


def export_gazepoint_permuco_cluster_input(data, path="permuco_cluster_input.csv", **kwargs):
    return _export_matrix(data, path)


def export_gazepoint_permutes_cluster_input(data, path="permutes_cluster_input.csv", **kwargs):
    return _export_matrix(data, path)


def export_gazepoint_to_bids(
    data, output_dir, subject_col=None, task="gazepoint", **kwargs
) -> dict:
    df = ensure_dataframe(data)
    subject_col = subject_col or infer_column(df, "subject")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    if subject_col and subject_col in df:
        iterator = df.groupby(subject_col, dropna=False)
    else:
        iterator = [("01", df)]
    for subject, g in iterator:
        sid = str(subject).replace("sub-", "")
        d = root / f"sub-{sid}" / "eyetrack"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"sub-{sid}_task-{task}_eyetrack.tsv.gz"
        g.to_csv(p, sep="\t", index=False, compression="gzip")
        files.append(str(p))
    desc = {
        "Name": "gp3tools Gazepoint export",
        "BIDSVersion": "1.10.0",
        "GeneratedBy": [{"Name": "gp3tools-python"}],
    }
    (root / "dataset_description.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    return {"output_dir": str(root), "files": files}

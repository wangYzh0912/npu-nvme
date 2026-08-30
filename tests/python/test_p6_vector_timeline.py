import csv

import pytest

from experiments.benchmarks.p6_vector_timeline import read_hbm_average, read_pmu, windows


def test_arithmetic_pmu_does_not_invent_hbm_zero(tmp_path):
    path = tmp_path / "op_summary.csv"
    fields = [
        "Task Start Time(us)",
        "Task Duration(us)",
        "aiv_vec_fp32_ratio",
        "aiv_vec_fp16_ratio",
        "aiv_vec_int32_ratio",
        "aiv_vec_misc_ratio",
        "aic_mac_fp16_ratio",
        "aic_mac_int8_ratio",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Task Start Time(us)": 100,
            "Task Duration(us)": 10,
            "aiv_vec_fp32_ratio": 0.25,
            "aiv_vec_fp16_ratio": 0,
            "aiv_vec_int32_ratio": 0,
            "aiv_vec_misc_ratio": 0,
            "aic_mac_fp16_ratio": 0.5,
            "aic_mac_int8_ratio": 0,
        })

    pmu, has_vector, has_hbm = read_pmu([path])
    bins = windows([(100, 110, "vector", "kernel")], 10, pmu,
                   has_vector, has_hbm)

    assert has_vector is True
    assert has_hbm is False
    assert bins[0]["vector_util"] == 0.25
    assert bins[0]["cube_util"] == 0.5
    assert bins[0]["hbm_read_gb_s"] is None
    assert bins[0]["hbm_write_gb_s"] is None


def test_hbm_average_is_read_from_device_export(tmp_path):
    path = tmp_path / "hbm.csv"
    path.write_text(
        "Device_id,Metric,Read(MB/s),Write(MB/s)\n"
        "6,Average,20546.703,21708.906\n"
        "6,0,20552.849,21709.529\n"
    )

    read_gb_s, write_gb_s = read_hbm_average([path])

    assert read_gb_s == pytest.approx(20.546703)
    assert write_gb_s == pytest.approx(21.708906)

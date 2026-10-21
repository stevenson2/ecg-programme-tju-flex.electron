import pytest

from splits import assert_no_leakage, patient_id, split_table


def test_mit_201_202_same_patient():
    assert patient_id("mitdb", "201") == patient_id("mitdb", "202")
    assert patient_id("mitdb", "100") != patient_id("mitdb", "101")


def test_split_table_no_leakage():
    rows = split_table(
        {
            "afdb": ["04015", "04043"],
            "ltafdb": ["00", "01"],
            "mitdb": ["100", "201", "202"],
        }
    )
    assert {r["split"] for r in rows} == {"dev", "test_ltaf", "test_mit"}
    assert_no_leakage(rows)
    mit = [r for r in rows if r["db"] == "mitdb"]
    assert mit[0]["patient"].startswith("mitdb/")


def test_dev_must_be_afdb():
    with pytest.raises(AssertionError):
        assert_no_leakage(
            [{"db": "ltafdb", "rec": "00", "split": "dev"}]
        )


def test_cross_split_same_pid_rejected():
    with pytest.raises(AssertionError):
        assert_no_leakage(
            [
                {"db": "afdb", "rec": "04015", "split": "dev"},
                {"db": "afdb", "rec": "04015", "split": "test_ltaf"},
            ]
        )

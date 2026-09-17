"""The Excel repository, round-tripped through a workbook written by pandas."""
import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from examples.workbook import SPEC, sample_tables, write_sample_workbook  # noqa: E402
from pmsim.data import (  # noqa: E402
    ExcelRepository, FrameRepository, Orchestrator, SheetNames, SimulationSpec, load_workbook, run_workbook,
)
from tests.conftest import check_identities  # noqa: E402


def test_workbook_round_trip_matches_in_memory_tables(tmp_path):
    path = write_sample_workbook(tmp_path / "book.xlsx")
    repository = ExcelRepository(path)
    assert repository.sheet_names == ["fund_spec", "fund_market_data", "market_data", "commitment_rates"]
    assert repository.fund_specs()["fund_name"].tolist() == ["A", "B"]
    assert repository.commitment_rates().loc[2027, "BUYOUT"] == 0.1
    from_excel = Orchestrator(repository, SPEC).run()
    frames = sample_tables()
    from_frames = Orchestrator(FrameRepository(frames["fund_spec"], frames["fund_market_data"],
                                               frames["market_data"], frames["commitment_rates"]), SPEC).run()
    pd.testing.assert_frame_equal(from_excel.periods, from_frames.periods)
    pd.testing.assert_frame_equal(from_excel.funds, from_frames.funds)
    pd.testing.assert_frame_equal(from_excel.commitments, from_frames.commitments)
    assert from_excel.periods["total_close"].iloc[-1] == pytest.approx(1_207_318.75)
    assert from_excel.periods["fx_translation"].iloc[-1] == pytest.approx(-1_031.25)
    check_identities(from_excel)


def test_sheet_lookup_is_forgiving_and_missing_sheets_are_named(tmp_path):
    path = tmp_path / "odd.xlsx"
    frames = sample_tables()
    with pd.ExcelWriter(path) as writer:
        frames["fund_spec"].to_excel(writer, sheet_name="Fund Spec", index=False)
        frames["fund_market_data"].to_excel(writer, sheet_name="FUND-MARKET-DATA", index=False)
        frames["market_data"].to_excel(writer, sheet_name="Market_Data", index=False)
    repository = ExcelRepository(path)
    assert repository.fund_specs()["fund_name"].tolist() == ["A", "B"]
    assert repository.commitment_rates() is None  # optional sheet absent
    spec_with_rates = SimulationSpec(**{**SPEC.__dict__, "commitment_rates": {"BUYOUT": {2027: 0.1}}})
    assert run_workbook(path, spec_with_rates).status == "completed"
    with pytest.raises(ValueError, match=r"no sheet named 'funds'; sheets are \['Fund Spec', 'FUND-MARKET-DATA', 'Market_Data'\]"):
        ExcelRepository(path, SheetNames(fund_spec="funds")).fund_specs()
    with pytest.raises(ValueError, match="commitment_rates are needed"):
        load_workbook(path, SPEC).portfolio


def test_missing_workbook(tmp_path):
    with pytest.raises(FileNotFoundError, match="workbook not found"):
        ExcelRepository(tmp_path / "nowhere.xlsx")

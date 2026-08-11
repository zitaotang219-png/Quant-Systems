from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alpha_mining.config import SelectedFactor
from alpha_mining.dsl import parse_expression
from alpha_research.factor_diagnostics import generate_factor_diagnostics
from backtest.trading_convention import DEFAULT_TRADING_CONVENTION


def _panel() -> pd.DataFrame:
    rows=[]
    for symbol, base in (("AAA",10.0),("BBB",20.0),("CCC",30.0)):
        for offset,date in enumerate(pd.date_range("2024-01-01",periods=6)):
            rows.append({"date":date,"symbol":symbol,"open":base+offset,"high":base+offset+1,"low":base+offset-1,"close":base+offset+0.5,"volume":100.0+offset})
    return pd.DataFrame(rows)


def _factor() -> SelectedFactor:
    return SelectedFactor("volume",parse_expression("volume"),1,1.0,{},1,1.0)


def test_diagnostics_are_deterministic_and_use_trading_convention(tmp_path) -> None:
    panel=_panel(); first=generate_factor_diagnostics(factors=[_factor()],panel=panel,output_dir=tmp_path/"one")
    second=generate_factor_diagnostics(factors=[_factor()],panel=panel,output_dir=tmp_path/"two")
    pdt.assert_frame_equal(first["statistics"],second["statistics"])
    expected=DEFAULT_TRADING_CONVENTION.forward_return(panel.sort_values(["symbol","date"]).reset_index(drop=True))
    observed=first["timeseries"].loc[first["timeseries"]["factor"]=="volume","forward_return"].reset_index(drop=True)
    pdt.assert_series_equal(observed,expected,check_names=False)


def test_signal_values_do_not_use_future_prices(tmp_path) -> None:
    panel=_panel(); baseline=generate_factor_diagnostics(factors=[_factor()],panel=panel,output_dir=tmp_path/"base")["timeseries"]
    changed=panel.copy(); changed.loc[changed["date"]==pd.Timestamp("2024-01-06"),"close"]=999.0
    revised=generate_factor_diagnostics(factors=[_factor()],panel=changed,output_dir=tmp_path/"changed")["timeseries"]
    pdt.assert_series_equal(baseline["signal_value"],revised["signal_value"],check_names=False)

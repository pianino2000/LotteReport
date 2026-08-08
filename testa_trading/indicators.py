"""이동평균선(5/25/75일) 계산 유틸리티."""

import pandas as pd

SHORT, MID, LONG = 5, 25, 75


def add_moving_averages(df: pd.DataFrame, short: int = SHORT, mid: int = MID, long: int = LONG) -> pd.DataFrame:
    """OHLCV 데이터프레임에 5/25/75일 이동평균 컬럼을 추가해 반환한다.

    df는 'close', 'volume' 컬럼을 포함해야 하며 날짜 오름차순으로 정렬되어 있어야 한다.
    """
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short).mean()
    out["ma_mid"] = out["close"].rolling(mid).mean()
    out["ma_long"] = out["close"].rolling(long).mean()
    out["volume_ma20"] = out["volume"].rolling(20).mean()
    return out


def is_aligned_bullish(row: pd.Series) -> bool:
    """정배열(5 > 25 > 75) 여부."""
    return row["ma_short"] > row["ma_mid"] > row["ma_long"]


def is_trend_broken(row: pd.Series) -> bool:
    """75일선(장기 추세선) 하향 이탈 여부 - 진입 금지 조건."""
    return row["close"] < row["ma_long"]

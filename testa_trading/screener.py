"""1. 투자대상 추적 (범인 찾기)

거래량이 평소 대비 급증하며, 이동평균선이 정배열(주도주 후보)인 종목을 스크리닝한다.
관심 종목의 최우선 순위는 '당일 거래량이 터지는 종목'이라는 테스타의 원칙을 그대로 반영한다.
"""

from dataclasses import dataclass

import pandas as pd

from .indicators import add_moving_averages, is_aligned_bullish


@dataclass
class ScreenResult:
    ticker: str
    close: float
    volume: float
    volume_ratio: float  # 당일 거래량 / 20일 평균 거래량
    is_bullish_aligned: bool


def screen_leading_stocks(
    universe: dict[str, pd.DataFrame],
    volume_surge_ratio: float = 2.0,
    require_bullish_alignment: bool = True,
) -> list[ScreenResult]:
    """종목별 OHLCV 데이터를 받아 주도주 후보를 거래량 급증 순으로 반환한다.

    universe: {ticker: df}, df는 날짜 오름차순 정렬된 'close', 'volume' 컬럼 포함 OHLCV.
    volume_surge_ratio: 당일 거래량이 20일 평균 거래량의 몇 배 이상이면 '급증'으로 볼지의 기준.
    require_bullish_alignment: True면 정배열(5>25>75) 종목만 후보로 남긴다.
    """
    candidates: list[ScreenResult] = []

    for ticker, raw in universe.items():
        df = add_moving_averages(raw)
        if len(df) < 75:
            continue  # 75일선 계산에 필요한 최소 데이터 없음

        today = df.iloc[-1]
        if pd.isna(today["volume_ma20"]) or today["volume_ma20"] == 0:
            continue

        volume_ratio = today["volume"] / today["volume_ma20"]
        if volume_ratio < volume_surge_ratio:
            continue

        bullish = bool(is_aligned_bullish(today))
        if require_bullish_alignment and not bullish:
            continue

        candidates.append(
            ScreenResult(
                ticker=ticker,
                close=float(today["close"]),
                volume=float(today["volume"]),
                volume_ratio=float(volume_ratio),
                is_bullish_aligned=bullish,
            )
        )

    candidates.sort(key=lambda c: c.volume_ratio, reverse=True)
    return candidates

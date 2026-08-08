"""매매 타점 판단 (상승 추세 기준)

- 3개 이평선 정배열(5 > 25 > 75) 상태에서
- 가격이 5일선 아래로 눌림목을 형성한 뒤, 다시 5일선을 강하게(거래량 동반) 돌파하는 시점을 진입 신호로 본다.
- 75일선을 하향 이탈하면 추세 붕괴로 보고 진입을 금지한다.
"""

from dataclasses import dataclass

import pandas as pd

from .indicators import add_moving_averages, is_aligned_bullish, is_trend_broken


@dataclass
class EntrySignal:
    triggered: bool
    reason: str
    entry_price: float | None = None


def detect_entry_signal(
    df: pd.DataFrame,
    pullback_lookback: int = 5,
    volume_confirm_ratio: float = 1.5,
) -> EntrySignal:
    """가장 최근 봉을 기준으로 진입 신호 여부를 판단한다.

    df: 날짜 오름차순 정렬된 OHLCV. 이평선 컬럼이 없으면 자동으로 계산한다.
    pullback_lookback: 최근 몇 봉 내에 5일선 눌림이 있었는지 확인할 기간.
    volume_confirm_ratio: 돌파일 거래량이 20일 평균 대비 몇 배 이상이어야 '강한 돌파'로 볼지.
    """
    if "ma_short" not in df.columns:
        df = add_moving_averages(df)

    if len(df) < 75:
        return EntrySignal(False, "데이터 부족(75일선 계산 불가)")

    today = df.iloc[-1]

    if is_trend_broken(today):
        return EntrySignal(False, "75일선 하향 이탈 - 추세 붕괴, 진입 금지")

    if not is_aligned_bullish(today):
        return EntrySignal(False, "정배열(5>25>75) 아님 - 상승 추세 아님")

    recent = df.iloc[-(pullback_lookback + 1):-1]
    had_pullback = bool((recent["close"] < recent["ma_short"]).any())
    if not had_pullback:
        return EntrySignal(False, "최근 5일선 눌림목 없음")

    breakout = bool(today["close"] > today["ma_short"])
    if not breakout:
        return EntrySignal(False, "오늘 5일선 돌파 실패")

    volume_ok = bool(
        pd.notna(today["volume_ma20"])
        and today["volume_ma20"] > 0
        and today["volume"] >= today["volume_ma20"] * volume_confirm_ratio
    )
    if not volume_ok:
        return EntrySignal(False, "돌파 거래량 부족 - 가짜 신호 가능성")

    return EntrySignal(True, "눌림목 후 5일선 강한 돌파 - 진입 조건 충족", entry_price=float(today["close"]))

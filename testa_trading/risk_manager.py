"""2. 매수 후 손절/익절 값 자동 산출 + 3. 확정손절/추적손절 자동 전환

- 고정 손익비 전략: 손절선은 이전 저점 아래, 익절 목표는 손익비 3:1.
- 저항 구간(1차 익절가) 도달 시 절반 익절 후, 남은 물량은 추적손절로 전환해 추세를 따라간다.
- 그 전까지는 확정손절(진입 시 계산된 손절가 고정)로 계좌를 보호한다.
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class StopMode(Enum):
    FIXED = "fixed"       # 확정손절: 손절가 고정
    TRAILING = "trailing"  # 추적손절: 신고점 갱신에 따라 손절가 상향


@dataclass
class TradeSetup:
    entry_price: float
    stop_loss: float             # 확정손절가 (이전 저점 기준)
    partial_take_profit: float   # 1차 익절/저항구간 (절반 익절 + 추적손절 전환 트리거)
    take_profit: float           # 최종 목표가 (손익비 rr_ratio 기준)
    risk_reward_ratio: float


def calculate_fixed_risk_reward(
    df: pd.DataFrame,
    entry_price: float,
    lookback: int = 20,
    rr_ratio: float = 3.0,
    stop_buffer_pct: float = 0.0,
) -> TradeSetup:
    """진입가를 기준으로 손절가/익절가를 손익비 rr_ratio(기본 3:1)로 산출한다.

    stop_loss = 최근 lookback 기간의 최저가(이전 저점) * (1 - stop_buffer_pct)
    risk = entry_price - stop_loss
    take_profit = entry_price + risk * rr_ratio
    partial_take_profit = entry_price + risk * (rr_ratio / 2)  # 저항 구간 근사치(1차 익절)
    """
    if len(df) < lookback:
        raise ValueError("손절가 계산에 필요한 데이터가 부족합니다.")

    prior_low = float(df["low"].iloc[-lookback:].min())
    stop_loss = prior_low * (1 - stop_buffer_pct)

    risk = entry_price - stop_loss
    if risk <= 0:
        raise ValueError("진입가가 이전 저점(손절가)보다 낮습니다. 진입 조건을 다시 확인하세요.")

    take_profit = entry_price + risk * rr_ratio
    partial_take_profit = entry_price + risk * (rr_ratio / 2)

    return TradeSetup(
        entry_price=entry_price,
        stop_loss=stop_loss,
        partial_take_profit=partial_take_profit,
        take_profit=take_profit,
        risk_reward_ratio=rr_ratio,
    )


@dataclass
class PositionManager:
    """포지션 진입 이후 손절 방식을 자동으로 관리한다.

    - 확정손절(FIXED): 진입 시 계산된 손절가를 그대로 유지해 계좌를 보호한다.
    - 추적손절(TRAILING): 가격이 1차 익절/저항구간에 도달하면 절반 익절 후 자동 전환되며,
      이후 신고점을 갱신할 때마다(5일선 또는 % 트레일 기준) 손절선을 함께 끌어올려 추세를 최대한 따라간다.

    즉, "저항 구간 도달 전까지는 확정손절 → 도달 후에는 추적손절"로 전환 시점을 자동화한다.
    """

    setup: TradeSetup
    trail_pct: float = 0.03
    use_ma_trail: bool = True

    mode: StopMode = field(init=False)
    current_stop: float = field(init=False)
    highest_close: float = field(init=False)
    partial_taken: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.mode = StopMode.FIXED
        self.current_stop = self.setup.stop_loss
        self.highest_close = self.setup.entry_price

    def update(self, today_row: pd.Series) -> dict:
        """일별 종가 기준으로 손절가를 갱신하고 취해야 할 액션을 반환한다."""
        close = float(today_row["close"])

        if close <= self.current_stop:
            return {"mode": self.mode.value, "stop": self.current_stop, "actions": ["STOP_OUT"]}

        self.highest_close = max(self.highest_close, close)
        actions: list[str] = []

        if self.mode is StopMode.FIXED and close >= self.setup.partial_take_profit:
            if not self.partial_taken:
                actions.append("PARTIAL_TAKE_PROFIT")
                self.partial_taken = True
            self.mode = StopMode.TRAILING
            actions.append("SWITCH_TO_TRAILING")

        if self.mode is StopMode.TRAILING:
            if self.use_ma_trail and pd.notna(today_row.get("ma_short")):
                candidate_stop = float(today_row["ma_short"])
            else:
                candidate_stop = self.highest_close * (1 - self.trail_pct)
            if candidate_stop > self.current_stop:
                self.current_stop = candidate_stop
                actions.append("TRAIL_STOP_UPDATED")

        if self.mode is StopMode.TRAILING and close >= self.setup.take_profit:
            actions.append("FINAL_TAKE_PROFIT_ZONE")

        if not actions:
            actions.append("HOLD")

        return {"mode": self.mode.value, "stop": self.current_stop, "actions": actions}

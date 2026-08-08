"""테스타 이동평균선 매매 전술 파이프라인 데모.

1) screen_leading_stocks 로 거래량 급증 + 정배열 주도주 후보를 추린다.
2) detect_entry_signal 로 눌림목 후 5일선 돌파 진입 시점을 판단한다.
3) calculate_fixed_risk_reward 로 손절/익절가(손익비 3:1)를 산출한다.
4) PositionManager 로 확정손절 -> 추적손절 자동 전환을 시뮬레이션한다.

실행: python -m testa_trading.example
"""

import numpy as np
import pandas as pd

from testa_trading import (
    add_moving_averages,
    calculate_fixed_risk_reward,
    detect_entry_signal,
    screen_leading_stocks,
)
from testa_trading.risk_manager import PositionManager


def make_sample_ohlcv(n: int = 120, seed: int = 0, with_breakout_today: bool = True) -> pd.DataFrame:
    """상승 추세 + 눌림목 + (최근일 기준) 돌파 + 거래량 급증 패턴을 갖는 합성 OHLCV 데이터 생성(데모/테스트용)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    close = 10000 + np.cumsum(rng.normal(30, 60, n))
    close[70:80] -= np.linspace(0, 250, 10)  # 중기 눌림목(정배열 형성 과정)

    if with_breakout_today:
        close[-4:-1] -= np.linspace(0, 180, 3)  # 5일선 아래 최근 눌림목
        close[-1] += 450  # 오늘 5일선 강한 돌파

    high = close + rng.uniform(20, 80, n)
    low = close - rng.uniform(20, 80, n)
    volume = rng.uniform(80_000, 120_000, n)
    if with_breakout_today:
        volume[-1] *= 3.2  # 오늘 거래량 급증(주도주 스크리닝 + 돌파 거래량 확인)

    return pd.DataFrame({"date": dates, "close": close, "high": high, "low": low, "volume": volume})


def extend_with_future_path(df: pd.DataFrame, n_future: int = 30, seed: int = 42) -> pd.DataFrame:
    """진입 이후의 미래 가격 흐름을 합성해 이어붙인다 (추적손절 전환 데모용).

    실제로는 없는 '미래' 데이터이므로 실거래에는 사용하지 말 것 - PositionManager 동작 시연 전용.
    """
    rng = np.random.default_rng(seed)
    last_close = df["close"].iloc[-1]
    last_date = df["date"].iloc[-1]

    up_leg = np.linspace(0, 3200, n_future // 2) + rng.normal(0, 60, n_future // 2)
    down_leg = np.linspace(3200, 1200, n_future - n_future // 2) + rng.normal(0, 60, n_future - n_future // 2)
    future_close = last_close + np.concatenate([up_leg, down_leg])

    future_dates = pd.bdate_range(last_date, periods=n_future + 1)[1:]
    future = pd.DataFrame(
        {
            "date": future_dates,
            "close": future_close,
            "high": future_close + rng.uniform(20, 80, n_future),
            "low": future_close - rng.uniform(20, 80, n_future),
            "volume": rng.uniform(80_000, 120_000, n_future),
        }
    )
    return pd.concat([df, future], ignore_index=True)


def main() -> None:
    universe = {
        "005930": make_sample_ohlcv(seed=1, with_breakout_today=True),
        "000660": make_sample_ohlcv(seed=2, with_breakout_today=False),  # 오늘 기준 돌파/거래량 급증 없음
    }

    print("=== 1. 투자대상 추적 (거래량 급증 + 정배열 주도주 스크리닝) ===")
    candidates = screen_leading_stocks(universe, volume_surge_ratio=1.5)
    for c in candidates:
        print(f"  {c.ticker}: 종가={c.close:.0f} 거래량비율={c.volume_ratio:.2f}배 정배열={c.is_bullish_aligned}")

    if not candidates:
        print("  주도주 후보 없음")
        return

    ticker = candidates[0].ticker
    df = add_moving_averages(universe[ticker])

    print(f"\n=== 2. 진입 신호 판단 ({ticker}) ===")
    signal = detect_entry_signal(df)
    print(f"  진입여부={signal.triggered} 사유={signal.reason}")

    if not signal.triggered:
        return

    print("\n=== 3. 손절/익절 자동 산출 (손익비 3:1) ===")
    setup = calculate_fixed_risk_reward(df, entry_price=signal.entry_price, rr_ratio=3.0)
    print(
        f"  진입가={setup.entry_price:.0f} 손절가={setup.stop_loss:.0f} "
        f"1차익절(저항구간)={setup.partial_take_profit:.0f} 최종익절={setup.take_profit:.0f}"
    )

    print("\n=== 4. 확정손절 -> 추적손절 자동 전환 시뮬레이션 (진입 이후 가상 경로) ===")
    entry_index = len(df) - 1
    future_df = add_moving_averages(extend_with_future_path(universe[ticker]))

    manager = PositionManager(setup)
    for _, row in future_df.iloc[entry_index + 1:].iterrows():
        result = manager.update(row)
        print(f"  종가={row['close']:.0f} | 모드={result['mode']:>8} | 손절가={result['stop']:.0f} | {result['actions']}")
        if "STOP_OUT" in result["actions"]:
            break


if __name__ == "__main__":
    main()

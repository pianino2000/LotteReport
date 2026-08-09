"""백테스터 활용 데모: 랜덤 샘플링, 워크포워드, 파라미터 그리드서치.

실행: python -m testa_trading.backtest_example
"""

import numpy as np
import pandas as pd

from testa_trading import (
    add_moving_averages,
    grid_search,
    random_sample_backtest,
    summarize_trades,
    walk_forward_backtest,
)


def make_market_history(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """약 3년치 합성 일봉 데이터. 상승장/하락장/횡보장 국면(regime)을 랜덤하게 전환시켜
    기간 곳곳에서 눌림목/돌파 패턴이 서로 다른 시점에 자연스럽게 여러 번 발생하도록 만든다.

    순수 iid 랜덤워크로 만들면 정배열+눌림목+돌파+거래량급증이 동시에 맞는 경우가
    극히 드물어(수년에 1~2회) 랜덤 샘플링 통계가 사실상 같은 사건을 중복 관측하게 된다.
    국면 전환을 넣어야 서로 다른 상승 구간이 여러 번 생기고, 각 구간마다 독립적인
    매매 기회가 발생해 백테스트 표본이 의미를 가진다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)

    regimes = [
        (0.0022, 0.016),   # 상승장
        (-0.0018, 0.018),  # 하락장
        (0.0002, 0.010),   # 횡보장
    ]

    returns = np.empty(n)
    i = 0
    while i < n:
        drift, vol = regimes[rng.integers(0, len(regimes))]
        length = rng.integers(30, 80)
        end = min(i + length, n)
        returns[i:end] = rng.normal(drift, vol, end - i)
        i = end

    close = 10000 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))

    volume = rng.lognormal(mean=11.5, sigma=0.4, size=n)
    spike_days = rng.random(n) < 0.08
    volume[spike_days] *= rng.uniform(2.0, 4.0, spike_days.sum())

    return pd.DataFrame({"date": dates, "close": close, "high": high, "low": low, "volume": volume})


def main() -> None:
    df = add_moving_averages(make_market_history())
    base_params = dict(pullback_lookback=5, volume_confirm_ratio=1.5, stop_lookback=20, rr_ratio=3.0, trail_pct=0.03)

    print("=== 1. 랜덤 샘플 백테스트 (임의의 과거 구간 30회 반복) ===")
    trades = random_sample_backtest(df, n_samples=30, window=250, seed=1, **base_params)
    summary = summarize_trades(trades)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== 2. 워크포워드 백테스트 (기간을 5구간으로 나눠 구간별 성과 비교) ===")
    folds = walk_forward_backtest(df, n_folds=5, **base_params)
    for f in folds:
        print(
            f"  fold {f['fold']} ({f['start_date'].date()}~{f['end_date'].date()}): "
            f"거래수={f['num_trades']} 승률={f['win_rate']:.2f} 기대값={f['expectancy']:.2f}R "
            f"손익비={f['profit_factor']:.2f} MDD={f['max_drawdown_r']:.2f}R"
        )
    win_rates = [f["win_rate"] for f in folds if f["num_trades"] > 0]
    if win_rates:
        print(f"  -> 구간별 승률 표준편차={np.std(win_rates):.3f} (작을수록 과최적화 위험이 낮음)")

    print("\n=== 3. 파라미터 그리드서치 (기대값 기준 상위 5개) ===")
    grid = grid_search(
        df,
        param_grid={
            "rr_ratio": [2.0, 3.0, 4.0],
            "trail_pct": [0.02, 0.03, 0.05],
            "volume_confirm_ratio": [1.3, 1.5, 2.0],
        },
        fixed_params={"pullback_lookback": 5, "stop_lookback": 20},
    )
    cols = ["rr_ratio", "trail_pct", "volume_confirm_ratio", "num_trades", "win_rate", "expectancy", "profit_factor", "max_drawdown_r"]
    print(grid[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

"""임의의 과거 구간에서 전략을 반복 실행해 승률/손익비/기대값을 검증하는 백테스터.

- run_backtest: 한 종목의 전체 이력에 대해 순차적으로 진입/청산을 시뮬레이션한다.
- random_sample_backtest: 과거 데이터에서 임의의 시작점을 여러 번 뽑아 반복 실행하고
  결과를 합산한다 - "임의의 과거시점자료로 돌려보며 승률을 검증"하는 용도.
- walk_forward_backtest: 기간을 순서대로 나눠 구간별 성과를 비교한다 - 최근/과거 구간마다
  승률이 들쭉날쭉하면 과최적화(curve-fitting)를 의심할 신호가 된다.
- grid_search: 파라미터 조합별 성과를 비교해 기대값(승률×손익비) 기준으로 정렬한다.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import add_moving_averages
from .risk_manager import PositionManager, calculate_fixed_risk_reward
from .strategy import compute_entry_signal_series


@dataclass
class TradeResult:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    r_multiple: float  # 손절 대비 수익 배수. 1.0 = 최초 리스크(1R)만큼 이익.
    exit_reason: str

    @property
    def is_win(self) -> bool:
        return self.r_multiple > 0


def run_backtest(
    df: pd.DataFrame,
    pullback_lookback: int = 5,
    volume_confirm_ratio: float = 1.5,
    stop_lookback: int = 20,
    rr_ratio: float = 3.0,
    stop_buffer_pct: float = 0.0,
    trail_pct: float = 0.03,
    use_ma_trail: bool = True,
) -> list[TradeResult]:
    """단일 종목 이력 전체를 순회하며 진입 신호가 뜰 때마다 매매를 시뮬레이션한다.

    한 번에 한 포지션만 보유(피라미딩 없음)하며, 청산은 PositionManager의 STOP_OUT
    시점(확정손절 또는 추적손절 이탈) 또는 데이터 종료 시점(마지막 종가 기준 강제 청산)이다.
    """
    df = add_moving_averages(df) if "ma_short" not in df.columns else df
    signal = compute_entry_signal_series(df, pullback_lookback, volume_confirm_ratio)

    valid = df["ma_long"].notna().to_numpy()
    warmup = int(valid.argmax()) if valid.any() else len(df)

    trades: list[TradeResult] = []
    n = len(df)
    i = warmup

    while i < n:
        if not bool(signal.iloc[i]):
            i += 1
            continue

        entry_price = float(df["close"].iloc[i])
        try:
            setup = calculate_fixed_risk_reward(
                df.iloc[: i + 1], entry_price, lookback=stop_lookback, rr_ratio=rr_ratio, stop_buffer_pct=stop_buffer_pct
            )
        except ValueError:
            i += 1
            continue

        manager = PositionManager(setup, trail_pct=trail_pct, use_ma_trail=use_ma_trail)
        exit_price, exit_date, exit_reason = None, None, "OPEN_AT_END"
        j = i

        for j in range(i + 1, n):
            row = df.iloc[j]
            result = manager.update(row)
            if "STOP_OUT" in result["actions"]:
                exit_price = result["stop"]
                exit_date = row["date"] if "date" in df.columns else j
                exit_reason = "STOP_OUT"
                break

        if exit_price is None:
            j = n - 1
            exit_price = float(df["close"].iloc[j])
            exit_date = df["date"].iloc[j] if "date" in df.columns else j

        risk = entry_price - setup.stop_loss
        r_multiple = (exit_price - entry_price) / risk if risk else 0.0

        trades.append(
            TradeResult(
                entry_date=df["date"].iloc[i] if "date" in df.columns else i,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                r_multiple=r_multiple,
                exit_reason=exit_reason,
            )
        )
        i = j + 1

    return trades


def summarize_trades(trades: list[TradeResult]) -> dict:
    """승률/평균 R배수/기대값/손익비/최대낙폭(R 기준)을 계산한다."""
    if not trades:
        return {"num_trades": 0, "win_rate": 0.0, "avg_r": 0.0, "expectancy": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0}

    r = np.array([t.r_multiple for t in trades])
    wins = r[r > 0]
    losses = r[r <= 0]

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    max_drawdown_r = float((peak - cum).max())

    return {
        "num_trades": len(trades),
        "win_rate": float(len(wins) / len(r)),
        "avg_r": float(r.mean()),
        "expectancy": float(r.mean()),  # 거래당 평균 기대 수익(R 단위). avg_r과 동일하지만 의미를 명시.
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
    }


def random_sample_backtest(
    df: pd.DataFrame,
    n_samples: int = 30,
    window: int = 250,
    seed: int = 0,
    **strategy_params,
) -> list[TradeResult]:
    """과거 데이터에서 임의의 시작점을 n_samples번 뽑아 각각 window 구간만큼 백테스트하고 합산한다."""
    df = add_moving_averages(df) if "ma_short" not in df.columns else df
    n = len(df)
    if window >= n:
        raise ValueError("window가 데이터 길이보다 깁니다.")

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - window, size=n_samples)

    all_trades: list[TradeResult] = []
    for start in starts:
        segment = df.iloc[start : start + window].reset_index(drop=True)
        all_trades.extend(run_backtest(segment, **strategy_params))
    return all_trades


def walk_forward_backtest(
    df: pd.DataFrame,
    n_folds: int = 5,
    **strategy_params,
) -> list[dict]:
    """기간을 시간 순서대로 n_folds개 구간으로 나눠 구간별 성과를 반환한다.

    과최적화 점검용: 특정 구간에서만 성과가 좋고 나머지 구간은 저조하다면
    해당 파라미터가 우연히 그 구간에만 맞춰진 것일 가능성이 높다.
    """
    df = add_moving_averages(df) if "ma_short" not in df.columns else df
    n = len(df)
    fold_size = n // n_folds

    results = []
    for k in range(n_folds):
        start = k * fold_size
        end = n if k == n_folds - 1 else (k + 1) * fold_size
        segment = df.iloc[start:end].reset_index(drop=True)

        trades = run_backtest(segment, **strategy_params)
        summary = summarize_trades(trades)
        summary["fold"] = k
        summary["start_date"] = df["date"].iloc[start] if "date" in df.columns else start
        summary["end_date"] = df["date"].iloc[end - 1] if "date" in df.columns else end - 1
        results.append(summary)

    return results


def grid_search(
    df: pd.DataFrame,
    param_grid: dict[str, list],
    fixed_params: dict | None = None,
    sort_by: str = "expectancy",
) -> pd.DataFrame:
    """파라미터 조합별로 백테스트를 돌려 성과를 비교한다.

    param_grid 예: {"rr_ratio": [2.0, 3.0, 4.0], "trail_pct": [0.02, 0.03, 0.05]}
    sort_by 기준 내림차순 정렬된 결과 데이터프레임을 반환한다(기본: 기대값 우선).
    """
    df = add_moving_averages(df) if "ma_short" not in df.columns else df
    fixed_params = fixed_params or {}
    keys = list(param_grid.keys())

    rows = []
    for combo in itertools.product(*param_grid.values()):
        params = {**fixed_params, **dict(zip(keys, combo))}
        trades = run_backtest(df, **params)
        rows.append({**params, **summarize_trades(trades)})

    result = pd.DataFrame(rows)
    return result.sort_values(sort_by, ascending=False).reset_index(drop=True)

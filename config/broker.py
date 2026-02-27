"""
Backtest Engine — Broker Configuration
========================================
EXACT broker parameters for Moneta Markets BTCUSD.
All cost calculations are based on these values.
"""

BTCUSD_CONFIG = {
    "symbol": "BTCUSD",
    "digits": 2,
    "point": 0.01,
    "pip_size": 1.0,
    "spread_points": 1700,
    "spread_dollars_per_lot": 17.00,
    "commission_per_side_per_lot": 6.0,
    "commission_round_trip_per_lot": 12.0,
    "stop_level_pips": 20,
    "contract_size": 1,
    "backtest_lot": 0.01,
    "pip_value_per_lot": 1.0,
    "pip_value_at_001": 0.01,
    "spread_cost_at_001": 0.17,
    "commission_at_001": 0.12,
    "total_cost_per_trade_at_001": 0.29,
    "slippage_estimate_pips": 2,
    "mt5_path": r"C:\Program Files\Moneta Markets MT5 Terminal\terminal64.exe",
}

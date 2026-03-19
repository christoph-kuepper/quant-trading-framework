"""Central configuration for all experiments."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    symbol: str = "SPY"
    period: str = "10y"
    interval: str = "1d"
    cross_assets: list[str] = field(default_factory=lambda: ["^VIX"])


@dataclass
class FeatureConfig:
    rolling_windows: list[int] = field(default_factory=lambda: [10, 20, 50, 200])
    use_cross_asset: bool = True
    use_interactions: bool = True


@dataclass
class ModelConfig:
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.01
    xgb_reg_alpha: float = 1.0
    xgb_reg_lambda: float = 2.0
    n_regimes: int = 3
    lstm_lookback: int = 20
    lstm_epochs: int = 20
    lstm_hidden: int = 64
    use_lstm: bool = False


@dataclass
class BacktestConfig:
    ensemble_agree: bool = False
    long_bias: bool = False
    initial_capital: float = 10_000.0
    train_window_days: int = 378
    test_window_days: int = 21
    signal_threshold: float = 0.51
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    commission: float = 0.001
    slippage: float = 0.0001  # 1bp execution slippage vs close price


@dataclass
class ExperimentConfig:
    name: str = "default"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

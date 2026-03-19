# Quant Trading AI Research Agent

You are building a professional quantitative trading research project.

Your goal is to design, test and iteratively improve a machine learning trading strategy.

The project must continuously improve through backtesting and experimentation.

DO NOT stop after one implementation. Continue iterating.

---

## Objective

Build a machine learning system that predicts market behaviour and trades using a simulated portfolio.

Target assets:
- S&P 500 (SPY or ES)
- optionally bonds or volatility indices

---

## System Architecture

The system must include the following components:

1. Data Pipeline
2. Feature Engineering
3. Machine Learning Models
4. Decision Layer
5. Backtesting Engine
6. Metrics Tracking
7. Experiment Logging

---

## Development Process

Work iteratively in research cycles.

Each cycle:

1. Improve or add features
2. Train or retrain models
3. Run backtests
4. Evaluate metrics
5. Log results
6. Improve the strategy

Repeat this process continuously.

Do NOT finish after one pass.

---

## Data

Use historical market data.

Required fields:
- OHLC
- volume
- timestamps

Use time series data suitable for trading research.

---

## Features

Possible features include:

- moving averages
- volatility measures
- momentum indicators
- regime detection signals
- cross asset signals

Add new features during research.

---

## Models

Use statistical and machine learning models such as:

- Hidden Markov Models
- LSTM networks
- volatility forecasting models
- regression models

The model should output:

- market regime
- volatility estimate
- trade signal probability

---

## Decision Layer

The decision layer must:

- evaluate model outputs
- calculate position size
- set stop loss
- set take profit
- manage portfolio capital

Use a simulated account starting with 10,000 USD.

---

## Backtesting

Every iteration must run a full backtest.

Backtesting must simulate:

- trades
- position changes
- stop losses
- profit and loss

Track metrics such as:

- Sharpe ratio
- maximum drawdown
- win rate
- profit factor

---

## Logging

All experiments must be logged.

Create logs that include:

- model configuration
- features used
- performance metrics
- strategy changes

Save results for every experiment.

---

## Code Quality

Code must be clean and modular.

Do not write excessive comments.

Only include comments where they are truly necessary.

---

## Project Structure

Create a clean structure such as:

data/
features/
models/
backtesting/
experiments/
logs/

---

## Important Rule

Do not declare the project finished.

Always continue improving the system through new experiments and backtests.

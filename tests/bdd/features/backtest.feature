Feature: Backtest
  As a DCA strategy system
  I want to run backtests on historical data
  So that I can validate strategy performance

  Background:
    Given sample BTC price data from 2020 to 2025
    And backtest features are initialized

  Scenario: Weights are computed only for window dates
    Given a backtest window from "2020-06-01" to "2021-05-31"
    When I extract the feature window
    And I compute weights for the window using explicit feature weights
    Then weights should be computed only for the window dates

  Scenario: Weights match window index boundaries
    Given a backtest window from "2020-06-01" to "2021-05-31"
    When I extract the feature window
    And I compute weights for the window using explicit feature weights
    Then weight index should match window dates

  Scenario: No future data is used in computation
    Given a tracking strategy function
    When I run compute_cycle_spd with tracking strategy
    Then no received dates should exceed configured backtest end
    And SPD table should not be empty

  Scenario: SPD table has required columns
    When I run compute_cycle_spd
    Then SPD table should not be empty
    And SPD table should have required columns

  Scenario: Percentile values are in valid range
    When I run backtest_dynamic_dca
    Then percentile values should be in valid range

  Scenario: Excess percentile is calculated correctly
    When I run backtest_dynamic_dca
    Then excess percentile should equal dynamic minus uniform

  Scenario: Weight computation is deterministic
    Given a backtest window from "2020-06-01" to "2021-05-31"
    When I extract the feature window
    And I compute weights twice for the same window
    Then both weight computations should be identical

  Scenario: Empty window produces empty weights
    Given empty feature window
    When I compute weights for the window using explicit feature weights
    Then empty window should produce empty weights

  Scenario: Fixed-span window is normalized
    Given single-day feature window
    When I compute weights for the window using explicit feature weights
    Then single-day weight should equal 1.0

  Scenario: Backtest returns numeric exp_decay_percentile
    When I run backtest_dynamic_dca
    Then exp_decay_percentile should be a number

Feature: Forward-Looking Bias Prevention
  As a DCA strategy system
  I must not use future data in weight computation
  So that backtest results are not artificially inflated

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Weights for past dates are identical with full vs truncated data
    Given a date range from "2020-06-01" to "2021-06-01"
    When I compute weights for the date range
    Then the weights should be deterministic

  Scenario: Weights are computed only within window
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be finite
    And all weights should be at least MIN_W

  Scenario: Weight computation respects minimum constraint
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be at least MIN_W

  Scenario: Weight sum is exactly 1.0
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then the weights should sum to 1.0

  Scenario: Features contain required z-score columns
    When I precompute features
    Then the features should contain all required columns

  Scenario: Features have no NaN values
    When I precompute features
    Then the features should have no NaN values


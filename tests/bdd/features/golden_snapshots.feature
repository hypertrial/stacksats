Feature: Golden Snapshot Weights
  As a DCA strategy system
  I need stable and reproducible weight computations
  So that strategy changes are intentional and tracked

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Weights are deterministic across multiple calls
    Given a date range from "2024-01-01" to "2024-12-31"
    When I compute weights for the date range
    Then the weights should be deterministic

  Scenario: Features are deterministic across multiple calls
    When I precompute features
    Then the features should contain all required columns
    And the features should have no NaN values

  Scenario: Weight count for leap year is correct
    Given a date range from "2024-01-01" to "2024-12-31"
    When I compute weights for the date range
    Then the weight count should be 366

  Scenario: Weight sum is exactly 1.0
    Given a date range from "2024-06-01" to "2024-06-30"
    When I compute weights for the date range
    Then the weights should sum to 1.0

  Scenario: All weights are positive
    Given a date range from "2024-06-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be positive


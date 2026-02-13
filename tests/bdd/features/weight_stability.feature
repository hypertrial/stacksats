Feature: Weight Stability Over Time
  As a DCA strategy system
  I must ensure that past weights never change when new data arrives
  So that users can rely on consistent investment allocations

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Past weights unchanged when features are extended by one day
    Given a date range from "2022-01-01" to "2022-12-31"
    And features computed up to "2022-06-15"
    When I compute weights with current_date "2022-06-15"
    And features are extended to "2022-06-16"
    And I recompute weights with current_date "2022-06-16"
    Then weights for dates before "2022-06-15" should be identical

  Scenario: Past weights unchanged when features are extended by 30 days
    Given a date range from "2022-01-01" to "2022-12-31"
    And features computed up to "2022-06-01"
    When I compute weights with current_date "2022-06-01"
    And features are extended to "2022-07-01"
    And I recompute weights with current_date "2022-07-01"
    Then weights for dates before "2022-06-01" should be identical

  Scenario: Weights sum to 1.0 at all stages
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2022-06-15"
    Then the weights should sum to 1.0

  Scenario: All weights are non-negative
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2022-06-15"
    Then all weights should be non-negative

  Scenario: Future weights are distributed uniformly
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2022-06-15"
    Then weights for dates after "2022-06-15" should be uniform

  Scenario: Past weights locked across multiple current_date advances
    Given a date range from "2022-01-01" to "2022-12-31"
    And features computed up to "2022-03-01"
    When I compute weights with current_date "2022-03-01"
    And I store the past weights
    And features are extended to "2022-06-01"
    And I recompute weights with current_date "2022-06-01"
    Then the stored past weights should match the new past weights

  Scenario: Current date at start of range
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2022-01-01"
    Then the weights should sum to 1.0
    And all weights should be non-negative

  Scenario: Current date at end of range
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2022-12-31"
    Then the weights should sum to 1.0
    And all weights should be non-negative

  Scenario: Current date before start of range
    Given a date range from "2022-01-01" to "2022-12-31"
    When I compute weights with current_date "2021-12-01"
    Then the weights should sum to 1.0
    And all weights should be uniform


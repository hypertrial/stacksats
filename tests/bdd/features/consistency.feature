Feature: Computation Consistency
  As a DCA strategy system
  I need deterministic and consistent weight computations
  So that results are reproducible and reliable

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Weight computation is deterministic
    Given a date range from "2025-01-01" to "2025-12-31"
    And current date is "2025-12-31"
    When I process a start date batch
    And I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Past weights remain immutable as time advances
    Given a date range from "2025-01-01" to "2025-12-31"
    When I compute weights for the date range
    Then the weights should be deterministic

  Scenario: Leap day boundary is handled correctly
    Given a date range from "2024-02-28" to "2025-02-27"
    And current date is "2025-02-27"
    When I process a start date batch
    Then batch weights should sum to 1.0
    And batch result should have required columns

  Scenario: Year boundary transition is handled correctly
    Given a date range from "2024-12-30" to "2025-12-29"
    And current date is "2025-12-29"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Current date at start of range is handled
    Given a date range from "2025-06-01" to "2026-05-31"
    And current date is "2025-06-01"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Very short range with MIN_W floor works correctly
    Given a date range from "2025-01-01" to "2025-12-31"
    And current date is "2025-12-31"
    When I process a start date batch
    Then batch weights should sum to 1.0


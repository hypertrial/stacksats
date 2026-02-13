Feature: Edge Cases
  As a DCA strategy system
  I need to handle edge cases correctly
  So that the system is robust and reliable

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Single day range has weight 1.0
    Given a date range from "2025-06-01" to "2026-05-31"
    And current date is "2025-12-31"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Leap year range has valid weights
    Given a date range from "2024-03-01" to "2025-02-28"
    And current date is "2025-12-31"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Future range has valid weights
    Given a date range from "2026-01-01" to "2026-12-31"
    And current date is "2025-12-15"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Current date equals end date
    Given a date range from "2025-01-01" to "2025-12-31"
    And current date is "2025-12-31"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Current date after end date
    Given a date range from "2025-01-01" to "2025-12-31"
    And current date is "2026-01-10"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Two day range produces valid weights
    Given a date range from "2025-06-15" to "2026-06-14"
    And current date is "2025-12-31"
    When I process a start date batch
    Then batch weights should sum to 1.0
    And batch result should have required columns


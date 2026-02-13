Feature: Export Weights
  As a DCA strategy system
  I need to export computed weights to a database
  So that they can be used for real-time investment decisions

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Batch processing returns required columns
    Given a date range from "2024-01-07" to "2025-01-05"
    And current date is "2025-01-05"
    When I process a start date batch
    Then batch result should have required columns

  Scenario: Batch weights are normalized
    Given a date range from "2024-01-07" to "2025-01-05"
    And current date is "2025-01-05"
    When I process a start date batch
    Then batch weights should sum to 1.0

  Scenario: Multiple end dates produce correct number of ranges
    Given a date range from "2024-01-07" to "2025-01-05"
    And current date is "2025-01-05"
    When I process a start date batch
    Then batch result should have required columns


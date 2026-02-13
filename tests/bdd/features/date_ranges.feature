Feature: Date Range Generation
  As a DCA strategy system
  I need to generate valid date ranges for weight computation
  So that I can process all investment periods correctly

  Background:
    Given default date range configuration

  Scenario: Date ranges are generated
    When I generate date ranges
    Then date ranges should not be empty

  Scenario: All ranges span exactly one year
    When I generate date ranges
    Then all ranges should have 1-year span

  Scenario: All ranges are within configured bounds
    When I generate date ranges
    Then all ranges should be within configured bounds

  Scenario: Start dates are sequential daily
    When I generate date ranges
    Then start dates should be sequential daily

  Scenario: Grouping preserves all end dates
    When I generate date ranges
    And I group ranges by start date
    Then grouped ranges should preserve all end dates

  Scenario: Empty ranges for impossible configuration
    Given date range from "2025-01-05" to "2025-02-02" with min length 120
    When I generate date ranges
    Then date ranges should be empty

  Scenario: DATE_FREQ is configured as daily
    Then DATE_FREQ should be daily


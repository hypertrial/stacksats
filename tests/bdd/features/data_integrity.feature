Feature: Data Integrity
  As a DCA strategy system
  I need to ensure data integrity in weight outputs
  So that database operations are reliable and consistent

  Background:
    Given the sample weights DataFrame

  Scenario: No duplicate rows exist
    When I check for duplicate rows
    Then there should be no duplicate rows

  Scenario: Primary keys are unique
    When I check primary key uniqueness
    Then primary keys should be unique

  Scenario: IDs are sequential within each date range
    When I check sequential IDs within each range
    Then IDs should be sequential within each range

  Scenario: Dates are sequential within each date range
    When I check date sequentiality within each range
    Then dates should be sequential within each range

  Scenario: Row counts match expected for each range
    When I check row counts per range
    Then row counts should match expected

  Scenario: No missing dates within any range
    When I check for missing dates in ranges
    Then there should be no missing dates

  Scenario: Start date is before end date for all rows
    When I check date ordering constraints
    Then start_date should be before end_date

  Scenario: DCA date is within the range boundaries
    When I check DCA dates are within range
    Then DCA_date should be within the range

  Scenario: Data types match schema requirements
    When I check data types
    Then data types should match schema

  Scenario: Required columns have no null values
    When I check for null values in required columns
    Then required columns should have no null values


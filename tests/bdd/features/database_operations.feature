Feature: Database Operations
  As a DCA strategy system
  I need reliable database operations
  So that weight data is stored and retrieved correctly

  Scenario: Empty table detection
    Given an empty mock database table
    When I check if the table is empty
    Then table_is_empty should return True

  Scenario: Non-empty table detection
    Given a non-empty mock database table
    When I check if the table is empty
    Then table_is_empty should return False

  Scenario: Today's data exists check - data present
    Given a non-empty mock database table
    And today's data count is 50
    When I check if data exists for "2025-12-28"
    Then data_exists should return True

  Scenario: Today's data exists check - data missing
    Given an empty mock database table
    When I check if data exists for "2025-12-28"
    Then data_exists should return False

  Scenario: Create table executes SQL
    Given a mock database connection
    When I create the table if not exists
    Then CREATE TABLE should be executed
    And commit should be called

  Scenario: Missing DATABASE_URL raises error
    Given DATABASE_URL is not set
    When I get a database connection without DATABASE_URL
    Then a ValueError should be raised for missing DATABASE_URL


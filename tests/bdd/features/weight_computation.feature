Feature: Weight Computation
  As a DCA strategy system
  I want to compute dynamic investment weights
  So that I can allocate funds based on historical patterns

  Background:
    Given sample BTC price data from 2020 to 2025
    And precomputed features from the price data

  Scenario: Weights sum to one
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then the weights should sum to 1.0

  Scenario: All weights above minimum threshold
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be at least MIN_W

  Scenario: All weights are positive
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be positive

  Scenario: All weights are finite
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then all weights should be finite

  Scenario: Weight computation is deterministic
    Given a date range from "2024-01-01" to "2024-06-30"
    When I compute weights for the date range
    Then the weights should be deterministic

  Scenario: Empty date range produces empty weights
    Given a date range from "1990-01-01" to "1990-06-30"
    When I compute weights for the date range
    Then the weight count should be 0

  Scenario: Softmax produces valid probability distribution
    Given an array of values [1.0, 2.0, 3.0]
    When I apply softmax to the array
    Then softmax output should sum to 1.0
    And all softmax values should be positive
    And larger inputs should have larger softmax probabilities

  Scenario: Softmax with zeros produces uniform distribution
    Given all-zero input values
    When I apply softmax to the array
    Then softmax should produce uniform distribution

  Scenario: Softmax with large values does not overflow
    Given large input values for numerical stability test
    When I apply softmax to the array
    Then softmax should not overflow

  Scenario: Allocation sums to one
    Given an array of values [1.0, 2.0, 3.0, 4.0]
    When I apply allocate_sequential_stable
    Then allocation should sum to 1.0
    And all allocations should be at least MIN_W

  Scenario: Dynamic multiplier has correct length
    Given a window size of 100 days
    When I compute the dynamic multiplier
    Then dynamic multiplier should have correct length
    And all dynamic multiplier values should be positive
    And dynamic multiplier should have no NaN values


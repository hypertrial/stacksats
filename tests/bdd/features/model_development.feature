Feature: Model Development
  As a DCA strategy system
  I need reliable ML model computations
  So that investment weights are calculated correctly

  Background:
    Given sample BTC price data from 2020 to 2025

  Scenario: Precompute features creates all required columns
    When I precompute features
    Then the features should contain all required columns

  Scenario: Features have no NaN values
    When I precompute features
    Then the features should have no NaN values

  Scenario: Softmax with normal input
    Given an array of values [1.0, 2.0, 3.0]
    When I apply softmax to the array
    Then softmax output should sum to 1.0
    And all softmax values should be positive

  Scenario: Softmax preserves ordering
    Given an array of values [1.0, 2.0, 3.0]
    When I apply softmax to the array
    Then larger inputs should have larger softmax probabilities

  Scenario: Softmax with zeros is uniform
    Given all-zero input values
    When I apply softmax to the array
    Then softmax should produce uniform distribution

  Scenario: Softmax handles large values
    Given large input values for numerical stability test
    When I apply softmax to the array
    Then softmax should not overflow

  Scenario: Allocation sums to one
    Given an array of values [1.0, 2.0, 3.0, 4.0]
    When I apply allocate_sequential_stable
    Then allocation should sum to 1.0

  Scenario: Allocation is non-negative
    Given an array of values [0.0001, 100.0, 0.0001, 0.0001]
    When I apply allocate_sequential_stable
    Then all allocations should be non-negative

  Scenario: Dynamic multiplier is positive
    Given a window size of 100 days
    When I compute the dynamic multiplier
    Then all dynamic multiplier values should be positive

  Scenario: Dynamic multiplier has no NaN
    Given a window size of 50 days
    When I compute the dynamic multiplier
    Then dynamic multiplier should have no NaN values


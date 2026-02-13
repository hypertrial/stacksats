Feature: Weight Constraints
  As a DCA strategy system
  I need to enforce strict constraints on computed weights
  So that the investment strategy remains valid and balanced

  Background:
    Given the sample weights DataFrame

  # Note: MIN_W is not enforced in stable allocation for weight stability
  # Scenario: All weights are above minimum floor
  #   Then all weights should be above minimum

  Scenario: All weights are non-negative
    Then all weights should be non-negative

  Scenario: All weights are finite values
    Then all weights should be finite

  Scenario: Weights sum to one per date range
    When I check weight sum per range
    Then weights should sum to 1.0 per range

  Scenario: Weights have variance within each range
    Then weights should have variance


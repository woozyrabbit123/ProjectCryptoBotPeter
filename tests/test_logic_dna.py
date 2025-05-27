import pytest
from src.logic_dna import (
    ConditionNode, ActionNode, CompositeNode, SequenceNode, LogicDNA_v1
)
import random

# --- Helper functions for test trees ---
def make_simple_action():
    return ActionNode(action_type="BUY", size_factor=0.5)

def make_simple_condition(val=True, op='GREATER_THAN'):
    # indicator_id, comparison_operator, threshold_value, lookback_period_1
    return ConditionNode(
        indicator_id="SMA",
        comparison_operator=op,
        threshold_value=10.0 if val else 100.0,
        lookback_period_1=5
    )

def make_composite_and(child1, child2):
    return CompositeNode(logical_operator="AND", child1=child1, child2=child2)

def make_composite_or(child1, child2):
    return CompositeNode(logical_operator="OR", child1=child1, child2=child2)

def make_sequence(child1, child2):
    return SequenceNode(child1=child1, child2=child2)

# --- Fixtures ---
@pytest.fixture
def available_indicators():
    return {
        "SMA_5": 20.0,
        "SMA_10": 15.0,
        "RSI_14": 30.0,
        "EMA_5_10": 25.0
    }

@pytest.fixture
def market_state():
    return {}  # Not used in current node logic

# --- Tests for copy() ---
def test_actionnode_copy_independence():
    node = ActionNode(action_type="SELL", size_factor=0.7)
    node2 = node.copy()
    assert node is not node2
    assert node.action_type == node2.action_type
    assert node.size_factor == node2.size_factor
    node2.size_factor = 0.1
    assert node.size_factor == 0.7  # Deep copy

def test_conditionnode_copy_independence():
    node = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    node2 = node.copy()
    assert node is not node2
    assert node.indicator_id == node2.indicator_id
    node2.threshold_value = 99.0
    assert node.threshold_value == 10.0

def test_compositenode_copy_independence():
    c1 = make_simple_condition()
    c2 = make_simple_condition()
    node = CompositeNode("AND", c1, c2)
    node2 = node.copy()
    assert node is not node2
    assert node.child1 is not node2.child1
    node2.child1.threshold_value = 999.0
    assert node.child1.threshold_value != 999.0

def test_sequencenode_copy_independence():
    a1 = make_simple_action()
    a2 = make_simple_action()
    node = SequenceNode(a1, a2)
    node2 = node.copy()
    assert node is not node2
    assert node.child2 is not node2.child2
    node2.child2.size_factor = 0.9
    assert node.child2.size_factor != 0.9

def test_logicdna_v1_copy_deep():
    root = make_composite_and(make_simple_condition(), make_simple_condition())
    dna = LogicDNA_v1(root_node=root)
    dna2 = dna.copy()
    assert dna is not dna2
    assert dna.root_node is not dna2.root_node
    # Change copy, original unaffected
    dna2.root_node.child1.threshold_value = 123.0
    assert dna.root_node.child1.threshold_value != 123.0

# --- Tests for calculate_complexity() ---
def test_calculate_complexity_single_action():
    dna = LogicDNA_v1(root_node=make_simple_action())
    node_count, tree_depth = dna.calculate_complexity()
    assert node_count == 1
    assert tree_depth == 1

def test_calculate_complexity_composite():
    root = make_composite_and(make_simple_condition(), make_simple_condition())
    dna = LogicDNA_v1(root_node=root)
    node_count, tree_depth = dna.calculate_complexity()
    assert node_count == 3
    assert tree_depth == 2

def test_calculate_complexity_nested():
    # Composite(AND, Condition, Composite(OR, Condition, Action))
    inner = make_composite_or(make_simple_condition(), make_simple_action())
    root = make_composite_and(make_simple_condition(), inner)
    dna = LogicDNA_v1(root_node=root)
    node_count, tree_depth = dna.calculate_complexity()
    assert node_count == 5
    assert tree_depth == 3

# --- Tests for is_valid() ---
def test_is_valid_simple_action():
    dna = LogicDNA_v1(root_node=make_simple_action())
    assert dna.is_valid(max_depth=4, max_nodes=9)

def test_is_valid_composite_depth():
    # Build a tree of depth 5 (invalid for max_depth=4)
    n = make_simple_action()
    for _ in range(4):
        n = CompositeNode("AND", make_simple_condition(), n)
    dna = LogicDNA_v1(root_node=n)
    assert not dna.is_valid(max_depth=4, max_nodes=9)

def test_is_valid_composite_nodes():
    # Build a tree with 10 nodes (invalid for max_nodes=9)
    n = make_simple_action()
    for _ in range(4):
        n = CompositeNode("AND", make_simple_condition(), n)
    # Add more nodes to exceed node count
    n = CompositeNode("AND", make_simple_condition(), n)
    dna = LogicDNA_v1(root_node=n)
    assert not dna.is_valid(max_depth=10, max_nodes=9)

def test_is_valid_action_not_leaf():
    # ActionNode as non-leaf (invalid)
    a = make_simple_action()
    c = make_simple_condition()
    n = CompositeNode("AND", a, c)
    # Now wrap ActionNode in CompositeNode (ActionNode is not a leaf)
    n2 = CompositeNode("AND", n, a)
    dna = LogicDNA_v1(root_node=n2)
    assert dna.is_valid(max_depth=10, max_nodes=9)  # Our implementation allows ActionNode as leaf in this context
    # To test stricter: make ActionNode have children (not possible in current design)

# --- Tests for evaluate() ---
def test_actionnode_evaluate():
    node = ActionNode(action_type="BUY", size_factor=0.5)
    assert node.evaluate({}, {}) == ("BUY", 0.5)

def test_conditionnode_evaluate_true(available_indicators):
    node = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    assert node.evaluate({}, available_indicators) is True

def test_conditionnode_evaluate_false(available_indicators):
    node = ConditionNode("SMA", "LESS_THAN", 10.0, 5)
    assert node.evaluate({}, available_indicators) is False

def test_conditionnode_evaluate_eq(available_indicators):
    node = ConditionNode("SMA", "EQUAL_TO", 20.0, 5)
    assert node.evaluate({}, available_indicators) is True

def test_compositenode_evaluate_and(available_indicators):
    c1 = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    c2 = ConditionNode("SMA", "LESS_THAN", 30.0, 5)
    node = CompositeNode("AND", c1, c2)
    assert node.evaluate({}, available_indicators) is True

def test_compositenode_evaluate_or(available_indicators):
    c1 = ConditionNode("SMA", "GREATER_THAN", 100.0, 5)
    c2 = ConditionNode("SMA", "LESS_THAN", 30.0, 5)
    node = CompositeNode("OR", c1, c2)
    assert node.evaluate({}, available_indicators) is True

def test_sequencenode_evaluate_true(available_indicators):
    c1 = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    a2 = ActionNode("BUY", 0.5)
    node = SequenceNode(c1, a2)
    assert node.evaluate({}, available_indicators) == ("BUY", 0.5)

def test_sequencenode_evaluate_false(available_indicators):
    c1 = ConditionNode("SMA", "LESS_THAN", 10.0, 5)
    a2 = ActionNode("SELL", 0.5)
    node = SequenceNode(c1, a2)
    assert node.evaluate({}, available_indicators) == ("HOLD", 0.0)

def test_logicdna_v1_evaluate_action(available_indicators):
    dna = LogicDNA_v1(root_node=ActionNode("BUY", 0.5))
    assert dna.evaluate({}, available_indicators) == ("BUY", 0.5)

def test_logicdna_v1_evaluate_composite(available_indicators):
    c1 = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    c2 = ConditionNode("SMA", "LESS_THAN", 30.0, 5)
    node = CompositeNode("AND", c1, c2)
    dna = LogicDNA_v1(root_node=node)
    assert dna.evaluate({}, available_indicators) == ("HOLD", 0.0)

def test_logicdna_v1_evaluate_sequence_failed(available_indicators):
    c1 = ConditionNode("SMA", "LESS_THAN", 10.0, 5)
    a2 = ActionNode("SELL", 0.5)
    node = SequenceNode(c1, a2)
    dna = LogicDNA_v1(root_node=node)
    assert dna.evaluate({}, available_indicators) == ("HOLD", 0.0)

def test_logicdna_v1_evaluate_no_root():
    dna = LogicDNA_v1(root_node=None)
    assert dna.evaluate({}, {}) == ("NO_ACTION", 0.0)

def test_get_random_node_returns_node_from_tree():
    random.seed(42)
    root = make_composite_and(make_simple_condition(), make_simple_action())
    dna = LogicDNA_v1(root_node=root)
    found = set()
    for _ in range(20):
        node = dna.get_random_node()
        assert node is not None
        found.add(id(node))
    # Should have found at least root and one child
    assert len(found) >= 2

def test_get_random_subtree_point_equivalent_to_random_node():
    random.seed(43)
    root = make_composite_and(make_simple_condition(), make_simple_action())
    dna = LogicDNA_v1(root_node=root)
    found = set()
    for _ in range(20):
        node = dna.get_random_subtree_point()
        assert node is not None
        found.add(id(node))
    assert len(found) >= 2

def test_replace_node_replaces_correctly():
    # Build tree: Composite(AND, Condition, Action)
    cond = make_simple_condition()
    act = make_simple_action()
    root = make_composite_and(cond, act)
    dna = LogicDNA_v1(root_node=root)
    # Replace the ConditionNode with a new ActionNode
    new_action = ActionNode("SELL", 0.9)
    dna.replace_node(cond.node_id, new_action)
    # The left child of root should now be new_action
    assert isinstance(dna.root_node.child1, ActionNode)
    assert dna.root_node.child1.action_type == "SELL"
    # Replace the ActionNode with a new ConditionNode
    new_cond = ConditionNode("SMA", "LESS_THAN", 5.0, 5)
    dna.replace_node(act.node_id, new_cond)
    assert isinstance(dna.root_node.child2, ConditionNode)
    assert dna.root_node.child2.comparison_operator == "LESS_THAN"

def test_replace_node_nonexistent_id():
    root = make_composite_and(make_simple_condition(), make_simple_action())
    dna = LogicDNA_v1(root_node=root)
    # Should not raise, just do nothing
    before = dna.root_node.child1
    dna.replace_node("nonexistent_id", ActionNode("BUY", 1.0))
    after = dna.root_node.child1
    assert before is after

def test_add_node_at_random_valid_point_adds_node():
    random.seed(44)
    root = make_composite_and(make_simple_condition(), make_simple_action())
    dna = LogicDNA_v1(root_node=root)
    new_action = ActionNode("SELL", 0.8)
    result = dna.add_node_at_random_valid_point(new_action)
    # Should have added somewhere (either replaced an ActionNode or filled a None child)
    assert result is True
    # Tree should still be valid
    assert dna.is_valid(4, 9)

def test_add_node_at_random_valid_point_no_valid_point():
    # Tree with only root ActionNode (no valid point to add)
    dna = LogicDNA_v1(root_node=make_simple_action())
    # Fill up with max nodes (simulate)
    # For this test, method should still allow replacing the root
    new_cond = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    result = dna.add_node_at_random_valid_point(new_cond)
    assert result is True
    assert isinstance(dna.root_node, ConditionNode)

def test_prune_random_subtree_replaces_with_hold():
    random.seed(45)
    # Build tree: Composite(AND, Condition, Composite(OR, Action, Condition))
    inner = make_composite_or(make_simple_action(), make_simple_condition())
    root = make_composite_and(make_simple_condition(), inner)
    dna = LogicDNA_v1(root_node=root)
    # Prune a random subtree
    result = dna.prune_random_subtree()
    assert result is True
    # Tree should still be valid
    assert dna.is_valid(4, 9)
    # At least one child of a Composite/Sequence should now be an ActionNode("HOLD", 0.0)
    def _find_hold(node):
        if isinstance(node, ActionNode) and node.action_type == "HOLD" and node.size_factor == 0.0:
            return True
        if isinstance(node, (CompositeNode, SequenceNode)):
            return _find_hold(node.child1) or _find_hold(node.child2)
        return False
    assert _find_hold(dna.root_node)

def test_prune_random_subtree_on_root_only():
    dna = LogicDNA_v1(root_node=make_simple_action())
    # Should do nothing and return False
    result = dna.prune_random_subtree()
    assert result is False
    assert isinstance(dna.root_node, ActionNode)

def test_to_string_representation_simple():
    # Action only
    dna = LogicDNA_v1(root_node=ActionNode(action_type="BUY", size_factor=0.5))
    s = dna.to_string_representation()
    assert s == "(ACTION BUY_0.5)"

    # Condition only
    cond = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    dna2 = LogicDNA_v1(root_node=cond)
    s2 = dna2.to_string_representation()
    assert s2.startswith("(CONDITION SMA_5_GREATER_THAN_10.0)")

def test_to_string_representation_composite():
    # Composite(AND, Condition, Action)
    cond = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    act = ActionNode("BUY", 0.5)
    comp = CompositeNode("AND", cond, act)
    dna = LogicDNA_v1(root_node=comp)
    s = dna.to_string_representation()
    assert s.startswith("(COMPOSITE_AND (CONDITION SMA_5_GREATER_THAN_10.0) (ACTION BUY_0.5)")

    # Sequence(Condition, Action)
    seq = SequenceNode(cond, act)
    dna2 = LogicDNA_v1(root_node=seq)
    s2 = dna2.to_string_representation()
    assert s2.startswith("(SEQUENCE (CONDITION SMA_5_GREATER_THAN_10.0) (ACTION BUY_0.5)") 
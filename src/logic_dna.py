import random
import copy
import logging
import uuid
import operator
from abc import ABC, abstractmethod
from typing import Optional, Any

logger = logging.getLogger(__name__)

# === v1.0 Core Evolutionary System Classes (Scaffolding) ===

class LogicNode(ABC):
    """
    Abstract base class for all node types in LogicDNA trees.
    """
    def __init__(self, node_id: Optional[str] = None, parent_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.parent_id = parent_id

    @abstractmethod
    def evaluate(self, market_state: Any, available_indicators: dict) -> Any:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def copy(self) -> 'LogicNode':
        pass

class ConditionNode(LogicNode):
    def __init__(self, indicator_id: str, comparison_operator: str, threshold_value: float, lookback_period_1: int, lookback_period_2: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.indicator_id = indicator_id
        self.comparison_operator = comparison_operator
        self.threshold_value = threshold_value
        self.lookback_period_1 = lookback_period_1
        self.lookback_period_2 = lookback_period_2

    def evaluate(self, market_state: Any, available_indicators: dict) -> bool:
        # Compose indicator key (assume lookback2 optional)
        if self.lookback_period_2 is not None:
            indicator_key = f"{self.indicator_id}_{self.lookback_period_1}_{self.lookback_period_2}"
        else:
            indicator_key = f"{self.indicator_id}_{self.lookback_period_1}"
        value = available_indicators.get(indicator_key)
        if value is None:
            logger.warning(f"Indicator {indicator_key} not found in available_indicators.")
            return False
        op_map = {
            'GREATER_THAN': operator.gt,
            'LESS_THAN': operator.lt,
            'EQUAL_TO': operator.eq,
            'CROSSES_ABOVE': lambda v, t: v > t,  # Simplified for now
            'CROSSES_BELOW': lambda v, t: v < t,  # Simplified for now
        }
        op_func = op_map.get(self.comparison_operator)
        if op_func is None:
            logger.error(f"Unknown comparison operator: {self.comparison_operator}")
            return False
        return op_func(value, self.threshold_value)

    def to_string(self) -> str:
        return f"Condition({self.indicator_id} {self.comparison_operator} {self.threshold_value} [{self.lookback_period_1}{',' + str(self.lookback_period_2) if self.lookback_period_2 else ''}])"

    def copy(self) -> 'ConditionNode':
        return ConditionNode(
            indicator_id=self.indicator_id,
            comparison_operator=self.comparison_operator,
            threshold_value=self.threshold_value,
            lookback_period_1=self.lookback_period_1,
            lookback_period_2=self.lookback_period_2,
            node_id=self.node_id,
            parent_id=self.parent_id
        )

class ActionNode(LogicNode):
    def __init__(self, action_type: str, size_factor: float, **kwargs):
        super().__init__(**kwargs)
        self.action_type = action_type
        self.size_factor = size_factor

    def evaluate(self, market_state: Any, available_indicators: dict) -> tuple:
        return (self.action_type, self.size_factor)

    def to_string(self) -> str:
        return f"Action({self.action_type}, size={self.size_factor})"

    def copy(self) -> 'ActionNode':
        return ActionNode(
            action_type=self.action_type,
            size_factor=self.size_factor,
            node_id=self.node_id,
            parent_id=self.parent_id
        )

class CompositeNode(LogicNode):
    def __init__(self, logical_operator: str, child1: LogicNode, child2: LogicNode, **kwargs):
        super().__init__(**kwargs)
        self.logical_operator = logical_operator
        self.child1 = child1
        self.child2 = child2

    def evaluate(self, market_state: Any, available_indicators: dict) -> bool:
        left = self.child1.evaluate(market_state, available_indicators)
        right = self.child2.evaluate(market_state, available_indicators)
        if self.logical_operator == 'AND':
            return left and right
        elif self.logical_operator == 'OR':
            return left or right
        else:
            logger.error(f"Unknown logical operator: {self.logical_operator}")
            return False

    def to_string(self) -> str:
        return f"Composite({self.logical_operator}, {self.child1.to_string()}, {self.child2.to_string()})"

    def copy(self) -> 'CompositeNode':
        return CompositeNode(
            logical_operator=self.logical_operator,
            child1=self.child1.copy(),
            child2=self.child2.copy(),
            node_id=self.node_id,
            parent_id=self.parent_id
        )

class SequenceNode(LogicNode):
    def __init__(self, child1: LogicNode, child2: LogicNode, **kwargs):
        super().__init__(**kwargs)
        self.child1 = child1
        self.child2 = child2

    def evaluate(self, market_state: Any, available_indicators: dict) -> Any:
        result1 = self.child1.evaluate(market_state, available_indicators)
        # If child1 is a ConditionNode and False, sequence returns HOLD
        if isinstance(self.child1, ConditionNode) and not result1:
            return ("HOLD", 0.0)
        # If child1 is Action/Composite/Sequence, always continue
        return self.child2.evaluate(market_state, available_indicators)

    def to_string(self) -> str:
        return f"Sequence({self.child1.to_string()} -> {self.child2.to_string()})"

    def copy(self) -> 'SequenceNode':
        return SequenceNode(
            child1=self.child1.copy(),
            child2=self.child2.copy(),
            node_id=self.node_id,
            parent_id=self.parent_id
        )

class LogicDNA_v1:
    """
    v1.0 LogicDNA tree structure for evolutionary trading logic.
    """
    def __init__(self, dna_id: Optional[str] = None, root_node: Optional[LogicNode] = None, generation_born: Optional[int] = None):
        self.dna_id = dna_id or str(uuid.uuid4())
        self.root_node = root_node
        self.generation_born = generation_born

    def evaluate(self, market_state: Any, available_indicators: dict) -> Any:
        if self.root_node is None:
            return ("NO_ACTION", 0.0)
        result = self.root_node.evaluate(market_state, available_indicators)
        # If result is a tuple (action_type, size_factor), return it
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If result is True/False, return NO_ACTION or HOLD
        if result is True:
            return ("HOLD", 0.0)
        if result is False:
            return ("NO_ACTION", 0.0)
        # If result is a sequence failed signal
        if isinstance(result, tuple) and result[0] == "SEQUENCE_FAILED":
            return ("NO_ACTION", 0.0)
        return result

    def calculate_complexity(self) -> tuple:
        def _count_and_depth(node, depth=1):
            if node is None:
                return 0, 0
            if isinstance(node, (ActionNode, ConditionNode)):
                return 1, depth
            elif isinstance(node, (CompositeNode, SequenceNode)):
                left_count, left_depth = _count_and_depth(node.child1, depth+1)
                right_count, right_depth = _count_and_depth(node.child2, depth+1)
                return 1 + left_count + right_count, max(left_depth, right_depth)
            else:
                return 1, depth
        node_count, tree_depth = _count_and_depth(self.root_node)
        return node_count, tree_depth

    def to_string_representation(self) -> str:
        """
        Return a LISP-like, unambiguous string representation of the LogicDNA tree for MLE parsing.
        """
        def _to_lisp(node):
            if node is None:
                return "(NONE)"
            if isinstance(node, ConditionNode):
                op = node.comparison_operator
                return f"(CONDITION {node.indicator_id}_{node.lookback_period_1}{('_' + str(node.lookback_period_2)) if node.lookback_period_2 else ''}_{op}_{round(node.threshold_value, 4)})"
            elif isinstance(node, ActionNode):
                return f"(ACTION {node.action_type}_{round(node.size_factor, 4)})"
            elif isinstance(node, CompositeNode):
                return f"(COMPOSITE_{node.logical_operator} {_to_lisp(node.child1)} {_to_lisp(node.child2)})"
            elif isinstance(node, SequenceNode):
                return f"(SEQUENCE {_to_lisp(node.child1)} {_to_lisp(node.child2)})"
            else:
                return f"(UNKNOWN_NODE)"
        return _to_lisp(self.root_node)

    def copy(self) -> 'LogicDNA_v1':
        # Create a new UUID for the copy
        new_dna_id = str(uuid.uuid4())
        new_dna = LogicDNA_v1(
            dna_id=new_dna_id,
            root_node=self.root_node.copy() if self.root_node else None,
            generation_born=self.generation_born
        )
        if hasattr(self, 'fitness'):
            new_dna.fitness = self.fitness
        return new_dna

    def get_random_node(self) -> LogicNode:
        """Return a randomly selected node from the tree."""
        nodes = []
        def _collect(node):
            if node is None:
                return
            nodes.append(node)
            if isinstance(node, (CompositeNode, SequenceNode)):
                _collect(node.child1)
                _collect(node.child2)
        _collect(self.root_node)
        return random.choice(nodes) if nodes else None

    def get_random_subtree_point(self) -> LogicNode:
        """Return a randomly selected node that can serve as a subtree root (any node)."""
        return self.get_random_node()

    def replace_node(self, old_node_id: str, new_node_subtree_root: LogicNode):
        """Replace the node with old_node_id with new_node_subtree_root."""
        def _replace(node):
            if node is None:
                return None
            if getattr(node, 'node_id', None) == old_node_id:
                return new_node_subtree_root
            if isinstance(node, CompositeNode):
                node.child1 = _replace(node.child1)
                node.child2 = _replace(node.child2)
            elif isinstance(node, SequenceNode):
                node.child1 = _replace(node.child1)
                node.child2 = _replace(node.child2)
            return node
        self.root_node = _replace(self.root_node)

    def add_node_at_random_valid_point(self, new_node: LogicNode):
        """Insert new_node at a random valid point in the tree, e.g., as a child of a Composite/Sequence node or replacing an Action node."""
        candidates = []
        def _find_candidates(node, parent=None, child_idx=None):
            if node is None:
                return
            # If ActionNode, can be replaced
            if isinstance(node, ActionNode):
                candidates.append((parent, child_idx))
            # If Composite/Sequence, can add as child if any child is None
            if isinstance(node, (CompositeNode, SequenceNode)):
                if node.child1 is None:
                    candidates.append((node, 1))
                else:
                    _find_candidates(node.child1, node, 1)
                if node.child2 is None:
                    candidates.append((node, 2))
                else:
                    _find_candidates(node.child2, node, 2)
        _find_candidates(self.root_node)
        if not candidates:
            return False
        parent, child_idx = random.choice(candidates)
        if parent is None:
            # Replace root
            self.root_node = new_node
        else:
            if child_idx == 1:
                parent.child1 = new_node
            elif child_idx == 2:
                parent.child2 = new_node
        return True

    def prune_random_subtree(self):
        """Select a random non-root node and replace its subtree with a default ActionNode('HOLD', 0.0)."""
        nodes = []
        def _collect(node, parent=None, child_idx=None):
            if node is None:
                return
            if parent is not None:
                nodes.append((parent, child_idx))
            if isinstance(node, (CompositeNode, SequenceNode)):
                _collect(node.child1, node, 1)
                _collect(node.child2, node, 2)
        _collect(self.root_node)
        if not nodes:
            return False
        parent, child_idx = random.choice(nodes)
        default_action = ActionNode(action_type="HOLD", size_factor=0.0)
        if child_idx == 1:
            parent.child1 = default_action
        elif child_idx == 2:
            parent.child2 = default_action
        return True

    def is_valid(self, max_depth: int, max_nodes: int) -> bool:
        def _validate(node, depth=1):
            if node is None:
                return True, 0
            if depth > max_depth:
                return False, 0
            if isinstance(node, ActionNode):
                return True, 1
            elif isinstance(node, ConditionNode):
                return True, 1
            elif isinstance(node, CompositeNode):
                valid1, count1 = _validate(node.child1, depth+1)
                valid2, count2 = _validate(node.child2, depth+1)
                # Composite must have exactly two children
                return valid1 and valid2, 1 + count1 + count2
            elif isinstance(node, SequenceNode):
                valid1, count1 = _validate(node.child1, depth+1)
                valid2, count2 = _validate(node.child2, depth+1)
                return valid1 and valid2, 1 + count1 + count2
            else:
                return False, 0
        valid, node_count = _validate(self.root_node)
        if not valid or node_count > max_nodes:
            return False
        # Action nodes must be leaves
        def _action_leaves(node):
            if isinstance(node, ActionNode):
                return True
            elif isinstance(node, (CompositeNode, SequenceNode)):
                return _action_leaves(node.child1) and _action_leaves(node.child2)
            elif isinstance(node, ConditionNode):
                return True
            return False
        return _action_leaves(self.root_node)

# === Legacy LogicDNA (for migration/compatibility) ===

class LogicDNA:
    """
    Represents a trading logic DNA, encoding trigger conditions and actions.
    Each instance is assigned a unique ID for tracking and records its parent for lineage analysis.
    Args:
        trigger_indicator (str): Indicator name for the trigger (e.g., 'RSI_14').
        trigger_operator (str): Operator for the trigger ('<' or '>').
        trigger_threshold (float): Threshold value for the trigger.
        context_regime_id (str): Regime context for the DNA ('any', etc.).
        action_target (str): Target of the action (e.g., 'buy_confidence_boost').
        action_type (str): Type of action ('add', etc.).
        action_value (float): Value for the action.
        resource_cost (float): Resource cost for this DNA (default 1).
        parent_id (str): Unique ID of the parent DNA (for lineage tracking).
        seed_type (str): Type of seed if this is a seed DNA (e.g., 'RSI_BUY', 'RSI_SELL').
    Attributes:
        id (str): Unique identifier for this DNA instance.
        parent_id (str): Unique identifier of the parent DNA (or None for seeds).
        seed_type (str): Seed type if applicable.
    """
    def __init__(self, trigger_indicator, trigger_operator, trigger_threshold, context_regime_id, action_target, action_type, action_value, resource_cost=1, parent_id=None, seed_type=None):
        self.id = uuid.uuid4().hex[:8]  # Unique ID for tracking
        self.trigger_indicator = trigger_indicator
        self.trigger_operator = trigger_operator
        self.trigger_threshold = trigger_threshold
        self.context_regime_id = context_regime_id
        self.action_target = action_target
        self.action_type = action_type
        self.action_value = action_value
        self.resource_cost = resource_cost
        self.parent_id = parent_id
        self.seed_type = seed_type
        logger.info(f"LogicDNA instantiated: {self}")

    @staticmethod
    def seed_rsi_buy():
        """
        Create a seed LogicDNA for an RSI buy signal.
        Returns:
            LogicDNA: Seed buy DNA instance.
        """
        return LogicDNA(
            trigger_indicator='RSI_14',
            trigger_operator='<',
            trigger_threshold=30.0,
            context_regime_id='any',
            action_target='buy_confidence_boost',
            action_type='add',
            action_value=0.1,
            parent_id=None,
            seed_type='RSI_BUY'
        )

    @staticmethod
    def seed_rsi_sell():
        """
        Create a seed LogicDNA for an RSI sell signal.
        Returns:
            LogicDNA: Seed sell DNA instance.
        """
        return LogicDNA(
            trigger_indicator='RSI_14',
            trigger_operator='>',
            trigger_threshold=70.0,
            context_regime_id='any',
            action_target='sell_confidence_boost',
            action_type='add',
            action_value=0.1,
            parent_id=None,
            seed_type='RSI_SELL'
        )

    def __repr__(self):
        return (f"LogicDNA(id={self.id}, parent_id={self.parent_id}, seed_type={self.seed_type}, trigger={self.trigger_indicator} {self.trigger_operator} {self.trigger_threshold}, "
                f"context={self.context_regime_id}, action={self.action_target} {self.action_type} {self.action_value}, "
                f"resource_cost={self.resource_cost})")

    def to_dict(self):
        """
        Return a flat dict of all relevant fields for JSON serialization.
        """
        return {
            'dna_id': self.id,
            'parent_id': self.parent_id,
            'seed_type': self.seed_type,
            'trigger_indicator': self.trigger_indicator,
            'trigger_operator': self.trigger_operator,
            'trigger_threshold': self.trigger_threshold,
            'context_regime_id': self.context_regime_id,
            'action_target': self.action_target,
            'action_type': self.action_type,
            'action_value': self.action_value,
            'resource_cost': self.resource_cost,
        }

def mutate_dna(dna_instance, mutation_strength=0.1):
    """
    Randomly mutates one numerical parameter (threshold or action_value) by ±mutation_strength (default 10%).
    Args:
        dna_instance (LogicDNA): The DNA instance to mutate.
        mutation_strength (float): The mutation strength as a fraction.
    Returns:
        LogicDNA: A new mutated DNA instance with parent_id set to the original's id.
    """
    new_dna = copy.deepcopy(dna_instance)
    param_to_mutate = random.choice(['trigger_threshold', 'action_value'])
    orig_value = getattr(new_dna, param_to_mutate)
    change_pct = random.uniform(-mutation_strength, mutation_strength)
    new_value = orig_value * (1 + change_pct)
    setattr(new_dna, param_to_mutate, new_value)
    new_dna.parent_id = dna_instance.id  # Track lineage
    new_dna.seed_type = None  # Not a seed
    logger.info(f"Mutated {param_to_mutate} from {orig_value} to {new_value:.4f} in DNA: {new_dna}")
    return new_dna 
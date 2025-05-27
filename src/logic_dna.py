"""
Defines the structure and manipulation of "Logic DNA," which represents
trading strategies or decision trees within the system.

This module includes:
- Abstract base class `LogicNode` for tree nodes.
- Concrete node types: `ConditionNode`, `ActionNode`, `CompositeNode`, `SequenceNode`.
- `LogicDNA_v1`: A class representing a tree-based strategy with methods for
  evaluation, complexity calculation, mutation, and crossover.
- Legacy `LogicDNA`: A simpler, flat-structure representation of a strategy,
  along with its mutation function.
"""
import random
import copy
import logging
import uuid
import operator
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Tuple, List, Callable # Added Dict, Tuple, List, Callable

logger = logging.getLogger(__name__)

# === v1.0 Core Evolutionary System Classes (Scaffolding) ===

class LogicNode(ABC):
    """
    Abstract base class for all node types in LogicDNA trees.
    """
    node_id: str
    parent_id: Optional[str]

    def __init__(self, node_id: Optional[str] = None, parent_id: Optional[str] = None) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.parent_id = parent_id

    @abstractmethod
    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def copy(self) -> 'LogicNode':
        pass

class ConditionNode(LogicNode):
    indicator_id: str
    comparison_operator: str
    threshold_value: float
    lookback_period_1: int
    lookback_period_2: Optional[int]

    def __init__(self, indicator_id: str, comparison_operator: str, threshold_value: float, lookback_period_1: int, lookback_period_2: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.indicator_id = indicator_id
        self.comparison_operator = comparison_operator
        self.threshold_value = threshold_value
        self.lookback_period_1 = lookback_period_1
        self.lookback_period_2 = lookback_period_2

    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> bool:
        # Compose indicator key (assume lookback2 optional)
        if self.lookback_period_2 is not None:
            indicator_key: str = f"{self.indicator_id}_{self.lookback_period_1}_{self.lookback_period_2}"
        else:
            indicator_key = f"{self.indicator_id}_{self.lookback_period_1}"
        
        value: Optional[float] = available_indicators.get(indicator_key) # Assuming indicators are float
        if value is None:
            logger.warning(f"Indicator {indicator_key} not found in available_indicators.")
            return False
        
        op_map: Dict[str, Callable[[Any, Any], bool]] = {
            'GREATER_THAN': operator.gt,
            'LESS_THAN': operator.lt,
            'EQUAL_TO': operator.eq,
            'CROSSES_ABOVE': lambda v, t: v > t,  # Simplified for now
            'CROSSES_BELOW': lambda v, t: v < t,  # Simplified for now
        }
        op_func: Optional[Callable[[Any, Any], bool]] = op_map.get(self.comparison_operator)
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
    action_type: str
    size_factor: float

    def __init__(self, action_type: str, size_factor: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_type = action_type
        self.size_factor = size_factor

    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> Tuple[str, float]:
        return (self.action_type, self.size_factor)

    def to_string(self) -> str:
        return f"Action({self.action_type}, size={self.size_factor})"

    def copy(self) -> 'ActionNode':
        return ActionNode(
            action_type=self.action_type,
            size_factor=self.size_factor,
            node_id=self.node_id, # Pass existing node_id
            parent_id=self.parent_id # Pass existing parent_id
        )

class CompositeNode(LogicNode):
    logical_operator: str
    child1: Optional[LogicNode] # Can be None if tree is malformed, though evaluate assumes not None
    child2: Optional[LogicNode]

    def __init__(self, logical_operator: str, child1: LogicNode, child2: LogicNode, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logical_operator = logical_operator
        self.child1 = child1
        self.child2 = child2
        # Update parent_id for children
        if self.child1: self.child1.parent_id = self.node_id
        if self.child2: self.child2.parent_id = self.node_id


    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> bool:
        if self.child1 is None or self.child2 is None: # Guard against malformed tree
            logger.error(f"CompositeNode {self.node_id} has missing child/children.")
            return False
        left: bool = self.child1.evaluate(market_state, available_indicators)
        right: bool = self.child2.evaluate(market_state, available_indicators)
        if self.logical_operator == 'AND':
            return left and right
        elif self.logical_operator == 'OR':
            return left or right
        else:
            logger.error(f"Unknown logical operator: {self.logical_operator}")
            return False

    def to_string(self) -> str:
        child1_str: str = self.child1.to_string() if self.child1 else "None"
        child2_str: str = self.child2.to_string() if self.child2 else "None"
        return f"Composite({self.logical_operator}, {child1_str}, {child2_str})"

    def copy(self) -> 'CompositeNode':
        return CompositeNode(
            logical_operator=self.logical_operator,
            child1=self.child1.copy() if self.child1 else None, # type: ignore 
            child2=self.child2.copy() if self.child2 else None, # type: ignore
            node_id=self.node_id, # Pass existing node_id
            parent_id=self.parent_id # Pass existing parent_id
        )

class SequenceNode(LogicNode):
    child1: Optional[LogicNode]
    child2: Optional[LogicNode]

    def __init__(self, child1: LogicNode, child2: LogicNode, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child1 = child1
        self.child2 = child2
        # Update parent_id for children
        if self.child1: self.child1.parent_id = self.node_id
        if self.child2: self.child2.parent_id = self.node_id


    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> Any:
        if self.child1 is None or self.child2 is None: # Guard against malformed tree
            logger.error(f"SequenceNode {self.node_id} has missing child/children.")
            return ("NO_ACTION", 0.0) # Or some other default error signal

        result1: Any = self.child1.evaluate(market_state, available_indicators)
        # If child1 is a ConditionNode and False, sequence returns HOLD
        if isinstance(self.child1, ConditionNode) and not result1:
            return ("HOLD", 0.0) # Explicitly a HOLD signal
        # If child1 is Action/Composite/Sequence, always continue
        return self.child2.evaluate(market_state, available_indicators)

    def to_string(self) -> str:
        child1_str: str = self.child1.to_string() if self.child1 else "None"
        child2_str: str = self.child2.to_string() if self.child2 else "None"
        return f"Sequence({child1_str} -> {child2_str})"

    def copy(self) -> 'SequenceNode':
        return SequenceNode(
            child1=self.child1.copy() if self.child1 else None, # type: ignore
            child2=self.child2.copy() if self.child2 else None, # type: ignore
            node_id=self.node_id,
            parent_id=self.parent_id
        )

class LogicDNA_v1:
    """
    v1.0 LogicDNA tree structure for evolutionary trading logic.
    """
    dna_id: str
    root_node: Optional[LogicNode]
    generation_born: Optional[int]
    fitness: float # Added attribute hint

    def __init__(self, dna_id: Optional[str] = None, root_node: Optional[LogicNode] = None, generation_born: Optional[int] = None) -> None:
        self.dna_id = dna_id or str(uuid.uuid4())
        self.root_node = root_node
        self.generation_born = generation_born
        self.fitness = 0.0 # Initialize fitness

    def evaluate(self, market_state: Any, available_indicators: Dict[str, Any]) -> Tuple[str, float]: # Return type changed
        if self.root_node is None:
            return ("NO_ACTION", 0.0)
        result: Any = self.root_node.evaluate(market_state, available_indicators)
        
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str) and isinstance(result[1], (int, float)):
            return result # type: ignore
        
        if result is True: # Typically from a ConditionNode that evaluated true but isn't followed by an ActionNode directly
            return ("HOLD", 0.0) 
        if result is False: # Typically from a ConditionNode
            return ("NO_ACTION", 0.0)
        
        # Default fallback if result is not a clear action tuple or boolean
        logger.warning(f"LogicDNA_v1 {self.dna_id} evaluation returned unexpected result type: {result}. Defaulting to NO_ACTION.")
        return ("NO_ACTION", 0.0)


    def calculate_complexity(self) -> Tuple[int, int]: # Return type changed
        def _count_and_depth(node: Optional[LogicNode], depth: int = 1) -> Tuple[int, int]:
            if node is None:
                return 0, 0
            if isinstance(node, (ActionNode, ConditionNode)):
                return 1, depth
            elif isinstance(node, (CompositeNode, SequenceNode)):
                left_count, left_depth = _count_and_depth(node.child1, depth+1)
                right_count, right_depth = _count_and_depth(node.child2, depth+1)
                return 1 + left_count + right_count, max(left_depth, right_depth)
            else: # Should not happen with proper node types
                return 1, depth 
        node_count, tree_depth = _count_and_depth(self.root_node)
        return node_count, tree_depth

    def to_string_representation(self) -> str:
        """
        Return a LISP-like, unambiguous string representation of the LogicDNA tree for MLE parsing.
        """
        def _to_lisp(node: Optional[LogicNode]) -> str: # Typed node parameter
            if node is None:
                return "(NONE)"
            if isinstance(node, ConditionNode):
                op: str = node.comparison_operator
                return f"(CONDITION {node.indicator_id}_{node.lookback_period_1}{('_' + str(node.lookback_period_2)) if node.lookback_period_2 else ''}_{op}_{round(node.threshold_value, 4)})"
            elif isinstance(node, ActionNode):
                return f"(ACTION {node.action_type}_{round(node.size_factor, 4)})"
            elif isinstance(node, CompositeNode):
                return f"(COMPOSITE_{node.logical_operator} {_to_lisp(node.child1)} {_to_lisp(node.child2)})"
            elif isinstance(node, SequenceNode):
                return f"(SEQUENCE {_to_lisp(node.child1)} {_to_lisp(node.child2)})"
            else:
                return f"(UNKNOWN_NODE_TYPE_{type(node).__name__})" # More informative
        return _to_lisp(self.root_node)

    def copy(self) -> 'LogicDNA_v1':
        new_dna_id: str = str(uuid.uuid4()) # Ensure new_dna_id is str
        new_dna: LogicDNA_v1 = LogicDNA_v1(
            dna_id=new_dna_id,
            root_node=self.root_node.copy() if self.root_node else None,
            generation_born=self.generation_born
        )
        if hasattr(self, 'fitness'): # Preserve fitness if it exists
            new_dna.fitness = self.fitness
        return new_dna

    def get_random_node(self) -> Optional[LogicNode]: # Changed return type
        """Return a randomly selected node from the tree."""
        nodes: List[LogicNode] = [] # Typed list
        def _collect(node: Optional[LogicNode]) -> None: # Typed node parameter
            if node is None:
                return
            nodes.append(node)
            if isinstance(node, (CompositeNode, SequenceNode)):
                _collect(node.child1)
                _collect(node.child2)
        _collect(self.root_node)
        return random.choice(nodes) if nodes else None

    def get_random_subtree_point(self) -> Optional[LogicNode]: # Changed return type
        """Return a randomly selected node that can serve as a subtree root (any node)."""
        return self.get_random_node()

    def replace_node(self, old_node_id: str, new_node_subtree_root: LogicNode) -> None:
        """Replace the node with old_node_id with new_node_subtree_root."""
        def _replace(node: Optional[LogicNode]) -> Optional[LogicNode]: # Typed parameters and return
            if node is None:
                return None
            if getattr(node, 'node_id', None) == old_node_id:
                new_node_subtree_root.parent_id = node.parent_id # Preserve parentage
                return new_node_subtree_root
            if isinstance(node, CompositeNode):
                node.child1 = _replace(node.child1)
                if node.child1: node.child1.parent_id = node.node_id # Update parent_id
                node.child2 = _replace(node.child2)
                if node.child2: node.child2.parent_id = node.node_id # Update parent_id
            elif isinstance(node, SequenceNode):
                node.child1 = _replace(node.child1)
                if node.child1: node.child1.parent_id = node.node_id # Update parent_id
                node.child2 = _replace(node.child2)
                if node.child2: node.child2.parent_id = node.node_id # Update parent_id
            return node
        self.root_node = _replace(self.root_node)

    def add_node_at_random_valid_point(self, new_node: LogicNode) -> bool:
        """Insert new_node at a random valid point in the tree, e.g., as a child of a Composite/Sequence node or replacing an Action node."""
        candidates: List[Tuple[Optional[LogicNode], Optional[int]]] = [] # parent, child_index (1 or 2, or None for root)
        
        def _find_candidates(node: Optional[LogicNode], parent: Optional[LogicNode]=None, child_idx: Optional[int]=None) -> None: # Typed parameters
            if node is None: # A potential spot in Composite/Sequence
                if parent and isinstance(parent, (CompositeNode, SequenceNode)): # Ensure parent can have children
                    candidates.append((parent, child_idx))
                return

            # If ActionNode, can be replaced by a new subtree (new_node could be Condition, Composite etc.)
            if isinstance(node, ActionNode):
                candidates.append((parent, child_idx)) # Store parent and which child it was
            
            if isinstance(node, (CompositeNode, SequenceNode)):
                _find_candidates(node.child1, node, 1)
                _find_candidates(node.child2, node, 2)

        if self.root_node is None: # If tree is empty, new_node becomes root
            candidates.append((None, None)) 
        else:
            _find_candidates(self.root_node)

        if not candidates:
            # This case should ideally not happen if the tree is not empty or if root can be replaced.
            # If root is not None and no candidates, means root is not Action and has no None children.
            # We can choose to replace the root in this case if new_node is not an Action node.
            if not isinstance(new_node, ActionNode) and self.root_node and not isinstance(self.root_node, ActionNode):
                 # Heuristic: if new_node is complex and root is complex, replace root.
                 # This might not always be ideal. A more robust strategy could be needed.
                 pass # For now, do nothing if no clear spot and root is not Action
            elif isinstance(self.root_node, ActionNode): # if root is action, it's a candidate
                 candidates.append((None,None))
            if not candidates: return False


        chosen_parent, chosen_child_idx = random.choice(candidates)
        new_node.parent_id = chosen_parent.node_id if chosen_parent else None

        if chosen_parent is None: # Replacing the root or setting root if tree was empty
            self.root_node = new_node
        elif isinstance(chosen_parent, (CompositeNode, SequenceNode)):
            if chosen_child_idx == 1:
                chosen_parent.child1 = new_node
            elif chosen_child_idx == 2:
                chosen_parent.child2 = new_node
        return True

    def prune_random_subtree(self) -> bool: # Return type was missing
        """Select a random non-root node and replace its subtree with a default ActionNode('HOLD', 0.0)."""
        nodes_with_parent_info: List[Tuple[LogicNode, Optional[LogicNode], Optional[int]]] = [] # node, parent, child_idx
        
        def _collect(node: Optional[LogicNode], parent: Optional[LogicNode]=None, child_idx: Optional[int]=None) -> None: # Typed parameters
            if node is None:
                return
            # Add any node to be potentially pruned, along with its parent context
            nodes_with_parent_info.append((node, parent, child_idx))
            if isinstance(node, (CompositeNode, SequenceNode)):
                _collect(node.child1, node, 1)
                _collect(node.child2, node, 2)
        
        if self.root_node is None: return False
        _collect(self.root_node)

        # Filter out the root node itself for pruning, unless it's the only node
        prunable_nodes = [(n, p, ci) for n, p, ci in nodes_with_parent_info if p is not None]
        if not prunable_nodes: # Only root node exists or tree is empty
            if self.root_node and not isinstance(self.root_node, ActionNode): # If root is not already a simple action
                self.root_node = ActionNode(action_type="HOLD", size_factor=0.0, parent_id=None)
                self.root_node.parent_id = None # Explicitly set parent_id for new root
                return True
            return False # Cannot prune if root is already a simple action or tree empty

        _, parent_of_pruned, child_idx_of_pruned = random.choice(prunable_nodes)
        
        default_action = ActionNode(action_type="HOLD", size_factor=0.0)
        default_action.parent_id = parent_of_pruned.node_id if parent_of_pruned else None # type: ignore

        if parent_of_pruned is None: # Should not happen due to filter, but defensive
             self.root_node = default_action
        elif isinstance(parent_of_pruned, (CompositeNode, SequenceNode)):
            if child_idx_of_pruned == 1:
                parent_of_pruned.child1 = default_action
            elif child_idx_of_pruned == 2:
                parent_of_pruned.child2 = default_action
        return True

    def is_valid(self, max_depth: int, max_nodes: int) -> bool:
        def _validate(node: Optional[LogicNode], depth: int =1) -> Tuple[bool, int]: # Typed node and return
            if node is None: # A None child for Composite/Sequence is considered valid in structure, but count as 0
                return True, 0 
            if depth > max_depth:
                return False, 1 # Count the node even if it violates depth
            
            current_node_count: int = 1
            children_valid: bool = True

            if isinstance(node, (CompositeNode, SequenceNode)):
                # Both children must exist for Composite/Sequence to be structurally sound for evaluation purposes
                if node.child1 is None or node.child2 is None:
                    return False, current_node_count # Invalid if children are missing for these types

                valid1, count1 = _validate(node.child1, depth+1)
                valid2, count2 = _validate(node.child2, depth+1)
                children_valid = valid1 and valid2
                current_node_count += count1 + count2
            # ActionNode and ConditionNode are leaves or have specific structures handled by their own validation if any
            
            return children_valid, current_node_count

        if self.root_node is None: # An empty tree can be considered valid (evaluates to NO_ACTION)
            return True

        valid_structure, total_nodes = _validate(self.root_node)
        if not valid_structure or total_nodes > max_nodes:
            return False
        
        # Check that Action nodes are leaves
        # This check is implicitly handled if Composite/Sequence nodes require non-None children
        # and ActionNodes don't have children attributes.
        # A more explicit check might be needed if node structure rules are complex.
        return True


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
    id: str
    trigger_indicator: str
    trigger_operator: str
    trigger_threshold: float
    context_regime_id: str
    action_target: str
    action_type: str
    action_value: float
    resource_cost: float
    parent_id: Optional[str]
    seed_type: Optional[str]

    def __init__(self, trigger_indicator: str, trigger_operator: str, trigger_threshold: float, 
                 context_regime_id: str, action_target: str, action_type: str, action_value: float, 
                 resource_cost: float = 1.0, parent_id: Optional[str] = None, seed_type: Optional[str] = None) -> None:
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
    def seed_rsi_buy() -> 'LogicDNA':
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
    def seed_rsi_sell() -> 'LogicDNA':
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

    def __repr__(self) -> str:
        return (f"LogicDNA(id={self.id}, parent_id={self.parent_id}, seed_type={self.seed_type}, trigger={self.trigger_indicator} {self.trigger_operator} {self.trigger_threshold}, "
                f"context={self.context_regime_id}, action={self.action_target} {self.action_type} {self.action_value}, "
                f"resource_cost={self.resource_cost})")

    def to_dict(self) -> Dict[str, Any]:
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

def mutate_dna(dna_instance: LogicDNA, mutation_strength: float = 0.1) -> LogicDNA:
    """
    Randomly mutates one numerical parameter (threshold or action_value) by ±mutation_strength (default 10%).
    Args:
        dna_instance (LogicDNA): The DNA instance to mutate.
        mutation_strength (float): The mutation strength as a fraction.
    Returns:
        LogicDNA: A new mutated DNA instance with parent_id set to the original's id.
    """
    new_dna: LogicDNA = copy.deepcopy(dna_instance)
    param_to_mutate: str = random.choice(['trigger_threshold', 'action_value'])
    
    orig_value: float = getattr(new_dna, param_to_mutate)
    change_pct: float = random.uniform(-mutation_strength, mutation_strength)
    new_value: float = orig_value * (1 + change_pct)
    
    setattr(new_dna, param_to_mutate, new_value)
    new_dna.parent_id = dna_instance.id  # Track lineage
    new_dna.seed_type = None  # Not a seed
    new_dna.id = uuid.uuid4().hex[:8] # Assign a new ID to the mutated instance
    logger.info(f"Mutated {param_to_mutate} from {orig_value} to {new_value:.4f} in DNA: {new_dna.id} (parent: {dna_instance.id})")
    return new_dna 
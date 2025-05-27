import os
import pandas as pd
import pytest
from src.mle_engine import MLE_v0_1

def make_mock_performance_log(path):
    data = [
        {'dna_id': 'dna1', 'generation_born': 0, 'current_generation_evaluated': 1, 'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 1.0, 'active_persona_name': 'HUNTER', 'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:00:00'},
        {'dna_id': 'dna2', 'generation_born': 0, 'current_generation_evaluated': 1, 'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 2.0, 'active_persona_name': 'HUNTER', 'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:01:00'},
        {'dna_id': 'dna3', 'generation_born': 1, 'current_generation_evaluated': 2, 'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 3.0, 'active_persona_name': 'HUNTER', 'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:02:00'},
        {'dna_id': 'dna4', 'generation_born': 1, 'current_generation_evaluated': 2, 'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 4.0, 'active_persona_name': 'HUNTER', 'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:03:00'},
    ]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def make_detailed_mock_performance_log(path):
    # Rows: some with motifs, some without, some in top performers, some not
    data = [
        # Generation 1
        {'dna_id': 'dna1', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '(CONDITION SMA_5_GREATER_THAN_10.0) (ACTION BUY_0.5)',
         'performance_metrics': '{}', 'fitness_score': 1.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:00:00'},
        {'dna_id': 'dna2', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '(CONDITION RSI_14_LESS_THAN_30.0) (ACTION SELL_0.7)',
         'performance_metrics': '{}', 'fitness_score': 2.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:01:00'},
        # Generation 2, high fitness, with composite/sequence
        {'dna_id': 'dna3', 'generation_born': 1, 'current_generation_evaluated': 2,
         'logic_dna_structure_representation': '(COMPOSITE_AND (CONDITION SMA_5_GREATER_THAN_10.0) (CONDITION RSI_14_GREATER_THAN_50.0))',
         'performance_metrics': '{}', 'fitness_score': 10.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:02:00'},
        {'dna_id': 'dna4', 'generation_born': 1, 'current_generation_evaluated': 2,
         'logic_dna_structure_representation': '(SEQUENCE (CONDITION SMA_5_GREATER_THAN_10.0) (ACTION BUY_0.5))',
         'performance_metrics': '{}', 'fitness_score': 12.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:03:00'},
        # Generation 3, low fitness, motifs present but not top
        {'dna_id': 'dna5', 'generation_born': 2, 'current_generation_evaluated': 3,
         'logic_dna_structure_representation': '(CONDITION EMA_20_GREATER_THAN_100.0) (ACTION HOLD_0.0)',
         'performance_metrics': '{}', 'fitness_score': 0.5, 'active_persona_name': 'GUARDIAN',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:04:00'},
        # Generation 3, no motifs
        {'dna_id': 'dna6', 'generation_born': 2, 'current_generation_evaluated': 3,
         'logic_dna_structure_representation': '',
         'performance_metrics': '{}', 'fitness_score': 0.2, 'active_persona_name': 'GUARDIAN',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:05:00'},
    ]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def test_mle_engine_initialization(tmp_path):
    log_path = tmp_path / "mock_performance_log.csv"
    make_mock_performance_log(log_path)
    mle = MLE_v0_1(str(log_path))
    assert mle.performance_log_path == str(log_path)

def test_analyze_recent_performance_returns_bias_dict(tmp_path):
    log_path = tmp_path / "mock_performance_log.csv"
    make_mock_performance_log(log_path)
    mle = MLE_v0_1(str(log_path))
    bias = mle.analyze_recent_performance(num_recent_generations=2, top_x_percent=0.5)
    assert isinstance(bias, dict)
    assert 'seed_motifs' in bias
    assert 'recommended_operator_biases' in bias
    assert isinstance(bias['seed_motifs'], dict)
    assert isinstance(bias['recommended_operator_biases'], dict)
    # For a simple log, seed_motifs may be empty
    assert bias['seed_motifs'] == {} or isinstance(bias['seed_motifs'], dict)

def test_mle_motif_extraction_top_performers(tmp_path):
    log_path = tmp_path / "mock_performance_log.csv"
    make_detailed_mock_performance_log(log_path)
    mle = MLE_v0_1(str(log_path))
    # Only top 50% by fitness in last 2 generations (gen 2,3): dna3, dna4 (high fitness)
    bias = mle.analyze_recent_performance(num_recent_generations=2, top_x_percent=0.5)
    motifs = bias['seed_motifs']
    # Should find composite AND motif, sequence motif, indicator usage, and condition-action pair
    assert any('(COMPOSITE_AND' in k for k in motifs)
    assert any('(SEQUENCE' in k for k in motifs)
    assert 'Indicator_SMA_5_Used' in motifs
    assert 'Indicator_RSI_14_Used' in motifs
    # Should count the sequence Condition-Action motif
    assert any('(CONDITION SMA_5_GREATER_THAN_#VAL) (ACTION BUY_#SIZE)' in k or '(CONDITION SMA_5_GREATER_THAN_#VAL)' in k for k in motifs)
    # Check counts (should be 1 for each in this mock)
    assert motifs['Indicator_SMA_5_Used'] >= 1
    assert motifs['Indicator_RSI_14_Used'] >= 1

def test_mle_motif_extraction_no_motifs_in_top(tmp_path):
    log_path = tmp_path / "mock_performance_log.csv"
    # All top performers have empty or non-motif strings
    data = [
        {'dna_id': 'dna1', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 10.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:00:00'},
        {'dna_id': 'dna2', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 9.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:01:00'},
    ]
    pd.DataFrame(data).to_csv(log_path, index=False)
    mle = MLE_v0_1(str(log_path))
    bias = mle.analyze_recent_performance(num_recent_generations=1, top_x_percent=1.0)
    assert bias['seed_motifs'] == {}

def test_mle_motif_extraction_motifs_not_in_top(tmp_path):
    log_path = tmp_path / "mock_performance_log.csv"
    # Motifs present but only in low fitness rows
    data = [
        {'dna_id': 'dna1', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '(CONDITION SMA_5_GREATER_THAN_10.0) (ACTION BUY_0.5)', 'performance_metrics': '{}', 'fitness_score': 1.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:00:00'},
        {'dna_id': 'dna2', 'generation_born': 0, 'current_generation_evaluated': 1,
         'logic_dna_structure_representation': '', 'performance_metrics': '{}', 'fitness_score': 10.0, 'active_persona_name': 'HUNTER',
         'ces_vector_at_evaluation_time': '{}', 'timestamp_of_evaluation': '2024-01-01T00:01:00'},
    ]
    pd.DataFrame(data).to_csv(log_path, index=False)
    mle = MLE_v0_1(str(log_path))
    bias = mle.analyze_recent_performance(num_recent_generations=1, top_x_percent=0.5)
    # Only dna2 is top performer, but has no motifs
    assert bias['seed_motifs'] == {} 
"""
Meta-Learning Engine (MLE) for Project Crypto Bot Peter.

This module is responsible for analyzing performance data from evolutionary
runs to identify successful patterns, motifs, or characteristics within
LogicDNA structures. The insights (bias) derived are then used to guide
subsequent evolutionary processes, aiming to accelerate the discovery of
effective trading strategies.

Key Components:
- MLE_v0_1: Analyzes performance logs (CSV) to extract common motifs
  from top-performing LogicDNA instances and generates a bias dictionary.
"""
import pandas as pd
import os
import re
import logging # Added
import time # Added
from typing import Dict, Any, List, Pattern, Match # Added typing imports

class MLE_v0_1:
    """
    Meta-Learning Engine v0.1 for Project Crypto Bot Peter.
    Analyzes recent performance logs and returns a bias dictionary for evolutionary guidance.
    """
    performance_log_path: str
    logger: logging.Logger
    _motif_monoculture_history: List[str] # Attribute hint

    def __init__(self, performance_log_path: str) -> None:
        self.performance_log_path = performance_log_path
        self.logger = logging.getLogger(__name__) # Added logger initialization
        self._motif_monoculture_history = [] # Initialize attribute

    def analyze_recent_performance(self, num_recent_generations: int = 5, top_x_percent: float = 0.10) -> Dict[str, Any]: # Return type changed
        """
        Analyze the last num_recent_generations in the performance log and return a bias dictionary.
        Extract actual motifs from logic_dna_structure_representation in top performers.
        """
        if not os.path.exists(self.performance_log_path):
            self.logger.error(f"Performance log not found: {self.performance_log_path}") # Changed to error
            raise FileNotFoundError(f"Performance log not found: {self.performance_log_path}")
        
        start_time_csv: float = time.time() 
        df: pd.DataFrame = pd.read_csv(self.performance_log_path)
        duration_csv: float = time.time() - start_time_csv 
        self.logger.info(f"MLE: Time taken to read performance log CSV: {duration_csv:.4f} seconds") 
        
        if 'current_generation_evaluated' not in df.columns or 'fitness_score' not in df.columns:
            self.logger.error("Performance log missing required columns: 'current_generation_evaluated' or 'fitness_score'") # Changed to error
            raise ValueError("Performance log missing required columns.")
        
        start_time_filter: float = time.time() 
        # Get the last N generations
        recent_gens: List[Any] = sorted(df['current_generation_evaluated'].unique())[-num_recent_generations:]
        df_recent: pd.DataFrame = df[df['current_generation_evaluated'].isin(recent_gens)]
        
        if len(df_recent) == 0:
            self.logger.info("MLE: No recent data to process after filtering for recent generations.") 
            return {'seed_motifs': {}, 'recommended_operator_biases': {}}
        
        threshold: float = df_recent['fitness_score'].quantile(1 - top_x_percent)
        top_df: pd.DataFrame = df_recent[df_recent['fitness_score'] >= threshold]
        duration_filter: float = time.time() - start_time_filter 
        self.logger.info(f"MLE: Time taken for data processing/filtering top performers: {duration_filter:.4f} seconds") 
        
        motif_counts: Dict[str, int] = {}
        indicator_counts: Dict[str, int] = {}
        triplet_counts: Dict[str, int] = {}
        
        start_time_motif: float = time.time() 
        dna_strings: List[str] = top_df.get('logic_dna_structure_representation', pd.Series(dtype=str)).tolist()

        for s in dna_strings:
            if not isinstance(s, str) or not s:
                continue
            
            # DEBUG: print the string being processed
            # self.logger.debug(f"MLE: [DEBUG] Processing DNA string: {s}") # Changed print to logger.debug
            
            # 1. Condition-Action pairs (within Sequence or anywhere)
            cond_act_pattern: Pattern[str] = re.compile(r'\(CONDITION [^\)]+\)\s*\(ACTION [^\)]+\)')
            for match in cond_act_pattern.finditer(s):
                motif: str = match.group(0)
                motif = re.sub(r'(_[0-9.]+\))', r'_#VAL)', motif)
                motif = re.sub(r'(ACTION [A-Z]+)_([0-9.]+)\)', r'\1_#SIZE)', motif)
                # self.logger.debug(f"MLE: [DEBUG] Found Condition-Action motif: {motif}")
                motif_counts[motif] = motif_counts.get(motif, 0) + 1
            
            # 2. Indicator usage
            indicators: List[Tuple[str, str]] = re.findall(r'CONDITION ([A-Z]+)_([0-9]+)', s)
            for ind, period in indicators:
                ind_motif: str = f"Indicator_{ind}_{period}_Used"
                # self.logger.debug(f"MLE: [DEBUG] Found Indicator motif: {ind_motif}")
                indicator_counts[ind_motif] = indicator_counts.get(ind_motif, 0) + 1
            
            # 3. Composite/Sequence node triplets (match nested parenthesis)
            triplet_pattern: Pattern[str] = re.compile(r'\((COMPOSITE_[A-Z]+|SEQUENCE) ((?:[^()]+|\([^()]*\))+?) ((?:[^()]+|\([^()]*\))+?)\)')
            for match in triplet_pattern.finditer(s):
                triplet: str = match.group(0)
                triplet = re.sub(r'(_[0-9.]+\))', r'_#VAL)', triplet)
                # self.logger.debug(f"MLE: [DEBUG] Found Composite/Sequence motif: {triplet}")
                triplet_counts[triplet] = triplet_counts.get(triplet, 0) + 1
        
        seed_motifs: Dict[str, int] = {}
        seed_motifs.update(motif_counts)
        seed_motifs.update(indicator_counts)
        seed_motifs.update(triplet_counts)
        
        bias_dict: Dict[str, Any] = {
            'seed_motifs': seed_motifs,
            'recommended_operator_biases': {
                'structural_mutation_rate_adjustment_factor': 1.1 # Example, could be dynamic
            }
        }
        
        total_motifs: int = sum(seed_motifs.values())
        if total_motifs > 0:
            top_motif, top_count = max(seed_motifs.items(), key=lambda item: item[1]) # Corrected max() usage
            if top_count / total_motifs > 0.7:
                # _motif_monoculture_history is already initialized in __init__
                self._motif_monoculture_history.append(top_motif)
                self._motif_monoculture_history = self._motif_monoculture_history[-5:]
                if self._motif_monoculture_history.count(top_motif) >= 3:
                    self.logger.warning(f"MLE: Potential Motif Monoculture for motif {top_motif}") # Changed print to logger.warning
        
        duration_motif: float = time.time() - start_time_motif 
        self.logger.info(f"MLE: Time taken for motif identification loop: {duration_motif:.4f} seconds") 
        return bias_dict

    def process_new_data(self) -> None: # Added return type hint
        """
        Placeholder for future incremental learning.
        """
        pass 
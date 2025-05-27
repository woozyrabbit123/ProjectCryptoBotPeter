import pandas as pd
import os
import re

class MLE_v0_1:
    """
    Meta-Learning Engine v0.1 for Project Crypto Bot Peter.
    Analyzes recent performance logs and returns a bias dictionary for evolutionary guidance.
    """
    def __init__(self, performance_log_path: str):
        self.performance_log_path = performance_log_path

    def analyze_recent_performance(self, num_recent_generations: int = 5, top_x_percent: float = 0.10) -> dict:
        """
        Analyze the last num_recent_generations in the performance log and return a bias dictionary.
        Extract actual motifs from logic_dna_structure_representation in top performers.
        """
        if not os.path.exists(self.performance_log_path):
            raise FileNotFoundError(f"Performance log not found: {self.performance_log_path}")
        df = pd.read_csv(self.performance_log_path)
        if 'current_generation_evaluated' not in df.columns or 'fitness_score' not in df.columns:
            raise ValueError("Performance log missing required columns.")
        # Get the last N generations
        recent_gens = sorted(df['current_generation_evaluated'].unique())[-num_recent_generations:]
        df_recent = df[df['current_generation_evaluated'].isin(recent_gens)]
        # Filter for top X percent by fitness_score
        if len(df_recent) == 0:
            return {'seed_motifs': {}, 'recommended_operator_biases': {}}
        threshold = df_recent['fitness_score'].quantile(1 - top_x_percent)
        top_df = df_recent[df_recent['fitness_score'] >= threshold]
        # Motif extraction
        motif_counts = {}
        indicator_counts = {}
        triplet_counts = {}
        for s in top_df.get('logic_dna_structure_representation', []):
            if not isinstance(s, str) or not s:
                continue
            # DEBUG: print the string being processed
            print(f"[DEBUG] Processing DNA string: {s}")
            # 1. Condition-Action pairs (within Sequence or anywhere)
            cond_act_pattern = re.compile(r'\(CONDITION [^\)]+\)\s*\(ACTION [^\)]+\)')
            for match in cond_act_pattern.finditer(s):
                motif = match.group(0)
                # Generalize threshold and action size
                motif = re.sub(r'(_[0-9.]+\))', r'_#VAL)', motif)  # Generalize all trailing numbers to _#VAL)
                motif = re.sub(r'(ACTION [A-Z]+)_([0-9.]+)\)', r'\1_#SIZE)', motif)  # For action size
                print(f"[DEBUG] Found Condition-Action motif: {motif}")
                motif_counts[motif] = motif_counts.get(motif, 0) + 1
            # 2. Indicator usage
            indicators = re.findall(r'CONDITION ([A-Z]+)_([0-9]+)', s)
            for ind, period in indicators:
                ind_motif = f"Indicator_{ind}_{period}_Used"
                print(f"[DEBUG] Found Indicator motif: {ind_motif}")
                indicator_counts[ind_motif] = indicator_counts.get(ind_motif, 0) + 1
            # 3. Composite/Sequence node triplets (match nested parenthesis)
            # This regex matches (COMPOSITE_AND ... ...) or (SEQUENCE ... ...), where ... can be nested
            triplet_pattern = re.compile(r'\((COMPOSITE_[A-Z]+|SEQUENCE) ((?:[^()]+|\([^()]*\))+?) ((?:[^()]+|\([^()]*\))+?)\)')
            for match in triplet_pattern.finditer(s):
                triplet = match.group(0)
                triplet = re.sub(r'(_[0-9.]+\))', r'_#VAL)', triplet)
                print(f"[DEBUG] Found Composite/Sequence motif: {triplet}")
                triplet_counts[triplet] = triplet_counts.get(triplet, 0) + 1
        # Merge all motifs
        seed_motifs = {}
        seed_motifs.update(motif_counts)
        seed_motifs.update(indicator_counts)
        seed_motifs.update(triplet_counts)
        bias_dict = {
            'seed_motifs': seed_motifs,
            'recommended_operator_biases': {
                'structural_mutation_rate_adjustment_factor': 1.1
            }
        }
        # Motif monoculture detection
        total_motifs = sum(seed_motifs.values())
        if total_motifs > 0:
            top_motif, top_count = max(seed_motifs.items(), key=lambda x: x[1])
            if top_count / total_motifs > 0.7:
                if not hasattr(self, '_motif_monoculture_history'):
                    self._motif_monoculture_history = []
                self._motif_monoculture_history.append(top_motif)
                # Only keep last 5
                self._motif_monoculture_history = self._motif_monoculture_history[-5:]
                if self._motif_monoculture_history.count(top_motif) >= 3:
                    print(f"WARNING: Potential Motif Monoculture for motif {top_motif}")
        return bias_dict

    def process_new_data(self):
        """
        Placeholder for future incremental learning.
        """
        pass 
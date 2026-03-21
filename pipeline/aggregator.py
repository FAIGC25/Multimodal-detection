from typing import Dict
from core.interfaces import BaseAggregator
from core.entities import DetectionResult

class WeightedAggregator(BaseAggregator):
    def __init__(self, weights: Dict[str, float] = None):
        # По умолчанию веса равны (заглушка на будущее)
        self.weights = weights or {"spatial": 0.5, "temporal": 0.5}

    def aggregate(self, results: Dict[str, DetectionResult]) -> DetectionResult:
        print("[Aggregator] Агрегация результатов со всех модальностей...")
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for mod_name, res in results.items():
            w = self.weights.get(mod_name, 0.0)
            weighted_score += res.score * w
            total_weight += w
            
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return DetectionResult(
            score=final_score,
            metadata={"individual_scores": {k: v.score for k, v in results.items()}}
        )

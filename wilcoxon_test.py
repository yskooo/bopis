import math
import dataclasses
from typing import List, Sequence, Optional, Dict

def _rank_data(data: Sequence[float]) -> List[float]:
    order = sorted(range(len(data)), key=data.__getitem__)
    ranks = [0.0] * len(data)
    position = 0
    while position < len(data):
        span = position + 1
        while span < len(data) and data[order[span]] == data[order[position]]:
            span += 1
        average = (position + span + 1) / 2.0
        for idx in range(position, span):
            ranks[order[idx]] = average
        position = span
    return ranks

@dataclasses.dataclass
class WilcoxonResult:
    n_effective: int
    w_statistic: float
    z_statistic: float
    p_value: float
    tie_correction: float
    condition_names: List[str]
    alpha: float = 0.05
    
    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_effective": self.n_effective,
            "w_statistic": self.w_statistic,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "tie_correction": self.tie_correction,
        }

def wilcoxon(
    blocks: Sequence[Sequence[float]],
    condition_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> WilcoxonResult:
    if not blocks:
        raise ValueError("wilcoxon requires at least one block")
    k = len(blocks[0])
    if k != 2:
        raise ValueError("wilcoxon requires exactly two conditions")
    
    if condition_names is None:
        condition_names = ["Condition A", "Condition B"]
        
    diffs = []
    for block in blocks:
        diff = block[0] - block[1]
        if diff != 0:
            diffs.append(diff)
            
    n = len(diffs)
    if n == 0:
        return WilcoxonResult(0, 0.0, 0.0, 1.0, 0.0, list(condition_names), alpha)

    abs_diffs = [abs(d) for d in diffs]
    ranks = _rank_data(abs_diffs)
    
    w_pos = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_neg = sum(r for d, r in zip(diffs, ranks) if d < 0)
    
    w_stat = min(w_pos, w_neg)
    
    t_sum = 0
    unique_vals = set(abs_diffs)
    for v in unique_vals:
        t = abs_diffs.count(v)
        if t > 1:
            t_sum += (t**3 - t)
            
    expected_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1)) / 24.0 - (t_sum / 48.0)
    
    if var_w == 0:
        z = 0.0
        p_val = 1.0
    else:
        correction = 0.5 if abs(w_stat - expected_w) > 0.5 else 0.0
        z = (w_stat - expected_w + correction) / math.sqrt(var_w) if w_stat < expected_w else (w_stat - expected_w - correction) / math.sqrt(var_w)
        p_val = math.erfc(abs(z) / math.sqrt(2.0))
        
    return WilcoxonResult(n, w_stat, z, p_val, t_sum / 48.0, list(condition_names), alpha)

res = wilcoxon([[20, 15], [18, 12], [23, 23], [14, 11], [15, 12]])
print(res)

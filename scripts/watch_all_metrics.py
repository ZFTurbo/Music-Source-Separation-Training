"""
    Script to print metrics for checkpoint file of new format
"""

import argparse
import torch


def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')

    all_metrics = checkpoint['all_metrics']

    import numpy as np
    from typing import Dict, Any, List

    def _fmt_list(xs: List[float], n: int, p: int) -> str:
        if not xs:
            return "[]"
        xs = [float(x) for x in xs]
        head = ", ".join(f"{v:.{p}f}" for v in xs[:n])
        if len(xs) > n:
            return f"[{head}, … {len(xs) - n} more]"
        return f"[{head}]"

    def pretty_metrics(
            all_time_all_metrics: Dict[str, Any],
            *,
            precision: int = 4,
            show_items: int = 6
    ) -> str:
        lines = []
        for epoch_key in sorted(all_time_all_metrics.keys(), key=lambda k: int(k.split('_')[-1])):
            m = all_time_all_metrics[epoch_key]
            lines.append(f"\n=== {epoch_key} ===")
            for metric_name, per_instr in m.items():
                lines.append(f"{metric_name}:")
                for instr, values in per_instr.items():
                    arr = np.array(values, dtype=float)
                    mean = np.mean(arr) if arr.size else float('nan')
                    std = np.std(arr) if arr.size else float('nan')
                    preview = _fmt_list(values, show_items, precision)
                    lines.append(
                        f"  {instr:>10s}: mean={mean:.{precision}f}  std={std:.{precision}f}  "
                        f"n={arr.size}  values={preview}"
                    )
        return "\n".join(lines)


    print(pretty_metrics(all_metrics, precision=4, show_items=6))


if __name__ == '__main__':
    main()
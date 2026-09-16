"""Combine frozen development gates with unchanged hold-out evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dev = json.loads((ROOT / "outputs/formal_analysis/development_gate_completion.json").read_text())
    hold = json.loads((ROOT / "outputs/holdout_analysis/holdout_stratum_method_summary.json").read_text())
    policy = json.loads((ROOT / "configs/compound_event_qualification_policy.json").read_text())
    dev_map = {(x["stratum"], x["method"]): x for x in dev["groups_detail"]}
    rows = []
    for item in hold["summaries"]:
        key = (item["stratum"], item["method"])
        d = dev_map[key]
        qualified = bool(
            d["all_development_gates_pass"]
            and item["bias_gate_pass"]
            and item["rmse_gate_pass"]
            and item["all_record_95_percent_gate_pass"]
            and item["extraction_failure_gate_pass"]
        )
        rows.append({
            "stratum": item["stratum"],
            "method": item["method"],
            "development_all_gates_pass": d["all_development_gates_pass"],
            "holdout_bias_gate_pass": item["bias_gate_pass"],
            "holdout_rmse_gate_pass": item["rmse_gate_pass"],
            "holdout_95_percent_gate_pass": item["all_record_95_percent_gate_pass"],
            "holdout_extraction_gate_pass": item["extraction_failure_gate_pass"],
            "final_status": "Qualified" if qualified else "Not qualified",
            "decision_basis": "all frozen development gates and unchanged hold-out numerical gates",
        })
    out = ROOT / "outputs/qualification_matrix"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "primary_qualification_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compound = {
        "status": "DESCRIPTIVE_ONLY",
        "policy_id": policy["policy_id"],
        "compound_strata": policy["compound_strata"],
        "rows": hold["compound_rows_descriptive_only"],
        "primary_qualification_label": None,
        "reason": policy["rationale"],
    }
    (out / "compound_descriptive_status.json").write_text(json.dumps(compound, indent=2) + "\n")
    audit = {
        "status": "PASS",
        "primary_groups": len(rows),
        "qualified": sum(r["final_status"] == "Qualified" for r in rows),
        "not_qualified": sum(r["final_status"] == "Not qualified" for r in rows),
        "compound_groups": len(policy["compound_strata"]),
        "compound_status": "DESCRIPTIVE_ONLY",
        "source_development": "outputs/formal_analysis/development_gate_completion.json",
        "source_holdout": "outputs/holdout_analysis/holdout_stratum_method_summary.json",
        "errors": [],
    }
    (out / "qualification_matrix_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

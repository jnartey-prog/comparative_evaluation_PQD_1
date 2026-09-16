"""Audit frozen-design and synthetic-generator consistency without confirmation outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tfpq_qualifier.synthetic import COMPOUND_COMPONENTS, SyntheticDataGenerator, load_conditions

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main() -> int:
    generator = SyntheticDataGenerator(ROOT / "configs/scientific_design.json")
    design = generator.design
    lock = json.loads((ROOT / "manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))
    config_for_hash = dict(design)
    config_for_hash["configuration_sha256"] = None
    calculated_design_hash = hashlib.sha256(canonical(config_for_hash)).hexdigest()
    development = list(load_conditions(ROOT / "manifests/development_conditions.csv"))
    confirmation = list(load_conditions(ROOT / "manifests/sealed_confirmation_conditions.csv"))
    for condition in development + confirmation:
        generator.validate_condition(condition)
    findings: list[dict[str, str]] = []

    def finding(severity: str, code: str, message: str) -> None:
        findings.append({"severity": severity, "code": code, "message": message})

    if calculated_design_hash != design["configuration_sha256"]:
        finding("BLOCKER", "DESIGN_HASH", "Canonical scientific-design hash does not reproduce.")
    if (
        generator.verify_manifest(ROOT / "manifests/development_conditions.csv")
        != lock["development_manifest_sha256"]
    ):
        finding("BLOCKER", "DEV_HASH", "Development manifest hash mismatch.")
    if (
        generator.verify_manifest(ROOT / "manifests/sealed_confirmation_conditions.csv")
        != lock["sealed_confirmation_manifest_sha256"]
    ):
        finding("BLOCKER", "CONF_HASH", "Confirmation manifest hash mismatch.")
    if {x.condition_id for x in development} & {x.condition_id for x in confirmation}:
        finding("BLOCKER", "ID_LEAKAGE", "Condition identifiers overlap partitions.")
    if {x.seed_namespace for x in development} & {x.seed_namespace for x in confirmation}:
        finding("BLOCKER", "SEED_LEAKAGE", "Seed namespaces overlap partitions.")
    expected_strata = set(design["parameter_domains"]) | set(COMPOUND_COMPONENTS)
    if expected_strata != {x.stratum for x in development}:
        finding(
            "BLOCKER",
            "STRATA",
            "Development manifest does not cover the complete approved stratum set.",
        )

    sag_conditions = [x for x in development if x.stratum == "sag"]
    realized_positions = {
        round(float(generator.generate(condition).truth["event_start_s"]), 12)
        for condition in sag_conditions[:32]
    }
    if len(realized_positions) == 1:
        finding(
            "AMENDMENT_REQUIRED",
            "POSITION_INACTIVE",
            "Event-position quantiles do not change event onset because the adaptive record has no positioning slack.",
        )
    flicker = min(
        (x for x in development if x.stratum == "flicker"),
        key=lambda x: x.parameters["modulation_frequency_hz"],
    )
    flicker_record = generator.generate(flicker)
    observed_cycles = (
        len(flicker_record.time_s)
        / flicker_record.sample_rate_hz
        * float(flicker.parameters["modulation_frequency_hz"])
    )
    if observed_cycles < 3:
        finding(
            "AMENDMENT_REQUIRED",
            "FLICKER_DURATION",
            f"The lowest-frequency flicker record contains only {observed_cycles:.3f} modulation cycles; frequency characterization needs a declared longer record policy.",
        )
    compound_flicker = min(
        (x for x in development if x.stratum == "flicker+harmonics"),
        key=lambda x: x.parameters["overlap_fraction"],
    )
    compound_record = generator.generate(compound_flicker)
    compound_frequency = 0.5 + 24.5 * float(compound_flicker.parameters["overlap_fraction"])
    compound_cycles = (
        len(compound_record.time_s) / compound_record.sample_rate_hz * compound_frequency
    )
    if compound_cycles < 3:
        finding(
            "AMENDMENT_REQUIRED",
            "COMPOUND_FLICKER_DURATION",
            f"The lowest-frequency flicker compound contains only {compound_cycles:.3f} cycles.",
        )
    harmonic_keys = set(design["parameter_domains"]["harmonics"])
    expected_harmonic_keys = {"a3_pu", "a5_pu", "a7_pu", "phi3_deg", "phi5_deg", "phi7_deg"}
    if harmonic_keys != expected_harmonic_keys:
        finding(
            "AMENDMENT_REQUIRED",
            "HARMONIC_CONVENTION",
            "Harmonic amplitudes and phases are not independently enumerated in the frozen design.",
        )
    actual_development_records = sum(x.noise_realizations for x in development)
    planned = sum(
        int(stage["records"])
        for name, stage in design["staged_generation"].items()
        if name != "formal_repeatability"
    )
    if actual_development_records != planned:
        finding(
            "AMENDMENT_REQUIRED",
            "REPLICATION_BUDGET",
            f"Manifest schedules {actual_development_records} initial development realizations, versus {planned} in the amended staged design.",
        )
    stage_counts: dict[str, int] = {}
    for condition in development:
        stage_counts[condition.stage] = (
            stage_counts.get(condition.stage, 0) + condition.noise_realizations
        )
    expected_stage_counts = {
        "property_response": 480,
        "broad_nuisance_screen": 5760,
        "compound_screen": 4000,
    }
    if stage_counts != expected_stage_counts:
        finding("BLOCKER", "STAGE_COUNTS", f"Stage record counts differ: {stage_counts}")
    summary = {
        "status": "PASS" if not findings else "AMENDMENT_REQUIRED",
        "design_sha256": generator.design_sha256,
        "generator_sha256": generator.generator_sha256,
        "development_conditions": len(development),
        "confirmation_conditions": len(confirmation),
        "development_realizations": actual_development_records,
        "hashes_and_partition_firewall_passed": not any(
            item["severity"] == "BLOCKER" for item in findings
        ),
        "findings": findings,
    }
    output = ROOT / "outputs/synthetic_generator_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not any(item["severity"] == "BLOCKER" for item in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())

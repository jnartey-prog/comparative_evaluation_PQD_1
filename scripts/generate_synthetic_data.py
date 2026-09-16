"""Stream frozen manifest conditions into auditable per-condition data files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tfpq_qualifier.synthetic import SyntheticDataGenerator, load_conditions, save_condition_batch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_formal_manifest(manifest: Path, generator: SyntheticDataGenerator) -> None:
    """Verify the formal manifest against the public release lock."""
    lock_path = Path("manifests/PUBLIC_RELEASE_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_manifest = lock.get("formal_repeatability_manifest_sha256")
    if sha256(manifest) != expected_manifest:
        raise ValueError("formal-repeatability manifest hash does not match PUBLIC_RELEASE_LOCK.json")
    if generator.design_sha256 != lock.get("scientific_design_canonical_sha256"):
        raise ValueError("scientific-design hash does not match PUBLIC_RELEASE_LOCK.json")


def verify_confirmation_authorization() -> None:
    """Require the frozen confirmation manifest recorded in the public release lock."""
    lock = json.loads(Path("manifests/PUBLIC_RELEASE_LOCK.json").read_text(encoding="utf-8"))
    manifest = Path("manifests/sealed_confirmation_conditions.csv")
    if sha256(manifest) != lock.get("sealed_confirmation_manifest_sha256"):
        raise SystemExit("Refusing confirmation generation: manifest hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/development_conditions.csv")
    parser.add_argument("--design", default="configs/scientific_design.json")
    parser.add_argument("--output", default="outputs/synthetic/development")
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--realizations", type=int, help="Cap realizations per condition")
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if "confirmation" in manifest.name.lower() and not args.allow_confirmation:
        raise SystemExit("Refusing sealed confirmation generation without --allow-confirmation")
    generator = SyntheticDataGenerator(args.design)
    if "formal_repeatability" in manifest.name.lower():
        verify_formal_manifest(manifest, generator)
    else:
        generator.verify_manifest(manifest)
    if "confirmation" in manifest.name.lower():
        verify_confirmation_authorization()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    realizations_total = 0
    for condition in load_conditions(manifest):
        if args.max_conditions is not None and completed >= args.max_conditions:
            break
        target = output / f"{condition.condition_id}.npz"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {target}; use --overwrite to replace")
        count = condition.noise_realizations
        if args.realizations is not None:
            count = min(count, args.realizations)
        records = [generator.generate(condition, index) for index in range(count)]
        save_condition_batch(records, output)
        completed += 1
        realizations_total += count
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "design_sha256": generator.design_sha256,
        "conditions_written": completed,
        "realizations_written": realizations_total,
        "output": str(output),
        "confirmation_generation_authorized": bool(args.allow_confirmation),
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build study manifests locally on the chosen data disk; no downloads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.data import DEFAULT_SIZES, prepare_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing original train/val/test.json")
    parser.add_argument("--image-root", type=Path, help="Recursively indexed extraction root; paths in JSONL are relative to this root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-image-verification", action="store_true", help="Explicitly skip Pillow RGB decoding; mark files not_decoded")
    parser.add_argument("--data-seed", type=int, default=20261104)
    parser.add_argument("--presentation-seed", type=int, default=20261105)
    parser.add_argument("--audit-seed", type=int, default=20261106)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--training-passes", type=int, default=2)
    for name, default in DEFAULT_SIZES.items():
        parser.add_argument(f"--{name}-size", type=int, default=default)
    args = parser.parse_args()
    report = prepare_manifests(
        args.data_dir, args.output_dir, image_root=args.image_root,
        verify_images=not args.skip_image_verification,
        data_seed=args.data_seed, presentation_seed=args.presentation_seed,
        audit_seed=args.audit_seed,
        training_seed=args.training_seed, training_passes=args.training_passes,
        sizes={name: getattr(args, f"{name}_size") for name in DEFAULT_SIZES},
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "manifests": {name: {k: value for k, value in info.items() if k in {"records", "questions", "recipes", "technical_issue_questions", "unverified_questions", "sha256"}} for name, info in report["manifests"].items()},
        "image_report": report["image_report"],
    }, indent=2))


if __name__ == "__main__":
    main()

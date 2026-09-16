"""Runnable, non-interactive workflow in fewer than ten user operations."""

from pathlib import Path

import tfpq_qualifier as tq

result = tq.run(output_dir="outputs/example")
artifacts = tq.generate_artifacts(result)
print(result.summary())
print(f"Generated {len(artifacts)} manuscript files in {Path('manuscript/artifacts').resolve()}")

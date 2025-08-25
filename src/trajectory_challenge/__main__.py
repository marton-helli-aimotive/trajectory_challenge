from __future__ import annotations

from . import ngsim


def main() -> None:  # pragma: no cover
	try:
		df = ngsim.load_ngsim_portal(limit=10, json_endpoint=True)
		print(f"Loaded {len(df)} rows")
	except Exception as exc:
		print(f"trajectory_challenge CLI: data fetch skipped ({exc})")


if __name__ == "__main__":  # pragma: no cover
	main()

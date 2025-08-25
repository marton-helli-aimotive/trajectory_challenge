"""CLI entrypoint to launch the Streamlit GUI programmatically.

Users can still run `streamlit run -m trajectory_challenge.gui` directly.
This wrapper avoids importing Streamlit app code outside a ScriptRunContext.
"""
from __future__ import annotations

import runpy
import sys
import subprocess

def run() -> int:  # pragma: no cover
    """Launch the Streamlit GUI using the public CLI.

    Falls back to a bare module run (no Streamlit chrome) if streamlit is missing.
    Returns an exit code suitable for sys.exit().
    """
    try:
        import streamlit  # noqa: F401
    except Exception:  # noqa: BLE001
        print("Streamlit not installed in this environment. Running bare module (limited).")
        _bare_run()
        return 0

    # Use streamlit run with the module's __file__ path
    try:
        from trajectory_challenge import gui
        gui_file = gui.__file__
        if gui_file is None:
            raise ImportError("Cannot locate gui module file")
        cmd = [sys.executable, "-m", "streamlit", "run", gui_file]
    except ImportError:
        print("Cannot import trajectory_challenge.gui module.")
        return 1
    
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to launch Streamlit: {exc}. Falling back to bare run.")
        _bare_run()
        return 1


def _bare_run():  # pragma: no cover
    runpy.run_module("trajectory_challenge.gui", run_name="__main__")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())

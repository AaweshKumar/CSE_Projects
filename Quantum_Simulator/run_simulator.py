"""Quantum Atom Simulator - Standalone Application Entry Point."""
import os
import sys

if getattr(sys, "frozen", False):
    # Running in a PyInstaller bundle
    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

if bundle_dir not in sys.path:
    sys.path.insert(0, bundle_dir)

from ui import main

if __name__ == "__main__":
    main()

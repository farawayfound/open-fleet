"""python3 -m fleetctl ... -- the entry point the bootstrap scripts call.

Run as a module rather than a script so `fleetctl` works from a checkout, from
a staged tarball, and from a zipapp, with no install step and nothing on PATH.
"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

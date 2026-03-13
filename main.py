#!/usr/bin/env python3
"""RGB-D Suction Viewer
The folder must contain matching pairs of RGB images and NumPy depth arrays
with the same filename stem, e.g. ``frame_001.png`` + ``frame_001.npy``.
"""

import os
import argparse


def main() -> None:

    if not os.environ.get("XDG_RUNTIME_DIR"):
        d = "/tmp/runtime-root"
        os.makedirs(d, mode=0o700, exist_ok=True)
        os.environ["XDG_RUNTIME_DIR"] = d

    parser = argparse.ArgumentParser(
        description="RGB-D point cloud viewer with suction quality scoring.")
    parser.add_argument(
        "folder", nargs="?",
        help="Folder containing RGB images and matching .npy depth arrays.")
    args = parser.parse_args()

    from viewer import Viewer
    Viewer(preload_folder=args.folder).run()


if __name__ == "__main__":
    main()

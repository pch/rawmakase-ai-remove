#!/usr/bin/env python3
"""Check the shipped Linux package on the Ubuntu 24.04 release baseline."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

with tempfile.TemporaryDirectory() as directory:
    with zipfile.ZipFile(sys.argv[1]) as package:
        program = Path(directory) / "rawmakase-ai-retouch"
        program.write_bytes(package.read("bin/rawmakase-ai-retouch"))
    program.chmod(0o755)
    versions = subprocess.check_output(
        ["readelf", "--version-info", str(program)], text=True
    )
    required = {tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_([0-9.]+)", versions)}
    assert required, "no GLIBC requirements found"
    assert max(required) <= (2, 39), f"requires GLIBC {max(required)}, newer than Ubuntu 24.04"
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAWMAKASE_EXTENSION_")}
    result = subprocess.run([str(program)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1 and "RAWMAKASE_EXTENSION_FD is not set" in result.stderr, result
    print("Packaged binary loads on the release baseline; GLIBC requirements <= 2.39")

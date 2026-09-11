"""Regenerate Python gRPC stubs from proto/inference.proto.

The generated *_pb2_grpc.py imports its sibling by bare module name, which
only works if the output directory is on sys.path. Rewriting it to a package
import keeps engine/pb self-contained.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "engine" / "pb"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc", "-Iproto",
         f"--python_out={OUT}", f"--grpc_python_out={OUT}", f"--pyi_out={OUT}",
         "proto/inference.proto"],
        cwd=ROOT,
    ).returncode
    if rc:
        return rc

    grpc_file = OUT / "inference_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    grpc_file.write_text(
        text.replace("import inference_pb2 as inference__pb2",
                     "from engine.pb import inference_pb2 as inference__pb2"),
        encoding="utf-8",
    )
    (OUT / "__init__.py").write_text(
        '"""Generated protobuf/gRPC stubs. Regenerate with scripts/gen_proto.py."""\n',
        encoding="utf-8",
    )
    print(f"regenerated stubs in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

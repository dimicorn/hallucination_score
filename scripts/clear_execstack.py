"""Clear the executable-stack flag on jaxlib's shared libraries (needed on glibc >= 2.41).

    python clear_execstack.py [DIR_OR_SO ...]     # default: the installed jaxlib package
"""

import importlib.util
import struct
import sys
from pathlib import Path

PT_GNU_STACK, PF_X = 0x6474E551, 0x1


def patch(path):
    with open(path, "r+b") as f:
        head = f.read(64)
        if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return "skipped (not a little-endian ELF64 file)"
        (phoff,) = struct.unpack_from("<Q", head, 32)
        phentsize, phnum = struct.unpack_from("<HH", head, 54)
        for i in range(phnum):
            off = phoff + i * phentsize
            f.seek(off)
            p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type == PT_GNU_STACK:
                if not p_flags & PF_X:
                    return "already clean"
                f.seek(off + 4)
                f.write(struct.pack("<I", p_flags & ~PF_X))
                return "CLEARED"
        return "NO GNU_STACK header (stack defaults to executable; cannot fix by flag)"


def main():
    targets = [Path(a) for a in sys.argv[1:]]
    if not targets:
        spec = importlib.util.find_spec("jaxlib")
        if spec is None:
            sys.exit("jaxlib not found in this environment")
        targets = [Path(spec.origin).parent]
    files = [p for t in targets for p in ([t] if t.is_file() else sorted(t.rglob("*.so*")))]
    if not files:
        sys.exit(f"no .so files under {targets}")
    for p in files:
        print(f"{patch(p):>16}  {p}")


if __name__ == "__main__":
    main()

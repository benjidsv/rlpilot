"""ctypes binding to librlpilot_client and management of the worker processes.

Field and action indices are parsed from native/protocol.h so the C header stays the single source
of truth for the shared-memory layout.
"""

from __future__ import annotations

import ctypes as C
import os
import re
import subprocess
import tempfile
import weakref
from pathlib import Path

import numpy as np

from . import workdirs

PROTOCOL_H = workdirs.ROOT / "native" / "protocol.h"


def _parse_protocol(text: str):
    def enum(name: str, prefix: str) -> list[str]:
        body = re.search(r"enum %s \{(.*?)\};" % name, text, re.S).group(1)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        return [m.lower() for m in re.findall(r"\b%s(\w+)" % prefix, body)]

    def define(name: str) -> float:
        return float(re.search(r"#define %s ([0-9.]+)" % name, text).group(1))

    return enum("rlp_field", "RLP_F_"), enum("rlp_action", "RLP_A_"), {
        k: define("RLP_" + k.upper()) for k in ("version", "n_rays", "ray_range", "max_cars")
    }


_FIELDS, _ACTIONS, _DEFS = _parse_protocol(PROTOCOL_H.read_text())
F = {name: i for i, name in enumerate(_FIELDS)}  # raw obs field -> column
A = {name: i for i, name in enumerate(_ACTIONS)}  # action -> column
N_FIELDS = len(_FIELDS)
N_RAYS = int(_DEFS["n_rays"])
RAY_RANGE = _DEFS["ray_range"]
OBS_DIM = N_FIELDS + N_RAYS
ACT_DIM = len(_ACTIONS)
PROTOCOL_VERSION = int(_DEFS["version"])

CMD_NONE, CMD_STEP, CMD_RESET, CMD_QUIT = 0, 1, 2, 3
FLAG_AUTO_GEAR = 1
ST_READY = 1
_ERRORS = {-1: "timeout", -2: "worker error/exited", -3: "system error"}


class Header(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("magic", "version", "n_workers", "n_cars", "obs_dim", "act_dim",
                                         "frame_skip", "flags", "spin_us", "_pad")] + \
               [(n, C.c_uint64) for n in ("total_size", "workers_off", "obs_off", "act_off")]


class Worker(C.Structure):  # mirrors rlp_worker (two 64-byte aligned halves)
    _fields_ = [("cmd_seq", C.c_uint32), ("cmd_waiters", C.c_uint32), ("cmd", C.c_uint32), ("_p0", C.c_uint8 * 52),
                ("done_seq", C.c_uint32), ("done_waiters", C.c_uint32), ("status", C.c_uint32), ("pid", C.c_int32),
                ("n_steps", C.c_uint64), ("n_resets", C.c_uint64), ("reset_seconds", C.c_double),
                ("sim_seconds", C.c_double), ("_p1", C.c_uint8 * 16)]


def _load_lib() -> C.CDLL:
    lib = C.CDLL(str(workdirs.CLIENT_LIB))
    lib.rlp_create.restype = C.c_void_p
    lib.rlp_create.argtypes = [C.c_char_p] + [C.c_uint32] * 5
    lib.rlp_destroy.argtypes = [C.c_void_p, C.c_char_p]
    lib.rlp_wait_ready.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_int32)]
    lib.rlp_exchange.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_int32)]
    return lib


class WorkerError(RuntimeError):
    pass


class Client:
    """Shared-memory block + n_workers rlpilot-worker processes, one race each.

    `obs` [n_workers, n_cars, OBS_DIM] and `act` [n_workers, n_cars, ACT_DIM] are zero-copy views on
    the shared memory: write `act`, call `exchange(cmds)`, read `obs`.
    """

    def __init__(self, n_workers: int, n_cars: int = 1, frame_skip: int = 1, auto_gear: bool = True,
                 spin_us: int = 50, race: str = "rlpilot", run_dir: str | Path | None = None,
                 deterministic: bool = True, trace_level: int = 1, timeout_ms: int = 30000):
        self.lib = _load_lib()
        self.n_workers, self.n_cars, self.timeout_ms = n_workers, n_cars, timeout_ms
        self.name = f"/rlp{os.getpid()}_{id(self) % 100000}".encode()  # macOS: <= 31 chars
        ptr = self.lib.rlp_create(self.name, n_workers, n_cars, frame_skip,
                                  FLAG_AUTO_GEAR if auto_gear else 0, spin_us)
        if not ptr:
            raise OSError(C.get_errno(), "rlp_create failed")
        self._ptr = ptr
        h = Header.from_address(ptr)
        assert (h.magic, h.version, h.obs_dim, h.act_dim) == (0x31504C52, PROTOCOL_VERSION, OBS_DIM, ACT_DIM)
        assert h.obs_off - h.workers_off >= n_workers * C.sizeof(Worker) == n_workers * 128
        self.header = h
        self.workers = (Worker * n_workers).from_address(ptr + h.workers_off)
        self.obs = np.ctypeslib.as_array((C.c_float * (n_workers * n_cars * OBS_DIM)).from_address(ptr + h.obs_off)
                                         ).reshape(n_workers, n_cars, OBS_DIM)
        self.act = np.ctypeslib.as_array((C.c_float * (n_workers * n_cars * ACT_DIM)).from_address(ptr + h.act_off)
                                         ).reshape(n_workers, n_cars, ACT_DIM)
        self._cmds = np.zeros(n_workers, dtype=np.uint32)
        self._bad = C.c_int32(-1)
        self._finalizer = weakref.finalize(self, Client._cleanup, self.lib, ptr, self.name, [])

        self._tmp = None
        if run_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="rlpilot-")
            run_dir = self._tmp.name
        self.run_dir = Path(run_dir)
        self.dirs = workdirs.prepare(self.run_dir, n_workers, race)
        self.procs: list[subprocess.Popen] = []
        for i, d in enumerate(self.dirs):
            args = [str(workdirs.WORKER_BIN), "-lc", f"{d}/", "-s", race, "-m", self.name.decode(), "-w", str(i),
                    "-tl", str(trace_level)] + (["-nr"] if deterministic else [])
            with open(d / "worker.log", "wb") as log:
                self.procs.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL))
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, Client._cleanup, self.lib, ptr, self.name, self.procs)
        rc = self.lib.rlp_wait_ready(ptr, timeout_ms, C.byref(self._bad))
        if rc:
            self._raise(rc, "startup")

    def exchange(self, cmds: np.ndarray | int) -> None:
        """Send one command per worker (CMD_NONE skips it) and wait for all answers."""
        if np.isscalar(cmds):
            self._cmds.fill(cmds)
        else:
            self._cmds[:] = cmds
        rc = self.lib.rlp_exchange(self._ptr, self._cmds.ctypes.data, self.timeout_ms, C.byref(self._bad))
        if rc:
            self._raise(rc, "exchange")

    def stats(self) -> dict[str, float]:
        w = self.workers
        return {
            "worker_steps": float(sum(x.n_steps for x in w)),
            "worker_resets": float(sum(x.n_resets for x in w)),
            "worker_reset_seconds": float(sum(x.reset_seconds for x in w)),
            "worker_sim_seconds": float(sum(x.sim_seconds for x in w)),
        }

    def _raise(self, rc: int, what: str):
        i = self._bad.value
        tail = ""
        if 0 <= i < len(self.dirs):
            log = (self.dirs[i] / "worker.log").read_text(errors="replace").splitlines()
            tail = "\n".join(l for l in log if "Error" in l)[-2000:] or "\n".join(log[-10:])
        raise WorkerError(f"{what} failed: {_ERRORS.get(rc, rc)} (worker {i})\n{tail}")

    def close(self):
        if self._ptr is None:
            return
        try:
            self._cmds.fill(CMD_QUIT)
            self.lib.rlp_exchange(self._ptr, self._cmds.ctypes.data, 2000, C.byref(self._bad))
        finally:
            self._finalizer()
            self._ptr = None
            self.obs = self.act = None
            if self._tmp:
                self._tmp.cleanup()

    @staticmethod
    def _cleanup(lib, ptr, name, procs):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        lib.rlp_destroy(C.c_void_p(ptr), name)

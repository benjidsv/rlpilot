"""The native side, seen from Python: worker processes, their local dirs, and the shared-memory block.

Field and action indices are parsed from native/protocol.h, so the C header stays the single source of truth for the
shared-memory layout. `Client.obs` [n_workers, n_cars, OBS_DIM] and `Client.act` [n_workers, n_cars, ACT_DIM] are
zero-copy views on it: write `act`, call `exchange(cmds)`, read `obs`.
"""

from __future__ import annotations

import ctypes as C
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TEMPLATE = ROOT / "sd-local"  # initialized Speed Dreams local dir: config/, drivers/, and the heavy cars/, tracks/
INSTALL_RACEMAN = ROOT / "sd-install/share/games/speed-dreams-2/config/raceman"
WORKER_BIN = ROOT / "sd-install/games/rlpilot-worker"
CLIENT_LIB = ROOT / f"sd-install/lib/librlpilot_client.{'dylib' if sys.platform == 'darwin' else 'so'}"


def _parse_protocol(text: str):
    def enum(name: str, prefix: str) -> list[str]:
        body = re.search(r"enum %s \{(.*?)\};" % name, text, re.S).group(1)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        return [m.lower() for m in re.findall(r"\b%s(\w+)" % prefix, body)]

    def define(name: str) -> float:
        return float(re.search(r"#define %s ([0-9.]+)" % name, text).group(1))

    return enum("rlp_field", "RLP_F_"), enum("rlp_action", "RLP_A_"), {
        k: define("RLP_" + k.upper()) for k in ("version",)}


_FIELDS, _ACTIONS, _DEFS = _parse_protocol((ROOT / "native" / "protocol.h").read_text())
F = {name: i for i, name in enumerate(_FIELDS)}  # raw obs field -> column
A = {name: i for i, name in enumerate(_ACTIONS)}  # action slot -> column
N_FIELDS = len(_FIELDS)
OBS_DIM = N_FIELDS
ACT_DIM = len(_ACTIONS)
PROTOCOL_VERSION = int(_DEFS["version"])

CMD_NONE, CMD_STEP, CMD_RESET, CMD_QUIT = 0, 1, 2, 3
ST_READY = 1
MAX_WORKERS = 256  # rlp_exchange (client.c) only commands this many
_ERRORS = {-1: "timeout", -2: "worker error/exited", -3: "system error"}


def prepare_dir(run_dir: Path, worker: int, race: str) -> Path:
    """A fresh local dir (`-lc`) for one worker: the engine writes results and regenerated driver lists into it on every
    (re)start, so workers can't share one. Small copies of config/ and drivers/, symlinks to cars/ and tracks/. The race
    config is copied from data/ every time, because the engine rewrites it when loading."""
    if not (TEMPLATE / "config").is_dir():
        raise FileNotFoundError(f"{TEMPLATE} is not initialized: run the game once with -lc {TEMPLATE}/")
    race_xml = DATA / "raceman" / f"{race}.xml"
    INSTALL_RACEMAN.mkdir(parents=True, exist_ok=True)  # the race list is enumerated from the install dir
    shutil.copyfile(race_xml, INSTALL_RACEMAN / race_xml.name)
    d = Path(run_dir) / f"w{worker}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(TEMPLATE / "config", d / "config")
    shutil.copytree(TEMPLATE / "drivers", d / "drivers", ignore=shutil.ignore_patterns("rlpilot"))
    for heavy in ("cars", "tracks"):
        (d / heavy).symlink_to(TEMPLATE / heavy, target_is_directory=True)
    shutil.copytree(DATA / "drivers" / "rlpilot", d / "drivers" / "rlpilot")
    shutil.copyfile(race_xml, d / "config" / "raceman" / race_xml.name)
    return d


def run_worker_once(race: str, args: list[str], timeout: float = 600) -> None:
    """Run a worker without a client (trace / geometry modes, see native/worker/main.cpp) in a temporary dir."""
    with tempfile.TemporaryDirectory(prefix="rlpilot-once-") as tmp:
        d = prepare_dir(Path(tmp), 0, race)
        log = d / "worker.log"
        with open(log, "wb") as out:
            rc = subprocess.run([str(WORKER_BIN), "-lc", f"{d}/", "-s", race, "-nr", "-tl", "1", *args],
                                stdout=out, stderr=subprocess.STDOUT, timeout=timeout).returncode
        if rc != 0:
            raise RuntimeError(f"worker {args} on {race} failed (rc={rc}):\n{log.read_text(errors='replace')[-2000:]}")


class Header(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("magic", "version", "n_workers", "n_cars", "obs_dim", "act_dim",
                                         "frame_skip", "_reserved", "spin_us", "_pad")] + \
               [(n, C.c_uint64) for n in ("total_size", "workers_off", "obs_off", "act_off")]


class Worker(C.Structure):  # mirrors rlp_worker (two 64-byte aligned halves)
    _fields_ = [("cmd_seq", C.c_uint32), ("cmd_waiters", C.c_uint32), ("cmd", C.c_uint32), ("_p0", C.c_uint8 * 52),
                ("done_seq", C.c_uint32), ("done_waiters", C.c_uint32), ("status", C.c_uint32), ("pid", C.c_int32),
                ("n_steps", C.c_uint64), ("n_resets", C.c_uint64), ("reset_seconds", C.c_double),
                ("sim_seconds", C.c_double), ("_p1", C.c_uint8 * 16)]


def _load_lib() -> C.CDLL:
    lib = C.CDLL(str(CLIENT_LIB))
    lib.rlp_create.restype = C.c_void_p
    lib.rlp_create.argtypes = [C.c_char_p] + [C.c_uint32] * 4
    lib.rlp_destroy.argtypes = [C.c_void_p, C.c_char_p]
    lib.rlp_wait_ready.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_int32)]
    lib.rlp_exchange.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_int32)]
    lib.rlp_send.argtypes = [C.c_void_p, C.c_void_p]
    lib.rlp_wait.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_int32)]
    lib.rlp_answered.argtypes = [C.c_void_p, C.c_uint32]
    return lib


class WorkerError(RuntimeError):
    pass


class Client:
    """Shared-memory block + one rlpilot-worker process per race in `races` (one car each)."""

    def __init__(self, races: list[str], frame_skip: int = 2, spin_us: int = 50, run_dir: str | Path | None = None,
                 timeout_ms: int = 30000):
        n = len(races)
        if not 0 < n <= MAX_WORKERS:
            raise ValueError(f"{n} workers: rlp_exchange supports 1 to {MAX_WORKERS}")
        self.lib = _load_lib()
        self.n_workers, self.timeout_ms = n, timeout_ms
        self.name = f"/rlp{os.getpid()}_{id(self) % 100000}".encode()  # macOS: <= 31 chars
        ptr = self.lib.rlp_create(self.name, n, 1, frame_skip, spin_us)
        if not ptr:
            raise OSError(C.get_errno(), "rlp_create failed")
        self._ptr = ptr
        h = Header.from_address(ptr)
        assert (h.magic, h.version, h.obs_dim, h.act_dim) == (0x31504C52, PROTOCOL_VERSION, OBS_DIM, ACT_DIM), \
            "the installed client library doesn't match native/protocol.h: rebuild native/"
        self.workers = (Worker * n).from_address(ptr + h.workers_off)
        self.obs = np.ctypeslib.as_array((C.c_float * (n * OBS_DIM)).from_address(ptr + h.obs_off)).reshape(n, 1, OBS_DIM)
        self.act = np.ctypeslib.as_array((C.c_float * (n * ACT_DIM)).from_address(ptr + h.act_off)).reshape(n, 1, ACT_DIM)
        self._cmds = np.zeros(n, dtype=np.uint32)
        self._bad = C.c_int32(-1)

        self._tmp = None
        if run_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="rlpilot-")
            run_dir = self._tmp.name
        self.run_dir = Path(run_dir)
        self.races = list(races)
        self.dirs = [prepare_dir(self.run_dir, i, r) for i, r in enumerate(self.races)]
        self.procs: list[subprocess.Popen] = []
        self._finalizer = weakref.finalize(self, Client._cleanup, self.lib, ptr, self.name, self.procs)
        self.procs += [self._spawn(i) for i in range(n)]
        self.resets_since_spawn = np.zeros(n, np.int64)
        rc = self.lib.rlp_wait_ready(ptr, timeout_ms, C.byref(self._bad))
        if rc:
            self._raise(rc, "startup")

    def _spawn(self, i: int) -> subprocess.Popen:
        args = [str(WORKER_BIN), "-lc", f"{self.dirs[i]}/", "-s", self.races[i], "-m", self.name.decode(),
                "-w", str(i), "-tl", os.environ.get("RLP_WORKER_TRACELEVEL", "1"), "-nr"]
        # -nr: no srand(time), the engine is deterministic. -tl: the engine's log level, 1 = errors only; 3 shows its
        # info messages (race states, eliminations), for debugging a worker (they land in the WorkerError's tail)
        with open(self.dirs[i] / "worker.log", "wb") as log:
            return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def respawn(self, i: int) -> None:
        """Replace worker i by a fresh process on the same race. Its first observation is the start of a new race, as
        after a reset. Caps the engine's memory growth across in-process restarts (~0.14 MB each, root cause unknown)."""
        p = self.procs[i]
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        w = self.workers[i]
        w.status = 0  # STARTING; the new worker answers the current cmd seq once its race is running
        self.dirs[i] = prepare_dir(self.run_dir, i, self.races[i])
        self.procs[i] = self._spawn(i)
        deadline = time.monotonic() + self.timeout_ms / 1000
        while w.status != ST_READY:
            if self.procs[i].poll() is not None or time.monotonic() > deadline:
                self._bad.value = i
                self._raise(-2 if self.procs[i].poll() is not None else -1, "respawn")
            time.sleep(0.002)
        self.resets_since_spawn[i] = 0

    def exchange(self, cmds: np.ndarray | int) -> None:
        """Send one command per worker (CMD_NONE skips it) and wait for all answers."""
        if np.isscalar(cmds):
            self._cmds.fill(cmds)
        else:
            self._cmds[:] = cmds
        self.resets_since_spawn += self._cmds == CMD_RESET
        rc = self.lib.rlp_exchange(self._ptr, self._cmds.ctypes.data, self.timeout_ms, C.byref(self._bad))
        if rc:
            self._raise(rc, "exchange")

    def send(self, cmds: np.ndarray) -> None:
        """Send commands (CMD_NONE skips a worker) without waiting: `answered` polls each worker, `wait` collects
        them. A worker must have answered before it gets another command (exchange, send or close)."""
        cmds = np.ascontiguousarray(cmds, np.uint32)
        self.resets_since_spawn += cmds == CMD_RESET
        self.lib.rlp_send(self._ptr, cmds.ctypes.data)

    def wait(self, cmds: np.ndarray) -> None:
        """Wait for the answers of the workers `send` gave cmds (not CMD_NONE)."""
        cmds = np.ascontiguousarray(cmds, np.uint32)
        rc = self.lib.rlp_wait(self._ptr, cmds.ctypes.data, self.timeout_ms, C.byref(self._bad))
        if rc:
            self._raise(rc, "wait")

    def answered(self, i: int) -> bool:
        """Whether worker i has answered its last command, without blocking (raises if it died)."""
        rc = self.lib.rlp_answered(self._ptr, i)
        if rc < 0:
            self._bad.value = i
            self._raise(rc, "answer")
        return rc == 1

    def stats(self) -> dict[str, float]:
        w = self.workers
        return {"worker_steps": float(sum(x.n_steps for x in w)), "worker_resets": float(sum(x.n_resets for x in w)),
                "worker_reset_seconds": float(sum(x.reset_seconds for x in w)),
                "worker_sim_seconds": float(sum(x.sim_seconds for x in w))}

    def _raise(self, rc: int, what: str):
        i = self._bad.value
        tail = ""
        if 0 <= i < len(self.dirs):
            log = (self.dirs[i] / "worker.log").read_text(errors="replace").splitlines()
            tail = "\n".join(l for l in log if "Error" in l)[-2000:] or "\n".join(log[-10:])
            if os.environ.get("RLP_WORKER_TRACELEVEL", "1") != "1":  # debugging: what the engine said last
                tail += "\n-- last lines --\n" + "\n".join(log[-40:])
        raise WorkerError(f"{what} failed: {_ERRORS.get(rc, rc)} (worker {i}, {self.races[i] if 0 <= i < self.n_workers else '?'})\n{tail}")

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

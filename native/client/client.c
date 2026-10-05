/* librlpilot_client: owns the shared-memory block and runs the batched command/answer handshake.
 * Called from Python through ctypes (which releases the GIL during calls) and from bench.c. */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../protocol.h"
#include "../sync.h"

#define RLP_API __attribute__((visibility("default")))

enum { RLP_OK = 0, RLP_ERR_TIMEOUT = -1, RLP_ERR_WORKER = -2, RLP_ERR_SYS = -3 };

RLP_API rlp_header *rlp_create(const char *name, uint32_t n_workers, uint32_t n_cars, uint32_t frame_skip,
                               uint32_t flags, uint32_t spin_us)
{
    if (!n_workers || !n_cars || n_cars > RLP_MAX_CARS || !frame_skip)
        return NULL;
    rlp_header layout;
    memset(&layout, 0, sizeof layout);
    rlp_layout(&layout, n_workers, n_cars);

    shm_unlink(name);
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0)
        return NULL;
    if (ftruncate(fd, (off_t)layout.total_size) < 0) {
        close(fd);
        shm_unlink(name);
        return NULL;
    }
    void *p = mmap(NULL, layout.total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        shm_unlink(name);
        return NULL;
    }
    memset(p, 0, layout.total_size);
    rlp_header *h = (rlp_header *)p;
    *h = layout;
    h->frame_skip = frame_skip;
    h->flags = flags;
    h->spin_us = spin_us;
    return h;
}

RLP_API void rlp_destroy(rlp_header *h, const char *name)
{
    if (h)
        munmap(h, h->total_size);
    if (name)
        shm_unlink(name);
}

static int worker_alive(rlp_worker *w)
{
    uint32_t st = rlp_load(&w->status);
    if (st == RLP_ST_ERROR || st == RLP_ST_EXITED)
        return 0;
    return !(w->pid > 0 && kill(w->pid, 0) < 0 && errno == ESRCH);
}

/* Wait until every worker has published its first observation. Returns RLP_OK, or the error code
 * with *bad set to the failing worker index. */
RLP_API int rlp_wait_ready(rlp_header *h, uint32_t timeout_ms, int32_t *bad)
{
    uint64_t deadline = rlp_now_ns() + (uint64_t)timeout_ms * 1000000;
    for (uint32_t i = 0; i < h->n_workers; i++) {
        rlp_worker *w = rlp_worker_at(h, i);
        while (rlp_load(&w->status) != RLP_ST_READY) {
            if (rlp_load(&w->status) == RLP_ST_ERROR || rlp_now_ns() > deadline) {
                if (bad)
                    *bad = (int32_t)i;
                return rlp_load(&w->status) == RLP_ST_ERROR ? RLP_ERR_WORKER : RLP_ERR_TIMEOUT;
            }
            usleep(1000);
        }
    }
    return RLP_OK;
}

/* Send cmds[w] (RLP_CMD_NONE = leave that worker alone) to every worker, then wait for all answers.
 * Actions must already be written to the act array. */
RLP_API int rlp_exchange(rlp_header *h, const uint32_t *cmds, uint32_t timeout_ms, int32_t *bad)
{
    uint32_t want[256];
    uint32_t n = h->n_workers < 256 ? h->n_workers : 256;
    for (uint32_t i = 0; i < n; i++) {
        if (cmds[i] == RLP_CMD_NONE)
            continue;
        rlp_worker *w = rlp_worker_at(h, i);
        w->cmd = cmds[i];
        want[i] = w->cmd_sync.seq + 1;
        rlp_signal(&w->cmd_sync, want[i]);
    }
    for (uint32_t i = 0; i < n; i++) {
        if (cmds[i] == RLP_CMD_NONE)
            continue;
        rlp_worker *w = rlp_worker_at(h, i);
        uint64_t deadline = rlp_now_ns() + (uint64_t)timeout_ms * 1000000;
        uint32_t v;
        while ((v = rlp_load(&w->done_sync.seq)) != want[i]) {
            rlp_wait_change(&w->done_sync, v, h->spin_us ? 1000 : 0, 200);
            if (rlp_load(&w->done_sync.seq) == want[i])
                break;
            if (!worker_alive(w) || rlp_now_ns() > deadline) {
                if (bad)
                    *bad = (int32_t)i;
                return worker_alive(w) ? RLP_ERR_TIMEOUT : RLP_ERR_WORKER;
            }
        }
        if (rlp_load(&w->status) != RLP_ST_READY && cmds[i] != RLP_CMD_QUIT) {
            if (bad)
                *bad = (int32_t)i;
            return RLP_ERR_WORKER;
        }
    }
    return RLP_OK;
}

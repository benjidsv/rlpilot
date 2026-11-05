/* Cross-process wait/wake on a sequence word in shared memory: spin briefly, then sleep with a futex
 * (macOS 14.4+ os_sync_wait_on_address, Linux futex(2)). The waker only pays for a syscall when someone sleeps. */
#ifndef RLPILOT_SYNC_H
#define RLPILOT_SYNC_H

#include <stdint.h>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <os/os_sync_wait_on_address.h>
#elif defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#error "rlpilot's sync.h needs a cross-process futex: macOS or Linux only (Windows: use WSL2)"
#endif

#include "protocol.h"

static inline uint32_t rlp_load(const uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

#define RLP_WAIT_SLICE_NS 100000000ull /* a sleeper wakes up every 100 ms to honour its timeout */

#if defined(__APPLE__)
static inline uint64_t rlp_now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static inline void rlp_futex_wake_one(uint32_t *addr)
{
    os_sync_wake_by_address_any(addr, sizeof(uint32_t), OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}

static inline void rlp_futex_wait_slice(uint32_t *addr, uint32_t old)
{
    static mach_timebase_info_data_t tb;
    if (!tb.denom)
        mach_timebase_info(&tb);
    os_sync_wait_on_address_with_timeout(addr, old, sizeof(uint32_t), OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME, RLP_WAIT_SLICE_NS * tb.denom / tb.numer);
}
#else
static inline uint64_t rlp_now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Not FUTEX_*_PRIVATE: the word is shared between processes. */
static inline void rlp_futex_wake_one(uint32_t *addr) { syscall(SYS_futex, addr, FUTEX_WAKE, 1, NULL, NULL, 0); }

static inline void rlp_futex_wait_slice(uint32_t *addr, uint32_t old)
{
    struct timespec slice = {0, (long)RLP_WAIT_SLICE_NS};
    syscall(SYS_futex, addr, FUTEX_WAIT, old, &slice, NULL, 0);
}
#endif

static inline void rlp_cpu_relax(void)
{
#if defined(__x86_64__)
    __asm__ volatile("pause");
#else
    __asm__ volatile("yield");
#endif
}

/* Publish seq (release) and wake a sleeper if any. */
static inline void rlp_signal(rlp_sync *s, uint32_t seq)
{
    __atomic_store_n(&s->seq, seq, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&s->waiters, __ATOMIC_SEQ_CST))
        rlp_futex_wake_one(&s->seq);
}

/* Wait until s->seq != old. Returns the new seq, or `old` if timeout_ms elapsed (0 = no timeout). */
static inline uint32_t rlp_wait_change(rlp_sync *s, uint32_t old, uint32_t spin_us, uint32_t timeout_ms)
{
    uint32_t v = rlp_load(&s->seq);
    if (v != old)
        return v;
    uint64_t t0 = rlp_now_ns(), spin_end = t0 + (uint64_t)spin_us * 1000;
    do {
        for (int i = 0; i < 64; i++) {
            rlp_cpu_relax();
            if ((v = rlp_load(&s->seq)) != old)
                return v;
        }
    } while (rlp_now_ns() < spin_end);

    uint64_t deadline = timeout_ms ? t0 + (uint64_t)timeout_ms * 1000000 : 0;
    for (;;) {
        __atomic_fetch_add(&s->waiters, 1, __ATOMIC_SEQ_CST);
        if ((v = __atomic_load_n(&s->seq, __ATOMIC_SEQ_CST)) == old) {
            rlp_futex_wait_slice(&s->seq, old);
            v = rlp_load(&s->seq);
        }
        __atomic_fetch_sub(&s->waiters, 1, __ATOMIC_SEQ_CST);
        if (v != old)
            return v;
        if (deadline && rlp_now_ns() >= deadline)
            return old;
    }
}

#endif

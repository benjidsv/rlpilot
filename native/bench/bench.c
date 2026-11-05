/* Throughput benchmark of the worker + shm transport, without Python.
 *   rlpilot-bench <run_dir> <n_workers> <steps> [episode_steps=2000] [frame_skip=1] [spin_us=50]
 * <run_dir>/w<i>/ must be prepared local dirs (python -m rlpilot.workdirs). A centreline follower
 * drives every car; every episode_steps steps all workers reset. */
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "../protocol.h"
#include "../sync.h"

extern char **environ;
rlp_header *rlp_create(const char *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void rlp_destroy(rlp_header *, const char *);
int rlp_wait_ready(rlp_header *, uint32_t, int32_t *);
int rlp_exchange(rlp_header *, const uint32_t *, uint32_t, int32_t *);

static void control(const float *o, float *a)
{
    float w = o[RLP_F_TRACK_WIDTH] > 1 ? o[RLP_F_TRACK_WIDTH] : 10;
    a[RLP_A_STEER] = 1.5f * (o[RLP_F_ANGLE] - 0.5f * o[RLP_F_TO_MIDDLE] / w);
    float target = 10 + 0.25f * o[RLP_N_FIELDS + RLP_N_RAYS / 2]; /* faster when the road ahead is clear */
    a[RLP_A_ACCEL] = o[RLP_F_SPEED_X] < target ? 0.8f : 0;
    a[RLP_A_BRAKE] = o[RLP_F_SPEED_X] > target + 5 ? 0.5f : 0;
    a[RLP_A_GEAR] = 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <run_dir> <n_workers> <steps> [episode_steps] [frame_skip] [spin_us]\n", argv[0]);
        return 1;
    }
    const char *run = argv[1];
    uint32_t n = atoi(argv[2]), steps = atoi(argv[3]);
    uint32_t ep = argc > 4 ? atoi(argv[4]) : 2000, fs = argc > 5 ? atoi(argv[5]) : 1;
    uint32_t spin = argc > 6 ? atoi(argv[6]) : 50;

    char exe[1024], worker[1100], shm[32];
#if defined(__APPLE__)
    uint32_t sz = sizeof exe;
    _NSGetExecutablePath(exe, &sz);
#else
    ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[len > 0 ? len : 0] = 0;
#endif
    snprintf(worker, sizeof worker, "%.*s/rlpilot-worker", (int)(strrchr(exe, '/') - exe), exe);
    snprintf(shm, sizeof shm, "/rlpb%d", getpid());

    rlp_header *h = rlp_create(shm, n, 1, fs, RLP_FLAG_AUTO_GEAR, spin);
    if (!h) {
        perror("rlp_create");
        return 1;
    }
    pid_t *pids = calloc(n, sizeof(pid_t));
    for (uint32_t i = 0; i < n; i++) {
        char lc[1100], log[1100], wi[16];
        snprintf(lc, sizeof lc, "%s/w%u/", run, i);
        snprintf(log, sizeof log, "%s/w%u/worker.log", run, i);
        snprintf(wi, sizeof wi, "%u", i);
        char *args[] = {worker, "-lc", lc, "-s", "rlpilot", "-m", shm, "-w", wi, "-nr", "-tl", "1", NULL};
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 1, log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        posix_spawn_file_actions_adddup2(&fa, 1, 2);
        if (posix_spawn(&pids[i], worker, &fa, NULL, args, environ)) {
            perror("posix_spawn");
            return 1;
        }
    }

    int32_t bad = -1;
    uint64_t t0 = rlp_now_ns();
    int rc = rlp_wait_ready(h, 60000, &bad);
    printf("startup: %.2f s (rc=%d)\n", (rlp_now_ns() - t0) * 1e-9, rc);
    if (rc)
        goto out;

    uint32_t *cmds = malloc(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++)
        cmds[i] = RLP_CMD_RESET;
    rc = rlp_exchange(h, cmds, 10000, &bad);

    double lat_max = 0, lat_sum = 0;
    float dist_max = 0;
    t0 = rlp_now_ns();
    for (uint32_t s = 1; s <= steps && !rc; s++) {
        uint32_t cmd = (s % ep == 0) ? RLP_CMD_RESET : RLP_CMD_STEP;
        if (getenv("RLP_DEBUG") && s % 250 == 0) {
            const float *o = rlp_obs_at(h, 0, 0);
            printf("step %u t=%.2f v=%.1f gear=%.0f rpm=%.0f angle=%.2f mid=%.2f dist=%.0f lap=%.0f off=%.0f state=%.0f ray0=%.1f ray9=%.1f\n",
                   s, o[RLP_F_TIME], o[RLP_F_SPEED_X], o[RLP_F_GEAR], o[RLP_F_RPM], o[RLP_F_ANGLE], o[RLP_F_TO_MIDDLE],
                   o[RLP_F_DIST_RACED], o[RLP_F_LAPS], o[RLP_F_WHEELS_OFF], o[RLP_F_CAR_STATE], o[RLP_N_FIELDS], o[RLP_N_FIELDS + 9]);
        }
        for (uint32_t i = 0; i < n; i++) {
            const float *o = rlp_obs_at(h, i, 0);
            if (o[RLP_F_DIST_RACED] > dist_max)
                dist_max = o[RLP_F_DIST_RACED];
            control(o, rlp_act_at(h, i, 0));
            cmds[i] = cmd;
        }
        uint64_t ts = rlp_now_ns();
        rc = rlp_exchange(h, cmds, 10000, &bad);
        double lat = (rlp_now_ns() - ts) * 1e-3;
        lat_sum += lat;
        if (lat > lat_max)
            lat_max = lat;
    }
    double wall = (rlp_now_ns() - t0) * 1e-9;
    if (rc)
        printf("exchange failed rc=%d worker=%d\n", rc, bad);

    double reset_s = 0, sim_s = 0;
    uint64_t resets = 0;
    for (uint32_t i = 0; i < n; i++) {
        rlp_worker *w = rlp_worker_at(h, i);
        reset_s += w->reset_seconds;
        sim_s += w->sim_seconds;
        resets += w->n_resets;
    }
    printf("workers=%u frame_skip=%u steps=%u episode=%u\n", n, fs, steps, ep);
    printf("  env steps/s: %.0f total, %.0f per worker (%.0f sim-s per wall-s)\n", n * steps / wall,
           steps / wall, n * steps * fs * 0.02 / wall);
    printf("  exchange latency: mean %.1f us, max %.0f us\n", lat_sum / steps, lat_max);
    printf("  resets: %llu, mean %.1f ms; reset share of worker time %.1f%%\n", resets,
           resets ? reset_s / resets * 1e3 : 0, 100 * reset_s / (n * wall));
    printf("  max distance raced in an episode: %.0f m\n", dist_max);
    for (uint32_t i = 0; i < n; i++)
        cmds[i] = RLP_CMD_QUIT;
    rlp_exchange(h, cmds, 5000, &bad);
out:
    for (uint32_t i = 0; i < n; i++) {
        kill(pids[i], SIGTERM);
        waitpid(pids[i], NULL, 0);
    }
    rlp_destroy(h, shm);
    return rc ? 1 : 0;
}

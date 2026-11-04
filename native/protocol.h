/* Shared-memory protocol between the Python client and rlpilot-worker processes.
 *
 * One block holds: header | rlp_worker[n_workers] | obs[n_workers][n_cars][RLP_OBS_DIM] (float)
 *                  | act[n_workers][n_cars][RLP_ACT_DIM] (float).
 * Handshake per worker: the client writes actions + cmd, then bumps cmd.seq; the worker answers
 * by writing obs, then setting done.seq = cmd.seq. Sequence words are accessed with __atomic builtins.
 *
 * python/rlpilot/_native.py parses this file (enums and defines): bump RLP_VERSION on any layout change.
 */
#ifndef RLPILOT_PROTOCOL_H
#define RLPILOT_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
#define RLP_ALIGN64 alignas(64)
#else
#define RLP_ALIGN64 _Alignas(64)
#endif

#define RLP_MAGIC 0x31504C52u /* "RLP1" */
#define RLP_VERSION 5
#define RLP_MAX_CARS 8
#define RLP_N_RAYS 19
#define RLP_RAY_RANGE 200.0f

enum rlp_cmd { RLP_CMD_NONE = 0, RLP_CMD_STEP = 1, RLP_CMD_RESET = 2, RLP_CMD_QUIT = 3 };
enum rlp_status { RLP_ST_STARTING = 0, RLP_ST_READY = 1, RLP_ST_ERROR = 2, RLP_ST_EXITED = 3 };
/* Gearbox: AUTO_GEAR = the worker's fixed shift points; SHIFT_POINTS = the worker shifts at the points the
 * client sends (RLP_A_SHIFT_UP/DOWN); neither = the client sends the gear (RLP_A_GEAR). */
enum rlp_flags { RLP_FLAG_AUTO_GEAR = 1, RLP_FLAG_SHIFT_POINTS = 2 };

/* Raw per-car fields, followed by RLP_N_RAYS track-edge distances (m). */
enum rlp_field {
    RLP_F_TIME,            /* race time (s) */
    RLP_F_SPEED_X,         /* car frame (m/s) */
    RLP_F_SPEED_Y,
    RLP_F_SPEED_Z,
    RLP_F_YAW_RATE,        /* rad/s */
    RLP_F_ANGLE,           /* track tangent minus car yaw, [-pi, pi] */
    RLP_F_TO_MIDDLE,       /* m, + = left */
    RLP_F_TO_LEFT,         /* m to left edge of the main road (< 0 = outside) */
    RLP_F_TO_RIGHT,
    RLP_F_TRACK_WIDTH,
    RLP_F_RPM,             /* engine rad/s */
    RLP_F_RPM_REDLINE,
    RLP_F_GEAR,
    RLP_F_WHEEL_SPIN0,     /* rad/s, FR FL RR RL */
    RLP_F_WHEEL_SPIN1,
    RLP_F_WHEEL_SPIN2,
    RLP_F_WHEEL_SPIN3,
    RLP_F_DIST_RACED,      /* m since race start */
    RLP_F_DIST_FROM_START, /* m along the lap */
    RLP_F_LAPS,
    RLP_F_CUR_LAP_TIME,
    RLP_F_LAST_LAP_TIME,
    RLP_F_BEST_LAP_TIME,
    RLP_F_DAMAGE,
    RLP_F_COLL_COUNT,
    RLP_F_WHEELS_OFF,      /* wheels not on the main road */
    RLP_F_CAR_STATE,       /* RM_CAR_STATE_* bits */
    RLP_F_RACE_OVER,       /* 1 if the engine ended this car's race: episode must reset */
    RLP_F_POS_X,
    RLP_F_POS_Y,
    RLP_F_YAW,
    RLP_F_ACCEL_X,
    RLP_F_ACCEL_Y,
    RLP_F_TRACK_LENGTH,    /* m, constant */
    RLP_N_FIELDS
};
#define RLP_OBS_DIM (RLP_N_FIELDS + RLP_N_RAYS)

enum rlp_action {
    RLP_A_STEER,
    RLP_A_ACCEL,
    RLP_A_BRAKE,
    RLP_A_GEAR,
    RLP_A_SHIFT_UP,   /* SHIFT_POINTS: upshift when the engine reaches this fraction of the redline */
    RLP_A_SHIFT_DOWN, /* SHIFT_POINTS: downshift when the lower gear would be at or below this fraction */
    /* Car factors, applied when the race (re)starts (a RESET restarts it if they changed). 0 = 1 = stock. */
    RLP_A_CAR_GRIP,   /* x tyre friction (mu) */
    RLP_A_CAR_MASS,   /* x car mass */
    RLP_A_CAR_POWER,  /* x engine torque curve */
    /* Where the race (re)start puts the car (a RESET restarts it if they changed). LAUNCH_POS 0 = on the grid. */
    RLP_A_LAUNCH_POS,     /* fraction of the lap from the start line, in (0, 1) */
    RLP_A_LAUNCH_LATERAL, /* fraction of the road's width from its right edge */
    RLP_A_LAUNCH_YAW,     /* rad from the track direction, + = left */
    RLP_A_LAUNCH_SPEED,   /* m/s, along the car */
    RLP_ACT_DIM
};

typedef struct {
    uint32_t magic, version;
    uint32_t n_workers, n_cars;
    uint32_t obs_dim, act_dim;
    uint32_t frame_skip;   /* robot ticks (20 ms) per agent step */
    uint32_t flags;        /* rlp_flags */
    uint32_t spin_us;      /* busy-wait budget before sleeping */
    uint32_t _pad;
    uint64_t total_size, workers_off, obs_off, act_off;
} rlp_header;

typedef struct {
    uint32_t seq;
    uint32_t waiters;
} rlp_sync;

typedef struct {
    /* written by the client */
    RLP_ALIGN64 rlp_sync cmd_sync;
    uint32_t cmd;
    /* written by the worker */
    RLP_ALIGN64 rlp_sync done_sync;
    uint32_t status;
    int32_t pid;
    uint64_t n_steps, n_resets;
    double reset_seconds;  /* cumulative wall time spent in restartRace() */
    double sim_seconds;    /* cumulative wall time spent stepping between handshakes */
} rlp_worker;

static inline uint64_t rlp_align64(uint64_t x) { return (x + 63) & ~(uint64_t)63; }

static inline void rlp_layout(rlp_header *h, uint32_t n_workers, uint32_t n_cars)
{
    h->magic = RLP_MAGIC;
    h->version = RLP_VERSION;
    h->n_workers = n_workers;
    h->n_cars = n_cars;
    h->obs_dim = RLP_OBS_DIM;
    h->act_dim = RLP_ACT_DIM;
    h->workers_off = rlp_align64(sizeof(rlp_header));
    h->obs_off = rlp_align64(h->workers_off + (uint64_t)n_workers * sizeof(rlp_worker));
    h->act_off = rlp_align64(h->obs_off + (uint64_t)n_workers * n_cars * RLP_OBS_DIM * sizeof(float));
    h->total_size = rlp_align64(h->act_off + (uint64_t)n_workers * n_cars * RLP_ACT_DIM * sizeof(float));
}

static inline rlp_worker *rlp_worker_at(rlp_header *h, uint32_t w)
{
    return (rlp_worker *)((char *)h + h->workers_off) + w;
}

static inline float *rlp_obs_at(rlp_header *h, uint32_t w, uint32_t car)
{
    return (float *)((char *)h + h->obs_off) + ((uint64_t)w * h->n_cars + car) * RLP_OBS_DIM;
}

static inline float *rlp_act_at(rlp_header *h, uint32_t w, uint32_t car)
{
    return (float *)((char *)h + h->act_off) + ((uint64_t)w * h->n_cars + car) * RLP_ACT_DIM;
}

#endif

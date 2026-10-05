// rlpilot-worker: headless Speed Dreams host driven over shared memory.
//
// Replaces speed-dreams-2's main.cpp + textonly UI. Runs one race in results-only mode and pumps the
// race engine itself. The rlpilot robot forwards its callbacks to rlp_host (below); on every decision
// tick the host publishes observations for all agent cars and blocks until the client sends a
// command. Resets restart the race in-process (no relaunch).
//
// Usage: rlpilot-worker -lc <localdir>/ -s <race> -m <shm name> -w <worker index> [-nr] [-tl <level>]
//        rlpilot-worker -lc <localdir>/ -s <race> -o <out prefix> [-n <laps>] [-nr]
// The second form (trace mode) needs no client: it runs the race as configured (e.g. with a built-in
// bot), records car 0 every 20 ms to <prefix>_laps.csv until it completes <laps> laps, and samples the
// track borders to <prefix>_track.csv. Used to build reference lap times / traces for evaluation.

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <portability.h>
#include <tgf.hpp>
#include <tgfdata.h>
#include <racemanagers.h>
#include <race.h>
#include <raceman.h>
#include <robottools.h>
#include <iraceengine.h>
#include <iuserinterface.h>

#include "../host_api.h"
#include "../protocol.h"
#include "../sync.h"

namespace {

// Track-edge rangefinder angles (deg, + = left), as in the TORCS SCR championship.
const float kRayAngles[RLP_N_RAYS] = {-90, -75, -60, -45, -30, -20, -15, -10, -5, 0,
                                      5,   10,  15,  20,  30,  45,  60,  75,  90};

struct Host
{
    rlp_header *shm = nullptr;
    rlp_worker *w = nullptr;
    uint32_t wid = 0;
    IRaceEngine *engine = nullptr;

    tCarElt *cars[RLP_MAX_CARS] = {};
    // Robot index -> agent slot, in order of appearance: the engine also generates random drivers
    // for every robot module, so our race's driver indices are not 0..n_cars-1.
    int slot_of[256];
    uint32_t n_slots = 0;
    tSituation *s = nullptr;

    uint32_t last_cmd_seq = 0;  // last command seen; answered by setting done.seq to it
    bool fresh = true;          // no STEP since the race (re)started: a RESET is free
    bool hold = true;           // countdown / reset pending: keep cars braked
    bool reset_requested = false;
    bool race_over = false;
    bool quit = false;
    double last_tick_time = -1e9;
    uint64_t tick = 0;
    uint64_t sim_t0 = 0;
} H;

void fatal(const char *msg)
{
    GfLogError("rlpilot-worker: %s\n", msg);
    if (H.w) {
        __atomic_store_n(&H.w->status, (uint32_t)RLP_ST_ERROR, __ATOMIC_RELEASE);
        rlp_signal(&H.w->done_sync, H.last_cmd_seq);
    }
    std::_Exit(3);
}

float trackRay(tTrackSeg *seg, float x0, float y0, float dx, float dy)
{
    // Sphere tracing: distance to the nearest road edge bounds how far the ray can go without
    // leaving the road (exact on straights and arcs), then bisect the crossing.
    tTrkLocPos p;
    float t = 0, tIn = 0;
    for (int it = 0; it < 48; it++) {
        RtTrackGlobal2Local(seg, x0 + t * dx, y0 + t * dy, &p, TR_LPOS_MAIN);
        seg = p.seg;
        float d = std::min(p.toLeft, p.toRight);
        if (d < 0) {
            if (t == 0)
                return 0;
            float lo = tIn, hi = t;
            for (int b = 0; b < 6; b++) {
                float mid = 0.5f * (lo + hi);
                RtTrackGlobal2Local(seg, x0 + mid * dx, y0 + mid * dy, &p, TR_LPOS_MAIN);
                (std::min(p.toLeft, p.toRight) < 0 ? hi : lo) = mid;
            }
            return 0.5f * (lo + hi);
        }
        tIn = t;
        t += std::max(d, 0.5f + 0.02f * t);
        if (t >= RLP_RAY_RANGE)
            return RLP_RAY_RANGE;
    }
    return t;
}

void writeObs(uint32_t k)
{
    tCarElt *car = H.cars[k];
    float *o = rlp_obs_at(H.shm, H.wid, k);
    if (!car) {
        memset(o, 0, sizeof(float) * RLP_OBS_DIM);
        return;
    }
    tTrkLocPos &tp = car->_trkPos;
    float angle = RtTrackSideTgAngleL(&tp) - car->_yaw;
    FLOAT_NORM_PI_PI(angle);
    int off = 0;
    for (int i = 0; i < 4; i++)
        off += car->_wheelSeg(i) && car->_wheelSeg(i)->type2 != TR_MAIN;

    o[RLP_F_TIME] = H.s->currentTime;
    o[RLP_F_SPEED_X] = car->_speed_x;
    o[RLP_F_SPEED_Y] = car->_speed_y;
    o[RLP_F_SPEED_Z] = car->_speed_z;
    o[RLP_F_YAW_RATE] = car->_yaw_rate;
    o[RLP_F_ANGLE] = angle;
    o[RLP_F_TO_MIDDLE] = tp.toMiddle;
    o[RLP_F_TO_LEFT] = tp.toLeft;
    o[RLP_F_TO_RIGHT] = tp.toRight;
    o[RLP_F_TRACK_WIDTH] = tp.seg ? tp.seg->width : 0;
    o[RLP_F_RPM] = car->_enginerpm;
    o[RLP_F_RPM_REDLINE] = car->_enginerpmRedLine;
    o[RLP_F_GEAR] = car->_gear;
    for (int i = 0; i < 4; i++)
        o[RLP_F_WHEEL_SPIN0 + i] = car->_wheelSpinVel(i);
    o[RLP_F_DIST_RACED] = car->_distRaced;
    o[RLP_F_DIST_FROM_START] = car->_distFromStartLine;
    o[RLP_F_LAPS] = car->_laps;
    o[RLP_F_CUR_LAP_TIME] = car->_curLapTime;
    o[RLP_F_LAST_LAP_TIME] = car->_lastLapTime;
    o[RLP_F_BEST_LAP_TIME] = car->_bestLapTime;
    o[RLP_F_DAMAGE] = car->_dammage;
    o[RLP_F_COLL_COUNT] = car->_collCount;
    o[RLP_F_WHEELS_OFF] = off;
    o[RLP_F_CAR_STATE] = car->_state;
    o[RLP_F_RACE_OVER] = H.race_over || (car->_state & RM_CAR_STATE_NO_SIMU);
    o[RLP_F_POS_X] = car->_pos_X;
    o[RLP_F_POS_Y] = car->_pos_Y;
    o[RLP_F_YAW] = car->_yaw;
    o[RLP_F_ACCEL_X] = car->_accel_x;
    o[RLP_F_ACCEL_Y] = car->_accel_y;
    o[RLP_F_TRACK_LENGTH] = H.engine->inData()->track->length;

    float *rays = o + RLP_N_FIELDS;
    for (int r = 0; r < RLP_N_RAYS; r++) {
        float a = car->_yaw + kRayAngles[r] * (float)(PI / 180.0);
        rays[r] = tp.seg ? trackRay(tp.seg, car->_pos_X, car->_pos_Y, cosf(a), sinf(a)) : 0;
    }
}

void publish()
{
    for (uint32_t k = 0; k < H.shm->n_cars; k++)
        writeObs(k);
    H.w->sim_seconds += (rlp_now_ns() - H.sim_t0) * 1e-9;
    if (H.w->status != RLP_ST_READY)
        __atomic_store_n(&H.w->status, (uint32_t)RLP_ST_READY, __ATOMIC_RELEASE);
    rlp_signal(&H.w->done_sync, H.last_cmd_seq);
}

// Publish observations, then block until a command we must act on arrives.
void decide()
{
    publish();
    for (;;) {
        uint32_t seq = rlp_wait_change(&H.w->cmd_sync, H.last_cmd_seq, H.shm->spin_us, 1000);
        if (seq == H.last_cmd_seq) {
            if (getppid() == 1) { // client died
                H.quit = true;
                break;
            }
            continue;
        }
        H.last_cmd_seq = seq;
        uint32_t cmd = __atomic_load_n(&H.w->cmd, __ATOMIC_ACQUIRE);
        if (cmd == RLP_CMD_QUIT) {
            H.quit = true;
            break;
        }
        if (cmd == RLP_CMD_RESET && !H.fresh) {
            H.reset_requested = true;
            H.hold = true;
            break;
        }
        if (cmd == RLP_CMD_RESET || H.race_over) { // nothing to simulate: answer right away
            publish();
            continue;
        }
        H.fresh = false; // RLP_CMD_STEP
        H.w->n_steps++;
        break;
    }
    H.sim_t0 = rlp_now_ns();
}

void onTick()
{
    if (H.s->currentTime < 0 || H.reset_requested || H.quit) {
        H.hold = true;
        return;
    }
    H.hold = false;
    if (H.tick++ % H.shm->frame_skip == 0)
        decide();
}

int autoGear(tCarElt *car)
{
    int gear = car->_gear;
    if (gear <= 0)
        return 1;
    // Shift-up point as a fraction of redline (RLP_SHIFT_UP overrides, for sweeps) and downshift margin (m/s).
    static const float up = [] { const char *e = getenv("RLP_SHIFT_UP"); return e ? (float)atof(e) : 0.9f; }();
    const float margin = 4.0f;
    float wr = car->_wheelRadius(2), v = car->_speed_x;
    int i = gear + car->_gearOffset;
    if (i + 1 < car->_gearNb && v > up * car->_enginerpmRedLine / car->_gearRatio[i] * wr)
        return gear + 1;
    if (gear > 1 && v < up * car->_enginerpmRedLine / car->_gearRatio[i - 1] * wr - margin)
        return gear - 1;
    return gear;
}

float clampf(float v, float lo, float hi) { return std::isfinite(v) ? std::min(hi, std::max(lo, v)) : 0; }

void applyAction(int k, tCarElt *car)
{
    tCarCtrl &c = car->ctrl;
    if (H.hold) {
        c.steer = 0;
        c.accelCmd = 0;
        c.brakeCmd = 1;
        c.clutchCmd = 1;
        c.gear = 1;
        return;
    }
    const float *a = rlp_act_at(H.shm, H.wid, k);
    c.steer = clampf(a[RLP_A_STEER], -1, 1);
    c.accelCmd = clampf(a[RLP_A_ACCEL], 0, 1);
    c.brakeCmd = clampf(a[RLP_A_BRAKE], 0, 1);
    c.gear = (H.shm->flags & RLP_FLAG_AUTO_GEAR)
                 ? autoGear(car)
                 : (int)lrintf(clampf(a[RLP_A_GEAR], -1, car->_gearNb - 1 - car->_gearOffset));
    // The engine holds the clutch down during the pre-start and never releases it: slip it out
    // when pulling away in first or reverse, otherwise keep it engaged (the simu handles shifts).
    float v = fabsf(car->_speed_x);
    c.clutchCmd = (abs(c.gear) == 1 && v < 5) ? 0.5f * (1 - v / 5) : 0;
}

// ---- rlp_host callbacks (called from the robot module) ----

void hostNewTrack(int, tTrack *) {}

int slotOf(int index)
{
    if (index < 0 || index >= 256)
        fatal("robot index out of range");
    if (!H.shm)
        fatal("an rlpilot driver is in the race but no shm was given");
    if (H.slot_of[index] < 0) {
        if (H.n_slots >= H.shm->n_cars)
            fatal("the race has more rlpilot drivers than n_cars");
        H.slot_of[index] = H.n_slots++;
    }
    return H.slot_of[index];
}

void hostNewRace(int index, tCarElt *car, tSituation *s)
{
    H.cars[slotOf(index)] = car;
    H.s = s;
}

void hostDrive(int index, tCarElt *car, tSituation *s)
{
    int k = slotOf(index);
    H.cars[k] = car;
    H.s = s;
    if (s->currentTime != H.last_tick_time) { // first agent car called this robot tick
        H.last_tick_time = s->currentTime;
        onTick();
    }
    applyAction(k, car);
}

void hostEndRace(int, tCarElt *, tSituation *) {}

// ---- minimal user interface: no output, the race engine just runs ----

class RlUI : public IUserInterface
{
public:
    bool activate() override { return true; }
    void quit() override { H.quit = true; }
    void shutdown() override {}
    void onRaceConfiguring() override { H.quit = true; } // all race events over
    void onRaceEventInitializing() override {}
    bool onRaceEventStarting(bool) override { return true; }
    void onRaceInitializing() override {}
    bool onRaceStarting() override { return true; }
    void onRaceLoadingDrivers() override {}
    int onRaceDriversLoaded() override { return 0; }
    void onRaceSimulationReady() override {}
    void onRaceStarted() override {}
    void onRaceResuming() override {}
    void onLapCompleted(int) override {}
    void onRaceInterrupted() override {}
    void onRaceFinishing() override {}
    bool onRaceFinished(bool) override { return true; }
    void onRaceEventFinishing() override {}
    bool onRaceEventFinished(bool, bool) override { return true; }
    void onOptimizationInitializing() override {}
    bool onRaceStartingPaused() override { return false; }
    bool onRaceCooldownStarting() override { return false; }
    void addLoadingMessage(const char *) override {}
    void addOptimizationMessage(const char *) override {}
    void addOptimizationParameterMessage(int, char **, char **, char **) override {}
    void addOptimizationStatusMessage(int, int, double, double, double, double) override {}
    void setResultsTableTitles(const char *, const char *) override {}
    void setResultsTableHeader(const std::vector<std::string> &) override {}
    void addResultsTableRow(const std::vector<std::string> &) override {}
    void setResultsTableRow(int, const std::vector<std::string> &, bool) override {}
    void removeResultsTableRow(int) override {}
    void eraseResultsTable() override {}
    int getResultsTableRowCount() const override { return 50; }
    void setRaceEngine(IRaceEngine &) override {}
    void addHumanCfg(const std::vector<tCarElt *> &, void *) override {}
};

rlp_header *mapShm(const std::string &name)
{
    int fd = shm_open(name.c_str(), O_RDWR);
    if (fd < 0)
        return nullptr;
    struct stat st;
    fstat(fd, &st);
    void *p = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
        return nullptr;
    rlp_header *h = (rlp_header *)p;
    if (h->magic != RLP_MAGIC || h->version != RLP_VERSION || h->total_size > (uint64_t)st.st_size)
        return nullptr;
    return h;
}

// ---- trace mode ----

void writeTrack(const std::string &path, tTrack *track)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        fatal("cannot write the track file");
    fprintf(f, "# length=%.3f width=%.3f\n# left_x left_y right_x right_y\n", track->length, track->width);
    tTrackSeg *seg = track->seg->next; // track->seg is the last segment
    for (int i = 0; i < track->nseg; i++, seg = seg->next) {
        float span = seg->type == TR_STR ? seg->length : seg->arc;
        int n = std::max(1, (int)ceilf(seg->length / 2.0f));
        for (int j = 0; j < n; j++) {
            tTrkLocPos p = {};
            p.seg = seg;
            p.toStart = span * j / n;
            float lx, ly, rx, ry;
            p.toRight = 0;
            RtTrackLocal2Global(&p, &rx, &ry, TR_TORIGHT);
            p.toRight = seg->startWidth + (seg->endWidth - seg->startWidth) * j / n;
            RtTrackLocal2Global(&p, &lx, &ly, TR_TORIGHT);
            fprintf(f, "%.3f %.3f %.3f %.3f\n", lx, ly, rx, ry);
        }
    }
    fclose(f);
}

int runTrace(const std::string &prefix, int laps)
{
    tRmInfo *re = H.engine->inData();
    writeTrack(prefix + "_track.csv", re->track);
    FILE *f = fopen((prefix + "_laps.csv").c_str(), "w");
    if (!f)
        fatal("cannot write the trace file");
    fprintf(f, "# time laps dist_from_start dist_raced speed_x x y last_lap_time steer accel brake\n");
    double next = 0;
    while (!H.quit && re->_reState == RE_STATE_RACE) {
        H.engine->updateState();
        tSituation *s = re->s;
        if (s->currentTime < next)
            continue;
        next += RCM_MAX_DT_ROBOTS;
        tCarElt *car = s->cars[0];
        fprintf(f, "%.3f %d %.3f %.3f %.3f %.3f %.3f %.3f %.4f %.4f %.4f\n", s->currentTime, car->_laps,
                car->_distFromStartLine, car->_distRaced, car->_speed_x, car->_pos_X, car->_pos_Y,
                car->_lastLapTime, car->_steerCmd, car->_accelCmd, car->_brakeCmd);
        if (car->_laps > laps || (car->_state & RM_CAR_STATE_NO_SIMU) || s->currentTime > 3600)
            break;
    }
    fclose(f);
    GfTraceShutdown();
    std::_Exit(0);
}

} // namespace

extern "C" __attribute__((visibility("default"))) rlp_host_api rlp_host = {
    RLP_HOST_API_VERSION, hostNewTrack, hostNewRace, hostDrive, hostEndRace};

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);

    GfApplication *app = new GfApplication("rlpilot-worker", "1", "Speed Dreams RL worker");
    app->initialize(/*bLoggingEnabled=*/true, argc, argv);
    app->registerOption("s", "startrace", true);
    app->registerOption("m", "shm", true);
    app->registerOption("w", "worker", true);
    app->registerOption("o", "trace", true);
    app->registerOption("n", "tracelaps", true);
    if (!app->parseOptions())
        return 1;

    std::string race, shmName, worker, tracePrefix, traceLaps = "3";
    bool trace = app->hasOption("trace", tracePrefix);
    app->hasOption("tracelaps", traceLaps);
    if (!app->hasOption("startrace", race) ||
        (!trace && (!app->hasOption("shm", shmName) || !app->hasOption("worker", worker)))) {
        fprintf(stderr, "usage: rlpilot-worker -lc <dir>/ -s <race> (-m <shm name> -w <index> | -o <prefix> [-n <laps>])\n");
        return 1;
    }
    std::fill(std::begin(H.slot_of), std::end(H.slot_of), -1);
    if (!trace) {
        H.wid = (uint32_t)std::stoul(worker);
        H.shm = mapShm(shmName);
        if (!H.shm || H.wid >= H.shm->n_workers) {
            fprintf(stderr, "rlpilot-worker: cannot map shm '%s' (or bad worker index)\n", shmName.c_str());
            return 1;
        }
        H.w = rlp_worker_at(H.shm, H.wid);
        H.w->pid = getpid();
        H.last_cmd_seq = rlp_load(&H.w->cmd_sync.seq);
    }

    std::string tmp = std::string(GfLocalDir()) + "/tmp/";
    portability::rmdir_r(tmp.c_str());
    if (chdir(GfDataDir()))
        fatal("cannot cd to the data dir");
    app->updateUserSettings();
    app->setEventLoop(new GfEventLoop);

    GfData::initialize();
    void *hREParams = GfParmReadFileLocal(RACE_ENG_CFG, GFPARM_RMODE_REREAD | GFPARM_RMODE_CREAT);
    GfModule *modEngine = GfModule::load("modules/racing", GfParmGetStr(hREParams, "Modules", "racing", "standardgame"));
    H.engine = modEngine ? modEngine->getInterface<IRaceEngine>() : nullptr;
    if (!H.engine)
        fatal("cannot load the race engine module");
    RlUI ui;
    H.engine->setUserInterface(ui);

    GfRaceManager *rm = GfRaceManagers::self()->getRaceManager(race);
    if (!rm)
        fatal(("no such race: " + race).c_str());
    H.engine->reset();
    H.engine->selectRaceman(rm, /*bKeepHumans=*/false);
    H.engine->configureRace(/*bInteractive=*/false);
    H.engine->race()->forceResultsOnly();
    H.engine->startNewRace();
    if (trace)
        return runTrace(tracePrefix, std::stoi(traceLaps));
    H.sim_t0 = rlp_now_ns();

    while (!H.quit) {
        H.engine->updateState();

        if (H.reset_requested) {
            if (H.engine->inData()->_reState != RE_STATE_RACE)
                fatal("reset requested after the race engine left the RACE state");
            uint64_t t0 = rlp_now_ns();
            H.engine->restartRace();
            H.w->reset_seconds += (rlp_now_ns() - t0) * 1e-9;
            H.w->n_resets++;
            H.reset_requested = H.race_over = false;
            H.fresh = H.hold = true;
            H.tick = 0;
            H.last_tick_time = -1e9;
            memset(H.cars, 0, sizeof(H.cars));
        }
        else if (!H.race_over && !H.quit && H.engine->inData()->_reState != RE_STATE_RACE) {
            // The engine ended the race (car out, race finished): report it and wait for a reset.
            H.race_over = true;
            decide();
        }
        else if (!H.race_over && !H.quit && H.cars[0] && (H.cars[0]->_state & RM_CAR_STATE_NO_SIMU)) {
            H.race_over = true; // the agent car stopped being simulated: rbDrive won't be called again
            decide();
        }
    }

    __atomic_store_n(&H.w->status, (uint32_t)RLP_ST_EXITED, __ATOMIC_RELEASE);
    rlp_signal(&H.w->done_sync, H.last_cmd_seq);
    GfParmReleaseHandle(hREParams);
    GfTraceShutdown();
    std::_Exit(0); // skip engine teardown: the process is disposable
}

// rlpilot robot module: forwards its callbacks to the rlpilot-worker host (rlp_host, looked up with
// dlsym) so that the agent's state survives the dlclose/reload done on every race restart.
// When loaded by the regular game (no host), it drives a simple centreline follower instead, which
// keeps the module usable in rendered runs and as a smoke test.

#include <dlfcn.h>
#include <string>
#include <vector>

#include <tgf.h>
#include <track.h>
#include <car.h>
#include <raceman.h>
#include <robottools.h>
#include <robot.h>

#include "../host_api.h"

static std::vector<std::string> names;
static rlp_host_api *host;

static void newTrack(int index, tTrack *track, void *carHandle, void **carParmHandle, tSituation *)
{
    *carParmHandle = nullptr; // no setup of our own: the host edits the car's params in place
    if (host)
        host->new_track(index, track, carHandle);
}

static void newRace(int index, tCarElt *car, tSituation *s)
{
    if (host)
        host->new_race(index, car, s);
}

static void fallbackDrive(tCarElt *car)
{
    float angle = RtTrackSideTgAngleL(&car->_trkPos) - car->_yaw;
    FLOAT_NORM_PI_PI(angle);
    angle -= 0.3f * car->_trkPos.toMiddle / car->_trkPos.seg->width;
    car->_steerCmd = angle / car->_steerLock;
    car->_accelCmd = car->_speed_x < 25 ? 0.5f : 0;
    car->_brakeCmd = 0;
    car->_gearCmd = car->_speed_x > 15 ? 2 : 1;
}

static void drive(int index, tCarElt *car, tSituation *s)
{
    if (host)
        host->drive(index, car, s);
    else
        fallbackDrive(car);
}

static void endRace(int index, tCarElt *car, tSituation *s)
{
    if (host)
        host->end_race(index, car, s);
}

static void shutdownRobot(int) {}

static int initFuncPt(int index, void *pt)
{
    tRobotItf *itf = (tRobotItf *)pt;
    itf->rbNewTrack = newTrack;
    itf->rbNewRace = newRace;
    itf->rbDrive = drive;
    itf->rbPitCmd = nullptr;
    itf->rbEndRace = endRace;
    itf->rbShutdown = shutdownRobot;
    itf->index = index;
    return 0;
}

extern "C" int moduleWelcome(const tModWelcomeIn *, tModWelcomeOut *welcomeOut)
{
    names.clear();
    if (void *h = GfParmReadFileLocal("drivers/rlpilot/rlpilot.xml", GFPARM_RMODE_STD)) {
        int n = GfParmGetEltNb(h, ROB_SECT_ROBOTS "/" ROB_LIST_INDEX);
        for (int i = 0; i < n; i++) {
            std::string sect = ROB_SECT_ROBOTS "/" ROB_LIST_INDEX "/" + std::to_string(i);
            names.push_back(GfParmGetStr(h, sect.c_str(), ROB_ATTR_NAME, ("rlpilot " + std::to_string(i)).c_str()));
        }
        GfParmReleaseHandle(h);
    }
    welcomeOut->maxNbItf = names.size();

    host = (rlp_host_api *)dlsym(RTLD_DEFAULT, "rlp_host");
    if (host && host->version != RLP_HOST_API_VERSION) {
        GfLogError("rlpilot: host API version %d, expected %d\n", host->version, RLP_HOST_API_VERSION);
        host = nullptr;
        return -1;
    }
    return 0;
}

extern "C" int moduleInitialize(tModInfo *modInfo)
{
    memset(modInfo, 0, names.size() * sizeof(tModInfo));
    for (size_t i = 0; i < names.size(); i++) {
        modInfo[i].name = names[i].c_str();
        modInfo[i].desc = "rlpilot RL agent";
        modInfo[i].fctInit = initFuncPt;
        modInfo[i].gfId = ROB_IDENT;
        modInfo[i].index = i;
    }
    return 0;
}

extern "C" int moduleTerminate() { return 0; }

extern "C" int rlpilot(tModInfo *modInfo) { return moduleInitialize(modInfo); }
extern "C" int rlpilotShut() { return moduleTerminate(); }

/* Hooks exported by the rlpilot-worker executable and looked up by the rlpilot robot module with
 * dlsym(RTLD_DEFAULT, "rlp_host"). The robot is dlclose'd and reloaded on every race restart, so all
 * state lives in the host; the robot only forwards its callbacks. */
#ifndef RLPILOT_HOST_API_H
#define RLPILOT_HOST_API_H

#define RLP_HOST_API_VERSION 2

struct CarElt;
struct Situation;
struct Track;

typedef struct {
    int version;
    void (*new_track)(int index, struct Track *track, void *car_handle); /* car params, editable */
    void (*new_race)(int index, struct CarElt *car, struct Situation *s);
    void (*drive)(int index, struct CarElt *car, struct Situation *s);
    void (*end_race)(int index, struct CarElt *car, struct Situation *s);
} rlp_host_api;

#endif

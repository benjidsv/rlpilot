"""Track splits. Validation tracks select checkpoints, so their results at best/ are optimistic. Test tracks never select
anything: scripts/eval_test_tracks.py runs them on a run's final best/ only, so they must stay out of training evals.

The test set was drawn before any proto-7 result, with
random.Random(20261103).sample(sorted(proto-6 train tracks - PROTO7_EVAL_TRAIN), 6). Don't redraw, even though it picked
the two longest tracks. These tracks were trained on in proto-4/5/6, so no test comparison with those runs exists.
"""

PROTO7_VALIDATION = ("Imula", "ardennen-spa", "brno", "corkscrew", "lombaro", "street-1")
PROTO7_TEST = ("alicante", "atlanta", "auriac", "brondehach", "nordschleife", "spring")
PROTO7_TRAIN = ("Summer-Olsen", "aalborg", "allondaz", "braga", "bueno", "charmey", "chemisay", "dijon", "e-track-4",
                "espie", "hidden-valley", "inuksuk", "jarama", "karwada", "kurpfalzring", "longjour", "migrants",
                "ole-road-1", "peters", "petit", "pinabashi", "prenzlau", "racepark", "racepark3", "ruudskogen", "salty",
                "sao-paulo", "torino", "tuna")
PROTO7_EVAL_TRAIN = ("jarama", "braga", "sao-paulo", "petit")  # the training tracks evaluated, as in proto-4/5/6

import math
import numpy as np
import random
from collections import namedtuple

"""
Configuration of the reset positions through env_setup in the experiment config. Everything you leave out stays random.

    "env_setup": {
        "drawer": "open",                 "open" / "closed" or a position in meters
        "sliding_door": "left",           "left" / "right" or a position in meters. The door always covers one cabinet compartment
        "switch": "on",                   "on" / "off" or a position in meters
        "green_light": "off",             "on" / "off" or 0 / 1
        "blocks": {
            "red": "table_center",                                  a place from PLACES
            "blue": {"place": "drawer", "yaw": 0.5},                a place with a chosen yaw and/or {"offset": [dx, dy]}
            "pink": {"pos": [0.1, -0.1, 0.46], "yaw": 0.0}          coordinates. Doesn't follow the drawer, see below
        }
    }

A place is a region with random position and yaw, or a spot with fixed position and yaw 0. Passing yaw fixes it in both cases.
"hidden" parks a block far away from the table, outside the view of the camera.

Places in the drawer move along with it: the drawer opens towards -y, so they are defined for the closed drawer and shifted
by the drawer state of this reset. With the drawer closed the block is inside and can't be seen. To set your own position in
the drawer use a drawer place with an offset, because coordinates ("pos") stay where they are when the drawer moves.

All numbers are measured in the simulator with scene D (see preview_resets.py to check the result):
- table top z = 0.44, drawer floor z = 0.3424 (height inside 0.058), cabinet floor z = 0.441, blocks are 0.04 high
- the drawer interior spans x 0.004 to 0.356. The y range below is the part you can see when the drawer is open by 0.16
- the cabinet spans x -0.36 to 0.16 up to y = 0.16, and the back of the door leans in up to y = 0.048 at block height. The cabinet
  regions keep a block of any yaw off the door and the walls, so they are smaller than slider_left and slider_right of the scene
"""

TABLE_Z, DRAWER_Z, CABINET_Z = 0.46, 0.3624, 0.461 # height of the center of a block resting there
BLOCK_HEIGHT = 0.04
BLOCK_RADIUS = {"red" : 0.0344, "blue" : 0.0283, "pink" : 0.0447} # half diagonal of the footprint, so it holds for any yaw
BLOCK_INDEX = {"red" : 6, "blue" : 12, "pink" : 18} # index of the block pose in the lowdim state
HIDDEN_POS = {"red" : [5.0, 5.0, 0.02], "blue" : [5.2, 5.0, 0.02], "pink" : [5.4, 5.0, 0.02]} # resting on the floor plane

# what the words of the config mean, and the furthest each joint can go. The sliding door moves towards -x when it rises
STATE_WORDS = {"sliding_door" : {"right" : 0.0, "left" : 0.27},
               "drawer" : {"closed" : 0.0, "open" : 0.16},
               "switch" : {"off" : 0.0, "on" : 0.09},
               "green_light" : {"off" : 0.0, "on" : 1.0}}
STATE_MAX = {"sliding_door" : 0.28, "drawer" : 0.22, "switch" : 0.09, "green_light" : 1.0}

Place = namedtuple("Place", ["low", "high", "in_drawer"])

def _spot(x, y, z, in_drawer = False):
    return Place([x, y, z], [x, y, z], in_drawer)

PLACES = {
    "table" : Place([0.0, -0.15, TABLE_Z], [0.35, -0.03, TABLE_Z], False),
    "table_left" : _spot(0.05, -0.09, TABLE_Z),
    "table_center" : _spot(0.175, -0.09, TABLE_Z),
    "table_right" : _spot(0.30, -0.09, TABLE_Z),
    "drawer" : Place([0.06, -0.11, DRAWER_Z], [0.30, -0.06, DRAWER_Z], True),
    "drawer_left" : _spot(0.08, -0.085, DRAWER_Z, in_drawer = True),
    "drawer_center" : _spot(0.18, -0.085, DRAWER_Z, in_drawer = True),
    "drawer_right" : _spot(0.28, -0.085, DRAWER_Z, in_drawer = True),
    "cabinet_left" : Place([-0.31, 0.095, CABINET_Z], [-0.16, 0.115, CABINET_Z], False),
    "cabinet_left_center" : _spot(-0.24, 0.105, CABINET_Z),
    "cabinet_right" : Place([-0.05, 0.095, CABINET_Z], [0.11, 0.115, CABINET_Z], False),
    "cabinet_right_center" : _spot(0.04, 0.105, CABINET_Z),
}
HIDDEN = "hidden"


def articulated_value(name, value):
    """Turns a value of the config (word or number) into a joint position. Raises an error if it can't be one."""
    if isinstance(value, str):
        if value not in STATE_WORDS[name]:
            raise ValueError(f"env_setup {name}: unknown word '{value}', choose from {list(STATE_WORDS[name])} or give a number")
        return STATE_WORDS[name][value]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= STATE_MAX[name]:
        raise ValueError(f"env_setup {name}: {value} is not a position between 0 and {STATE_MAX[name]}")
    return float(value)


def _numbers(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length \
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def parse_block(name, spec):
    """Turns the config of one block into (low, high, in_drawer, yaw), where low == high is a fixed position and yaw None is random."""
    if name not in BLOCK_INDEX:
        raise ValueError(f"env_setup blocks: unknown block '{name}', choose from {list(BLOCK_INDEX)}")
    spec = {"place" : spec} if isinstance(spec, str) else spec
    if not isinstance(spec, dict) or not set(spec) <= {"place", "pos", "yaw", "offset"} or ("place" in spec) == ("pos" in spec):
        raise ValueError(f"env_setup block {name}: expected a place name, or a dict with either 'place' (and maybe 'offset') or 'pos', "
                         f"and maybe 'yaw'. Got {spec}")
    if "yaw" in spec and not _numbers([spec["yaw"]], 1):
        raise ValueError(f"env_setup block {name}: yaw must be a number in radians, got {spec['yaw']}")
    if "offset" in spec and ("pos" in spec or not _numbers(spec["offset"], 2)):
        raise ValueError(f"env_setup block {name}: offset must be [dx, dy] on a place, got {spec['offset']}")

    if "pos" in spec:
        if not _numbers(spec["pos"], 3):
            raise ValueError(f"env_setup block {name}: pos must be [x, y, z], got {spec['pos']}")
        low, high, in_drawer = spec["pos"], spec["pos"], False
    elif spec["place"] == HIDDEN:
        low, high, in_drawer = HIDDEN_POS[name], HIDDEN_POS[name], False
    elif spec["place"] in PLACES:
        low, high, in_drawer = PLACES[spec["place"]]
    else:
        raise ValueError(f"env_setup block {name}: unknown place '{spec['place']}', choose from {list(PLACES) + [HIDDEN]}")

    low, high = np.array(low, dtype = float), np.array(high, dtype = float)
    if "offset" in spec:
        low[:2] += spec["offset"]
        high[:2] += spec["offset"]
    default_yaw = 0.0 if np.array_equal(low, high) else None
    return low, high, in_drawer, spec.get("yaw", default_yaw)


def _find_overlap(positions):
    """Returns two blocks that would be in each other, if any. Blocks on top of each other are fine."""
    names = list(positions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            separated_in_height = abs(positions[a][2] - positions[b][2]) >= BLOCK_HEIGHT - 0.001
            if not separated_in_height and np.linalg.norm(positions[a][:2] - positions[b][:2]) < BLOCK_RADIUS[a] + BLOCK_RADIUS[b]:
                return a, b
    return None


def sample_block_poses(blocks, drawer_state, rng = random):
    """
    Returns the poses of all blocks as they are in the lowdim state (x, y, z, roll, pitch, yaw for each).
    Blocks that aren't listed stay NaN, which tells the scene to sample them itself like it always has.
    """
    parsed = {name : parse_block(name, spec) for name, spec in blocks.items()}
    num_sampling_iterations = 1000
    for _ in range(num_sampling_iterations):
        positions = dict()
        for name, (low, high, in_drawer, _) in parsed.items():
            positions[name] = low + np.array([rng.random() for _ in range(3)]) * (high - low)
            if in_drawer:
                positions[name][1] -= drawer_state
        overlap = _find_overlap(positions)
        if overlap is None:
            break
    else:
        raise ValueError(f"env_setup blocks: {overlap[0]} and {overlap[1]} overlap in every try. They need "
                         f"{BLOCK_RADIUS[overlap[0]] + BLOCK_RADIUS[overlap[1]]:.3f} m between their centers, or a bigger region")

    poses = np.full((18,), np.nan)
    for name, (_, _, _, yaw) in parsed.items():
        yaw = rng.uniform(-math.pi, math.pi) if yaw is None else yaw
        poses[BLOCK_INDEX[name] - 6 : BLOCK_INDEX[name]] = np.concatenate([positions[name], [0.0, 0.0, yaw]])
    return poses


def validate_env_setup(env_setup):
    """Raises an error for anything in env_setup that wouldn't do what it says. Call this before spending time on loading models."""
    allowed = list(STATE_WORDS) + ["blocks", "cubes_table_only"] # cubes_table_only is in the paper configs, but nothing reads it
    unknown = [key for key in env_setup if key not in allowed]
    if len(unknown) > 0:
        raise ValueError(f"Unknown keys {unknown} in env_setup, choose from {allowed}")
    for name in STATE_WORDS:
        if name in env_setup:
            articulated_value(name, env_setup[name])
    generate_reset_state(env_setup, rng = random.Random(0)) # a dry run, with its own random numbers so it can't change a seeded run


def generate_reset_state(sim_hold = {}, rng = random):
    """
    This function creates a reset state that is randomized while maintaining certain states required for tasks
    - sim_hold: the env_setup of the experiment config, see the top of this file
    - rng: where the random numbers come from
    """
    """
    Reference: this is what the lowdim states represent: 
    {"sliding_door" : state_obs[0],
        "drawer" : state_obs[1],
        "button" : state_obs[2],
        "switch" : state_obs[3],
        "lightbulb" : state_obs[4],
        "green_light" : state_obs[5],
        "red_block": state_obs[6:9], # we are ignoring rotations 
        "blue_block" : state_obs[12 : 15],
        "pink_block" : state_obs[18 :21]}, {
        "red_rot": R.from_euler("XYZ", state_obs[9:12]).as_matrix(), # we are ignoring rotations 
        "blue_rot" : R.from_euler("XYZ", state_obs[15:18]).as_matrix(),
        "pink_rot" : R.from_euler("XYZ", state_obs[21:]).as_matrix()
        }
    """

    adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]] # hard coded from simulator 
    state = np.zeros((24,))
    binary_list = list()
    # return state, [False, False, False, False]
    for adjustable, idx, limits in zip(adjustables, adjustable_index, adjustable_limits):
        if adjustable in sim_hold:
            state[idx] = articulated_value(adjustable, sim_hold[adjustable])
            binary_list.append(state[idx] > (limits[1] - limits[0]) / 2)
        else:
            select = rng.random() > 0.5
            state[idx] = limits[1] if select else limits[0]
            binary_list.append(select)
    state[6:] = sample_block_poses(sim_hold.get("blocks", {}), drawer_state = state[1], rng = rng)
    return state, binary_list #binary list is used to detect behaviors

def articulated_binaries_from_start_state(start_state): # given any start state, tell us what things are articulated in which direction 
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    binary_list = list()
    for idx, limits in zip(adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        binary_list.append(start_state[idx] > midpoint)
    return binary_list 


def check_state_difference(start_state, state, robot_pos, binaries, for_display = False):
    """
    This function checks if something has changed (warranting the end of episode). 
    Normally for speed, it only checks for a slight change in state. For display purposes
    when you want to see a full articulation, the flag changes the end-of-episode criteria 
    """
    # if you want to use this code for clean display, we use different thresholds 
    # adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    for binary, idx, limits in zip(binaries, adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        near_low = limits[0] + 0.25 * (limits[1] - limits[0])
        near_high = limits[0] + 0.75 * (limits[1] - limits[0])

        if not for_display:
            if binary and state[idx] < midpoint:
                return True 
            if not binary and state[idx] > midpoint:
                return True 
        else: # this is for display where we want the task to fully finish 
            if binary and state[idx] < near_low:
                return True 
            if not binary and state[idx] > near_high:
                return True 

    if not for_display and np.linalg.norm(robot_pos - state[6:9]) < 0.06 and np.linalg.norm(state[6:8] - start_state[6:8]) > 0.001:
        return True 

    if not for_display and np.linalg.norm(robot_pos - state[12:15]) < 0.06 and np.linalg.norm(state[12:14] - start_state[12:14]) > 0.001:
        return True 

    if not for_display and np.linalg.norm(robot_pos - state[18:21]) < 0.06 and np.linalg.norm(state[18:20] - start_state[18:20]) > 0.001:
        return True 
    
    if for_display and np.linalg.norm(robot_pos - state[6:9]) < 0.06 and np.linalg.norm(state[6:8] - start_state[6:8]) > 0.03:
        return True 

    if for_display and np.linalg.norm(robot_pos - state[12:15]) < 0.06 and np.linalg.norm(state[12:14] - start_state[12:14]) > 0.03:
        return True 

    if for_display and np.linalg.norm(robot_pos - state[18:21]) < 0.06 and np.linalg.norm(state[18:20] - start_state[18:20]) > 0.03:
        return True 
    

    return False 
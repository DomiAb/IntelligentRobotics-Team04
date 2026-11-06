"""
Guidance-collection environments for DynaGuide dual-nut steering experiments.

The base policy was trained on NutAssemblySquare with the gold nut always at
x ∈ [-0.115, -0.11], y ∈ [0.11, 0.225].  It navigates by position, not colour.

Strategy: place the TARGET nut at the training position → policy reliably picks
it up regardless of colour.  Place the DISTRACTOR nut randomly on the opposite
side of the table so it never draws the policy away.

Two environments:
  NutAssemblyGuidanceGold — gold at training position → clean gold pickup examples
  NutAssemblyGuidanceRed  — red at training position  → clean red pickup examples

Use the rollouts from these envs to extract pos/neg guidance HDF5 files for
DynaGuide's dual_pick_gold / dual_pick_red configs without needing a
successful dual-square policy.
"""

from robosuite.environments.manipulation.nut_assembly import NutAssemblyDualSquare
from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RedSquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

# Where the base policy expects to find the nut (NutAssemblySquare training zone)
_TRAIN_X = [-0.115, -0.11]
_TRAIN_Y = [0.11,   0.225]

# Distractor zone: opposite side of table, clearly away from training zone.
# Minimum separation from training zone centre (y≈0.168) is > 0.33 m, well
# above the 0.22 m minimum required by ensure_valid_placement.
_DISTRACTOR_X = [-0.15, 0.1]
_DISTRACTOR_Y = [-0.25, -0.05]

_SAMPLER_COMMON = dict(
    rotation=None,
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    z_offset=0.02,
)


def _load_guidance_model(env, target_cls, target_name, distractor_cls, distractor_name):
    """
    Shared _load_model implementation for guidance-collection environments.

    Places the TARGET nut at the training spawn zone and the DISTRACTOR nut
    randomly on the opposite side of the table.  The SequentialCompositeSampler
    places target first; the distractor sampler's ensure_valid_placement then
    guarantees no overlap.
    """
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

    SingleArmEnv._load_model(env)

    xpos = env.robots[0].robot_model.base_xpos_offset["table"](env.table_full_size[0])
    env.robots[0].robot_model.set_base_xpos(xpos)

    mujoco_arena = PegsArena(
        table_full_size=env.table_full_size,
        table_friction=env.table_friction,
        table_offset=env.table_offset,
    )
    mujoco_arena.set_origin([0, 0, 0])

    # (nut_cls, nut_name, x_range, y_range) — target always first so the
    # SequentialCompositeSampler places it at the fixed training zone before
    # trying to fit the distractor anywhere that doesn't overlap.
    nut_order = [
        (target_cls,     target_name,     _TRAIN_X,      _TRAIN_Y),
        (distractor_cls, distractor_name, _DISTRACTOR_X, _DISTRACTOR_Y),
    ]

    env.nuts = []

    if env.placement_initializer is None:
        env.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for _, name, x_range, y_range in nut_order:
            env.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    reference_pos=env.table_offset,
                    **_SAMPLER_COMMON,
                )
            )

    env.placement_initializer.reset()

    for nut_cls, nut_name, _, _ in nut_order:
        nut = nut_cls(name=nut_name)
        env.nuts.append(nut)
        if isinstance(env.placement_initializer, SequentialCompositeSampler):
            env.placement_initializer.add_objects_to_sampler(
                sampler_name=f"{nut_name}Sampler", mujoco_objects=nut
            )
        else:
            env.placement_initializer.add_objects(nut)

    env.model = ManipulationTask(
        mujoco_arena=mujoco_arena,
        mujoco_robots=[robot.robot_model for robot in env.robots],
        mujoco_objects=env.nuts,
    )


class NutAssemblyGuidanceGold(NutAssemblyDualSquare):
    """
    Gold nut spawns at the training position; red nut spawns randomly elsewhere.

    The base policy reliably picks up the gold nut → use successful rollouts
    as positive guidance examples for 'steer toward gold' and negative examples
    for 'steer toward red'.

    Inherits _check_success(), which_nut_assembled(), assembled_nut_yaw() and
    all stats/video-overlay logic from NutAssemblyDualSquare unchanged.
    """

    def _load_model(self):
        _load_guidance_model(
            self,
            target_cls=SquareNutObject,    target_name="SquareNut",
            distractor_cls=RedSquareNutObject, distractor_name="RedSquareNut",
        )


class NutAssemblyGuidanceRed(NutAssemblyDualSquare):
    """
    Red nut spawns at the training position; gold nut spawns randomly elsewhere.

    The base policy reliably picks up the red nut → use successful rollouts
    as positive guidance examples for 'steer toward red' and negative examples
    for 'steer toward gold'.

    Inherits _check_success(), which_nut_assembled(), assembled_nut_yaw() and
    all stats/video-overlay logic from NutAssemblyDualSquare unchanged.
    """

    def _load_model(self):
        _load_guidance_model(
            self,
            target_cls=RedSquareNutObject, target_name="RedSquareNut",
            distractor_cls=SquareNutObject,    distractor_name="SquareNut",
        )

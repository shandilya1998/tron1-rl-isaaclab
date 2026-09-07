"""Stance load analysis for a sole footed biped.

The companion script ``kscale_physical_analysis.py`` computes the effective inertia of each
joint's distal subtree about its own axis, which is the inertia a leg presents while swinging
freely in the air. That is not the quantity which decides whether a robot can stand up. With the
foot planted the joint reacts the ground reaction force, and the static torque it must hold is
set by the body weight and a moment arm rather than by the limb's inertia.

This script supplies the three quantities that follow from that observation. The static stance
torque at each joint, obtained by balancing the distal free body against the ground reaction
force at the centre of pressure. The settled configuration, obtained by minimising the sum of
the gravitational and the proportional spring potential energy with the sole planted, which
answers the question of where the robot actually comes to rest under its own weight. And the
stance reflected inertia, being the body mass seen through the joint's own Jacobian, which is
the inertia the derivative term should be sized against.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kscale_sole_analysis import forward_kinematics, parse_urdf, rpy_to_matrix  # noqa: E402

GRAVITY = 9.80665

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environments", "environments", "assets", "urdf", "solefoot", "kscale", "kscale.urdf",
)

# Mirrors init_state.joint_pos at assets/config/kscale_identified_cfg.py.
NOMINAL_POSE = {
    "right_hip_pitch_04": -0.1, "left_hip_pitch_04": 0.1,
    "right_hip_roll_03": 0.0, "left_hip_roll_03": 0.0,
    "right_hip_yaw_03": 0.0, "left_hip_yaw_03": 0.0,
    "right_knee_04": 0.4, "left_knee_04": 0.4,
    "right_foot_pitch_02": -0.3, "left_foot_pitch_02": -0.3,
    "right_foot_roll_02": 0.0, "left_foot_roll_02": 0.0,
}

# joint -> (stiffness, damping, armature, effort limit, lower limit, upper limit)
#
# The calculated set is the one derived in KScale.md sections 6 and 11 from the stance load
# requirement, the saturation ceiling and the disturbance rejection criterion, at the action
# scale of 0.25 the configuration carries. The identified set comes from actuator system
# identification rather than from derivation. Armature, effort limits and joint travel are
# common to both, only the stiffness and the damping differing.
CALCULATED_GAINS = {
    "right_hip_pitch_04": (200.0, 22.4, 0.010, 60.0, -2.21657, 1.04720),
    "right_hip_roll_03": (150.0, 17.3, 0.010, 60.0, -2.26893, 0.20944),
    "right_hip_yaw_03": (120.0, 2.5, 0.010, 60.0, -1.57080, 1.57080),
    "right_knee_04": (200.0, 10.0, 0.015, 60.0, 0.0, 2.70526),
    "right_foot_pitch_02": (50.0, 1.7, 0.005, 17.0, -0.87267, 0.52360),
    "right_foot_roll_02": (20.0, 0.5, 0.005, 17.0, -0.26180, 0.26180),
}

# Mirrors the actuator configurations at assets/config/kscale_identified_cfg.py.
IDENTIFIED_GAINS = {
    "right_hip_pitch_04": (200.0, 8.0, 0.010, 60.0, -2.21657, 1.04720),
    "right_hip_roll_03": (250.0, 10.0, 0.010, 60.0, -2.26893, 0.20944),
    "right_hip_yaw_03": (500.0, 18.0, 0.010, 60.0, -1.57080, 1.57080),
    "right_knee_04": (300.0, 10.0, 0.015, 60.0, 0.0, 2.70526),
    "right_foot_pitch_02": (170.0, 9.0, 0.005, 17.0, -0.87267, 0.52360),
    "right_foot_roll_02": (170.0, 9.0, 0.005, 17.0, -0.26180, 0.26180),
}

GAIN_SETS = {"calculated": CALCULATED_GAINS, "identified": IDENTIFIED_GAINS}

# The module level functions below read GAINS as a global. It is rebound by main() when a
# gain set other than the default is asked for, so that the no argument invocation reproduces
# the tables of KScale.md exactly as it did before the identified set was added.
GAINS = CALCULATED_GAINS

# Measured 90th percentile of the absolute applied joint torque, and the fraction of samples
# at or above 99 per cent of the effort limit, taken from the replay of 2026-08-28_04-50-51 at
# seed 42 on 2026-09-04, right leg. A joint that saturates does not report its disturbance, the
# clipped torque being a lower bound on the torque actually demanded, so the disturbance
# criterion below is reported for the unsaturated joints alone.
# joint -> (p90 |tau| in Nm, saturated fraction)
MEASURED_DISTURBANCE = {
    "right_hip_pitch_04": (60.338, 0.464),
    "right_hip_roll_03": (60.305, 0.493),
    "right_hip_yaw_03": (12.134, 0.000),
    "right_knee_04": (28.106, 0.003),
    "right_foot_pitch_02": (17.118, 0.501),
    "right_foot_roll_02": (17.027, 0.509),
}

ANKLE_ROLL_JOINT = "right_foot_roll_02"
SOLE_DEPTH = 0.0430          # metres below the ankle roll origin, along the foot frame's plus y
SOLE_NORMAL = (0.0, 1.0, 0.0)
SOLE_HALF_WIDTH = 0.0846 / 2

# The three sagittal joints, in the order the settled pose solver varies them.
SAGITTAL = ("right_hip_pitch_04", "right_knee_04", "right_foot_pitch_02")


def parse_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Mass, centre of mass offset and inertia tensor of every link with an inertial block.

    The tensor is rotated out of the inertial frame into the link frame by ``R I R^T``, since
    these inertial frames carry a non trivial rpy and are not aligned with the link.
    """
    out: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for link in ET.parse(path).getroot().iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        text = origin.get("xyz") if origin is not None else None
        com = np.array([float(v) for v in (text or "0 0 0").split()])
        rpy_text = origin.get("rpy") if origin is not None else None
        rpy = np.array([float(v) for v in (rpy_text or "0 0 0").split()])
        node = inertial.find("inertia")
        ixx, ixy, ixz = float(node.get("ixx")), float(node.get("ixy")), float(node.get("ixz"))
        iyy, iyz, izz = float(node.get("iyy")), float(node.get("iyz")), float(node.get("izz"))
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        rotation = rpy_to_matrix(*rpy)
        out[link.get("name", "")] = (
            float(inertial.find("mass").get("value")), com, rotation @ tensor @ rotation.T,
        )
    return out


def descendants(joints, link: str) -> set[str]:
    """Every link at or below the given link in the kinematic tree."""
    children: dict[str, list[str]] = {}
    for joint in joints.values():
        children.setdefault(joint.parent, []).append(joint.child)
    seen: set[str] = set()
    stack = [link]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return seen


def joint_axis(joints, rot, name: str) -> np.ndarray:
    """The joint's rotation axis expressed in the root frame, normalised."""
    joint = joints[name]
    axis = rot[joint.parent] @ joint.origin_rot @ joint.axis
    return axis / np.linalg.norm(axis)


class Model:
    """Forward kinematics, mass properties and ground contact for one URDF."""

    def __init__(self, urdf: str):
        self.links, self.joints, self.root = parse_urdf(urdf)
        self.inertials = parse_inertials(urdf)
        self.mass = {n: m for n, (m, _, _) in self.inertials.items()}
        self.total_mass = sum(self.mass.values())
        self.weight = self.total_mass * GRAVITY
        self.foot = self.joints[ANKLE_ROLL_JOINT].child

    def pose_from(self, hip: float, knee: float, ankle: float) -> dict[str, float]:
        """Both legs at the mirrored equivalent of the three sagittal angles given.

        The hip pitch axes are anti parallel between the legs, so the left hip takes the
        negation of the right for one physical posture, while the knee and the ankle pitch
        axes are parallel and take the same sign on both legs.
        """
        pose = dict(NOMINAL_POSE)
        pose.update({
            "right_hip_pitch_04": hip, "left_hip_pitch_04": -hip,
            "right_knee_04": knee, "left_knee_04": knee,
            "right_foot_pitch_02": ankle, "left_foot_pitch_02": ankle,
        })
        return pose

    def kinematics(self, pose: dict[str, float]):
        """Link rotations and positions, plus the height shift that plants the sole."""
        rot, pos = forward_kinematics(self.joints, self.root, pose)
        sole = pos[self.foot] + rot[self.foot] @ (SOLE_DEPTH * np.array(SOLE_NORMAL))
        return rot, pos, -sole[2]

    def com_world(self, rot, pos, shift: float) -> np.ndarray:
        """Whole body centre of mass with the sole resting on the ground plane."""
        total = sum(
            self.mass[n] * (pos[n] + rot[n] @ com)
            for n, (_, com, _) in self.inertials.items() if n in pos
        )
        return total / self.total_mass + np.array([0.0, 0.0, shift])

    def base_height(self, pose: dict[str, float]) -> float:
        rot, pos, shift = self.kinematics(pose)
        return pos[self.root][2] + shift


def stance_torques(model: Model, pose: dict[str, float], support: float) -> dict[str, float]:
    """Static torque at each right leg joint with the sole planted.

    ``support`` is the fraction of the body weight borne by this foot, being one half in
    symmetric double support and one in single support. The centre of pressure is placed
    directly beneath the ankle roll axis, which is the balanced case.
    """
    rot, pos, shift = model.kinematics(pose)
    ground = np.array([pos[model.foot][0], pos[model.foot][1], -shift])
    grf = np.array([0.0, 0.0, model.weight * support])
    gravity = np.array([0.0, 0.0, -GRAVITY])
    out: dict[str, float] = {}
    for name in GAINS:
        origin = pos[model.joints[name].child]
        axis = joint_axis(model.joints, rot, name)
        torque = float(axis @ np.cross(ground - origin, grf))
        for link in descendants(model.joints, model.joints[name].child):
            if link not in model.inertials:
                continue
            mass, com, _ = model.inertials[link]
            world = pos[link] + rot[link] @ com
            torque += mass * float(axis @ np.cross(world - origin, gravity))
        out[name] = torque
    return out


def swing_inertia(model: Model, name: str) -> float:
    """Distal subtree inertia about the joint axis at the nominal pose, armature included.

    Each link contributes its own inertia resolved along the axis, ``a . R I R^T . a``, plus
    the parallel axis term ``m r_perp^2``. The own inertia term is taken in full three
    dimensions rather than by reading a diagonal entry, because these inertial frames are not
    in general aligned with the axis acting upon them.
    """
    rot, pos, _ = model.kinematics(NOMINAL_POSE)
    origin = pos[model.joints[name].child]
    axis = joint_axis(model.joints, rot, name)
    total = 0.0
    for link in descendants(model.joints, model.joints[name].child):
        if link not in model.inertials:
            continue
        mass, com, tensor = model.inertials[link]
        world_tensor = rot[link] @ tensor @ rot[link].T
        offset = pos[link] + rot[link] @ com - origin
        perpendicular = offset - float(offset @ axis) * axis
        total += float(axis @ world_tensor @ axis) + mass * float(perpendicular @ perpendicular)
    return total + GAINS[name][2]


def settled_pose(model: Model, stiffness: dict[str, float]):
    """Minimise gravitational plus spring potential energy with the sole planted.

    Both legs are constrained to the same posture, so each spring term is counted twice, and
    the constant factor of one half in the spring energy is dropped since it scales the whole
    objective. The result is where the robot comes to rest under its own weight.
    """
    q0 = np.array([NOMINAL_POSE[j] for j in SAGITTAL])
    bounds = [(GAINS[j][4], GAINS[j][5]) for j in SAGITTAL]
    gains = np.array([stiffness[j] for j in SAGITTAL])

    def energy(q: np.ndarray) -> float:
        rot, pos, shift = model.kinematics(model.pose_from(*q))
        potential = model.total_mass * GRAVITY * model.com_world(rot, pos, shift)[2]
        return potential + float(np.sum(gains * (q - q0) ** 2))

    best = None
    for start in (q0, np.array([-0.1, 0.8, -0.5]), np.array([-0.3, 1.5, -0.8])):
        result = minimize(energy, start, bounds=bounds, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    return best.x, model.base_height(model.pose_from(*best.x))


def stance_inertia(model: Model) -> dict[str, float]:
    """Swing inertia plus the body mass reflected through the joint's own Jacobian.

    A joint that lifts the whole body accelerates far more than its own distal subtree. The
    reflected term is ``M (d z_com / d q)^2``, taken by central difference at the nominal pose,
    and it applies only to the sagittal joints, the remaining joints not raising the body to
    first order in a symmetric stance.
    """
    q0 = np.array([NOMINAL_POSE[j] for j in SAGITTAL])
    step = 1e-4
    out: dict[str, float] = {}
    for name in GAINS:
        swing = swing_inertia(model, name)
        if name not in SAGITTAL:
            out[name] = swing
            continue
        index = SAGITTAL.index(name)
        heights = []
        for sign in (+1.0, -1.0):
            q = q0.copy()
            q[index] += sign * step
            rot, pos, shift = model.kinematics(model.pose_from(*q))
            heights.append(model.com_world(rot, pos, shift)[2])
        jacobian = (heights[0] - heights[1]) / (2 * step)
        out[name] = swing + model.total_mass * jacobian ** 2
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Stance load analysis for a sole footed biped.")
    parser.add_argument("--urdf", default=os.path.normpath(DEFAULT_URDF))
    parser.add_argument("--budget", type=float, default=0.05,
                        help="acceptable static sag in radians before a stiffness is flagged")
    parser.add_argument("--gains", choices=sorted(GAIN_SETS), default="calculated",
                        help="which actuator gain set to analyse")
    parser.add_argument("--action-scale", type=float, default=0.25,
                        help="joint position action scale, for the saturation check")
    parser.add_argument("--control-period", type=float, default=0.02,
                        help="control period in seconds, for the Nyquist check")
    args = parser.parse_args()

    global GAINS
    GAINS = GAIN_SETS[args.gains]

    model = Model(args.urdf)
    print(f"gain set      {args.gains}")
    print(f"URDF          {args.urdf}")
    print(f"total mass    {model.total_mass:.4f} kg, weight {model.weight:.2f} N")
    print(f"nominal pose  base height {model.base_height(NOMINAL_POSE):.5f} m")

    double = stance_torques(model, NOMINAL_POSE, 0.5)
    single = stance_torques(model, NOMINAL_POSE, 1.0)
    print("\nStatic stance load, centre of pressure beneath the ankle roll axis")
    print(f"{'joint':22}{'tau_2sup':>10}{'tau_1sup':>10}{'K':>8}{'sag_2sup':>10}"
          f"{'sag_1sup':>10}{'effort':>9}{'verdict':>22}")
    for name, (K, _D, _a, effort, lo, hi) in GAINS.items():
        t2, t1 = double[name], single[name]
        sag2, sag1 = abs(t2) / K, abs(t1) / K
        if max(abs(t2), abs(t1)) > effort:
            verdict = "EXCEEDS EFFORT LIMIT"
        elif sag1 > (hi - lo):
            verdict = "DEFLECTS PAST RANGE"
        elif sag2 > args.budget:
            verdict = "sag over budget"
        else:
            verdict = "ok"
        print(f"{name:22}{t2:10.3f}{t1:10.3f}{K:8.1f}{sag2:10.3f}{sag1:10.3f}"
              f"{effort:9.1f}{verdict:>22}")

    q, height = settled_pose(model, {n: g[0] for n, g in GAINS.items()})
    nominal = model.base_height(NOMINAL_POSE)
    print("\nSettled configuration under body weight, symmetric double support")
    print(f"  hip pitch {q[0]:+.4f}  knee {q[1]:.4f}  ankle pitch {q[2]:+.4f}")
    print(f"  base height {height:.5f} m against a nominal {nominal:.5f} m, "
          f"sag {(nominal - height) * 1000:.1f} mm")

    print("\nStance reflected inertia against swing inertia, and the damping each implies")
    reflected = stance_inertia(model)
    print(f"{'joint':22}{'I_swing':>10}{'I_stance':>10}{'ratio':>8}{'K':>7}{'D':>7}"
          f"{'z_swing':>9}{'z_stance':>10}{'w_stance':>10}")
    for name, (K, D, _a, _e, _lo, _hi) in GAINS.items():
        swing = swing_inertia(model, name)
        stance = reflected[name]
        print(f"{name:22}{swing:10.5f}{stance:10.5f}{stance / swing:8.2f}{K:7.1f}{D:7.2f}"
              f"{D / (2 * (K * swing) ** 0.5):9.3f}{D / (2 * (K * stance) ** 0.5):10.3f}"
              f"{(K / stance) ** 0.5:10.2f}")

    torque = (model.weight / 2) * SOLE_HALF_WIDTH
    K_roll, _, _, _, lo, hi = GAINS[ANKLE_ROLL_JOINT]
    print("\nAnkle roll lateral criterion, centre of pressure held at the edge of the sole")
    print(f"  demand {torque:.2f} Nm, deflection at K = {K_roll:.1f} is "
          f"{torque / K_roll:.3f} rad against a travel of {lo:+.4f} to {hi:+.4f} rad")
    print(f"  minimum stiffness to stay inside the travel "
          f"{torque / max(abs(lo), abs(hi)):.2f} Nm/rad")

    nyquist = np.pi / args.control_period
    print(f"\nSaturation headroom at an action scale of {args.action_scale:.2f} rad, "
          f"and bandwidth against a Nyquist bound of {nyquist:.2f} rad/s")
    print(f"{'joint':22}{'K':>7}{'K*scale':>9}{'tau_2sup':>10}{'sum':>9}{'effort':>8}"
          f"{'headroom':>10}{'q_sat':>8}{'w_stance':>10}{'w/nyq':>8}")
    for name, (K, _D, _a, effort, _lo, _hi) in GAINS.items():
        commanded = K * args.action_scale
        total = commanded + abs(double[name])
        w = (K / reflected[name]) ** 0.5
        print(f"{name:22}{K:7.1f}{commanded:9.2f}{abs(double[name]):10.3f}{total:9.2f}"
              f"{effort:8.1f}{100 * (effort - total) / effort:9.1f}%{effort / K:8.3f}"
              f"{w:10.2f}{w / nyquist:8.3f}")

    print(f"\nFeasible stiffness band at an action scale of {args.action_scale:.2f} rad, "
          f"the load floor taken at a sag budget of {args.budget:.2f} rad")
    print(f"{'joint':22}{'K_min_load':>12}{'K_max_noclip':>14}{'K':>7}{'verdict':>22}")
    for name, (K, _D, _a, effort, _lo, _hi) in GAINS.items():
        floor = abs(double[name]) / args.budget
        ceiling = (effort - abs(double[name])) / args.action_scale
        if floor > ceiling:
            verdict = "BAND EMPTY"
        elif not floor <= K <= ceiling:
            verdict = "K outside band"
        else:
            verdict = "ok"
        print(f"{name:22}{floor:12.1f}{ceiling:14.1f}{K:7.1f}{verdict:>22}")

    print("\nDisturbance rejection, K needed to hold the measured p90 torque within a deflection")
    print(f"{'joint':22}{'p90 tau':>9}{'sat':>7}{'K':>7}{'dev at K':>10}"
          f"{'K@0.05':>9}{'K@0.10':>9}{'K@0.20':>9}{'note':>14}")
    for name, (K, _D, _a, _e, _lo, _hi) in GAINS.items():
        tau, sat = MEASURED_DISTURBANCE[name]
        note = "SATURATED" if sat > 0.05 else "valid"
        print(f"{name:22}{tau:9.2f}{sat:7.3f}{K:7.1f}{tau / K:10.3f}"
              f"{tau / 0.05:9.1f}{tau / 0.10:9.1f}{tau / 0.20:9.1f}{note:>14}")
    print("  A saturated joint reports a lower bound on its disturbance, so its K columns are")
    print("  not a requirement. The criterion binds only where the joint never reaches its limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

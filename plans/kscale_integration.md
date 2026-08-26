# KScale Biped Integration, Implementation Plan

> Status, MIXED. Chapters 1 to 4 are IMPLEMENTED 2026-08-24, every section of chapter 4 having been carried out, and the outcome together with seven divergences from what this document proposed is recorded in section 7 at the foot of the page. Chapter 5, KBot Specific Reward Tuning, is IMPLEMENTED 2026-08-26, its outcome recorded in section 7.8. The physical parameterisation it establishes has been promoted to [../context/KScale.md](../context/KScale.md), which supersedes section 3 of this document wherever the two disagree, and is the source a later reader should consult. See [README.md](README.md) for the register.

This document is the implementation brief for bringing the KScale biped to parity with the SD_BRS1 biped, which this repository refers to throughout as the BRS. The KScale configuration entered the tree as a direct copy of the BRS configuration taken at an earlier point in its development, retargeted onto a different set of link and joint names, and it has since fallen behind the BRS on every axis the gait work stream advanced, the sole aware clearance and landing rewards, the impact penalty, the graced single support term, the symmetry augmentation, and the location of the sources themselves. The object of this plan is that the KScale environment and agent configuration mimic the BRS, differing only where the two robots genuinely differ, and that the BRS, the TRON1 SoleFoot, PointFoot and WheelFoot tasks, and the quadruped are left bit for bit unaltered.

The plan carries its whole codebase investigation as validation material, so that the implementing agent needs no further exploration, and it states the design rationale for every proposed value rather than only the value. Its conclusions rest on four parallel investigations recorded during the writing of this document, on direct measurement of the KScale meshes, and on the accumulated record of the BRS gait work stream in `../../context/brs_gait.md` and the workspace plans it indexes.

## 1. Introduction

The two robots are both sole footed bipeds of six actuated degrees of freedom per leg, arranged in the same order from the torso outward, a hip pitch, a hip roll, a hip yaw, a knee, an ankle pitch and an ankle roll, terminating in a flat plate that meets the ground over an area rather than at a point. That shared topology is what makes the BRS configuration a legitimate starting point for the KScale, and it is why the original port succeeded in producing a configuration that at least reads as coherent. The resemblance is close enough that a reader comparing the two environment configuration files side by side finds them structurally identical, term for term, differing only in the regular expressions that name the bodies and the joints.

The resemblance is nonetheless superficial in three respects that this plan exists to address, and each of the three defeats a different assumption the original port made.

The first is scale. The KScale masses 20.281 kg against the BRS at 59.85 kg, a ratio near one third, and its hip to ankle leg length is 0.505 m against 0.87 m. Every reward parameter carrying units of length, force or height is therefore wrong by roughly that ratio when copied across unchanged, and several were copied across unchanged. The consequences are not uniform, since a Gaussian kernel width and a contact force threshold scale differently, so the correction has to be made term by term against the physics of each rather than by applying one factor throughout.

The second is geometry. The BRS carries a near rectangular sole plate whose lowest surface sits 0.124 m below the ankle frame, and the reward set that the gait work stream converged upon depends on knowing the shape of that plate, not merely its depth. The KScale sole is a tapered blade of a different shape at a different depth, and, decisively, its foot link frame is a full axis permutation away from the BRS convention, its own y axis pointing downward where the BRS uses z. The original port read the minimum of the foot mesh's link frame z coordinate, obtained 0.19, compared it against the BRS figure of 0.124, found the discrepancy inexplicable, and recorded its own suspicion in a comment at the point of use. That suspicion was correct. The figure 0.19 is the foot's length and not its depth, and the four reward terms that depend on the sole were dropped rather than shipped against a number nobody trusted, which is the immediate reason the KScale reward set is four terms short of the BRS.

The third, and the most consequential, is the frame convention of the root link itself, which no comment in the tree records and which the original port did not detect. The KScale root frame is rotated by ninety degrees about its vertical axis relative to the convention Isaac Lab assumes, so the axis the configuration commands as forward velocity is in fact the robot's lateral axis. This is established in section 3.2 and its remedy is section 4.2. It is stated here, at the head of the document, because it governs the interpretation of everything that follows, a velocity command, a height scanner footprint, a symmetry mirror plane and a foot separation measurement all being expressed in that frame.

The document proceeds by establishing the differences between the two robots in detail, then by reporting the measurement of the KScale URDF and its meshes together with the script that performs it, and finally by setting out the changes, each with the rationale that justifies it.

## 2. BRS and KScale Biped Comparison

### 2.1 Morphological Differences

The KScale is the smaller robot throughout, and the ratios are not uniform across its segments, which matters because a uniform scale factor would otherwise be a legitimate shortcut for the reward parameters.

| Quantity | BRS | KScale | Ratio | Source |
|---|---|---|---|---|
| Total mass | 59.85 kg | 20.281 kg | 0.339 | `../context/BRS.md:33`, `assets/urdf/solefoot/kscale/kscale.urdf` inertial blocks |
| Torso mass | not restated here | 4.319 kg | | `kscale.urdf:7-11` |
| Thigh length, hip yaw to knee | see `../context/BRS.md:18` | 0.21417 m | | `kscale.urdf` joint origins |
| Shank length, knee to ankle | see `../context/BRS.md:19` | 0.29051 m | | `kscale.urdf` joint origins |
| Hip to ankle leg length | 0.87 m | 0.505 m | 0.580 | `../context/BRS.md:18-19` |
| Nominal standing height | 1.15 m | 0.7953 m | 0.692 | `cfg/SF/brs_base_env_cfg.py:750`, measured at the KScale nominal pose |
| Sole depth below the ankle frame | 0.124 m | 0.0430 m | 0.347 | `../../context/brs_gait.md:89`, section 3.3 below |
| Sole length | 0.2612 m | 0.2100 m | 0.804 | `cfg/SF/brs_base_env_cfg.py:644-657`, section 3.3 below |
| Sole width | 0.194 m | 0.0846 m | 0.436 | as above |
| Foot lateral separation at the nominal pose | 0.259 m | 0.252 m | 0.973 | `../../context/literature.md` cluster 13 correction, section 3.2 below |

Three observations follow from the table and each bears on a later section.

The sole depth scales with the mass ratio, 0.347 against 0.339, whereas the sole length scales far more weakly, 0.804, and the standing height weakly again, 0.692. The KScale therefore stands on a foot that is nearly as long as the BRS foot but less than half as wide and sits at a third of the height beneath the ankle. A parameter derived from the depth may be scaled by the mass ratio with some confidence, and a parameter derived from the footprint may not.

The foot lateral separation is very nearly the same on the two robots, 0.252 m against 0.259 m, despite the threefold mass difference. The KScale is a narrow robot in the fore and aft sense and a comparatively broad one across the hips, which means the stance width parameters transfer across almost unchanged where the height and force parameters do not.

The hip yaw joint is live on the KScale and mechanically absent on the BRS. The BRS declares both hip yaw joints with `type="fixed"` in its URDF, so they never enter `robot.joint_names` at all, and the KScale declares both as revolute over the full range of plus and minus 1.5708 rad at `kscale.urdf:614` and `:1094`. The KScale consequently actuates twelve joints where the BRS actuates ten, which propagates into the observation width, the action width and the symmetry permutation alike. A note in the tree describes the BRS hip yaw as disabled by a zero width limit rather than as a fixed joint, and that description is inaccurate, the correction being recorded in section 4.9.

The joint limits of the KScale URDF carry a uniform effort of 10 Nm and a uniform velocity of 10 rad/s on every joint without exception, from the hip pitch that carries the whole leg to the ankle roll that carries only the foot. This is the signature of an unedited export default rather than a measured hardware limit, and it is consistent with the docstring at `assets/config/kscale_identified_cfg.py:11-20` recording that the robot arrived with no actuator specification. The limits that actually bind in simulation come from the per joint actuator configuration rather than from the URDF, which is fortunate, but the URDF figures should not be mistaken for data.

### 2.2 Reward Functions

The BRS reward set carries twenty six live terms and the KScale twenty three, and the difference is not a simple subset relation. The KScale is missing four BRS terms and carries one term the BRS deliberately removed.

The four missing terms are `rew_keep_ankle_roll_zero_in_air`, `rew_foot_clearance`, `pen_foot_landing_vel` and `pen_feet_impact`. Three of the four depend on the sole geometry, which is why they were dropped together, and the comment at `cfg/SF/kscale_base_env_cfg.py:809-816` states as much, recording that a per vertex table of the true lowest points of the sole mesh had not been measured and that the term would be restored once it had. That comment is an accurate account of why the terms are absent, and section 3 supplies the missing measurement.

The one extra term is `pen_ankle_deviation`, a `joint_deviation_l1` over the four ankle joints at weight minus 0.1. The BRS carries the same term commented out at `cfg/SF/brs_base_env_cfg.py:738-746` at weight minus 0.2. The history is that the term was introduced on the BRS as an experiment against the defect of the ankles resting at their mechanical stops, recorded at `../../plans/GAIT_EFFICIENCY_PLAN.md:9`, and was subsequently removed for producing no observable effect on policy performance. The KScale revives it at half the BRS weight, which is to say it revives an experiment the BRS abandoned, and section 4.5 removes it.

Two further differences are behavioural rather than structural.

The first is that `rew_no_fly` calls the free function `no_fly` on the KScale and the stateful class `NoFlyWithGrace` on the BRS. The distinction is the grace window. A single support reward without a window prices every instant of double support at zero, and therefore drives the number of support transitions downward, which is the opposite of what a walking gait requires, since the double support interval is where weight transfer occurs and occupies roughly 24 percent of the human gait cycle [10]. Van Marum and colleagues define their single foot contact term as awarding credit if single contact occurred at least once in the preceding 0.2 s, so that a brief double support during transfer is not punished [6]. That window is twenty control steps at this task's 0.01 s period, which is deeper than the four sample contact sensor history, and is the reason the term had to become a stateful class rather than acquire an argument. The migration is nonetheless free of risk, because `NoFlyWithGrace` with `grace_steps` set to zero is bit for bit identical to `no_fly` at the same history index, a fact recorded at `../../context/brs_gait.md:593`.

The second is that `rew_keep_ankle_pitch_zero_in_air` on the KScale omits four of the six parameters the BRS passes. The BRS sets `history_index` to 0, `force_threshold` to 1.0, `pitch_scale` to 0.2 and `use_default_offset` to False, and the KScale sets only `require_airborne`. The `history_index` omission is the material one. The contact sensor writes its newest sample to index 0, so the function's default of minus 1 reads the oldest of the four buffered samples, roughly 15 ms stale, and the comment at `cfg/SF/brs_base_env_cfg.py:681-684` records exactly this. The KScale therefore grades its ankle posture against a contact state three control steps out of date, and `rew_no_fly` does so as well, its `history_index` of 0 being the one parameter of the BRS term the KScale did carry across.

The remaining terms are common to both configurations and differ only in their parameters, which section 2.5 tabulates.

### 2.3 Agent Configuration

`KscaleFlatPPORunnerCfg` at `tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:296-331` and `SD_BRS1FlatPPORunnerCfg` at `:241-282` agree on every hyperparameter of the algorithm and on the shape of every network. The value loss coefficient, the clipping parameter, the entropy coefficient at 0.005, the five learning epochs, the four minibatches, the learning rate, the adaptive schedule, the discount and trace decay factors, the desired divergence and the gradient norm ceiling are identical, as are the actor and critic hidden dimensions of 512, 256 and 128, the ELU activation, the two observation normalisation flags and the encoder geometry.

They differ in exactly two places.

The first is `max_iterations`, 15000 on the KScale against 30000 on the BRS. Nothing in the tree justifies the halving and the KScale is the harder learning problem of the two by virtue of carrying two additional actuated degrees of freedom, so the figure should be restored.

The second is `symmetry_cfg`, which the BRS sets and the KScale omits. The BRS passes an `RslRlSymmetryCfg` enabling data augmentation and mirror loss with a mirror loss coefficient of 0.0, referring the augmentation function to `tasks/locomotion/mdp/symmetry/brs.py`. The comment at `limx_rsl_rl_ppo_cfg.py:286-294` explains the omission, that the BRS augmentation matches joint names by a trailing L and R suffix and would silently mismatch against the KScale naming convention, and it is right to have declined rather than to have risked a silent bad augmentation. The omission is nonetheless costly, the symmetry augmentation being the single change that first produced a coordinated alternating gait on the BRS, recorded in the status banner of `../../plans/SYMMETRY_PLAN.md`, and section 4.7 supplies the KScale specific module.

One detail of the BRS configuration should not be copied without deliberation. Setting `mirror_loss_coeff` to 0.0 while leaving `use_mirror_loss` true causes the mirror loss to be computed on every minibatch and then multiplied by zero, so the additional forward pass is executed and its gradient contribution discarded. The loss in question is the squared discrepancy between the policy's action at a state and the mirrored action at the mirrored state, introduced by Yu, Turk and Liu [5] and adopted as the standard form thereafter [2], and it is exactly what the Isaac Lab symmetry configuration computes when the flag is set. The data augmentation, which is the mechanism that carries the benefit, is gated independently by `use_data_augmentation` and is genuinely active. The literature supports that division, both the primary symmetry paper and the Isaac Lab symmetry note reporting that augmentation outperforms the loss, the former excluding the loss from its comparison altogether on the strength of the latter's finding [1][4]. The recommendation for the KScale is therefore to enable augmentation and to set `use_mirror_loss` to False rather than to carry a coefficient of zero, which obtains the same learning behaviour without the wasted computation.

### 2.4 Other Environment Configurations

The observations, the actions and the events were, as the task brief supposed, ported faithfully, and this was verified term by term rather than assumed. Every noise standard deviation, every clip, every scale and every `__post_init__` flag of `ObservationsCfg` and `HIMObservationsCfg` matches the BRS, across all of the policy, critic, history, target encoder, commands and estimator ground truth groups. Every event range and distribution parameter matches. The only edits are the necessary retargeting of body and joint name patterns. Four exceptions were found and each is small but real.

The first is that `add_link_mass` addresses `_LEG_LINKS_NO_FOOT` on the KScale and therefore excludes the feet from mass randomisation, where the BRS includes them.

The second is that `gait_command.durations` is 0.6 on the KScale against 0.62 on the BRS, while the comment above the term at `cfg/SF/kscale_base_env_cfg.py:143-146` states that the block was ported unchanged. The comment misstates the value it claims to have copied. The BRS raised its own figure from 0.6 to 0.62 as part of the double support work recorded at `../../context/brs_gait.md:522`.

The third is that both `_PLAY` variants drop three of the four command and episode overrides the BRS play configurations carry, retaining only `lin_vel_x`, and set that to the range the BRS uses for its HIM play task even on the non HIM task.

The fourth is that the KScale defines five leaf environment configuration classes against the BRS exemplar of eight, lacking `KscaleHIMBlindRoughEnvCfg` and its play sibling, so the combination of the hybrid internal model with rough terrain cannot be launched for this robot at all.

Two structural matters lie outside the configuration files and are more serious than any of the four above.

The task registry at `tasks/locomotion/robots/__init__.py` carried an unclosed brace in the `Isaac-Limx-Kscale-Blind-Flat-Play-v0` registration, which is a syntax error in the module that registers every task of every robot in this repository, so that the BRS, the three TRON1 variants and the quadruped were all unloadable on this branch alongside the KScale. This was repaired in the pass that wrote this document and is recorded in section 4.1.

The launcher at `/ws/djinn` carries no KScale branch in either its training dispatch at `djinn:118-154` or its play dispatch at `djinn:192-206`. Both are chains of `elif` clauses over a task variable that has already been assigned a default of `Isaac-Limx-SF-Identified-Blind-Flat-v0`, so a request for the KScale does not fail, it falls through every clause and launches the TRON1 SoleFoot instead. A user would obtain a plausible looking run of the wrong robot with no diagnostic of any kind. This is the most dangerous defect in the integration precisely because it is silent, and section 4.8 repairs it.

A defect unrelated to the KScale was found during the validation pass and is recorded here rather than repaired, in the manner rule 6 of `../../CLAUDE.md` requires. `assets/__init__.py:2` executes `from .usd import *`, and `assets/usd/` contains only payload directories and no `__init__.py`, so the import cannot resolve. It dates from the quadruped integration commit `bdc40ee` and is latent only because nothing imports `environments.assets` directly, every call site reaching past it to `environments.assets.config` and below. Repairing it would change what `from environments.assets import *` exports, which is a decision for the quadruped work stream rather than for this plan.

### 2.5 Summary of the discussed differences

The table gathers every difference established above, together with the consequence of leaving it standing and the section that proposes its remedy. Values marked as derived are computed in section 3 and justified in section 4.

| Aspect | BRS | KScale as it stands | Same | Consequence if left | Remedy |
|---|---|---|---|---|---|
| Root frame convention | lateral axis y, forward x | lateral axis x, forward minus y | No | Forward velocity commands drive lateral motion | 4.2 |
| Source location | `environments/environments/` | was `exts/bipedal_locomotion/` | No | Unimportable, no package `__init__.py` anywhere in the chain | 4.1, done |
| Task registry syntax | valid | unclosed brace at the play registration | No | Every robot's tasks unloadable | 4.1, done |
| `djinn` dispatch | `brs`, `brs-simplified` clauses | no clause | No | Silently launches the TRON1 SoleFoot | 4.8 |
| Leaf env cfg classes | 8 | 5 | No | HIM with rough terrain unlaunchable | 4.6 |
| `sole_offsets` table | 12 points, `brs_base_env_cfg.py:644` | absent | No | Four reward terms dropped | 3.3, 4.3 |
| `rew_foot_clearance` | `foot_clearance_reward_v3`, weight 10.0 | absent | No | No swing shaping, the plateau defect returns | 4.4 |
| `pen_foot_landing_vel` | `foot_landing_vel_v2`, weight minus 30.0 | absent | No | Impact unpriced on the descent side | 4.4 |
| `pen_feet_impact` | `feet_impact_force`, minus 3.0e-2 at 850 N | absent | No | Impact unpriced on the force side | 4.4 |
| `rew_keep_ankle_roll_zero_in_air` | weight 0.25 | absent | No | Ankle roll unregulated in swing | 4.4 |
| `pen_ankle_deviation` | removed after producing no effect | live at minus 0.1 | No | Revives an abandoned experiment | 4.5 |
| `rew_no_fly` function | `NoFlyWithGrace`, `grace_steps` 20 | `no_fly` | No | Double support priced at zero, transitions suppressed | 4.4 |
| `rew_keep_ankle_pitch_zero_in_air` params | 6 parameters set | 1 parameter set | No | Graded against a 15 ms stale contact state | 4.4 |
| `pen_feet_regulation` `foot_radius` | 0.124, measured | 0.19, an admitted misread | No | A grounded foot reports 0.147 m of false clearance | 4.3 |
| `pen_base_height` `target_height` | 1.15 | 1.0, a guess | No | Height penalty centred 0.2 m above the true stance | 4.3 |
| `pen_feet_regulation` `base_height_target` | 1.15 | 1.0, tracking the guess | No | As above | 4.3 |
| `low_height` `minimum_height` | 0.4 | 0.35, scaled from the guess | No | Termination threshold at 44 percent of stance, not 35 | 4.3 |
| `pen_feet_distance` `min_feet_distance` | 0.25 | 0.21 | No | Stance floor set below the BRS ratio | 4.3 |
| `gait_command.durations` | 0.62 | 0.6, with a comment claiming parity | No | Diverges from the BRS double support setting | 4.3 |
| `gait_command.swing_height` | 0.08 | 0.08 | Yes | Proportionally a much larger swing on a shorter leg | 4.3 |
| `add_link_mass` bodies | includes the feet | excludes the feet | No | Feet escape mass randomisation | 4.3 |
| `_PLAY` overrides | 4 | 1 | No | Play runs at training command ranges | 4.6 |
| Hip yaw | fixed in the URDF | revolute, plus and minus 1.5708 | No | Twelve actuated joints against ten, no gain randomisation, no reset perturbation | 4.3, 4.7 |
| `enabled_self_collisions` | see 3.4 | False, while `self_collision` is True | No | Contradictory, and the BRS forged contact exploit is undetectable | 4.3 |
| Actuator damping ratios | targeted near 0.7 | 1.64 to 16.4, every joint | No | Near rigid joints, two ankles at or above Nyquist | 4.3 |
| `max_iterations` | 30000 | 15000 | No | Halved training budget on the harder problem | 4.7 |
| `symmetry_cfg` | set, augmentation active | absent | No | The change that first produced a walking BRS gait is missing | 4.7 |
| Observations, all groups | | identical | Yes | | none |
| Events, all terms | | identical but for `add_link_mass` | Yes | | none |
| PPO hyperparameters and network shapes | | identical | Yes | | none |
| Curriculum, all terms | | identical | Yes | | none |

## 3. KScale URDF And STL Analysis

The URDF alone does not answer the questions the reward set asks of it. It gives the joint origins and the link inertias, and it names a mesh file for each collision shape, but the shape of the sole, the depth of the sole below the ankle frame and the orientation of the foot link frame are properties of the mesh and must be measured from it. This section reports that measurement, states the script that performs it, and records two findings the measurement produced that were not being sought.

### 3.1 The script

The script is `scripts/analysis/kscale_sole_analysis.py`, written in the manner of `scripts/analysis/stats.py`, pure numpy and the standard library with no dependency on Isaac Lab, on torch or on the task package, so that it runs inside the simulation container and equally in a plain interpreter against a checked out URDF. It is robot agnostic, taking the URDF path and a regular expression naming the foot links, so that it serves the BRS and any future robot as readily as the KScale.

It proceeds in five steps. It parses the URDF into links and joints and identifies the root as the unique link that is never a joint child. It computes forward kinematics at the zero pose to obtain the world rotation of every link frame. It uses that rotation to determine which axis of the foot link frame points downward, rather than assuming that the foot's own z is its vertical, which is the assumption that produced the erroneous 0.19 figure. It transforms every collision mesh vertex through the collision origin into the link frame, isolates the vertices lying within one millimetre of the extreme along the downward axis, and takes the convex hull of that set in the two remaining axes. It then reduces the hull to a requested number of points by repeatedly deleting the vertex whose removal costs the least polygon area, which preserves the extremes that dominate the tilted minimum where a naive truncation would discard the widest part of a rounded toe.

Two checks are emitted beside the table and both earn their place.

The first is a frame convention check, reporting which root frame axis separates the two feet. Isaac Lab evaluates the velocity command and the base linear velocity observation in the root body frame, so a robot whose lateral axis is not y has its forward command pointing sideways, and this check is what detected the defect of section 3.2.

The second is a fidelity sweep, rotating both the full vertex set and the reduced table through the ankle's own roll and pitch limits, read from the two joints immediately proximal to the foot, and reporting the largest height by which the reduced table overestimates the true clearance. The comparison is performed after aligning the sole to face world down, so that the sweep exercises the foot's real roll and pitch axes rather than the arbitrary axes of a permuted link frame. This distinction is not pedantic. Sweeping in the unaligned frame reported an error of 8.399 mm for the twelve point table, and the same table measured correctly commits 0.836 mm.

The sweep also settles the point count, which would otherwise be arbitrary.

| Points retained | Worst height error over the ankle's travel |
|---|---|
| 4 | 25.116 mm |
| 8 | 2.530 mm |
| 12 | 0.836 mm |
| 16 | 0.657 mm |
| 24 | 0.389 mm |

Twelve points is the knee of that curve, improving on eight by a factor of three and improved upon by sixteen by less than a quarter. It also matches the cardinality of the BRS table and lands within a twentieth of a millimetre of the 0.85 mm fidelity the BRS table achieves over its own travel, which is the closest thing to an independent calibration available. Four points, which an earlier analysis proposed on the reasoning that a straight sided trapezoid's corners bound its hull, commits 25 mm of error, because the KScale toe is a rounded arc rather than a straight taper and a chord drawn across that arc falls some two centimetres inside the true boundary.

### 3.2 The root frame convention

The two feet of the KScale are separated in the root frame by the vector 0.2520, 0.0001, 0.0000 in metres, so the lateral axis is x. The two feet of the BRS are separated by 0.000, minus 0.259, 0.000, so its lateral axis is y, which is the Isaac Lab convention. The KScale root frame is rotated by ninety degrees about the vertical relative to that convention.

Three independent lines of evidence agree and the finding does not rest on the separation vector alone. The two hip pitch joint origins at `kscale.urdf:631` and its left mirror differ only in x, at plus and minus 0.055, and are identical in y and z. The foot's long axis of 0.21 m lies along the root frame y. The toe extends toward minus y from the ankle while the heel sits 0.02 m to plus y, so forward is minus y. The joint naming is nonetheless correct throughout, the joints named as pitch rotating about the lateral axis and those named as roll about the fore and aft axis, so nothing in the URDF is mislabelled. It is the frame convention alone that differs.

The consequences reach four places in the configuration. The velocity command's `lin_vel_x` range of minus 0.3 to 0.8 is evaluated in the root body frame and therefore commands a leftward sidestep of up to 0.8 m/s, while `lin_vel_y` at plus and minus 0.01 pins the true fore and aft velocity near zero. The height scanner's `GridPatternCfg` of size 1.6 by 1.0 lays its long axis across the robot rather than along its path. The symmetry mirror plane is the y and z plane rather than the x and z plane, which inverts the flip set as section 3.5 records. And any per axis foot separation statistic reads the stride where it intends to read the stance width.

### 3.3 The sole

The foot link frame is a full axis permutation away from the BRS convention. Its x is the sole's width, its y is the vertical with the positive sense pointing downward, and its z is the fore and aft length. Both feet use the same mesh under the same collision origin, at `kscale.urdf:582-587`, the visual origin being identical to the collision origin, so the two feet share one table and the mirroring is carried entirely by the link rotations.

The sole is a genuinely flat manufactured surface. Of the 51228 vertices, 462 lie within one millimetre of the extreme, and those 462 span a range of one micron, which is a real machined face and not a numerical accident. A second, shallower flat band exists four millimetres above it, an interior recessed face bordered by the deeper rim, and the outer rim is the physically correct contact surface since it is what first touches the ground under any orientation.

| Quantity | Value |
|---|---|
| Sole plane in the link frame | y equal to plus 0.0430 |
| Sole depth below the link origin | 0.0430 m |
| Sole width, the x extent | 0.0846 m |
| Sole length, the z extent | 0.2100 m |
| Convex hull vertices on the sole plane | 78 |
| Shape | full width at the heel, tapering through a rounded toe |

The figure 0.19 that the superseded comment reported is arithmetically correct as a minimum of the link frame z, and it is the foot's length rather than its depth. The comment's arithmetic was sound and its choice of which axis to call the lowest point was not, because it carried the BRS convention that a foot link's z is its vertical onto a robot for which that is false. The comparable figure is 0.0430 m against the BRS 0.124 m, a ratio of 0.347 which tracks the mass ratio of 0.339 closely, and is therefore evidence that the foot is a normally proportioned sole plate rather than the tall bracket the comment feared.

The table the script produces is reproduced in section 4.3.

### 3.4 Self collision and link penetration

No left leg link overlaps any right leg link at either the zero pose or the nominal pose, the two chains remaining separated throughout by the 0.126 m per side lateral offset of the hip. No link penetrates the ground plane at the configured spawn height of 1.0 m, the lowest point of the robot clearing the ground by 0.2047 m at the nominal pose, which is a large margin and suggests the spawn height is set well above the true stance.

Two non adjacent axis aligned bounding box overlaps exist at both poses. The torso overlaps each hip roll link across a region of 0.003 by 0.116 by 0.089 m, two joints removed. Each shank overlaps its own foot, one hop around the ankle bearing, over a region growing from 0.085 by 0.096 by 0.035 m at the zero pose to 0.085 by 0.202 by 0.055 m at the nominal pose. Bounding box overlap is a conservative test and neither pair is necessarily a true mesh intersection, but both warrant a visual check.

The configuration is internally contradictory on the matter. `assets/config/kscale_identified_cfg.py:113` sets `enabled_self_collisions` to False on the articulation properties while `:160` sets `self_collision` to True on the spawn configuration. This is the same flag combination that `../../context/brs_gait.md:151` and `:171` record as central to a confirmed BRS exploit, in which a trained policy pressed its legs together to forge a contact signal and thereby defeated every contact keyed reward term at once. The KScale carries the same exposure with the same geometry and, unlike the BRS, carries no instrumentation that would reveal it, no equivalent of the sole clearance logging the BRS work stream built. Resolving the contradiction and adding the forged contact check, a non zero contact force on a foot whose true sole clearance is well above zero, should precede any trust in a KScale contact keyed reward.

### 3.5 The symmetry mirror

Under a reflection, a rotation about an axis maps to a rotation about the image of that axis carrying the determinant's sign, so for the reflection matrix the axis transforms as a pseudovector. Composing each joint's origin rotation chain to the root and comparing each left joint's mirrored world axis against its right partner's actual axis settles the flip set without any appeal to a naming convention or to a roll against pitch rule, which is the method the MorphoSymm framework establishes [3] and the reason the BRS hip pitch flips despite being a pitch joint.

Every KScale joint declares its axis as the local 0, 0, 1, with the whole orientation carried in the origin rotation, so the flip set cannot be read off the axis vectors and must be composed. Doing so about the correct y and z mirror plane gives the following.

| Joint | Left world axis | Right world axis | Flips |
|---|---|---|---|
| `hip_pitch_04` | minus x | plus x | Yes |
| `hip_roll_03` | minus y | minus y | Yes |
| `hip_yaw_03` | plus z | plus z | Yes |
| `knee_04` | plus x | plus x | No |
| `foot_pitch_02` | plus x | plus x | No |
| `foot_roll_02` | plus y | plus y | Yes |

This is the physically expected pattern, the roll and yaw degrees of freedom flipping and the pitch degrees of freedom not, with the single exception of the hip pitch, which flips because its left and right URDF axes are genuinely anti parallel. That exception is corroborated independently by the joint limits, the hip pitch and hip roll being the only two joints whose left and right limits are negations of one another, at minus 1.0472 to 2.21657 against minus 2.21657 to 1.0472 and at minus 0.20944 to 2.26893 against minus 2.26893 to 0.20944, while the remaining four carry identical limits on both sides. It is also exactly the anomaly the BRS symmetry module documents for itself at `../../plans/SYMMETRY_PLAN.md:220`. The two robots are therefore structurally identical under the mirror once the frame is corrected, the KScale simply adding the live hip yaw to the flip set where the BRS hip yaw is a fixed joint.

Reflecting about the wrong plane, which is what a reader assuming the Isaac Lab convention would do, yields the flip set of hip yaw, knee and foot pitch, which is the physically implausible pattern of the pitch joints flipping and the roll joints not. That an incorrect mirror plane produces a plausible looking table is the reason section 4.2 corrects the frame before section 4.7 writes the mirror, rather than writing a mirror against the frame as it stands.

### 3.6 Actuator parameterisation

Following the parallel axis method of `../context/BRS.md:112-231`, and computing in full three dimensions because the KScale link inertial frames are not generally aligned with the joint axes acting on them, the effective inertia at each joint of the right leg at the nominal pose is as follows, with the natural frequency and damping ratio implied by the placeholder gains at `assets/config/kscale_identified_cfg.py:22-105`.

| Joint | Stiffness | Damping | Effective inertia | Natural frequency | Damping ratio |
|---|---|---|---|---|---|
| `hip_yaw` | 40 | 5 | 0.0133 | 54.9 | 3.43 |
| `hip_roll` | 150 | 45 | 1.0162 | 12.2 | 1.82 |
| `hip_pitch` | 200 | 50 | 1.1657 | 13.1 | 1.64 |
| `knee` | 200 | 22 | 0.1074 | 43.2 | 2.37 |
| `foot_roll` | 20 | 4 | 0.00074 | 164.3 | 16.42 |
| `foot_pitch` | 50 | 4 | 0.00260 | 138.6 | 5.54 |

Units are newton metres per radian, newton metre seconds per radian, kilogramme metres squared, and radians per second, the damping ratio being dimensionless.

Every joint is overdamped, the ratios running from 1.64 to 16.4 against the 0.7 that `../context/BRS.md` targets, and the two ankle joints carry a natural frequency at or above the Nyquist bound of 157.08 rad/s imposed by the 50 Hz control loop, the foot roll exceeding it outright. This is the direct consequence of copying the BRS gains by joint role onto effective inertias roughly a sixth as large, and it is the opposite failure mode from the one the BRS itself suffered, where the same class of copied gain was too soft and left the proximal joints ringing at ratios between 0.07 and 0.16. The KScale joints as configured will resist a commanded change in position almost as though position controlled open loop, and the ankles operate at the edge of what the discrete loop can represent without aliasing.

A further caution applies at the distal joints. The armature values of 0.005 to 0.015 kg m squared are of the same order as the effective inertias of the hip yaw, ankle pitch and ankle roll joints themselves, so the reflected rotor inertia is not the safely ignorable correction there that it is at the hips, where the effective inertia exceeds 1 kg m squared.

The three actuator housing meshes, `rs02.stl`, `rs03.stl` and `rs04.stl`, are placed largest at the hip pitch and smallest at the shank, with mesh volumes of 514.71, 181.16 and 60.63 cubic centimetres respectively. The naming and the size ordering are consistent with the Robstride RS02, RS03 and RS04 quasi direct drive actuator line, which would be a far better source of gains than any scaling argument. This identification is plausible speculation and nothing more, no manufacturer name, part number or specification reference appearing anywhere in the URDF or the surrounding configuration, and it is recorded so that it may be confirmed or refuted rather than relied upon.

## 4. Proposed Changes

The changes are ordered so that each rests only on those before it. Section 4.2 must precede section 4.7, because a mirror written against the uncorrected frame would carry the wrong flip set, and section 4.3 must precede any training run, because the parameters it corrects govern terms that are already live.

Throughout, the governing constraint of `../../CLAUDE.md` is that no existing caller may change behaviour. That constraint turns out to be easy to satisfy here, for a reason worth stating plainly. No reward function in `tasks/locomotion/mdp/rewards.py` hard codes a link name, a body count or a sole geometry. Every robot specific quantity is already an explicit configuration parameter, `sole_offsets`, `foot_radius`, `min_feet_distance`, `force_threshold`, `base_height_target` and the rest, and every body set arrives through a `SceneEntityCfg` pattern. The reward terms this plan restores are therefore a data problem and not a code problem, and no function in the shared `mdp` package need be edited, no optional argument added and no version two created. The BRS, the three TRON1 variants and the quadruped are untouched by every change below, and the only shared file this plan modifies at all is the task registry, where it adds registrations without altering existing ones.

### 4.1 Relocation of the sources, completed

Carried out in the pass that wrote this document, and recorded here because the plan's file references depend upon it.

The assets moved from `exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/kscale/` to `environments/environments/assets/urdf/solefoot/kscale/`, being the URDF and its 35 STL meshes. The environment configuration moved from `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/kscale/kscale_base_env_cfg.py` to `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py`, the KScale being a sole footed biped and therefore belonging with the BRS and the TRON1 SoleFoot rather than in a directory of its own. The now empty `exts/` tree was deleted in its entirety.

Five import statements were rewritten, two in `kscale_base_env_cfg.py`, one in `assets/config/kscale_identified_cfg.py` and three in `tasks/locomotion/robots/kscale_solefoot_env_cfg.py`, each changing the package root from `bipedal_locomotion` to `environments`. In `tasks/locomotion/robots/__init__.py` the dead `from ..cfg.kscale import kscale_base_env_cfg` was folded into the existing `from ..cfg.SF import ...` group, and the unclosed brace in the `Isaac-Limx-Kscale-Blind-Flat-Play-v0` registration was closed.

The relocation is a repair rather than a tidying. The `exts/` tree contained no `__init__.py` at any level, so `bipedal_locomotion` was never an importable package and every import referring to it was unresolvable, and the unclosed brace was a syntax error in the module that registers every task of every robot. The KScale had never been launchable and, on this branch, neither had anything else.

The relocation was validated four ways. Byte compilation succeeds across `environments/`, `scripts/` and `co_optimisation/`. No reference to `bipedal_locomotion` survives in any Python source. All 114 mesh references in the URDF resolve relative to the URDF's own directory, so the relative paths survived the move. And a static walk of the import graph over all 60 modules resolves every intra package module and every imported symbol, the single exception being the pre existing and unrelated `assets/__init__.py` defect recorded at the end of section 2.4.

### 4.2 Correct the root frame convention

This change must be made before any training run and before the symmetry module of section 4.7.

The object is that the root link frame carry the Isaac Lab convention, x forward, y to the left and z up. The frame is presently rotated by ninety degrees about z from that, so the correction is to re-express the root frame by the rotation that carries the robot's forward direction of minus y onto plus x, which is a rotation of plus ninety degrees about z.

Two mechanisms are available and the second is preferred.

The first is to insert a massless `base_link` above the torso joined by a fixed joint carrying `rpy="0 0 1.5708"`. It is the more conventional fix and the less invasive to the existing link definitions, but it adds a body to the articulation, which changes the width of every critic observation that iterates over bodies, `robot_mass`, `robot_inertia` and `robot_material_properties` among them, and it introduces a root link that is not the torso, so the height scanner prim path and the `_TORSO_LINK` references would each need review.

The second, and the one this plan adopts, is to re-express the existing root frame in place, which preserves the body count exactly and therefore changes no observation width. Every quantity expressed in the root frame is premultiplied by the rotation, being the root link's own inertial, visual and collision origins, the origins of the two joints whose parent is the root, and the root's inertia tensor, which transforms as `R I R^T`. Nothing below the root changes, since each subtree is rigidly attached to its parent joint and moves with it.

The transformation is mechanical and should be performed by a script rather than by hand, and it is self checking. Re-running `scripts/analysis/kscale_sole_analysis.py` after the edit must report the lateral axis as y and the verdict as conventional, where it presently reports x and non conventional. The foot separation vector must become approximately 0.000, 0.252, 0.000. The composed flip set of section 3.5, recomputed about the x and z plane in the corrected frame, must reproduce the same table, since the flip set is a physical property of the robot and not of the frame it is expressed in.

Two configuration values become correct as a consequence and neither should be edited before the frame is fixed, on pain of correcting the same defect twice in opposite directions. The velocity command ranges at `cfg/SF/kscale_base_env_cfg.py:135-140` then mean what they say, `lin_vel_x` being forward. The height scanner pattern at `:105` then lays its 1.6 m axis along the direction of travel.

Should the frame correction be deferred, which this plan does not recommend, the only coherent alternative is to swap the command ranges and the scanner dimensions and to write the symmetry mirror about the y and z plane, and to record loudly at each site that the robot's x is lateral. That path is worse in every respect except immediacy, because it distributes one defect across four files instead of repairing it in one.

### 4.3 Correct the parameters that are already live

These terms exist in the KScale configuration and carry values that were copied or guessed rather than derived. Each row states the derivation.

The sole table is the prerequisite for section 4.4 and is placed here because it belongs beside the geometry it summarises. It is emitted by the script of section 3.1 and should be inserted into `cfg/SF/kscale_base_env_cfg.py` as a module constant beside the `RewardsCfg` class, in the manner of `SD_BRS1_SOLE_OFFSETS` at `cfg/SF/brs_base_env_cfg.py:644`, so that the clearance reward and the landing gate cannot drift apart.

```python
# Twelve points on the sole rim in the foot link frame, measured from the collision mesh by
# scripts/analysis/kscale_sole_analysis.py. NOTE the axis permutation, this foot's link frame
# carries its vertical on +y and its fore-aft length on z, unlike SD_BRS1 where the vertical is
# z. The sole plane is y = +0.0430, i.e. 0.0430 m below the ankle. Reproduces the true lowest
# point of the mesh to within 0.836 mm over the ankle's full roll and pitch travel.
KSCALE_SOLE_OFFSETS = [
    [-0.0423, +0.0430, +0.0138],
    [-0.0362, +0.0430, -0.1572],
    [-0.0326, +0.0430, -0.1712],
    [-0.0258, +0.0430, -0.1805],
    [-0.0108, +0.0430, -0.1889],
    [+0.0051, +0.0430, -0.1899],
    [+0.0213, +0.0430, -0.1841],
    [+0.0326, +0.0430, -0.1712],
    [+0.0362, +0.0430, -0.1572],
    [+0.0423, +0.0430, +0.0138],
    [+0.0363, +0.0430, +0.0200],
    [-0.0363, +0.0430, +0.0200],
]
```

| Parameter | Present | Proposed | Derivation |
|---|---|---|---|
| `pen_base_height.target_height` | 1.0 | 0.795 | Measured torso height above the lowest mesh point at the nominal pose |
| `pen_feet_regulation.base_height_target` | 1.0 | 0.795 | Kept in step with the above, as the BRS keeps its own pair in step |
| `pen_feet_regulation.foot_radius` | 0.19 | 0.043 | The measured sole depth of section 3.3, against a figure that was the foot's length |
| `low_height.minimum_height` | 0.35 | 0.28 | The BRS ratio of 0.4 against 1.15 is 0.348, applied to 0.795 |
| `pen_feet_distance.min_feet_distance` | 0.21 | 0.24 | The BRS ratio of 0.25 against a 0.259 m separation is 0.965, applied to 0.252 |
| `gait_command.durations` | 0.6 | 0.62 | Parity with the BRS, which raised its own figure for the double support work |
| `gait_command.swing_height` | 0.08 | 0.05 | The BRS swing is 9.2 percent of its 0.87 m leg, applied to the 0.505 m leg |
| `init_state.pos` z | 1.0 | 0.85 | Approximately 0.05 m above the corrected 0.795 m stance, so the robot settles rather than drops |
| `add_link_mass` bodies | legs without feet | legs with feet | Parity with the BRS, which randomises foot mass |
| `enabled_self_collisions` | False, against `self_collision` True | resolve to one value | Section 3.4, the contradiction and the forged contact exposure |

The `foot_radius` correction is the largest single error in the live configuration and its magnitude deserves stating. The term computes a ground clearance as the body frame height less `foot_radius`, so at 0.19 against a true 0.043 a foot resting flat on the ground reports a clearance of minus 0.147 m. Paired with the `height_decay_scale` of 0.03, the gate `exp(-(z - r)/s)` evaluates to `exp(4.9)`, so the penalty is amplified by a factor near 134 rather than being focused near the ground, which inverts the term's intent entirely. This is the same defect family the BRS suffered in the opposite direction, where an inherited point foot radius of 0.03 against a true 0.124 made a grounded foot report 0.094 m of false clearance and retained only four percent of the configured weight, recorded at `../../plans/GAIT_STRATEGY.md:184` and `../../context/brs_gait.md:89`.

The `min_feet_distance` figure is carried across by the BRS ratio rather than derived afresh, and the biomechanical standard corroborates the result rather than the method, a stance width of 1.0 to 1.3 times hip width [11] placing the KScale floor of 0.24 m against a hip separation of 0.252 m within that band. It should be re-examined against the corrected frame, since a scalar separation term measures the planar norm and therefore reads the stride where it intends to read the stance width.

The `swing_height` reduction warrants its own note because it is the one row above that changes a command rather than a penalty. The gait command implements the periodic contact schedule of Siekmann and colleagues [7] in the form Walk These Ways gives it [8], and it declares a swing height that, until the BRS Phase 3 work, no reward read. It is now read by `foot_clearance_reward_v3` as the amplitude of the raised cosine reference, so it has become a physical setpoint rather than an unused declaration, and a swing of 0.08 m on a 0.505 m leg is proportionally half again what the BRS asks of its own leg.

Two matters in this section are directional rather than settled and are marked as such. The actuator gains of section 3.6 require a retuning pass in the manner of `../context/BRS.md` section 9, targeting a damping ratio near 0.7 and a natural frequency comfortably below the 157.08 rad/s Nyquist bound, which points toward hip stiffnesses on the order of a third of the present values and ankle stiffnesses an order of magnitude below them. That derivation is deliberately not attempted here, both because it deserves the full treatment `../context/BRS.md` gives the BRS and because confirming the Robstride identification of section 3.6 would supersede any scaling argument with measured data. The nominal standing pose likewise remains the placeholder that `assets/config/kscale_identified_cfg.py:118-125` admits it to be, and the 0.795 m figure above is measured against that placeholder, so it must be re-measured once the pose is settled.

Finally, the hip yaw joint is presently actuated and observed while receiving neither actuator gain randomisation nor a reset perturbation, where every other joint receives both. Two event terms should be extended to cover it, `hip_joint_stiffness_and_damping` at `cfg/SF/kscale_base_env_cfg.py:513` acquiring `(right|left)_hip_yaw_03` in its joint list, and a `reset_hip_yaw_joints` term added beside the existing reset terms with a position range comparable to the hip roll's.

### 4.4 Restore the four missing reward terms and repair two more

Each block below is the BRS term retargeted onto the KScale names and parameters. No reward function changes.

```python
    # Repointed from the stateless no_fly. NoFlyWithGrace with grace_steps 0 is bit for bit
    # identical to no_fly, so this is a strict extension. 20 steps is van Marum's 0.2 s window
    # at this task's 0.01 s control period.
    rew_no_fly = RewTerm(
        func=mdp.NoFlyWithGrace,
        weight=15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "threshold": 1.0,
            "history_index": 0,
            "grace_steps": 20,
        },
    )

    # The four parameters the port omitted are restored. history_index 0 reads the CURRENT
    # contact frame, the default of -1 reading the oldest of the four buffered samples.
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_foot_pitch_02"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": False,
        },
    )

    # New. The BRS counterpart, absent from the port. Same function, quarter weight, roll joints.
    rew_keep_ankle_roll_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_foot_roll_02"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
        },
    )

    # New. Tracks a raised cosine reference read from the gait clock, so the whole swing path is
    # determined rather than only its extremum. std scaled with swing_height, 0.03 * 0.05 / 0.08.
    rew_foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_v3,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "command_name": "gait_command",
            "std": 0.02,
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 1.0,
        },
    )

    # New. Charges the vertical velocity of the LOWEST SOLE POINT, gated on the true sole
    # clearance, so the gate cannot be defeated by tilting the foot. Threshold 0.75 of the
    # swing height, matching the BRS ratio of 0.06 against 0.08.
    pen_foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel_v2,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "about_landing_threshold": 0.04,
            "force_threshold": 1.0,
        },
    )

    # New. Prices the impact from the force side. 850 N is 1.448 body weights on the 59.85 kg
    # BRS, and 1.448 body weights on the 20.281 kg KScale is 288 N.
    pen_feet_impact = RewTerm(
        func=mdp.feet_impact_force,
        weight=-3.0e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 290.0,
        },
    )
```

The weights transfer unchanged and this is deliberate. A weight multiplies a kernel whose argument has already been normalised by a parameter carrying the units, so the scaling belongs in the parameter and not in the weight. The two impact terms are the exception worth watching, since `feet_impact_force` is a hinge on an absolute force rather than a normalised kernel, and its threshold has been scaled while its weight has not, which preserves the price per newton of excess. Whether the price per newton should itself scale with the robot is a question the BRS record does not answer, and the recommendation is to launch with the weight unchanged and to read the term's logged value against the BRS baseline before adjusting it.

The clearance reward should be introduced with attention to the failure it is meant to prevent. A Gaussian on instantaneous foot height multiplied by a tanh of foot speed, which is the form of the superseded version two, has an integrand depending on the instantaneous height alone, so its maximiser over a swing of fixed duration is the trajectory that reaches the target soonest, holds longest and leaves latest, which is a plateau and not an arc. A reward on an extremum determines only that extremum, whereas a reward on a reference determines the whole path, which is the reasoning behind the phase conditioned tracking form that Humanoid-Gym adopts [9] and that version three implements here. The human swing profile has an interior minimum near mid swing rather than a plateau [10], so the reference form is also the biomechanically faithful one.

### 4.5 Remove `pen_ankle_deviation`

Delete the term at `cfg/SF/kscale_base_env_cfg.py:712-720`. It is a `joint_deviation_l1` over the four ankle joints at weight minus 0.1, and it revives on the KScale an experiment the BRS ran and abandoned. The BRS introduced the same term at weight minus 0.2 against the defect of the ankles resting at their mechanical stops, recorded at `../../plans/GAIT_EFFICIENCY_PLAN.md:9`, and removed it after it produced no observable effect on policy performance, leaving it commented at `cfg/SF/brs_base_env_cfg.py:738-746` as the record of the attempt.

Two considerations argue for removal beyond mere parity. The term opposes `rew_keep_ankle_pitch_zero_in_air` and `rew_keep_ankle_roll_zero_in_air`, which section 4.4 restores and which already regulate the ankle posture, but does so unconditionally rather than only when the foot is airborne, so it penalises the ankle articulation that stance requires. And it is being carried at half the weight the BRS found ineffective, so the KScale would be running a weaker version of a null result.

`pen_hip_deviation` should be retained as it stands, including its coverage of the hip yaw joints, which the BRS cannot penalise because its own hip yaw is a fixed joint. That coverage follows the Isaac Lab G1 recipe of penalising deviation on the hip roll and yaw while leaving the hip pitch and the knee to swing freely, and it is the appropriate treatment of a degree of freedom the KScale has and the BRS does not.

### 4.6 Complete the environment class hierarchy and the registrations

Add `KscaleHIMBlindRoughEnvCfg` and `KscaleHIMBlindRoughEnvCfg_PLAY` to `tasks/locomotion/robots/kscale_solefoot_env_cfg.py`, following `brs_solefoot_env_cfg.py:247-265` as the template, so that the eight leaf classes of the BRS exemplar are matched. Register both in `tasks/locomotion/robots/__init__.py` with `HIMManagerBasedRLEnv` as the entry point, in the manner of the existing HIM registrations.

Restore the three dropped overrides to both `_PLAY` variants, being `lin_vel_y`, `ang_vel_z` and `episode_length_s`, and set `lin_vel_x` on the non HIM play task to the range the BRS non HIM play task uses rather than to its HIM range.

The six existing registrations are otherwise correct, each pointing at a class that exists, with the HIM tasks correctly using `HIMManagerBasedRLEnv` and the remainder the standard manager based environment.

### 4.7 Symmetry augmentation and the agent configuration

The new module is `tasks/locomotion/mdp/symmetry/kscale.py`, written closely against `symmetry/brs.py` so that the two are reviewable side by side. It must not be a copy with substituted strings, for a reason that is easy to miss. The BRS module discovers each joint's mirror partner by swapping a trailing L for a trailing R, at `symmetry/brs.py:82-91`. Every KScale joint name ends in a digit, so that rule matches nothing and degenerates silently to an identity permutation, raising no error and producing an augmentation that teaches the policy the robot is symmetric under doing nothing. The partner rule must instead swap the `right_` and `left_` prefixes. The body names need an explicit dictionary rather than any rule at all, since the right hip roll link is named `kd_d_201r_6061` and its left partner is named `rs03`, sharing no stem whatever.

Four constants change from the BRS module and each is established above. The joint partner rule becomes the prefix swap. The body partner map becomes an explicit dictionary. The sign flip set becomes the table of section 3.5, being the hip pitch, the hip roll, the hip yaw and the ankle roll, which differs from the BRS set by the addition of the hip yaw. And the height scan grid shape must be recomputed from whatever `GridPatternCfg` the corrected configuration carries after section 4.2.

Two elements transfer unchanged and should be understood rather than merely copied. The mirror of the observation groups follows the reflection physics, a polar vector such as the base linear velocity or the projected gravity flipping only its lateral component, a pseudovector such as the base angular velocity flipping its roll and yaw components and keeping its pitch, and the velocity command flipping its lateral and yaw components [3]. The gait phase, which the KScale policy group observes and which is a sine and cosine pair of one shared clock, negates both channels together, because the two feet are placed in antiphase by the command's offset of 0.5 rather than by the observation, so exchanging the feet is exactly a half cycle shift of the shared clock [1]. A naive treatment of the pair as an even and an odd channel would negate only one and is wrong.

The critic group must be mirrored as well as the policy group. A value function that observes an unmirrored privileged state cannot be invariant under the reflection, and this was raised late in the BRS work and is recorded as established fact in `../../context/rsl_rl.md`.

One quantity cannot be determined statically. The number of collision shapes per body, which the mirror of `robot_material_properties` requires, is known only at runtime, and the BRS module carries it as a constant obtained from a one time print. The KScale module should follow the same procedure, leaving the constant `None` until the count has been read from a running environment, under which the term falls back to an identity mirror rather than to a wrong one.

In `agents/limx_rsl_rl_ppo_cfg.py`, `KscaleFlatPPORunnerCfg` acquires the symmetry configuration and restores its iteration budget.

```python
from environments.tasks.locomotion.mdp.symmetry.kscale import (
    compute_symmetric_states as kscale_compute_symmetric_states,
)

    max_iterations = 30000
    ...
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            # False rather than the BRS's True with a zero coefficient, which computes the loss
            # every minibatch and multiplies it by zero. Augmentation is the mechanism the
            # literature finds effective, the loss being consistently outperformed by it.
            use_mirror_loss=False,
            data_augmentation_func=kscale_compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
```

The choice of augmentation over the mirror loss is grounded rather than inherited. The catalogue of four mechanisms distinguishes duplication of transitions through the mirror, an auxiliary equivariance penalty, phase replay and a hard equivariant network, and reports that no single method dominates across robots [2]. The primary symmetry paper excludes the loss from its comparison on the strength of the Isaac Lab finding that augmentation converges faster and behaves better [1][4], and it recommends augmentation specifically for intrinsic motion symmetry on a real biped, where actuator and mass asymmetries break the perfect symmetry assumption and a hard constraint becomes brittle under distribution shift. The KScale, like the BRS, carries an asymmetric inertial model, so augmentation is the grounded first choice.

A caution from the same literature bears on the reset events. A strictly symmetric policy cannot leave a symmetric neutral pose, which is the neutral state problem, so training must begin from a noised non neutral posture [2]. The KScale reset events already perturb every joint, so this is satisfied, and it is recorded so that the perturbations are not removed as an economy.

A self test should precede any launch. Mirroring twice must return the original tensors to within floating point tolerance, the permutation must be a genuine involution over the twelve joints, and the augmented batch must be exactly twice the original in its leading dimension.

### 4.8 Register the KScale with the launcher

Add a `kscale` clause and a `kscale-him` clause to both dispatch chains of `/ws/djinn`, at `djinn:118-154` for training and `djinn:192-206` for play, pointing at the identifiers the registry already declares and setting `policy_type` to `HIMPPO` on the HIM clause in the manner of the existing `him` clause.

The change is small and its warrant is not. Both chains assign a default before testing any clause, so an unrecognised argument does not fail, it silently selects the TRON1 SoleFoot. A user asking for the KScale today receives a complete, plausible, converging run of a different robot, with the wrong task identifier appearing only in the run's own dumped parameters where nobody looks until something is already wrong. A defensive improvement worth making in the same pass, though strictly beyond this plan's scope, is a final `else` arm that fails loudly on an unrecognised robot argument, which would have made this defect self reporting.

This section is the one place where the plan reaches outside the repository, `djinn` being workspace level tooling. It is recorded here rather than split into a second document because it is two clauses and because a plan that left the robot unlaunchable would not be complete.

### 4.9 Record the findings

Add `../context/KScale.md`, recording the physical parameterisation established in section 3, in the manner `../context/BRS.md` records the BRS and `../context/quadruped.md` records the quadruped. It should carry the link and joint inventories, the segment lengths, the sole geometry with its measurement method, the effective inertias with the natural frequencies and damping ratios they imply, the self collision audit, and the frame convention finding. Register it in `../context/README.md`, whose document register and summaries section both require an entry.

Register this plan in `README.md`, which presently states that the directory is empty and that no plan is specific to this repository alone. That statement becomes false with this document and the surrounding paragraph needs rewriting rather than merely extending, since its argument is that every plan so far has spanned the workspace.

Three corrections belong in the context record and are listed so that they are not lost. The BRS hip yaw joints are declared `type="fixed"` in the URDF and are absent from the joint list entirely, rather than being present with a zero width limit, which several comments in the tree assert. The figure of roughly 0.13 m by which the KScale foot was said to stand off to the side of the hip is measured from the torso centreline and is correct as such, being the sum of two fixed bracket offsets, but the foot sits directly beneath the hip roll axis to within four microns, so the claim as phrased, that the zero pose is anomalous, is refuted. And `scripts/rsl_rl/play.py` applies the BRS sole table to whatever robot is played, a defect recorded at `../../context/brs_gait.md:698` and deliberately left standing, which now acquires a second affected robot and should be revisited before any KScale dump is read.

## 5. KBot Specific Reward Tuning

> Status, IMPLEMENTED 2026-08-26. Every proposal of this chapter was carried out as specified and the outcome, together with the two divergences it produced, is recorded in section 7.8. The chapter is retained in its proposing voice rather than rewritten into the past tense, so that what was predicted before the run may be read against what the run reports.

The integration of chapter 4 was judged a success on a narrow and deliberate criterion, that the KScale reward set should match the SD_BRS1 reward set term for term, and it met that criterion exactly, twenty seven terms carrying identical functions and identical weights with every differing parameter a derived robot specific quantity. That criterion was the right one for a first pass, because a reward set whose every departure from a working exemplar is deliberate is a reward set whose failures can be attributed, whereas a set retuned in the same pass in which it is ported confounds the two sources of error beyond recovery. The criterion has now served its purpose and it must be retired, because the run of 2026-08-25 produced a walking policy and therefore produced, for the first time, behaviour to judge.

The reason a ported reward set cannot remain a copy indefinitely is that a reward set is not a specification of behaviour in the abstract. It is a specification of behaviour over a particular mechanism, and a term is silent about every degree of freedom the exemplar mechanism does not possess. Where two robots differ in degree, in mass or in leg length or in sole width, a ported term specifies the same intent and the derived parameters carry the difference, which is what chapter 4 achieved. Where they differ in kind, in the presence or absence of a joint, a ported term specifies nothing whatever about the joint that is present in one and absent in the other, and no amount of care in transcribing parameters will supply the missing specification. The KScale differs from the SD_BRS1 in kind at exactly one place, and the defect this section addresses is located there.

The physical differences established across section 2 of this document and sections 2, 7 and 12 of [../context/KScale.md](../context/KScale.md) are gathered below, with the last column stating whether the difference is one of degree, which a derived parameter absorbs, or one of kind, which it cannot.

| Property | SD_BRS1 | KScale | Ratio | Character |
|---|---|---|---|---|
| Total mass | 59.85 kg | 20.2814 kg | 0.339 | Degree |
| Standing height | 1.15 m | 0.77161 m | 0.671 | Degree |
| Leg length, hip to ankle | 0.87 m | 0.505 m | 0.580 | Degree |
| Sole length | 0.2612 m | 0.2100 m | 0.804 | Degree |
| Sole width | 0.194 m | 0.0846 m | 0.436 | Degree |
| Sole depth below the ankle | 0.124 m | 0.0430 m | 0.347 | Degree |
| Ankle effort limit | 131 to 420 Nm | 17 Nm | 0.130 at best | Degree |
| Ankle roll travel | wider | plus and minus 0.26180 rad | | Degree |
| Active revolute joints | 10 | 12 | | Kind |
| Hip yaw joint | `type="fixed"` | `type="revolute"`, plus and minus 1.57080 rad | | Kind |
| Hip yaw axis in the root frame | absent | -0.0998, 0.0000, 0.9950 | | Kind |

Every difference of degree was addressed in chapter 4 and again in the gain derivation recorded in section 7.7, and none of them is at issue here. The single difference of kind is the hip yaw, and it is the widest range of travel on the robot, plus and minus ninety degrees, exceeding the ankle roll's travel by a factor of six. The SD_BRS1 declares both hip yaw joints with `type="fixed"` at `environments/environments/assets/urdf/solefoot/SD_BRS1/SD_BRS.urdf:31` and `:36`, so they never enter `robot.joint_names` at all and no reward term of that robot's set was ever required to say anything about them. The KScale declares both as revolute at `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf:614` and `:1094`. The KScale therefore inherits a reward specification with a blind spot whose width is exactly the two joints the exemplar lacks, and it inherits it silently, because a term set that omits a joint looks no different from a term set that regulates it well.

Section 2.5 of this document anticipated the consequence in a single clause, noting that the KScale trains one more degree of freedom per leg than the SD_BRS1 and would need its own `joint_deviation_l1` treatment if the extra axis wandered off in training. It has wandered off. What follows establishes by what margin, why the one term that nominally covers the axis does not restrain it, and what should be added.

### 5.1 Feet Heading Direction

The behaviour is visible in the training video at `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-25_10-44-21/videos/train/rl-video-step-400000.mp4`, which records the policy at 400000 environment steps, near the end of a run of 17217 iterations. The robot walks, tracks the commanded forward velocity, and survives 1425 steps of the 2000 step episode, so the gross failure recorded in section 7.7 is repaired and the policy has learned a gait. Within that gait the feet do not point where the robot is going. Across the sampled stride the swing foot is placed with its long axis rotated markedly out of the plane of travel, the two feet frequently point in appreciably different directions within the same double support interval, and the legs cross at the shank so that the stance leg is twisted beneath a torso facing forward. The commanded heading over the sampled interval is constant and forward, indicated by the command arrow the debug visualiser draws, so none of the rotation is attributable to a turn.

The logs establish the magnitude, and they establish it more precisely than the video can, because the one term that reads the axis reports a number. `pen_hip_deviation` at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:791` is a `joint_deviation_l1` over four joints, being both hip rolls and both hip yaws, at weight minus 0.1. Isaac Lab logs an episode reward as the episode sum divided by the nominal episode length in seconds, so dividing the logged value by the weight and by the realised fraction of the nominal episode recovers the time averaged sum of absolute deviations across the four joints.

| Iteration | `pen_hip_deviation` | Mean episode length | Recovered L1 sum, four joints | Mean per joint |
|---|---|---|---|---|
| 2000 | -0.0385 | 1290.65 | 0.596 rad | 0.149 rad, 8.5 degrees |
| 8000 | -0.2739 | 1888.92 | 2.900 rad | 0.725 rad, 41.5 degrees |
| 17150 | -0.1342 | 1425.23 | 1.884 rad | 0.471 rad, 27.0 degrees |

The comparison that settles whether 0.471 rad is large is the SD_BRS1 itself, whose own `pen_hip_deviation` at `cfg/SF/brs_base_env_cfg.py:733` carries the identical function and the identical weight of minus 0.1 over its two hip roll joints, those being the only off axis leg joints it has. The mature SD_BRS1 run at `IsaacLab/logs/rsl_rl/sd_brs1_flat/2026-08-17_04-26-44`, at iteration 29999 and a mean episode length of 1950.7, logs minus 0.0217, which recovers to 0.2225 rad across two joints and therefore 0.111 rad, being 6.4 degrees, per joint. The KScale runs its off axis hip joints at 4.2 times the per joint excursion of the robot it was copied from, and it does so while surviving a shorter fraction of its episode.

That figure is an average across the roll and the yaw axes and the logged term cannot separate them, so the yaw excursion alone is bracketed rather than measured. If the KScale hip roll behaves as the SD_BRS1 hip roll does, at 0.111 rad, the two hip yaws carry the balance and average 0.83 rad, being 47.6 degrees. If instead the roll and the yaw share the excursion equally, each averages 0.471 rad, being 27.0 degrees. The lower end of that bracket already exceeds the SD_BRS1 figure by a factor of four, so the conclusion does not turn on which end is nearer the truth.

The mechanism by which the excursion is bought is visible in the correlation structure of the run. Across the 17009 logged iterations from iteration 200 onward, the magnitude of `pen_hip_deviation` correlates with `rew_ang_vel_z` at 0.966 and with `rew_lin_vel_xy` at 0.852. Both are high, because in a converging run most quantities improve together and a raw correlation across training is therefore a weak instrument. The partial correlations discriminate. Controlling for linear velocity tracking, the correlation of the hip deviation with yaw tracking is 0.927, whereas controlling for yaw tracking, its correlation with linear tracking falls to 0.648. The off axis hip excursion tracks the yaw tracking reward specifically and not merely the general improvement of the run, which is what one expects if the policy is generating yaw by rotating its legs about the vertical rather than by placing its feet.

That the policy should discover this is not surprising, and section 5.1.1 records the physics. What matters here is that nothing in the reward set prices it. The full episode reward breakdown at iteration 17217 places `pen_hip_deviation` at minus 0.1464 per second against a total of plus 26.0431 and a linear tracking reward of plus 27.0038, so the entire price charged for every degree of off axis hip excursion on both legs together is 0.54 per cent of the tracking reward it purchases. Seventeen of the twenty seven terms are larger in magnitude. A policy trading 47 degrees of leg twist for a measurable improvement in yaw tracking is not defeating the reward set, it is obeying it.

The three terms a reader might expect to catch the behaviour do not. `pen_feet_distance` at `cfg/SF/kscale_base_env_cfg.py:858` hinges on the separation of the two foot frames and is blind to their orientation, a splayed pair of feet at the correct separation costing exactly nothing. The two `rew_keep_ankle_*_zero_in_air` terms regulate the ankle pitch and the ankle roll joint coordinates and say nothing about the vertical axis, which those joints do not turn about. And `pen_joint_pos_limits` reads minus 0.4084, which is real but is a limit penalty rather than a posture penalty and is in any case diffuse across twelve joints. There is no term anywhere in the KScale set that reads the direction a foot points.

One kinematic result determines how the missing term should be built and is established here because both of the following sections depend on it. Composing the origin rotations of the KScale URDF through to the foot link at the nominal pose, and taking the toe direction as the foot link's negative z axis, which section 12 of [../context/KScale.md](../context/KScale.md) establishes as the fore and aft axis of that permuted frame, the heading of the toe in the root frame is 0.995 times the hip yaw joint coordinate across the whole of that joint's travel. The ankle roll contributes nothing, its axis lying along the toe direction itself and moving the heading by 0.0001 degrees at its limit, and the three sagittal joints cannot yaw the foot at all. The direction a KScale foot points is the hip yaw angle and nothing else.

The coefficient of 0.995, however, is a property of the POSE and not of the robot, and the distinction is the more important half of the result. Section 2.3 of that document records, in its correction of 2026-08-26, that the 0.0998 tilt of the hip yaw axis out of vertical is not a hardware cant but an artefact of the nominal hip pitch flexion of 0.1 rad propagating down a chain in which the hip yaw is rigidly downstream of the hip pitch. The heading gain follows the same pose. It is 1.000 at the URDF's own zero pose, 0.995 at the nominal stance, and 1.073 and 1.524 at hip pitch flexions of 0.5 and 1.0 rad, so it varies by more than half across the travel of a joint that swings through most of that range on every step.

| Hip pitch | Heading gain per radian of hip yaw |
|---|---|
| 0.0 rad, the URDF zero pose | 1.000 |
| -0.1 rad, the nominal stance | 0.995 |
| -0.5 rad | 1.073 |
| -1.0 rad | 1.524 |

The consequence bears directly on the choice of instrument. A given hip yaw excursion turns the foot furthest at the extremes of hip pitch flexion, which is to say during swing, which is exactly when the foot's direction is being decided, so a penalty written against the joint coordinate systematically under prices the heading error at the moment it matters most. A penalty written against the toe direction prices what the foot does regardless of the pose the leg is in. This is the strongest of the three arguments section 5.1.2 gives for the task space form and it was not available when that section was first drafted, the gain having then been believed constant.

That result cuts both ways and both halves are load bearing. It means a foot heading reward and a hip yaw deviation penalty are, on this robot, very nearly the same instrument, so the case for the former over simply raising the weight of the latter must be made rather than assumed, and section 5.1.2 makes it. It also means the bracket of 0.471 to 0.83 rad computed above transfers to the foot heading error to within the pose dependence just described, which is the quantity the new term will read, and therefore that the term's magnitude at present behaviour can be predicted before it is ever run, the gain's departure from unity making that prediction a lower bound rather than an estimate wherever the leg is deeply flexed.

#### 5.1.1 Literature Survey and Related Work

The regulation of foot orientation about the vertical is a recent and thinly reported concern in the learned locomotion literature, for a reason that section 5.1 has already supplied. The exemplars from which this repository's reward vocabulary descends are quadrupeds and point footed or fixed hip yaw bipeds, and a machine that cannot rotate its foot about the vertical needs no reward that says it should not. The concern appears in the literature at precisely the point at which humanoids with a full six degree of freedom leg become the common subject, and the treatments divide into three families.

The first family regulates the joint. The Isaac Lab reference configurations penalise the summed absolute deviation of the off axis hip joints from their default posture, applying `joint_deviation_l1` to the hip yaw and hip roll joints together at weight minus 0.1 for the Unitree G1 at `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py:57-61`, and deliberately not to the hip pitch or the knee, which must swing. The comment there names the principle, that the term covers the joints which are not essential for locomotion. The Digit configuration, which is the closest true biped analogue in that tree, separates the two axes and prices them differently, carrying `joint_deviation_hip_roll` at minus 0.1 and `joint_deviation_hip_yaw` at minus 0.2 at `.../config/digit/rough_env_cfg.py:97-105`, so the vertical axis is charged at twice the rate of the fore and aft axis. This is the family the KScale already belongs to, and the comparison is unflattering, the KScale lumping both axes into a single term at the roll's rate rather than the yaw's, which is to say at half the price the nearest published analogue assigns.

The second family regulates the foot in task space, and Booster Gym is its most complete published reference [12]. Its Table II carries a feet yaw term of the form given below at weight minus 1.0, beside a feet roll term at minus 0.1, a feet slip term at minus 0.1 and a feet distance term at minus 1.0, against velocity tracking weights of 1.0 for each linear axis and 0.5 for the yaw rate.

```
Feet yaw    ||psi_feet - psi_base||^2    -1.0
Feet roll   ||phi_feet||^2               -0.1
```

The single tabulated row is not what the released work computes, and the difference is the most useful thing this survey recovers. The implementation at `envs/t1.py` carries two distinct methods. `_reward_feet_yaw_diff` squares the wrapped difference between the two feet's yaw angles and is entirely independent of the base. `_reward_feet_yaw_mean` squares the wrapped difference between the base yaw and the MEAN of the two feet's yaws, with a branch cut repair of pi added to that mean so that averaging two angles either side of the wrap does not produce a value lying between them. The configuration at `envs/T1.yaml` prices the two SEPARATELY AND EQUALLY, carrying `feet_yaw_diff` at minus 1.0 and `feet_yaw_mean` at minus 1.0, so the tabulated single weight of minus 1.0 is the weight of each of a pair rather than of one term.

The decomposition is the substance rather than a presentational detail, and the reason is that the two modes are orthogonal coordinates under the mean form and are not orthogonal under the summed form. Writing the two per foot errors against the base as e_1 and e_2, a pure differential fault, being one foot toed out by e and the other toed in by e, gives a mean form common mode of exactly zero and a differential of 4e squared, so the fault is attributed entirely to the mode that names it. The same fault under the summed form of e_1 squared plus e_2 squared gives 2e squared, so a splay registers on the common mode as well and the two terms cannot be varied independently. A pure common fault, both feet rotated together by e, gives e squared under the mean form and 2e squared under the summed. The consequence for this plan is that a design wishing to give each mode its own physical meaning, and to ablate one against the other, must adopt the mean form for the common mode, and section 5.1.2 does so.

The two modes describe physically distinct faults and both are present in the KScale run. The common mode is a stance rotated bodily away from the direction of travel, which a turning robot exhibits legitimately and which a walking one should not. The differential mode is one foot toed in against the other toed out, the pigeon toed and duck footed postures, which nothing legitimises at any commanded velocity, since a coordinated turn moves both feet the same way and therefore leaves the differential at zero by construction. The behaviour recorded in section 5.1 is predominantly differential, the feet pointing towards one another and away from one another within the same double support interval, which is the mode Booster Gym prices and which the summed form would have blurred into the common mode.

The existing `feet_yaw_alignment` in this repository at `environments/environments/tasks/locomotion/mdp/rewards.py:579` implements the summed form and therefore carries the two modes at a fixed and unadjustable ratio, pricing a splay at three times a common rotation of the same per foot magnitude rather than at the four to one the mean form gives. Its docstring states that Booster Gym carries the term at twice the linear tracking weight. Against Table II and the released configuration, each of the two weights is minus 1.0, equal to a single linear tracking component's 1.0 and twice the yaw tracking weight of 0.5, so the docstring names the wrong reference quantity. The correction is recorded here rather than in the function, which has no caller and is examined again in section 5.1.2.

The third family regulates the foot's orientation as part of a broader posture objective without isolating the vertical axis. Van Marum and colleagues carry a feet orientation term at weight 0.05 in a table whose largest entry is a sparse touchdown triggered air time reward at 1.0 [6], so foot orientation is present but is among the least consequential terms in that design, and their reported failure mode under velocity tracking alone is two footed hopping rather than foot misalignment. Humanoid-Gym shapes the swing trajectory and the contact schedule without a foot yaw term at all [9]. The absence is informative. Where a design obtains its turning from foot placement and its swing from a clock, the foot yaw axis is not the cheapest route to yaw and the policy does not exploit it, so no term is needed. The KScale's situation differs, and section 5.1 has measured the difference.

Two properties of the Booster Gym treatment bear directly on whether it may be ported, and both were checked against that work's own robot description rather than assumed. Neither term carries any contact gate whatever, the two methods being evaluated at every step irrespective of whether a foot is planted or swinging, and neither carries a dead band, the squared error being charged from the first milliradian. The plan proposed in section 5.1.2 follows the first of these and departs from the second, and section 5.1.2 quantifies both decisions rather than resting them on the precedent.

The physics of why the axis is exploited is established from the disturbance side rather than the control side. A swinging leg generates a yaw moment about the vertical which the stance foot must absorb through friction, and the effect is large enough that mechanisms have been designed for the express purpose of cancelling it [13]. Popovic, Hofmann and Herr established that whole body angular momentum is regulated to a small range throughout human walking, with segment to segment cancellation accounting for some eighty per cent of the horizontal component [14], which is the biomechanical statement that a walking machine both generates and must cancel yaw momentum continuously. A machine whose hip yaw joints are fixed, as the SD_BRS1's are, can obtain a commanded yaw only from the ground, as a friction moment beneath the sole or as the reaction to a change in whole body angular momentum, and this repository's own survey records that constraint. A machine whose hip yaw joints are free has a third and much cheaper route, which is to counter rotate the legs about the vertical and let the resulting reaction turn the torso. That route requires no friction margin, no change in foot placement and no reorganisation of the gait, so a policy rewarded for yaw rate will find it early, which is what the partial correlation of 0.927 in section 5.1 reports. The KScale's narrow sole aggravates the matter from the other side, its 0.0846 m width against the SD_BRS1's 0.194 m giving it a proportionally smaller friction moment about the vertical for the same coefficient and load, so the ground route is dearer on this robot exactly where the joint route is cheaper.

The counter rotation argument also explains why the observed fault is differential rather than common. A torque applied between the pelvis and one thigh about the vertical acts equally and oppositely upon the other, so a policy generating yaw internally does so by rotating the two legs in OPPOSITE senses about their own hip yaw axes, which is the differential mode exactly. A common mode rotation of both legs the same way carries no reaction against the torso at all and therefore buys no yaw, so a policy exploiting the joint route has no reason to produce one. The prediction is that the KScale fault should be predominantly differential, and the video of section 5.1 bears it out.

The biomechanical literature supplies the tolerance rather than the term. The foot progression angle, defined as the angle between the long axis of the foot and the line of progression, is the direct human analogue of the quantity to be regulated. Cibulka and colleagues measured it in sixty healthy adults and report a mean of 3.3 degrees of toe out with a standard deviation of 5.6 degrees and a range from 9.7 degrees of toe in to 14.3 degrees of toe out [15]. Human walking therefore does not hold the foot exactly along the line of progression, it holds it within roughly one tenth of a radian of it, and a reward that demands exact alignment demands something more than natural. One standard deviation of that distribution, 5.6 degrees or 0.098 rad, is the natural scale for a dead band on the common mode, and it is an order of magnitude below the 0.471 to 0.83 rad this policy exhibits, so the tolerance and the defect are not in danger of being confused. The same data bear on the differential mode differently. The difference of two independent draws from that distribution has a standard deviation of 5.6 times the square root of two, being 7.9 degrees or 0.138 rad, so a common tolerance applied to both modes holds the differential to a stricter standard in per foot terms than it holds the common mode. That asymmetry is deliberate and is defended in section 5.1.2, the mean toe out of 3.3 degrees being a COMMON mode which human walking does exhibit, against which no comparable systematic differential exists.

The survey leaves one question open and it must be settled by argument rather than by citation, because no surveyed source addresses it. Every published term references the foot to the robot's own base and none references it to the commanded heading, whereas the request that occasions this section is framed in terms of the direction of movement. The two coincide when the base tracks its heading command, and this configuration commands heading directly, `CommandsCfg.base_velocity` at `cfg/SF/kscale_base_env_cfg.py:156-170` setting `heading_command=True` with `rel_heading_envs=1.0` and a heading control stiffness of 0.5, so the yaw rate the robot is asked for is itself computed from the heading error. Referencing the feet to the base is therefore the correct choice on three grounds. It measures the quantity actually at fault, which is the foot against the body and not the body against the world. It avoids charging the same error twice, the base's own heading error being already priced by `rew_ang_vel_z` at weight 15. And it remains well defined when the commanded velocity is zero, where a heading referenced term would be regulating the feet against a direction of movement that does not exist. The base referenced form satisfies the requirement that the feet face the direction of movement precisely because the base is what faces the direction of movement, and it satisfies the requirement that the feet twist only to turn without any special case, since during a turn the base yaws and a foot that follows it incurs no error.

#### 5.1.2 Implementation Plan

##### The frame and sign convention audit against Booster Gym

A reward ported from another robot's codebase is safe only where the two robots express the regulated quantity in the same way, and this document has twice recorded the cost of assuming that they do. Two conventions govern a foot yaw term, being the sense in which each hip yaw joint rotates and the orientation of the foot link frame from which a heading is read, and both were measured on both robots rather than inferred from the source.

The KScale hip yaw axes were composed through the URDF origin rotation chain to the root frame at the nominal pose. The right axis at `kscale.urdf:614` reads minus 0.0998, plus 0.0000, plus 0.9950 and the left at `:1094` reads minus 0.0998, minus 0.0000, plus 0.9950, so the two are CO DIRECTED, both pointing upward and both carrying the same 0.0998 rad cant. The consequence is verified end to end rather than left as an axis comparison. Driving each hip yaw alone through its travel and reading the toe heading as the negative z axis of the foot link rotated into the root frame gives a gain of 0.995 on the right and 0.995 on the left, of the SAME sign, across 0.1 to 0.8 rad. A positive command at either hip turns that foot the same way. The apparent contradiction with the two joints' opposite origin rotations, minus 1.5708 about x on the right and plus 1.5708 on the left, is resolved by the parent links, which are themselves oppositely oriented, so the two cancel.

The Booster T1 was checked the same way from `resources/T1/T1_locomotion.urdf`. Both `Left_Hip_Yaw` and `Right_Hip_Yaw` declare `axis xyz="0 0 1"` with `origin rpy="0 0 0"` and identical limits of minus 1.0 to plus 1.0, and every joint origin in the leg chain from the hip roll down to the foot link carries an identity rotation. The two robots therefore share the convention. Both hip yaw pairs are co directed, a positive joint command turns the corresponding foot the same way on either leg, and no sign correction of any kind is required to carry the Booster Gym differential term onto the KScale.

The differential term operates on the two feet's WORLD headings rather than on their joint coordinates, so it is invariant to the joint sign convention by construction and would remain correct even had the two axes been anti directed. The audit matters not because the ported term is at risk but because the interpretation is. Under co directed axes the differential heading error equals 0.995 times the difference of the two hip yaw coordinates, so a reader may read the logged term back into joint space directly, whereas under anti directed axes the same term would correspond to their SUM and any such reading would invert. The audit therefore licenses the diagnostic use of the term as well as the term itself.

Three differences between the two robots are recorded in the same pass, none of which obstructs the port. The T1 hip yaw axes are exactly vertical in the robot's own zero pose and so are the KScale's, the 0.0998 rad tilt reported at the KScale's nominal stance being the hip pitch artefact of section 5.1 rather than a mounting cant, so the two robots agree on this as well and the KScale's 0.995 gain is a statement about its standing pose alone. The T1 hip yaw travel is plus and minus 1.0 rad against the KScale's plus and minus 1.5708, so the KScale has 57 per cent more room in which to misbehave. And the T1 nominal pose of hip pitch minus 0.2, knee 0.4 and ankle pitch minus 0.25, read from `envs/T1.yaml`, sits remarkably close to the KScale's minus 0.1, 0.4 and minus 0.3 established in section 13 of [../context/KScale.md](../context/KScale.md), which is an independent corroboration of that pose from a robot of comparable scale.

The foot link frame is where the two robots diverge, and it is the reason the ported term must be reformulated rather than transcribed. The T1 leg chain carries identity rotations throughout, its ankle pitch declaring `axis xyz="0 1 0"` and its ankle roll `axis xyz="1 0 0"`, so the foot link frame is canonical with x forward, y to the left and z upward. The yaw component of an Euler decomposition of that link's world quaternion therefore IS the toe heading, and Booster Gym's implementation is correct for its own robot. The KScale foot link is a full axis permutation away, its x being the sole's width, its y the vertical with the positive sense pointing downward and its z the fore and aft length, as section 12 of [../context/KScale.md](../context/KScale.md) establishes from the collision mesh.

The magnitude of the resulting discrepancy was measured rather than argued. Sweeping the right foot through 405 configurations spanning the full ankle pitch travel, the full ankle roll travel and the full hip yaw travel, and comparing the Euler yaw against the true toe heading, the two differ by between 70.16 and 109.84 degrees and at no configuration in the sweep do they agree. At the nominal standing pose the Euler yaw reads exactly 90 degrees while the toe points exactly forward.

The consequence is not that a transcribed term would be noisy. It is that it would be inverted. Wired as the existing function stands, at any negative weight, it would charge a squared error of 2.467 rad squared per foot at the nominal pose in which the feet are correctly aligned, and its gradient would push each foot towards a toe heading near minus 90 degrees, which is to say it would actively produce the pathology it was added to remove. The residual variation of some twenty degrees across the sweep is contributed by the ankle pitch and roll, so the term would additionally leak into the two joints that `rew_keep_ankle_pitch_zero_in_air` and `rew_keep_ankle_roll_zero_in_air` already regulate, and would oppose them. The Euler decomposition is well conditioned throughout, the smallest cosine of the extracted pitch across the sweep being 0.9354, so the defect is a frame convention error and not a numerical one and it would not have announced itself as instability.

This finding is offered as the principal argument for the review this section requests. The term would have trained. It would have produced a converged policy, a plausible learning curve and a robot with its feet turned outward, and the only evidence of the cause would have been a reward term that failed to fall.

##### The proposed change to the shared module

Rule 4 of `/ws/CLAUDE.md` prefers an optional argument whose default reproduces the existing behaviour exactly over a second version of a function, and the situation admits that treatment cleanly. The proposal extends `feet_yaw_alignment` with six optional arguments, every default reproducing the current behaviour bit for bit, and adds no new function. Backwards compatibility is preserved by construction and is additionally vacuous here, the function having no caller, and both facts should be stated in the review because the second does not excuse the first. The SD_BRS1, the three TRON1 variants and the quadruped are untouched.

```python
def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forward_axis: tuple[float, float, float] | None = None,
    common_mode: str = "sum",
    tolerance: float = 0.0,
    differential_scale: float = 0.0,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
    history_index: int = 0,
    airborne_only: bool = False,
) -> torch.Tensor:
    """Penalise the yaw of each foot relative to the base.

    The term follows the feet yaw rewards of Booster Gym (arXiv:2506.15132). That work's
    Table II tabulates a single squared norm at minus 1.0, but its released implementation
    and configuration carry TWO terms priced separately and equally, `feet_yaw_diff` at
    minus 1.0 over the two feet against each other and `feet_yaw_mean` at minus 1.0 over the
    mean foot yaw against the base. Both are reproduced here.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies. Exactly two feet are
            required whenever differential_scale is non zero or common_mode is "mean".
        forward_axis: Which axis of the FOOT LINK frame points at the toe. When None, the
            default, the heading is taken as the yaw component of an Euler decomposition of
            the foot's world quaternion, which is the original behaviour and is preserved
            exactly. That path is correct only where the foot link's forward axis is its own
            x and its vertical is its own z, which is the convention of the SD_BRS1 and of
            Booster Gym's own T1, and is NOT the KScale convention, whose foot link carries
            its width on x, its vertical on y pointing downward and its fore and aft length
            on z. Pass (0.0, 0.0, -1.0) for the KScale. See context/KScale.md section 12 and
            this plan's section 5.1.2, which measures the Euler path's error on that robot at
            70 to 110 degrees, an error that would INVERT the term rather than blur it.
        common_mode: How the per foot errors against the base are combined. "sum", the
            default, sums their squares, which is the original behaviour and is preserved
            exactly. "mean" squares the error of their circular MEAN against the base, which
            is Booster Gym's `_reward_feet_yaw_mean` and is the form that makes the common and
            differential modes ORTHOGONAL. Under "sum" a pure splay of plus and minus e
            registers 2 e squared on the common mode, so the two modes cannot be varied or
            ablated independently. Under "mean" it registers exactly zero. Prefer "mean"
            wherever differential_scale is non zero.
        tolerance: Half width of a dead band, in radians, applied to each mode's error before
            it is squared. Defaults to 0.0, which is Booster Gym's own behaviour and is
            preserved as the default. The human foot progression angle has a standard
            deviation of 5.6 degrees about a mean toe out of 3.3 degrees, so a tolerance near
            0.10 rad demands no more than natural walking does.
        differential_scale: Weight of the differential mode, being the squared wrapped
            difference between the two feet's headings, RELATIVE to the common mode. Defaults
            to 0.0, preserving the original behaviour. Pass 1.0 for Booster Gym's own equal
            pricing of the two modes.
        sensor_cfg: Contact sensor resolving the same feet, in the same order as asset_cfg.
            Required only when airborne_only is True. Defaults to None.
        force_threshold: Contact force, in newtons, above which a foot counts as planted.
        history_index: Which slot of the contact sensor's rolling history supplies the
            contact test. The sensor writes the NEWEST sample to index 0. Defaults to 0.
        airborne_only: When True the COMMON mode is evaluated only over airborne feet, the
            mean under common_mode "mean" being taken over those feet alone and the term
            being zero when none is airborne. The differential mode is never gated, both feet
            being required for it to be defined. Defaults to False, which is Booster Gym's
            own behaviour and is preserved as the default.

    Note:
            The gate exists to support the ablation of section 5.1.2 and is NOT recommended
            as a shipping default. Under common_mode "mean" a well executed turn moves both
            feet together and the mean tracks the base, so the common mode's time averaged
            value during a steady turn is zero to four decimal places at every commanded yaw
            rate in this configuration's curriculum, and the gate has nothing to protect. It
            was necessary only under the summed form, whose mode mixing makes a turning
            stance foot register on the common mode.

    Returns:
        The computed penalty tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w)[2].unsqueeze(1)

    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]           # (N, F, 4)
    if forward_axis is None:
        foot_yaw = math_utils.euler_xyz_from_quat(foot_quat.reshape(-1, 4))[2].view(
            foot_quat.shape[0], foot_quat.shape[1]
        )
    else:
        axis = torch.tensor(forward_axis, device=foot_quat.device, dtype=foot_quat.dtype)
        toe_w = math_utils.quat_apply(foot_quat, axis.expand_as(foot_quat[..., :3]))
        foot_yaw = torch.atan2(toe_w[..., 1], toe_w[..., 0])            # (N, F)

    def _band(err: torch.Tensor) -> torch.Tensor:
        if tolerance <= 0.0:
            return err
        return torch.sign(err) * torch.clamp(torch.abs(err) - tolerance, min=0.0)

    if airborne_only:
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history
        airborne = ~(
            torch.norm(forces[:, history_index, sensor_cfg.body_ids], dim=-1) > force_threshold
        )
    else:
        airborne = torch.ones_like(foot_yaw, dtype=torch.bool)

    if common_mode == "mean":
        # circular mean of the selected feet, taken on the unit circle so that no branch cut
        # repair is needed. Booster Gym adds pi to a linear mean where the two feet straddle
        # the wrap; resolving the mean as an angle is equivalent and has no special case.
        mask = airborne.to(foot_yaw.dtype)
        sin_m = torch.sum(torch.sin(foot_yaw) * mask, dim=1)
        cos_m = torch.sum(torch.cos(foot_yaw) * mask, dim=1)
        any_sel = torch.sum(mask, dim=1) > 0
        mean_yaw = torch.atan2(sin_m, cos_m)
        common = torch.where(
            any_sel, torch.square(_band(math_utils.wrap_to_pi(mean_yaw - base_yaw.squeeze(1)))),
            torch.zeros_like(sin_m),
        )
    elif common_mode == "sum":
        err = _band(math_utils.wrap_to_pi(foot_yaw - base_yaw))
        common = torch.sum(torch.square(err) * airborne.to(err.dtype), dim=1)
    else:
        raise ValueError(f"common_mode must be 'sum' or 'mean', got {common_mode!r}")

    if differential_scale <= 0.0:
        return common
    differential = _band(math_utils.wrap_to_pi(foot_yaw[:, 1] - foot_yaw[:, 0]))
    return common + differential_scale * torch.square(differential)
```

Four points of construction deserve comment at review. The heading is taken by rotating a named link axis into the world and calling `atan2` on its horizontal projection rather than by any Euler decomposition, which removes the frame convention dependence entirely rather than parameterising around it. The circular mean is resolved on the unit circle rather than by Booster Gym's additive branch cut repair, which is equivalent, carries no special case and generalises beyond two feet. The dead band is applied before the square and preserves the sign, so the term remains continuous and its gradient is zero inside the band rather than discontinuous at its edge. And the differential mode is computed from the raw headings of the two feet rather than from their errors against the base, matching the released Booster Gym form, since the base cancels from the difference exactly.

The function above was executed rather than reviewed by eye, the body being extracted verbatim from this document and run against stub objects supplying the two quaternion fields it reads, so that the claims made for it are measurements. Seven identities were checked and all seven hold. A splay of plus and minus 0.3 rad returns a common mode of exactly 0.0 under `common_mode="mean"` and 0.18 under `"sum"`, which is the orthogonality claim and its failure under the summed form. A common rotation of 0.3 rad returns 0.09 with the differential at zero. The splay to common ratio is 4.000 under the mean form and 3.000 under the summed, as section 5.1.1 derives. With every new argument at its default the function returns 0.1299999952 against the original form's 0.1299999952 on the same input, which is the backwards compatibility claim to ten decimal places. A dead band of 0.10 on a splay of plus and minus 0.3 returns 0.25, being 0.5 squared. And with the foot link placed at its measured nominal rotation and yawed through a range of angles, the `forward_axis` path recovers the applied yaw to four decimal places while the Euler path reports that value plus 1.5708 at every angle, which is the ninety degree offset quantified at the nominal pose rather than swept.

##### The proposed configuration terms

Three terms are proposed for `RewardsCfg` in `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py`. The first is the new foot heading penalty. The second and third replace the existing `pen_hip_deviation` at line 791 with a pair that carries the identical total price while logging the two axes separately.

```python
    # The KScale hip yaw is a live degree of freedom where the SD_BRS1's is type="fixed", so
    # the ported reward set says nothing whatever about the direction a foot points, and the
    # run at logs/rsl_rl/kscale_flat/2026-08-25_10-44-21 shows the policy buying yaw tracking
    # with leg twist. See plans/kscale_integration.md section 5.1 for the measurement.
    #
    # forward_axis is (0, 0, -1) because this foot link carries its fore and aft length on z
    # with the toe at negative z. The default None path takes the yaw of an Euler
    # decomposition, which is the convention of the SD_BRS1 and of Booster Gym's own T1, and
    # on this robot reports a heading 70 to 110 degrees away from the true one, reading 90
    # degrees at a nominal pose whose feet point exactly forward. It would INVERT the term.
    # Do not use the default here.
    #
    # common_mode "mean" is Booster Gym's `_reward_feet_yaw_mean` and is required rather than
    # preferred, being the only form under which the common and differential modes are
    # orthogonal. Under the "sum" default a splay of plus and minus e registers 2 e squared on
    # the common mode, which would make the two modes inseparable in the logs and defeat the
    # ablation this term is set up for.
    #
    # differential_scale 1.0 prices the two modes equally, which is what Booster Gym's own
    # envs/T1.yaml does, carrying feet_yaw_diff at -1.0 and feet_yaw_mean at -1.0. The
    # differential is the mode the observed fault occupies: a counter rotation of the two legs
    # is the only hip yaw motion that generates a yaw reaction against the torso, so a policy
    # buying yaw from the joint rather than from the ground produces a splay and not a common
    # rotation. A coordinated turn leaves the differential at zero by construction.
    #
    # tolerance 0.10 rad is this implementation's one departure from Booster Gym, which
    # carries no dead band. It is one standard deviation of the human foot progression angle,
    # 5.6 degrees about a 3.3 degree mean toe out, so the term demands no more than natural
    # walking does, and it is an order of magnitude below the 0.47 to 0.83 rad now observed.
    # Set it to 0.0 for exact Booster Gym parity.
    #
    # airborne_only is False, matching Booster Gym, and because the observed fault is present
    # in stance as well as in swing. Under the mean form the gate has nothing to protect: the
    # common mode's time averaged value during a steady turn is 0.0000 at every yaw rate in
    # this curriculum, because a well executed turn moves both feet together. The argument is
    # retained in the function so that the gate may be ablated against this baseline.
    pen_feet_heading = RewTerm(
        func=mdp.feet_yaw_alignment,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "forward_axis": (0.0, 0.0, -1.0),
            "common_mode": "mean",
            "tolerance": 0.10,
            "differential_scale": 1.0,
            "airborne_only": False,
            "history_index": 0,
            "force_threshold": 1.0,
        },
    )

    # pen_hip_deviation is SPLIT into its two axes at the SAME weight it carried as one term.
    # joint_deviation_l1 is a plain torch.sum of absolute deviations over the resolved joints
    # (IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:180-186), so w * sum(roll + yaw)
    # equals w * sum(roll) + w * sum(yaw) exactly and the total reward is bit for bit
    # unchanged. Nothing about the policy's incentives moves in this pass.
    #
    # The purpose is diagnostic and preparatory. Section 5.1 could bracket the hip yaw
    # excursion only between 0.471 and 0.83 rad because the single lumped term cannot separate
    # the axes, and that bracket is the widest uncertainty in the whole of section 5. Split,
    # the two axes report separately from the first iteration, which both measures the fault
    # directly and supplies the independent corroboration for pen_feet_heading, no term
    # reading the hip yaw coordinate itself.
    #
    # It also stages the first escalation. IsaacLab's Digit, the nearest published biped
    # analogue, prices the yaw axis at twice the roll's, -0.2 against -0.1
    # (.../config/digit/rough_env_cfg.py:97-105), where this robot has been carrying both at
    # the roll's rate. Raising pen_hip_yaw_deviation to -0.2 then becomes a one line change
    # against a logged baseline rather than a change confounded with a restructuring.
    pen_hip_roll_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_roll_03"]
            )
        },
    )
    pen_hip_yaw_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_yaw_03"]
            )
        },
    )
```

##### Design rationale

The case for a task space term over simply raising the weight of the hip yaw deviation penalty must be made rather than assumed, because section 5.1 establishes that on this robot the foot heading is the hip yaw angle to within half a per cent and the two instruments therefore read almost the same quantity. Three considerations decide it.

The first is that the joint space term cannot express the distinction that matters. The fault is a DIFFERENTIAL rotation of the two legs and the legitimate behaviour is a COMMON one, and `joint_deviation_l1` charges the sum of absolute deviations, which is very nearly blind to the difference between them, a splay of plus and minus e and a common rotation of e both summing to 2e. A policy told only to keep its hip yaws near zero is told nothing about which of the two ways of departing from zero is the objectionable one. The mean and differential decomposition is the whole content of the proposal, and no weighting of a summed absolute deviation can reproduce it.

The second is that the quadratic form discriminates the tail where the L1 form does not. `joint_deviation_l1` charges the same price for the first degree as for the fortieth, so a policy has no more reason to reduce a large excursion than a small one and the term's minimum is a uniform light bias towards the default rather than a strong objection to the extreme. Squaring, with a dead band that forgives the natural range, makes the first degree free and the fortieth expensive, which is the shape the evidence calls for, the SD_BRS1 running happily at 6.4 degrees while the KScale runs at 27 to 48.

The third is that the term is stated in the quantity that is actually wrong. The fault is that the feet do not point where the robot is going, and the fact that this reduces to the hip yaw angle is a property of the KScale's current kinematics rather than a definition. A term written against the toe direction remains correct if the leg is re-linked, if a joint is added, or if the co-optimisation of `../../CO_OPTIMISATION.md` alters a link length, whereas a hip yaw penalty tuned against the nominal pose's 0.995 gain would silently mis-price a redesigned leg, and mis-prices the present one already, the gain reaching 1.524 at the hip pitch flexions the swing phase passes through. Given that this workspace exists to optimise designs, a reward stated in task space is the more durable artefact.

The two instruments are therefore complementary rather than alternative, and the plan retains both. The joint space pair is kept at its existing total weight precisely so that it remains a MEASUREMENT rather than becoming a second intervention, which is what allows the run to attribute any improvement to the task space term alone.

The decision to leave the term ungated is the one place where the plan follows Booster Gym against its own earlier draft, and it is worth recording why the earlier draft was wrong. The gate was introduced to protect a turning robot, on the reasoning that a planted foot is held by friction and cannot re-align while the base yaws over it, and that at the curriculum's terminal 0.9 rad/s a stance foot accrues up to 0.558 rad of base relative error for which the policy is not responsible. That reasoning is sound under the SUMMED common mode and is void under the mean form. Under the mean form a well executed turn moves both feet the same way and their mean tracks the base, so the accrual cancels between the stance foot that lags and the swing foot that leads. The budgeting below measures the residue and finds it zero to four decimal places at every commanded yaw rate. The gate is retained as an optional argument solely so that the claim may be falsified by ablation rather than believed.

The dead band applies to both modes at the same 0.10 rad and therefore holds the differential to a stricter standard in per foot terms, forgiving 0.05 rad of splay per foot against 0.10 rad of common rotation. This is deliberate and rests on the biomechanical data of section 5.1.1. Human walking exhibits a systematic COMMON toe out of 3.3 degrees and no comparable systematic differential, so a common tolerance is the one with a physiological warrant and a differential of the same size is the less defensible of the two.

##### Reward budgeting

The term's magnitude can be predicted before it is run, because section 5.1 established that the foot heading error is 0.995 times the hip yaw angle and bracketed that angle between 0.471 and 0.83 rad. Two regimes must be budgeted separately, the fault the term is meant to price and the legitimate turning it must not.

The fault is budgeted at three per foot magnitudes and in three postures, being a pure splay in which the two feet are rotated oppositely, a pure common rotation in which both are rotated the same way, and a mixed posture in which one foot is rotated and the other is not. All figures are the raw term value after the 0.10 rad dead band, with `common_mode` set to "mean" and `differential_scale` to 1.0.

| Per foot error | Pure splay | Pure common | One foot only | Pure splay at weight minus 2.0 |
|---|---|---|---|---|
| 0.471 rad, the lower bracket | 0.7011 | 0.1359 | 0.1699 | -1.402 per second |
| 0.600 rad | 1.1968 | 0.2470 | 0.3088 | -2.394 per second |
| 0.830 rad, the upper bracket | 2.4078 | 0.5269 | 0.6586 | -4.816 per second |

The splay column carries five times the common column at every magnitude, which is the four to one orthogonal ratio of section 5.1.1 modified by the dead band, and it is the column that matters, section 5.1.1 having argued from the reaction torque that a policy buying yaw from the hip yaw joint necessarily produces a splay.

The turning tax was computed by simulating a steady turn under ideal foot placement against this configuration's own gait clock, being 1.0 Hz with a stance duration of 0.62 and an anti phase offset of 0.5 read from `cfg/SF/kscale_base_env_cfg.py:176-190`, with each foot planted at the base heading it will hold at mid stance and free to track the base during swing.

| Commanded yaw rate | Common mode | Differential mode | Total | At weight minus 2.0 |
|---|---|---|---|---|
| 0.3 rad/s, the present command cap | 0.0000 | 0.0006 | 0.0006 | -0.001 per second |
| 0.6 rad/s | 0.0000 | 0.0096 | 0.0096 | -0.019 per second |
| 0.9 rad/s, the curriculum's terminal rate | 0.0000 | 0.0299 | 0.0299 | -0.060 per second |

The common mode contributes nothing at any rate, which is the quantitative form of the argument that retired the swing gate. The whole of the residual tax is differential and arises because the two feet are replaced alternately, so that at any instant one carries a heading half a cycle staler than the other. It is bounded by 0.0299 raw, being between 23 and 80 times smaller than the fault the term is meant to price, and at the proposed weight it costs 0.060 per second against a total episode reward of 26.0431, which is 0.23 per cent.

Against the episode reward breakdown of the run at iteration 17217, whose largest terms are `rew_lin_vel_xy` at plus 27.0038, `rew_no_fly` at plus 10.2748, `rew_gait` at minus 6.5717, `pen_action_smoothness` at minus 4.8300 and `pen_ang_vel_xy` at minus 3.2293, a term reading between minus 1.40 and minus 4.82 at the present fault places it fourth or fifth in the set, falling to under minus 0.10 once the per foot error is inside 0.15 rad and to the turning tax alone thereafter. That is the intended shape, expensive now and very nearly free once the behaviour is corrected, and it displaces between five and eighteen per cent of the current total reward while the fault persists.

The alternative calibration should be recorded and its rejection justified. Booster Gym carries each of its two terms at minus 1.0 against linear tracking weights of 1.0 per axis and a yaw tracking weight of 0.5 [12]. This configuration carries `rew_lin_vel_xy` at 50 covering both linear axes and `rew_ang_vel_z` at 15, so the corresponding scale factors are 25 against a single linear component and 30 against the yaw component, and a direct transfer would give between minus 25 and minus 30. At the lower bracket of the present fault such a weight would read between minus 17 and minus 21 per second, which would make it by a wide margin the largest term in the entire set and would exceed `rew_ang_vel_z` at plus 2.7620 by a factor near seven. A term that outweighs the reward for turning sevenfold does not correct a turn, it forbids it. The proposed minus 2.0 is a twelfth of the scaled figure. It is offered as a deliberately conservative first value to be read from the logs rather than as a derived one, and the escalation path is set out below.

##### Implementation instructions

The order below is the order of dependency and each step is verifiable before the next is begun.

1. Extend `feet_yaw_alignment` at `environments/environments/tasks/locomotion/mdp/rewards.py:579` with the six optional arguments and the body given above. Confirm by inspection that with every new argument left at its default the function reduces to the existing four statements, and confirm by `grep -rn "feet_yaw_alignment"` across the tree that it still has no caller other than the one added in step 3.

2. Verify the orthogonality and the sign convention numerically before any training run. Add a self test beside `scripts/analysis/kscale_symmetry_selftest.py` which asserts that a pure splay of plus and minus 0.3 rad returns a common mode of zero to within one milliradian squared under `common_mode="mean"` and a non zero one under `"sum"`, that a pure common rotation returns a differential of zero under both, that driving each hip yaw alone yields a toe heading gain of 0.995 of the SAME sign on both legs at the nominal pose, that the same gain is unity at the URDF zero pose, and that it spreads by more than 0.4 across the hip pitch travel, and that the negative z axis of each foot link maps to the root frame positive x at the nominal pose to within one milliradian. All four figures are established in section 5.1 and in the audit above, and a regression in the asset would otherwise reach training silently.

3. Add `pen_feet_heading` to `RewardsCfg` in `cfg/SF/kscale_base_env_cfg.py`, and replace `pen_hip_deviation` at line 791 with the two split terms, all with the parameters and comment blocks given above. Add them to `RewardsCfg` alone, which both `KscaleEnvCfg` and `KscaleHIMEnvCfg` share, so no second edit is required.

4. Confirm that the split of `pen_hip_deviation` is inert. Sum the two new logged channels at any iteration and check the result against what the single term would have reported. The identity is exact by the plain summation at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:186`, so any discrepancy indicates a joint name pattern that resolves differently than intended and must be traced before it is dismissed.

5. Confirm that the KScale symmetry module at `environments/environments/tasks/locomotion/mdp/symmetry/kscale.py` requires no change. The new term reads body poses and contact forces rather than the observation or action vectors, so it lies outside the augmentation's scope, but the confirmation belongs in the record because section 4.7 established that module against a joint ordering the reward set does not otherwise touch. Note that the co directed hip yaw axes established in the audit above are consistent with that module carrying the hip yaw in its sign flip set, a vertical axis pseudo vector negating under a sagittal reflection.

6. Launch a run and read four quantities against the run of 2026-08-25 at matched iteration counts. `Episode_Reward/pen_feet_heading` should fall monotonically after the first two thousand iterations. `Episode_Reward/pen_hip_yaw_deviation` should fall towards the SD_BRS1's per joint 0.111 rad and is the independent corroboration, no term reading the hip yaw coordinate directly. `Episode_Reward/pen_hip_roll_deviation` should not rise materially, a rise indicating that the policy has moved the fault from one off axis joint to the other rather than abandoned it. And `Episode_Reward/rew_ang_vel_z` should not fall materially below the plus 2.7620 of the reference run. The last is the falsification condition. A term that fixes the feet by suppressing the turn has not fixed anything, and if yaw tracking degrades the weight is too high.

7. Record the outcome in section 7 of this document and promote the kinematic and frame results, being that the KScale foot heading is 0.995 times the hip yaw angle on both legs with the same sign at the nominal pose, with no contribution from any other joint and with a gain that is unity at the zero pose and 1.524 at a hip pitch of minus 1.0 rad, into section 2.3 of [../context/KScale.md](../context/KScale.md), beside the correction of 2026-08-26 which establishes that the 0.0998 tilt producing the 0.995 is a pose artefact rather than a hardware cant.

##### The ablation sequence this plan sets up

The parameters above are chosen so that each subsequent question is a one line change against a logged baseline rather than a restructuring, and the intended order is recorded here so that the arms are not run in a sequence that confounds them.

| Arm | Change from the baseline | Question it answers |
|---|---|---|
| Baseline | as specified above | Does the task space term correct the fault at all |
| A | `airborne_only=True` | Does gating the common mode to the swing help, hurt or do nothing, the budgeting predicting nothing |
| B | `differential_scale=0.0` | How much of the correction is the differential mode alone responsible for |
| C | `common_mode="sum"` | Does the mode mixing of the summed form measurably degrade the correction |
| D | `tolerance=0.0` | Is the dead band earning its place, this arm being exact Booster Gym parity |
| E | `pen_hip_yaw_deviation` raised to -0.2 | Does the Digit ratio add anything the task space term has not already obtained |
| F | `weight` raised towards -5.0 | Is the conservative first weight leaving correction unclaimed |

Arms A through D vary the new term alone and may be run in any order against the baseline. Arm E varies the joint space instrument and should follow rather than accompany them, since it is the only arm that changes a term the baseline holds fixed. Arm F should be run last, a weight change being the least informative of the six and the most likely to be confounded with any of the others.

##### What this section deliberately does not propose

Two adjacent changes are declined here and are listed so that they are not mistaken for omissions.

The addition of a feet roll term, which Booster Gym carries at minus 0.1 [12] and which van Marum carries as feet orientation at 0.05 [6], is not proposed, because the KScale already carries `rew_keep_ankle_roll_zero_in_air` over the same axis and no evidence in the run of 2026-08-25 implicates it.

The referencing of the term to the commanded heading rather than to the base is not proposed, for the three reasons section 5.1.1 sets out, and the decision should be revisited only if a future configuration drops `heading_command=True`, at which point the base ceases to be a proxy for the direction of travel and the argument lapses.

---

## 6. Conclusion of the Integration Phase

The KScale arrived as a faithful copy of the BRS configuration and the fidelity of that copy is the reason the defects in it are subtle. Nothing is obviously wrong on reading the file. The observations, the events, the curriculum and the network architecture are correct term for term, and the reward set is coherent, internally consistent and, taken on its own terms, reasonable. The defects are all of one kind, which is that a quantity carrying physical units was carried across from a robot three times the mass, or that a convention true of the source robot was assumed true of the target.

Three findings account for most of the work this plan prescribes and none of the three is visible in the configuration file that contains it.

The root frame is rotated ninety degrees from the Isaac Lab convention, so the velocity command that reads as forward is lateral. This is invisible in the configuration because the configuration is correct, it is the asset that differs, and it would have produced a policy that trained, converged and crab walked.

The foot link frame carries its vertical on y rather than z, so the sole depth is 0.043 m and not the 0.19 m a reader assuming the BRS convention obtains. This one was very nearly caught, the original author having recorded the suspicion that the number was a misread, and the four reward terms that depend on the sole were withheld rather than shipped against it. That was the right call, and this plan supplies the measurement that was missing rather than overturning a judgement.

The launcher has no branch for the robot and fails open rather than closed, so a request for the KScale trains the TRON1. This is the only defect of the three that would waste a training budget without leaving any evidence in the training run itself.

Against those, the reward work proves lighter than the task brief anticipated. Because every robot specific quantity in the reward package is already a configuration parameter and no reward function hard codes a name, a count or a geometry, the four missing terms are restored by wiring alone. No shared function is edited, no optional argument is introduced, no version two is created, and the BRS, the three TRON1 variants and the quadruped are provably untouched, which satisfies the backwards compatibility rule of `../../CLAUDE.md` not by careful handling but by construction.

The order of work is determined by the dependencies. The frame correction of section 4.2 comes first, because the symmetry mirror and two configuration blocks are wrong until it is done and would otherwise be corrected twice in opposite directions. The parameter corrections of section 4.3 come next, since they govern terms that are already live and are therefore already doing harm. The reward restoration of section 4.4 and the removal of section 4.5 follow, then the class hierarchy and registrations of section 4.6, then the symmetry module and agent configuration of section 4.7. The launcher clause of section 4.8 may be done at any point and should be done early, since without it none of the rest can be exercised.

Two matters are deliberately left open and should not be mistaken for oversights. The actuator gains require a retuning pass of their own, the present values leaving every joint overdamped and two ankle joints at or above the Nyquist bound of the control loop, and that derivation deserves the treatment `../context/BRS.md` gives the BRS rather than a scaling argument appended here, particularly since confirming the Robstride identification would replace any such argument with measured data. And the nominal standing pose remains the placeholder its own docstring admits it to be, so the standing height of 0.795 m that four parameters in section 4.3 derive from must be re-measured once that pose is settled.


## 7. Outcome and divergences from the integration plan

Every proposal of chapter 4 was implemented in the pass of 2026-08-24. The KScale reward set now matches the BRS term for term, twenty seven terms carrying identical functions and identical weights, and every parameter that differs between the two configurations is a derived robot specific quantity rather than an inherited one. No function in the shared `mdp` package was edited, no optional argument was added and no version two was created, so the BRS, the three TRON1 variants and the quadruped are untouched by construction rather than by careful handling, as section 4 predicted. The three shared files that were modified, the task registry, the agent configuration module and the symmetry package initialiser, were each audited line by line to confirm that no line outside KScale scope changed.

Six things diverge from what this document proposed and each is recorded here rather than silently absorbed.

The effective inertias of section 3.6 omitted the armature, and they should not have. Isaac Lab writes armature onto the PhysX joint, so it enters the mass matrix diagonal of the degree of freedom and is part of the inertia the proportional derivative loop actually sees, and at this robot's distal joints it exceeds the link inertia several times over, the ankle roll's 0.005 kg m squared against a link inertia of 0.00074. Including it moves the ankle roll from a natural frequency of 164.3 rad/s to 59.0 and the ankle pitch from 138.6 to 81.1, so the claim in section 3.6 that two ankle joints sit at or above the Nyquist bound of 157.08 rad/s is withdrawn. No joint does. The overdamping finding stands unchanged and is if anything the more clearly the dominant defect once the bandwidth alarm is removed. The corrected figures are in [../context/KScale.md](../context/KScale.md) section 6.

The actuator retuning that section 4.3 deliberately declined to attempt was carried out, at the explicit direction of the user. It follows the method of `../context/BRS.md` section 9 rather than a scaling argument, and its result is independently corroborated at the ankles and the hip yaw by the mass scaled BRS recommendation. The Robstride identification remains unconfirmed and would still supersede it.

The contradiction between `enabled_self_collisions` and `self_collision` that section 4.3 proposed resolving to one value was left as it stands. The reason emerged only on inspection of the BRS asset configuration, which carries the identical pairing, so resolving it on the KScale alone would have made the KScale differ from the exemplar it is meant to mimic. In effect the articulation root property governs at runtime and self collision is off for both robots, which is also why the standing bounding box overlaps do not produce persistent interpenetration forces. The forged contact exposure the section describes is real and is recorded in the context document rather than repaired here.

The defect in `scripts/rsl_rl/play.py` that section 4.9 listed as deliberately left standing was repaired instead, because section 4.8 makes KScale evaluation a one word command and a dump silently taken against the BRS sole table would have been the first thing that command produced. The foot pattern and the sole table are now resolved at runtime against the articulation's own body names, the BRS path resolving to exactly the values it carried before.

The flip set of section 3.5 was stated about the y and z plane, that being the correct mirror plane in the frame as exported. After the frame correction of section 4.2 the mirror plane is x and z, and recomputing the set about it reproduces the same four joints, which is the confirmation the section predicted, the flip set being a physical property of the robot rather than of the frame it is expressed in.

The launcher gained four clauses per chain rather than the two section 4.8 proposed, the rough terrain variants being registered and therefore worth reaching, and it gained the defensive final arm that section recommended in passing. That arm is the more valuable half of the change. Both dispatch chains assigned a default before testing any clause, so an unrecognised argument did not fail, it silently selected the TRON1 SoleFoot, and a user asking for a robot with no clause received a complete, plausible, converging run of a different robot with the wrong task identifier appearing only in that run's dumped parameters.

One matter this document left open remains open. The nominal standing pose is still the unverified knee bend guess its own docstring admits it to be, and the four reward parameters derived from the 0.7953 m standing height must be re-measured once it is settled by visual inspection in the simulator.

### 7.7 The gain derivation was wrong, and the first training run failed because of it

Recorded 2026-08-25, after the run this plan authorised was launched and failed.

Section 3.6 of this plan called for actuator gains derived from the effective inertia of each joint's distal subtree, and section 4.3 carried that call into the configuration. The derivation was performed as specified and the specification was wrong. The distal subtree inertia is the inertia a leg presents while swinging freely in the air, and it says nothing about whether a joint can hold the robot up, which is a question answered by the static ground reaction torque and never asked anywhere in this plan.

The consequence was a knee at 25 Nm/rad against a static double support load of 8.109 Nm and an ankle pitch at 7 Nm/rad against 3.150 Nm. Minimising the gravitational plus spring potential energy with the sole planted puts the settled stance at a base height of 0.4362 m against a nominal 0.77161 m, so the robot collapsed the instant it was set down. The training run at `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-24_11-22-46` shows an episode length pinned at 40 steps from iteration 1000 to iteration 18000, with every termination attributed to `low_height` and none whatever to `base_contact`, which is the signature of legs folding under a body rather than a body toppling over its feet.

Two further defects of the same derivation are recorded with it. The ankle roll at 5 Nm/rad deflected 0.841 rad under a centre of pressure held at the edge of the sole, being 3.2 times that joint's entire travel of plus and minus 0.2618 rad, so it sat pinned at its stop whenever the robot leaned. And the spawn height of 0.85 m, which this plan left unexamined, stood 0.0784 m above a standing height that had moved twice since it was chosen, so the robot was dropped rather than set down.

The remedy raised the knee to 200 Nm/rad, the ankle pitch to 50 and the ankle roll to 20, keeping the corrected damping which was the sound part of the derivation and sizing it against the stance reflected inertia rather than the swing inertia. The spawn height was lowered to 0.79 m, being the standing height plus the 0.02 m settling margin the SD_BRS1 already uses. `scripts/analysis/kscale_stance_analysis.py` was written so that the omitted calculation is reproducible rather than remembered, and [../context/KScale.md](../context/KScale.md) was restructured to the analytical order of `context/BRS.md` so that both inertia regimes and the stance load are computed for every robot added to this workspace hereafter.

The lesson this plan should have carried, and which any successor plan for a new robot must, is that a stiffness has two independent requirements. It must place the joint's bandwidth somewhere sensible, and it must hold the robot's weight. The second is the binding one and it is the cheaper to check.

### 7.8 The feet heading reward, implemented 2026-08-26

Every proposal of chapter 5 was implemented in a single pass on 2026-08-26 and the chapter's specification was followed without amendment. `feet_yaw_alignment` at `environments/environments/tasks/locomotion/mdp/rewards.py:579` gained the six optional arguments, `pen_feet_heading` was added to `RewardsCfg` at `cfg/SF/kscale_base_env_cfg.py:874` at weight minus 2.0 with the mean form, a differential scale of 1.0, a tolerance of 0.10 rad and no contact gate, and `pen_hip_deviation` was replaced by `pen_hip_roll_deviation` and `pen_hip_yaw_deviation`, both at the minus 0.1 the single term carried.

The verification the chapter's step 2 called for was written as `scripts/analysis/kscale_feet_heading_selftest.py` and passes twenty five checks. Twelve are kinematic and read the URDF directly, establishing that the negative z axis of each foot link maps to the root frame positive x at the nominal pose to within 1.1e-05, that both hip yaw axes read minus 0.0998, 0.0000, plus 0.9950 in the root frame with a mutual dot product of 1.000000, that a positive command at either hip produces a toe heading gain between 0.99416 and 0.99497 of the same sign on both legs, and that the ankle roll at its limit moves the toe heading by 0.0001 degrees. Thirteen are behavioural and execute the shipped reward, establishing the orthogonality of the mean form against the summed form's failure of it, the four to one and three to one splay ratios, the dead band, the backwards compatible default path to ten decimal places, and the ninety degree offset of the Euler path at three foot yaw angles.

The inertness of the hip deviation split was confirmed by resolving both joint name patterns against the URDF rather than by argument. The single term and the pair resolve to the identical four joint set, `left_hip_roll_03`, `left_hip_yaw_03`, `right_hip_roll_03` and `right_hip_yaw_03`, and the two new patterns are disjoint, so no joint is counted twice or dropped. With `joint_deviation_l1` a plain summation at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:186` and both weights equal, the total reward is unchanged exactly.

The blast radius is as chapter 5 predicted. Every hunk of the diff to the shared `mdp` package falls inside `feet_yaw_alignment`, whose only caller in the tree is the KScale term added in the same pass, so the SD_BRS1, the three TRON1 variants and the quadruped are untouched by construction. The KScale symmetry module needed no change, the new term reading body poses rather than the observation or action vectors, and it was confirmed to reference no reward term at all.

Two divergences from the chapter are recorded.

The self test was written to run outside the Isaac container as well as inside it, which the chapter did not ask for. Neither `isaaclab` nor the environments package imports on a bare interpreter, so a test that merely imported the reward would have been unrunnable in exactly the situation in which a frame regression is most likely to be introduced, which is an editing session rather than a training session. The script therefore imports the real modules where they are available and otherwise extracts the function body from `rewards.py` and executes it against a shim supplying the three mathematics helpers it uses. Both paths exercise the same source text, so the checks are meaningful either way, and the shim is validated against the real implementation whenever the container is available.

The module docstring of `cfg/SF/kscale_base_env_cfg.py` was corrected in the same pass, which the chapter did not list. It described the SD_BRS1 hip yaw joints as disabled by a zero width limit, which section 4.9 of this plan had already established to be false, they being declared `type="fixed"`, and it directed the reader to a `pen_hip_deviation` that no longer exists under that name. Leaving either standing would have pointed the next reader at a term that is absent and a mechanism that is imagined.

The chapter's remaining steps are not yet discharged and are the whole of what is outstanding. No training run has been launched, so the four quantities of step 6 have no values, the 0.0000 turning tax and the minus 1.40 to minus 4.82 fault magnitudes remain predictions from statics and geometry rather than observations, and the ablation table of section 5.1.2 has no arm completed, not even its baseline. The falsification condition should be read first. If `rew_ang_vel_z` falls materially below the plus 2.7620 of the reference run, the term has corrected the feet by suppressing the turn and the weight is too high.


---

## Bibliography
1. Su et al., Leveraging Symmetry in RL-based Legged Locomotion Control, IROS, 2024, arXiv:2403.17320.
2. Abdolhosseini et al., On Learning Symmetric Locomotion, Motion in Games, 2019, DOI 10.1145/3359566.3360070.
3. Ordonez Apraez et al., On discrete symmetries of robotics systems, a group theoretic and data driven analysis, RSS 2023 and IJRR 2024, arXiv:2302.10433.
4. Mittal et al., Symmetry Considerations for Learning Task Symmetric Robot Policies, 2024, arXiv:2403.04359.
5. Yu, Turk and Liu, Learning Symmetric and Low-Energy Locomotion, 2018, arXiv:1801.08093.
6. Van Marum et al., Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking, 2024, arXiv:2404.19173.
7. Siekmann et al., Learning Memory-Based Control for Human-Scale Bipedal Locomotion, 2020, arXiv:2011.01387. Cited for the periodic contact schedule construction the gait clock implements.
8. Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior, 2022, arXiv:2212.03238. Authors not recorded by the retrieved source, the short form is given.
9. Humanoid-Gym, Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer, 2024, arXiv:2404.05695. Authors not recorded by the retrieved source, the short form is given.
10. Nilsson and Thorstensson, Ground reaction forces at different speeds of human walking and running, Acta Physiologica Scandinavica, 136(2), 1989. Cited for the ground reaction force range and the double support fraction of the gait cycle.
11. Perry, Gait Analysis, Normal and Pathological Function, 1992. Cited for stance width as a multiple of hip width.
12. Wang, Chen, Han, Wu and Zhao, Booster Gym, An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion, 2025, arXiv:2506.15132. Cited for the feet yaw, feet roll and torque tiredness terms of its Table II, and for the common and differential decomposition its released implementation carries.
13. Bevel-geared mechanical foot, a bioinspired robotic foot compensating yaw moment of bipedal walking, Advanced Robotics, DOI 10.1080/01691864.2021.2017343. Cited for the yaw moment a swinging leg imposes upon the stance foot. Authors, venue year and page range are not recorded here, the publisher page having returned a 403 to retrieval, and the short form is given in accordance with the citation rule of `/ws/CLAUDE.md`.
14. Popovic, Hofmann and Herr, Angular momentum regulation during human walking, biomechanics and control, ICRA, 2004. Cited for the regulation of whole body angular momentum and the segment to segment cancellation of its horizontal component.
15. Cibulka, Winters, Kampwerth, McAfee, Payne, Roeckenhaus and Ross, Predicting foot progression angle during gait using two clinical measures in healthy adults, a preliminary study, International Journal of Sports Physical Therapy, 11(3), 2016, pages 400 to 408. Cited for the foot progression angle of 3.3 degrees plus or minus 5.6 degrees measured over sixty healthy adults.

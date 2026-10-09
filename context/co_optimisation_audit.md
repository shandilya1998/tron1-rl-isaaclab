# Co-Optimisation Audit

Defect audit of the design and policy co-optimisation pipeline, 2026-09-29, covering `co_optimisation/`, `design_generator/` and `environments/` for the quadruped and the TRON1 solefoot biped. Read against the live tree. Every finding carries its location, its arithmetic and the edit that closes it.

Already discharged, do not rediscover. The quadruped rough terrain built a 39,694,044 triangle collider that registered no contact, cause `horizontal_scale=0.05` at `terrains_cfg.py:28`, restored to 0.1 on 2026-09-29. See [quadruped.md](quadruped.md) section 8.

## 1. Finding register

| ID | Finding | Live for | Stops a run | Location |
|---|---|---|---|---|
| B1 | Design generator family hard coded to `"biped"` | quadruped | yes | `train.py:221` |
| B2 | `capture_env_state` reads `terrain_levels`, absent on a plane | both, flat tasks | yes, iteration 0 | `env_state.py:144` |
| B3 | Link editor indexes `env_paths[individual]`, no bound check | both | yes, iteration 12480 | `update.py:208`, `update.py:272` |
| B4 | `cma`, `cmaes`, `pandas`, `design_generator` undeclared | all runs, all robots | if absent | `co_optimisation/setup.py:12` |
| B5 | `--policy-type` has no `choices`, typo falls through to plain PPO | both | silent wrong run | `cli_args.py:60` |
| D1 | Fixed `pen_base_height` target biases fitness toward the donor design | both | no | `base_env_cfg.py:1088`, `limx_base_env_cfg.py:939` |
| D2 | Estimator MSE unscaled, shares `clip_grad_norm_` with the policy | both | no | `copt_ppo.py:258`, `copt_ppo.py:309` |
| D3 | `encoderOut` unnormalised into the actor, `output_detach` ignored | both | no | `copt_actor_critic.py:144` |
| D4 | `compute_returns_design_wise` has no caller | both | no | `copt_ppo.py:96` |
| D5 | Learning rate reset 25 times in the random phase | both | no | `copt_on_policy_runner.py:692` |
| D6 | 6 of 7 curriculum terms read truncated sums at every swap | both | no | `curriculums.py:476` |
| D7 | Spawn height constant while leg length is searched | both | no | `quadruped_identified_cfg.py:94` |
| D8 | Design lattice 144 points against a population of 256 | quadruped | no | `usd_generator.py:416` |
| L1 | `force_usd_conversion=True`, 55,164 imports per run | both | no | `usd_generator.py:483` |
| L2 | Observation width 12,872 floats, 7.19 GiB of rollout storage | both | no | `base_env_cfg.py:360-651` |
| L3 | `replicate_physics=False` at 6000 envs, 1500 prototypes | both | no | `quadruped_pointfoot_env_cfg.py:144` |
| L4 | Actuator write is 72,000 single element device writes | both | no | `respawn.py:276` |
| X1 | Reward and termination managers not rebound on respawn | KScale only | dormant | `respawn.py:176-221` |
| X2 | `apply_actuator_params` silently drops unknown keys | both | dormant | `respawn.py:282` |
| X3 | `effort_limit_sim` and `velocity_limit_sim` never written | random or cat search | dormant | `respawn.py:301` |
| X4 | MJCF declares no joint velocity limit, solver clamp unknown | both | verify | `actuator_base.py:183` |
| X5 | 11 of 14 quadruped catalogue entries state no armature | cat search | dormant | `actuators.json` |
| X6 | `check_geometry` validates only `trunk_width`, never varied | both | dormant | `quadruped/builder.py:311` |
| X7 | 3 of 4 solefoot experiment configurations unregistered | biped | no | `robots/__init__.py:46` |
| H1 | `.gitmodules` name disagrees with path, 3 stale docstrings | both | no | `.gitmodules` |
| H2 | Quadruped duplicates 300 lines as `CoptObservationsCfg` | quadruped | no | `base_env_cfg.py:360` |
| H3 | `robot_pos` puts absolute world coordinates in the critic | both | no | `observations.py:103` |
| H4 | `num_output_dim`, `orthogonal_init`, `self_collision` unread or inconsistent | both | no | `copt_actor_critic.py:79` |

## 2. The pipeline as built

```
scripts/rsl_rl/train.py  --policy-type COPT
  │
  ├─ _family = "biped"                              train.py:221   ◄── B1
  ├─ _num_individuals   = 256                       train.py:222
  ├─ ea_update_interval = 480 iterations            train.py:225
  ├─ ea_late_start      = 12000 iterations          train.py:226
  ├─ late_start_pop     = num_envs / 4 = 1500       train.py:259
  │
  ├─ GrowingDesignDistCMAESDesignGenerator ─────────────────┐
  └─ CoptOnPolicyRunner(env, generator, cfg)                │
       │                                                    │
       │  learn()                                           │
       │   ├─ _reload_morphology()        gen 0, respawn    │
       │   └─ every 480 iterations:                         │
       │        _update_morphology()                        │
       │          ├─ sim.stop()                             │
       │          ├─ generator.update_with_fitness(f) ◄─────┤ ask / tell
       │          ├─ generate_population(gen) ──────────────┤ emit MJCF + convert USD  ◄── L1, B7
       │          ├─ apply_link_length_params()   in place, edits USD prototypes  ◄── N1
       │          ├─ (nominal pose write)          ◄── D1, absent today
       │          ├─ env.reset()                   ◄── D6 curricula read truncated sums
       │          ├─ assert_link_lengths_applied() ◄── N1, first N envs only
       │          ├─ apply_actuator_params()       ◄── L4, X2, X3
       │          └─ episode_length_buf randomised
       │
       └─ design → environment assignment, round robin, four places that must agree
            copt_on_policy_runner.py:744   env_to_individual = i % 256
            wrappers.py:114                proto_prim_paths[index % len]
            respawn.py:279                 env_idx % num_individuals
            update.py:208                  env_paths[individual]          ◄── B3
```

| | quadruped | biped |
|---|---|---|
| Task ids | `Isaac-Quadruped-Copt-{Flat,Rough}-v0` | `Isaac-Limx-SF-Copt-MoRAL-{Flat,Rough}-v0` |
| Bodies / joints | 25 / 12 | 15 / 8 |
| Root prim | `/trunk/trunk` | `/base_Link/base_Link` |
| Joint expressions | `.._hip_joint`, `.._thigh_joint`, `.._calf_joint` | `abad_._Joint`, `hip_._Joint`, `knee_._Joint`, `ankle_._Joint` |
| Catalogue picks | `A1-ABAD`, `A1-HIP`, `A1-KNEE` | `TRON1-ABAD`, `-HIP`, `-KNEE`, `-ANKLE` |
| `num_envs` | 6000 | 4096 |
| `gtEncoderOut` | `privilegedDynamicsObs` | `morphologyObs` |

The three `A1-` catalogue entries reproduce `quadruped_identified_cfg.py` field for field, so the actuator write of a baseline design is a no operation. Preserve that deliberately.

## 3. Blocking defects

### B1 Design generator family hard coded

The quadruped copt task emits biped assets. Root resolution fails first, then every joint expression matches nothing.

```python
# scripts/rsl_rl/train.py:221  BEFORE
_family = "biped"
...
catalog_names=["TRON1-ABAD", "TRON1-HIP", "TRON1-KNEE", "TRON1-ANKLE"],
```

```python
# scripts/rsl_rl/train.py:221  AFTER
# Derived from the task so both registered quadruped tasks reach the quadruped
# package. An unrecognised task is refused rather than defaulted, the failure it
# would otherwise raise being an unresolvable articulation root, thousands of
# lines of importer output later. Quadruped shortlist is the three catalogue
# entries that state a rotor inertia; the other eleven give armature 0.
_FAMILY_BY_TASK = {
    "Quadruped-Copt": ("quadruped", ["A1-ABAD", "A1-HIP", "A1-KNEE"]),
    "SF-Copt": ("biped", ["TRON1-ABAD", "TRON1-HIP", "TRON1-KNEE", "TRON1-ANKLE"]),
}
_family = _catalog_names = None
for _key, (_fam, _names) in _FAMILY_BY_TASK.items():
    if _key in args_cli.task:
        _family, _catalog_names = _fam, _names
        break
if _family is None:
    raise ValueError(
        f"Task {args_cli.task!r} maps to no design generator family. "
        f"Known keys: {sorted(_FAMILY_BY_TASK)}."
    )
...
catalog_names=_catalog_names,
```

### B2 Checkpoint capture needs a terrain level a plane does not have

`terrain_levels` is created only by `_compute_env_origins_curriculum` (`terrain_importer.py:347`), and a plane takes `_compute_env_origins_grid` (`terrain_importer.py:312`). `save()` runs at `it % save_interval == 0`, true at iteration 0.

```python
# co_optimisation/utils/env_state.py:144  capture_env_state  BEFORE
"terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
```

```python
# co_optimisation/utils/env_state.py:144  capture_env_state  AFTER
# A plane carries no curriculum state to restore (terrain_importer.py:312).
# None round-trips as a skip, so generated-terrain behaviour is unchanged.
_levels = getattr(terrain, "terrain_levels", None)
"terrain_levels": None if _levels is None else _levels.detach().cpu().clone(),
```

```python
# co_optimisation/utils/env_state.py:196  restore_env_state  AFTER
if state["terrain_levels"] is not None and hasattr(terrain, "terrain_levels"):
    terrain.terrain_levels.copy_(state["terrain_levels"].to(device))
terrain.env_origins.copy_(state["env_origins"].to(device))
```

Same function assumes `push_robot` and `base_velocity` exist (`env_state.py:137-138`). True in every training configuration, false in every play configuration, which never saves. Recorded, not repaired.

### B3 Link editor indexes environments by individual

| phase | population | safe |
|---|---|---|
| random, iterations 0 to 12000 | `num_envs / 4` | always |
| co-optimisation, 12000 onward | 256, fixed | only if `num_envs >= 256` |

A `--num_envs 64` smoke test survives the whole random phase and dies at iteration 12480.

```python
# co_optimisation/runners/copt_on_policy_runner.py, in __init__ after
# _num_individuals is read.
# Four pathways index the first num_individuals environments, one per design:
# update.py:208, update.py:272, respawn.py:279, wrappers.py:114. A population
# larger than the scene indexes past the end, and only at the first in-place
# update after the late start.
assert self._num_individuals <= env.num_envs, (
    f"num_individuals={self._num_individuals} > num_envs={env.num_envs}. "
    "Each design needs an environment to be authored through. "
    "Lower copt.num_individuals or raise --num_envs."
)
if env.num_envs % self._num_individuals:
    print(
        f"[copt] warning: {env.num_envs} envs over {self._num_individuals} "
        f"individuals gives {env.num_envs % self._num_individuals} individuals "
        "one extra environment, so fitness sample sizes differ across the "
        "population. 5888 or 6144 envs divide evenly."
    )
```

### B4 Undeclared dependencies

| import | site | scope |
|---|---|---|
| `cma` | `usd_generator.py:55`, module scope | every run, `train.py:107` imports the package unconditionally |
| `pandas` | `copt_on_policy_runner.py:23` | every run |
| `cmaes.CatCMAwM` | `usd_generator.py:830`, lazy | `--design_search cat` only |
| `design_generator` | `usd_generator.py:57` | every run |

```python
# co_optimisation/setup.py
install_requires=[
    "torch>=1.4.0", "torchvision>=0.5.0", "numpy>=1.16.4",
    "cma",              # usd_generator.py:55, module scope
    "cmaes>=0.11.0",    # CatCMAwM, usd_generator.py:830
    "pandas",           # design table, copt_on_policy_runner.py:450
],
# design_generator is a submodule of this repository, not a distribution.
# git submodule update --init --recursive, then pip install -e design_generator/
```

Verify in the training container with `python -c "import cma, cmaes, pandas, design_generator"` before anything else.

### B5 Policy type flag admits any string

`--policy-type copt` builds the co-optimisation environment and trains plain PPO on it, with no generator, no swap and no estimator, and says nothing.

```python
# scripts/rsl_rl/cli_args.py:60
choices=["PPO", "HIMPPO", "COPT",
         "quadruped-debug", "go1-debug", "go1-default-debug"],
```

## 4. Divergence surfaces

### D1 Fixed base height target prices the design search

The search varies both length scales over [0.75, 1.25], and the reward target does not move.

```
quadruped, foot drop = (l_thigh + l_shank) * cos(0.884337) = 0.270 m at s = 1

  s = 0.75 ──── stands at 0.2245 m ───┐
  s = 1.00 ──── stands at 0.2920 m    ├── target pinned at 0.292 m
  s = 1.25 ──── stands at 0.3595 m ───┘        (base_env_cfg.py:1088)

  cost = weight * (h - 0.292)^2 * episode_length
       = 5 * 0.0675^2 * 958.68 = 21.84 per episode at either extreme
```

Measured against run `quadruped_flat/2026-09-16_07-01-06`, artefact `artefacts/quadruped_flat-2026-09-16_07-01-06-reward-budget/reward_budget.txt`.

| quantity | value |
|---|---|
| `Train/mean_reward` | 89.81 |
| `Train/mean_episode_length` | 958.68 |
| `pen_base_height` realised at the donor design | -0.2723 per episode, 0.23 % of the negative budget |
| implied RMS height error held by the policy | 7.5 mm |
| penalty at either extreme of the design box | -21.84 per episode, 80x the donor |
| share of that run's return | 24 % |
| share of a return rescaled to the tree's tracking weights of 25 and 12.5 | 5.5 % of ~397 |

That run trained at target 0.33, tracking 10 and 5, `feet_air_time` 20.0, `pen_feet_hold` -0.5, against the tree's 0.292, 25, 12.5, 10.0, -0.1. Both figures are given for that reason.

Five per cent of the fitness signal is a deterministic function of the design, symmetric about the donor. CMA-ES maximises exactly this quantity, so it converges on the design whose natural height is 0.292 m, which is the design it started from. The biped is worse placed, its legs carrying 0.65 m of the 0.804 m body height, so the extremes move its standing height 0.1625 m against the same weight of -5.0.

```python
# environments/.../mdp/rewards.py:1889  base_height_rough_l2
# Optional argument, default reproduces the old behaviour exactly (CLAUDE.md 4).
def base_height_rough_l2(
    env, target_height, sensor_cfg,
    asset_cfg=SceneEntityCfg("robot"),
    target_height_attr: str | None = None,   # NEW
) -> torch.Tensor:
    ...
    target = target_height
    if target_height_attr is not None:
        # Per-environment target the co-optimisation runner writes after each
        # design swap, so the target tracks the design instead of pinning every
        # design to the donor's standing height. See co_optimisation_audit.md 4.1.
        target = getattr(env, target_height_attr, target_height)
    return torch.square(clearance - target)
```

```python
# co_optimisation/runners/copt_on_policy_runner.py, beside each
# apply_actuator_params call (lines 600, 646, 726).
def _write_design_geometry(self, env, population) -> None:
    """Per-environment standing height, computed from the population.

    The drop is an affine function of the two searched lengths, so it is
    computed rather than measured. Feeds both the reward target (4.1) and the
    spawn height (4.7).
    """
    import torch
    COS_FOLD, FOOT_RADIUS, MARGIN = 0.633691, 0.022, 0.028
    drops = []
    for links in population.get_link_length_params():
        thigh = links["thigh_FR_Link"]["child_pos"]
        shank = links["shank_FR_Link"]["child_pos"]
        l1 = sum(v * v for v in thigh) ** 0.5
        l2 = sum(v * v for v in shank) ** 0.5
        drops.append((l1 + l2) * COS_FOLD + FOOT_RADIUS)
    per_design = torch.tensor(drops, device=self.device, dtype=torch.float)
    idx = torch.arange(env.num_envs, device=self.device) % len(drops)
    env._design_base_height = per_design[idx]                       # 4.1
    robot = env.scene.articulations["robot"]
    robot.data.default_root_state[:, 2] = env._design_base_height + MARGIN   # 4.7
```

The quadruped configuration then sets `target_height_attr="_design_base_height"`, and every other task leaves it unset. The biped's `pen_feet_regulation` takes the same treatment at its own `base_height_target` (`limx_base_env_cfg.py:978`).

Interim one line measure, if the reward change is too large to make before a run, is to drop the weight of `pen_base_height` and record that the height term is disabled for the co-optimisation arm. Worse, because the gait needs the term.

### D2 Estimator MSE unscaled and sharing one gradient clip

```
loss = surrogate + c_v * value_loss - c_e * entropy + 1.0 * mse(estimator)   copt_ppo.py:258
                                                      ▲
                                                      └── no coefficient
clip_grad_norm_(self.policy.parameters(), 1.0)                               copt_ppo.py:309
                 ▲
                 └── estimator is a submodule, so its gradient norm scales the
                     actor and critic gradients down by the same factor
```

| robot | `gtEncoderOut` | channels | units | magnitude |
|---|---|---|---|---|
| quadruped | `privilegedDynamicsObs` | 48 | N m, rad/s², N, m/s | torque to 35.2, acc to ~1e2, force to ~160 |
| biped | `morphologyObs` | 154 | m, kg, kg m² | length ~0.3, mass 1 to 10, inertia ~1e-4 |

No term in either group carries a `scale` (`base_env_cfg.py:455-473`, `limx_base_env_cfg.py:324-350`). The quadruped's MSE is of order 1e4 against a surrogate loss of order 1. The biped's MSE is the masses alone, so the estimator never learns the 135 inertia columns.

```python
# co_optimisation/algorithms/copt_ppo.py, __init__ and update.
# Defaults to 1.0, reproducing the present behaviour exactly.
self.estimator_loss_coef = estimator_loss_coef
...
loss = (surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + self.estimator_loss_coef * model_estimation_loss)
```

The coefficient cannot fix the disparity between channels. Better is a running per channel standardisation inside `CoptActorCritic.get_estimator_target`, or, cheapest, a `scale` on every term of `PrivilegedDynamicsCfg` and `MorphologyCfg`, visible in the dumped configuration. Diagnostic already exists, `model_estimation_loss` is returned at `copt_ppo.py:339`, and four or more orders above `surrogate` in the first 200 iterations confirms the defect.

### D3 Estimator output unnormalised into the actor

```
actor input = policy(48) ++ morphologyObs(258) ++ encoderOut(48)   = 354 channels
              order 1       order 1 to 1e2       order 1 to 1e2
                            ▲                    ▲
                            │                    └─ predicted torques and accelerations
                            └─ constant within a generation, so a large per-design bias

actor_obs_normalization = False   (empirical_normalization=False, quadruped_rsl_rl_ppo_cfg.py:20)
EncoderCfg.output_detach = True   (quadruped_rsl_rl_ppo_cfg.py:65)  read nowhere
```

```python
# environments/.../agents/quadruped_rsl_rl_ppo_cfg.py, PFQuadrupedCoptPPORunnerCfg
# CoptActorCritic.update_normalization (copt_actor_critic.py:160) was written to
# fold encoderOut into the actor statistics, so this is supported already.
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    actor_obs_normalization=True,
    critic_obs_normalization=True,
)
```

Either honour `output_detach` by detaching at `copt_actor_critic.py:144`, or delete the field. A configuration field that describes nothing is worse than no field.

### D4 Per individual advantage normalisation is dead code

```python
# co_optimisation/runners/copt_on_policy_runner.py:408
- self.alg.compute_returns(obs)                 # base PPO, global normalisation
+ self.alg.compute_returns_design_wise(obs)     # copt_ppo.py:96, no caller today
```

`COPT_INVESTIGATION_PLAN.md` claims three landed variance remedies, of which two landed. Correct its register entry in the same pass. Make the change only alongside a measurement, the remedy having never been tested.

### D5 Learning rate reset 25 times in the random phase

`reinit_learning_rate()` fires at every generation boundary before the late start (`copt_on_policy_runner.py:692`), so the adaptive schedule returns to 1e-3 every 480 iterations for 12000 iterations. Whether that is wanted is an experiment question. That it happens silently is not.

```python
# co_optimisation/algorithms/copt_ppo.py:349
def reinit_learning_rate(self):
    self.learning_rate = self.init_learning_rate
    # Written through for the same reason CoptOnPolicyRunner.load does at
    # copt_on_policy_runner.py:210. Invisible under the adaptive schedule, which
    # rewrites the groups each minibatch, and a no-op under a fixed one.
    for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
```

### D6 Curriculum gates read truncated episode sums at every swap

```
_reset_idx (manager_based_rl_env.py)
  356  curriculum_manager.compute(env_ids)   ◄── reads _episode_sums
  375  reward_manager.reset(env_ids)         ◄── zeroes them

morphology swap truncates every episode at once, so every sum read at 356 is partial.
Suppression flag exists but is read in exactly one place, curriculums.py:476.
```

`common_step_counter` is a multiple of `480 * 25 = 12000` at every swap.

| curriculum term | interval, steps | coincidence with the 12000 step swap |
|---|---|---|
| `modify_command_velocity_ang_z` | 250 x 24 = 6000 | every 2nd evaluation |
| `modify_command_velocity_lin_x` | 300 x 24 = 7200 | every 5th |
| `modify_command_velocity_lin_y` | 300 x 24 = 7200 | every 5th |
| `modify_push_force` | 400 x 24 = 9600 | every 5th |
| `modify_linear_tracking_reward_std` | 2000 x 24 = 48000 | every evaluation |
| `modify_angular_tracking_reward_std` | 2000 x 24 = 48000 | every evaluation |
| `terrain_levels` | per reset | suppressed already |

The two tracking standard deviations sharpen the velocity reward from 0.4 toward the 0.09 floor, which is how the task becomes hard. Under co-optimisation they are frozen at their initial values for the whole run.

```python
# co_optimisation/runners/copt_on_policy_runner.py:714, and the pair at 570.
- unwrapped_env._suppress_terrain_curriculum = True
+ unwrapped_env._suppress_curriculum = True          # keep the old name as an alias
  unwrapped_env.reset()
- unwrapped_env._suppress_terrain_curriculum = False
+ unwrapped_env._suppress_curriculum = False
```

```python
# environments/.../mdp/curriculums.py, opening each of the six terms, in the
# shape the terrain term already has at line 476.
if getattr(env, "_suppress_curriculum", False):
    return command_cfg.ranges.ang_vel_z[1]   # present value, unchanged
```

Second defect in the same block. Every interval is written as a multiple of 24, the biped's `num_steps_per_env`, while the quadruped runs 25 (`quadruped_rsl_rl_ppo_cfg.py:16`). The intended 300 iteration interval is 288 and the intended 6000 iteration start is 5760, a 4 % error inherited from the robot the configuration was copied from. Six edits from `* 24` to `* 25`.

### D7 Spawn height constant while leg length is searched

| design | standing height | spawn z | foot sphere bottom relative to ground |
|---|---|---|---|
| quadruped, s = 0.75 | 0.2245 | 0.32 | +0.0955 m, dropped |
| quadruped, s = 1.00 | 0.2920 | 0.32 | +0.0280 m, as documented |
| quadruped, s = 1.00, hip offset -0.2 rad | 0.3520 | 0.32 | -0.0320 m, already buried on the plain task |
| quadruped, s = 1.25 | 0.3595 | 0.32 | -0.0395 m |
| biped, s = 0.75 | ~0.641 | 0.80 | +0.159 m |
| biped, s = 1.25 | ~0.967 | 0.80 | -0.167 m |

`reset_joints_by_offset` adds up to 0.2 rad to every joint (`base_env_cfg.py:1040`), so the 28 mm of clearance the asset comment claims is already consumed at the nominal design. Closed by the `_write_design_geometry` helper of D1, which writes `default_root_state[:, 2]`, since `reset_root_state_uniform` reads it at every reset and nothing else changes.

### D8 Design lattice smaller than the population

`_spec_from_scales` rounds each scaled length to two decimals (`usd_generator.py:416`).

| robot | baseline lengths | rounded values each | lattice | population | distinct designs possible |
|---|---|---|---|---|---|
| quadruped | 0.213, 0.213 | 12, 12 | 144 | 256 | 56 % of the population at best |
| biped | 0.30, 0.35 | 17, 19 | 323 | 256 | duplication likely, not certain |

Extends [copt_ppo_nonstationarity.md](../../context/copt_ppo_nonstationarity.md) section 3.1 to the quadruped with exact numbers. 44 per cent of the quadruped's population re-evaluates designs the same generation already contains, and a duplicated design's fitness is split across individuals rather than pooled. The rounding is also what makes a conversion cache effective, and L1 disables the cache anyway, so settle the two together. Interim measure is a population at or below 144.

## 5. Latency and memory

### L1 Every generation reconverts every design

```
force_usd_conversion=True         usd_generator.py:483   bypasses the cache entirely
output dir keyed by generation    usd_generator.py:458   two generations never share a path
```

| phase | generations | designs each | conversions |
|---|---|---|---|
| random | 25 | 1500 | 37,500 |
| co-optimisation | ~69 | 256 | 17,664 |
| total | | | 55,164 MJCF imports, of at most 144 distinct designs |

```python
# co_optimisation/runners/usd_generator.py, replacing the generation-keyed paths
# of _generate_individual_mjcf and _generate_individual_usd.
# The two-decimal lattice is 144 points against a population of 256, so a
# generation-keyed path reimports the same document tens of times. Keying on the
# specification collapses the run onto the lattice and force_usd_conversion then
# applies only to a design never seen before.
import hashlib

def _design_key(self, spec) -> str:
    return hashlib.sha1(repr(sorted(spec.__dict__.items())).encode()).hexdigest()[:16]

# gen_dir  = os.path.join(self.output_dir, f"gen_{generation:04d}")
gen_dir    = os.path.join(self.output_dir, "designs", self._design_key(spec))
# force_usd_conversion=True
force_usd_conversion=False
```

Second remedy, reduce the random phase population, which at 1500 is ten times the distinct designs the lattice can supply. Timings are already printed at `respawn.py:121,161` and `update.py:236`, so measure before and after without new instrumentation.

### L2 Observation width

Computed from the term inventories at `base_env_cfg.py:360-651` and `limx_base_env_cfg.py:163-432`, with a 17 by 11 height scan and a 3 component velocity command.

| group | history | quadruped per step | quadruped total | biped per step | biped total |
|---|---|---|---|---|---|
| policy | 1 | 48 | 48 | 36 | 36 |
| morphologyObs | 1 | 258 | 258 | 154 | 154 |
| privilegedDynamicsObs | 1 | 48 | 48 | 28 | 28 |
| historyObs | 25 | 235 | 5875 | 223 | 5575 |
| critic | 10 | 664 | 6640 | 486 | 4860 |
| commands | 1 | 3 | 3 | 3 | 3 |
| total | | | 12872 | | 10656 |

| quantity | quadruped | biped |
|---|---|---|
| rollout storage, `25 x num_envs x width x 4 B` | 7.19 GiB | 4.06 GiB |
| same for the plain task | 3.94 GiB | n/a |
| one minibatch of observations, 4 minibatches | 1.80 GiB | 1.02 GiB |
| estimator MLP parameters | 6.95 M | 6.40 M |
| actor MLP parameters | 0.35 M | 0.15 M |

Three economies, none of which changes what the policy can see.

| economy | columns removed | quadruped width after |
|---|---|---|
| drop `heights` from `historyObs` (`base_env_cfg.py:628`), terrain not morphology | 4675 | 8197, storage 4.58 GiB |
| critic `history_length` 10 to 1 for the six constant terms (`base_env_cfg.py:531-538`) | 3393 | 4804 with the above |
| cache `robot_material_properties` as `robot_link_lengths` does (`observations.py:259`) | 0 | removes a physics view query from every control step |

### L3 Heterogeneous physics at 6000 environments

`replicate_physics=False` is necessary, the environments carrying different assets. Its cost scales with the environment count and with the 1500 distinct prototypes of the random phase. Price of the method, not a defect, recorded so a slow start is not mistaken for a hang. Worth checking on the first run whether 6000 environments are affordable at all under it, the plain task having been tuned at that size with replication on.

### L4 Actuator write

6000 envs x 3 groups x 4 attributes = 72,000 single element device writes per generation (`respawn.py:276-290`). Replace with one gather per attribute indexed by `torch.arange(num_envs) % num_individuals`. Seconds per generation, not a bottleneck. Make it when the function is next touched.

## 6. Latent defects

| ID | Mechanism | Why dormant | Trigger | Fix |
|---|---|---|---|---|
| X1 | `respawn_robots` rebinds action (`:176`), event (`:196`) and command (`:215`) managers, not reward or termination. `HipLateralSwayReward` (`rewards.py:873`), `GaitReward` (`:1968`), `JointTorqueRatePenalty` (`:2170`) cache the articulation at construction | Quadruped and TRON1 use only `ActionSmoothnessPenalty`, which caches nothing | Adopting a KScale term, or giving the KScale a copt task | Add a rebind step in the event manager's shape |
| X2 | `apply_actuator_params` has no else branch (`respawn.py:282-290`), so an unknown key vanishes | Every `SIM_PARAM_KEYS` entry happens to be a tensor attribute of `IdentifiedActuator` (`actuator_pd.py:23-25`) | A new catalogue field, or a scalar-valued attribute | `raise`, or print the key |
| X3 | Only model side `effort_limit` and `velocity_limit` written; `*_sim` pushed once at `articulation.py:1772` and never again | Length search calls `_actuator_params(spec)` without scales (`usd_generator.py:654`), and `A1-*` equals the asset config | `RandomDesignGenerator` or `--design_search cat` | Write `effort_limit_sim` and `velocity_limit_sim` too |
| X4 | `velocity_limit_sim` falls back to the USD value (`actuator_base.py:183`); MJCF declares no joint velocity where URDF did | Unknown, not measured | Always live | Read the resolution table `_record_actuator_resolution` prints at startup, record it in [quadruped.md](quadruped.md) |
| X5 | 11 of 14 quadruped catalogue entries state neither `armature_value` nor `rotor_inertia`, so armature is 0 | `catalog_names` restricts the shelf, and B1's fix keeps it to the three `A1-` entries | Widening the shelf | Keep the shortlist; the warning at `usd_generator.py:819` already names offenders |
| X6 | `check_geometry` validates only `trunk_width` (`quadruped/builder.py:311`), never varied | Fallback at `usd_generator.py:449` can therefore never fire | Always live | Add a length aware check beside D7's arithmetic |
| X7 | `SFCoptMorphologyFromDynamics`, `SFCoptDynamicsFromMorphology`, `SFCoptMorphologyAndVelocity` defined at `limx_rsl_rl_ppo_cfg.py:171-207`, only the first instantiated at `robots/__init__.py:46` | Experiments 2 to 4 simply cannot be launched | Wanting them | Register three more entry points |

## 7. Configuration hygiene and record

| ID | Observation | Remedy |
|---|---|---|
| H1 | `.gitmodules` names the submodule `co_optimisation/co_optimisation/runners/design_generator` with `path = design_generator`. `usd_generator.py:4`, `update.py` docstring and [xml_design_generator.md](../../context/xml_design_generator.md) all still place it under `co_optimisation/runners/` | Rename the submodule to its path, correct the three descriptions |
| H2 | Quadruped carries a second copy of its observation configuration as `CoptObservationsCfg` (`base_env_cfg.py:360-651`); the biped carries copt groups as `None`-defaulting attributes of one class (`limx_base_env_cfg.py:429-432`) enabled in `__post_init__`. The two quadruped copies have already drifted, `link_lengths` existing in one critic and not the other | Bring the quadruped onto the biped's pattern, which is what CLAUDE.md rule 4 prescribes |
| H3 | `robot_pos` returns `root_pos_w` (`observations.py:103`), unclipped and unnormalised into the critic (`base_env_cfg.py:536`). On a 6 by 64 grid of 15 m tiles the y coordinate spans about +/- 480 m, so the value function is a function of the tile index | Subtract `env.scene.env_origins`. Every surviving run carries this, so not fatal, but free to fix |
| H4 | `EncoderCfg.num_output_dim=19` and `orthogonal_init` read nowhere, width derived from `gtEncoderOut` at `copt_actor_critic.py:79`. `MjcfConverterCfg.self_collision` is `False` in the generator (`usd_generator.py:481`) and `True` in both asset configs, though the articulation level `enabled_self_collisions=True` governs and is preserved across respawn (`respawn.py:82,111`). `flatten_history_dim` carries the comment "Required by HIMActorCritic" in tasks that build no such class (`base_env_cfg.py:345,582`) | Delete or honour each; make `self_collision` agree; correct the comment to name `copt_actor_critic.py:118` |

## 8. Order of work

```
1.  B3 + B5            assertions and choices, so a cheap validation is possible
2.  B4                 python -c "import cma, cmaes, pandas, design_generator"
3.  B1 + B2            the two that stop a run outright
4.  validation run     both robots, reduced --num_envs, reduced ea_update_interval
                       must exercise: bootstrap, one in-place update, the late-start
                       respawn, one save and one resume
5.  D1 + D7            shared helper; settle before a long run, a completed run
                       carrying D1 cannot be reinterpreted
6.  D2 + D3            diagnose from the first 200 iterations of the loss log
7.  D6                 invisible in the curves, leaves the curriculum where it started
8.  L1                 largest single item in the generation boundary
9.  D4, D5, D8         experiment design decisions as much as defects, user's call
```

## 9. What was verified and what was not

| Claim class | Basis |
|---|---|
| Every `file:line` citation | Read in the tree on 2026-09-29 |
| Reward budget of D1 | `artefacts/quadruped_flat-2026-09-16_07-01-06-reward-budget/`, hand rolled TFRecord and protobuf reader, no TensorFlow or Isaac Sim. Weights from that run's `params/env.yaml`, md5 `36978aabc33e77cb62669c2383938ced` |
| Widths, lattice, triangle and drop arithmetic | Computed from constants in the tree |
| Any runtime behaviour | NOT reproduced. No simulator reachable, so every consequence follows from reading |
| B4 | NOT confirmed. The Python on this filesystem is not the training interpreter |
| `co_optimisation/tests/design_generator_test.py`, 43 tests | NOT run, `pytest` absent here |
| Round robin ordering, that `Usd.Prim.GetAllChildren` returns env prims in creation order so `wrappers.py:114` agrees with `scene.env_prim_paths` | Reasoned from the cloner's construction order, NOT measured. One print at the first respawn settles it permanently |
| `CatCMAESDesignGenerator` | Read, reachable only for the biped today by B1, never exercised in a run. Margin correction follows Hamano and colleagues [1] as implemented by `cmaes` [2] |

## 10. Bibliography

1. Hamano, R., Saito, S., Nomura, M., Shirakawa, S. CMA-ES with Margin, Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization. Genetic and Evolutionary Computation Conference (GECCO), 2022. arXiv:2205.13482. Registered in [literature.md](../../context/literature.md) cluster 16, whose entry records that the venue and author list were taken from the arXiv record and not verified against the ACM proceedings.
2. CyberAgent AI Lab. `cmaes`, a Python library for CMA Evolution Strategy. https://github.com/CyberAgentAILab/cmaes. Registered in [literature.md](../../context/literature.md) cluster 16, read at `README.md`, `examples/catcma_with_margin.py` and `cmaes/_catcmawm.py` on the `main` branch on 2026-09-25.

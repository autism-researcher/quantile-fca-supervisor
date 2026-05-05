#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA 0.9.13 — Episode logger (normal / ttc_baseline / innov)
Python 3.7 compatible.

Your requested fixes:
1) No abnormal high speeds:
   - Optional speed caps (ego + traffic) based on current speed limit.
   - Traffic speed distribution kept near-normal.
2) Smoother traffic:
   - Hybrid physics enabled.
   - Random lane-change disabled globally (major smoothness gain).
   - "keep_right_rule" + gentle lane-change.
   - Warm-up period so autopilot starts smoothly before logging.
3) Still some risky situations to compare baseline vs innov:
   - stopped_vehicle / sudden_brake_lead / cut_in
   - close lead distances (7–12 m) + early brake (0.8–1.8 s)
   - cut-in vehicle slightly faster but still capped.

Outputs per episode:
- <out_csv>                      (per-tick log)
- <out_csv without .csv>_episode_summary.csv
"""

import argparse
import csv
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import carla

LOGGER_VERSION = "v9.2_publication_v3_anti_brake_paradox_2026-04-29"

# --------------------------- Utility ---------------------------

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def clamp01(x):
    return clamp(x, 0.0, 1.0)

def vec_norm(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def now_s():
    return time.time()

def cleanup(actors):
    """Destroy actors safely (works even if actor is None or already destroyed)."""
    for a in actors:
        try:
            if a is not None:
                a.destroy()
        except Exception:
            pass


# --------------- Perception sensor-noise helper ---------------

def add_gaussian_noise(value, sigma, rng):
    """
    Add zero-mean Gaussian noise (stddev=sigma) to a scalar measurement.
    Returns original if value/sigma/rng invalid. Used to model perception
    sensor uncertainty BEFORE values reach the controller. Ground truth is
    always logged separately for analysis.
    """
    if value is None or sigma is None or float(sigma) <= 0.0 or rng is None:
        return value
    try:
        return float(value) + float(rng.normal(0.0, float(sigma)))
    except Exception:
        return value


# ----------------------- Risk definition -----------------------

@dataclass
class RiskConfig:
    v_max: float = 28.0          # m/s
    a_abs_max: float = 7.0       # m/s^2
    jerk_abs_max: float = 12.0   # m/s^3
    lane_offset_max: float = 1.5 # m

    ttc_min: float = 0.5
    ttc_max: float = 8.0
    headway_min: float = 2.0
    headway_max: float = 60.0
    density_max: float = 25.0

    # weights
    w_speed: float = 0.08
    w_accel: float = 0.10
    w_jerk: float = 0.10
    w_steer_var: float = 0.08
    w_lane_off: float = 0.12
    w_ttc: float = 0.25
    w_headway: float = 0.15
    w_density: float = 0.06

    softness: float = 0.12


def normalize_speed(v, cfg):
    return clamp01(v / cfg.v_max)

def normalize_accel(a, cfg):
    return clamp01(abs(a) / cfg.a_abs_max)

def normalize_jerk(j, cfg):
    return clamp01(abs(j) / cfg.jerk_abs_max)

def normalize_steer_var(steer_hist):
    if len(steer_hist) < 2:
        return 0.0
    s = float(np.std(np.array(steer_hist)))
    return clamp01(s / 0.35)

def normalize_lane_offset(off, cfg):
    return clamp01(abs(off) / cfg.lane_offset_max)

def normalize_ttc(ttc, cfg):
    if ttc is None or math.isinf(ttc) or (isinstance(ttc, float) and math.isnan(ttc)):
        return 0.0
    ttc = clamp(ttc, cfg.ttc_min, cfg.ttc_max)
    return clamp01((cfg.ttc_max - ttc) / (cfg.ttc_max - cfg.ttc_min))

def normalize_headway(d, cfg):
    if d is None or math.isinf(d) or (isinstance(d, float) and math.isnan(d)):
        return 0.0
    d = clamp(d, cfg.headway_min, cfg.headway_max)
    return clamp01((cfg.headway_max - d) / (cfg.headway_max - cfg.headway_min))

def normalize_density(n, cfg):
    return clamp01(n / cfg.density_max)

def compute_risk_score(v_mps, a_mps2, j_mps3, steer_hist, lane_off_m,
                       ttc_s, headway_m, density_count, cfg):
    x_speed = normalize_speed(v_mps, cfg)
    x_accel = normalize_accel(a_mps2, cfg)
    x_jerk = normalize_jerk(j_mps3, cfg)
    x_steer = normalize_steer_var(steer_hist)
    x_lane = normalize_lane_offset(lane_off_m, cfg)
    x_ttc = normalize_ttc(ttc_s, cfg)
    x_head = normalize_headway(headway_m, cfg)
    x_den = normalize_density(density_count, cfg)

    R =(cfg.w_speed*x_speed + cfg.w_accel*x_accel + cfg.w_jerk*x_jerk +
         cfg.w_steer_var*x_steer + cfg.w_lane_off*x_lane + cfg.w_ttc*x_ttc +
         cfg.w_headway*x_head + cfg.w_density*x_den)
    R = clamp01(R)

    feats = dict(
        x_speed=x_speed, x_accel=x_accel, x_jerk=x_jerk,
        x_steer_var=x_steer, x_lane_offset=x_lane,
        x_ttc=x_ttc, x_headway=x_head, x_density=x_den
    )
    return R, feats


# ---------------------- Weather / TM -----------------------

def set_weather(world, kind):
    w = carla.WeatherParameters.ClearNoon
    if kind == "rain_fog":
        w = carla.WeatherParameters(
            cloudiness=85.0, precipitation=75.0, precipitation_deposits=65.0,
            wetness=85.0, fog_density=45.0, fog_distance=25.0, sun_altitude_angle=20.0
        )
    world.set_weather(w)
    return w

def tm_try_call(tm, fname, *args):
    try:
        fn = getattr(tm, fname)
        fn(*args)
        return True
    except Exception:
        return False

def _pct_to_speed_mult(pct_speed_diff):
    # TM: positive => slower by pct; negative => faster by abs(pct)
    return 1.0 - float(pct_speed_diff) / 100.0

def _speed_mult_to_pct(mult):
    # mult = desired / limit
    # pct = 100*(1 - mult)
    return 100.0 * (1.0 - float(mult))

def speed_cap_pct_min(vehicle, cap_kmh):
    """Return minimum slowdown percentage to keep speed <= cap_kmh, based on current speed limit."""
    if cap_kmh is None:
        return None
    try:
        lim = float(vehicle.get_speed_limit())  # km/h
    except Exception:
        return None
    if lim <= 1e-3:
        return None
    cap = float(cap_kmh)
    if cap <= 1e-3:
        return None
    if cap >= lim:
        return 0.0
    mult = cap / lim  # <= 1
    return float(clamp(_speed_mult_to_pct(mult), 0.0, 95.0))

def apply_tm_profile(tm, risk_profile, hybrid_physics=True):
    """
    "aggressive" here means: riskier interactions but still realistic, not racing.
    """
    # Hybrid physics improves stability with many actors, but can look less smooth.
    # For maximum smoothness, set --hybrid_physics 0 (default).
    if hybrid_physics:
        tm_try_call(tm, "set_hybrid_physics_mode", True)
        tm_try_call(tm, "set_hybrid_physics_radius", 70.0)
    else:
        tm_try_call(tm, "set_hybrid_physics_mode", False)

    # Global following distance (small -> riskier but can jitter if too small)
# -------------------------------
# Traffic Manager configuration
# -------------------------------

    if risk_profile == "aggressive":

        # Risky enough for comparison, but still physically believable.
        tm_try_call(tm, "set_global_distance_to_leading_vehicle", 1.2)
        tm_try_call(tm, "global_percentage_speed_difference", 8.0)
        tm_try_call(tm, "random_left_lanechange_percentage", 2.0)
        tm_try_call(tm, "random_right_lanechange_percentage", 2.0)
        tm_try_call(tm, "ignore_lights_percentage", 10.0)
        tm_try_call(tm, "ignore_signs_percentage", 8.0)

    else:

        # Safer mode
        tm_try_call(tm, "set_global_distance_to_leading_vehicle", 1.8)

        tm_try_call(tm, "global_percentage_speed_difference", 12.0)

        tm_try_call(tm, "random_left_lanechange_percentage", 1.0)
        tm_try_call(tm, "random_right_lanechange_percentage", 1.0)

        tm_try_call(tm, "ignore_lights_percentage", 0.0)
        tm_try_call(tm, "ignore_signs_percentage", 0.0)
            # Do NOT ignore vehicles (causes chaos)

def _try_load_world(client, town_name):
    """
    Try loading a map by short name (Town03) and common full paths.
    If not found, raise with available map names for debugging.
    """
    candidates = [town_name]
    if not town_name.startswith("/Game/"):
        candidates += ["/Game/Carla/Maps/" + town_name, "/Game/Carla/Maps/" + town_name + "/" + town_name]
    last_err = None
    for t in candidates:
        try:
            return client.load_world(t)
        except RuntimeError as e:
            last_err = e
    # Build a friendly error
    try:
        maps = client.get_available_maps()
        short = []
        for m in maps:
            s = str(m)
            if "Town" in s:
                short.append(s.split("/")[-1])
        short = sorted(set(short))
    except Exception:
        maps = []
        short = []
    msg = "map not found: {}. Tried: {}.".format(town_name, candidates)
    if short:
        msg += " Available short names: {}".format(", ".join(short))
    elif maps:
        msg += " Available maps: {}".format(", ".join([str(x) for x in maps]))
    raise RuntimeError(msg) from last_err


def setup_world(client, town, fps, tm_port, tm_seed, risk_profile, hybrid_physics):
    world = _try_load_world(client, town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(fps)
    # Physics substepping improves smoothness (esp. in crowded traffic)
    try:
        dt = float(settings.fixed_delta_seconds)
        settings.substepping = True
        settings.max_substeps = int(max(8, min(16, math.ceil(dt / 0.005))))
        settings.max_substep_delta_time = float(min(0.01, dt / settings.max_substeps))
    except Exception:
        pass
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(int(tm_seed))

    apply_tm_profile(tm, risk_profile, hybrid_physics=bool(hybrid_physics))
    world.tick()
    return world, tm


# -------------------- Spawn helpers (retry-safe) --------------------

def safe_spawn_actor(world, bp, tf, max_tries=10, jitter_xy=0.8, yaw_jitter=10.0):
    for _ in range(int(max_tries)):
        a = world.try_spawn_actor(bp, tf)
        if a is not None:
            return a
        jtf = carla.Transform(tf.location, tf.rotation)
        jtf.location.x += random.uniform(-jitter_xy, jitter_xy)
        jtf.location.y += random.uniform(-jitter_xy, jitter_xy)
        jtf.rotation.yaw += random.uniform(-yaw_jitter, yaw_jitter)
        tf = jtf
    return None

def _is_good_spawn_point(world, sp, min_clear_m=50.0):
    """
    Reject spawn points that are at junctions or have insufficient clear
    road ahead. This prevents ego from getting trapped at red lights or
    in junction-stops at the start of an episode.
    """
    try:
        m = world.get_map()
        wp = m.get_waypoint(sp.location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        # Reject if spawn waypoint itself is a junction.
        if wp.is_junction:
            return False
        # Walk forward 50 m in 5 m steps; reject if we hit a junction
        # in the first 30 m (gives ego a chance to get up to speed).
        cur = wp
        walked = 0.0
        step = 5.0
        while walked < min_clear_m:
            nxts = cur.next(step)
            if not nxts:
                return False
            cur = nxts[0]
            walked += step
            if cur.is_junction and walked < 30.0:
                return False
        return True
    except Exception:
        return False


def spawn_ego_with_retries(world, tm, seed, spawn_index, tries=24):
    random.seed(seed)
    bp_lib = world.get_blueprint_library()
    preferred = bp_lib.filter("vehicle.tesla.model3")
    ego_bp = preferred[0] if preferred else random.choice(bp_lib.filter("vehicle.*"))

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in map.")

    # Pre-filter spawn points: keep only those with clear road ahead.
    # Falls back to all points if the filter is too aggressive.
    good_points = [sp for sp in spawn_points if _is_good_spawn_point(world, sp)]
    if len(good_points) < 5:
        good_points = spawn_points  # fallback

    base = int(clamp(spawn_index, 0, len(good_points) - 1))
    for k in range(int(tries)):
        idx = (base + k) % len(good_points)
        sp = good_points[idx]
        ego = safe_spawn_actor(world, ego_bp, sp, max_tries=10, jitter_xy=0.3, yaw_jitter=6.0)
        if ego is not None:
            ego.set_autopilot(True, tm.get_port())
            # Find the actual index in spawn_points so the logged
            # spawn_index field is consistent.
            try:
                actual_idx = spawn_points.index(sp)
            except ValueError:
                actual_idx = idx
            return ego, actual_idx
    raise RuntimeError("Failed to spawn ego after retries (spawn collisions everywhere).")

def tm_set_headway(tm, ego, headway_m):
    headway_m = float(clamp(headway_m, 0.6, 20.0))
    if ego is None:
        tm_try_call(tm, "set_global_distance_to_leading_vehicle", headway_m)
        return
    if not tm_try_call(tm, "set_distance_to_leading_vehicle", ego, headway_m):
        tm_try_call(tm, "distance_to_leading_vehicle", ego, headway_m)

def apply_vehicle_profile(tm, v, risk_profile, traffic_speed_cap_kmh=None):
    """
    Keep traffic close to normal driving. Small variability is enough for realism.
    """
    r = random.random()
    if risk_profile == "aggressive":
        if r < 0.70:
            pct = random.uniform(0.0, 8.0)      # slightly slower than limit
            headway = random.uniform(0.6, 1.2)
        elif r < 0.90:
            pct = random.uniform(-12.0, -5.0)    # slightly faster
            headway = random.uniform(1.2, 1.8)
        else:
            pct = random.uniform(10.0, 20.0)    # clearly slower
            headway = random.uniform(1.8, 2.8)
    else:
        if r < 0.80:
            pct = random.uniform(2.0, 10.0)
            headway = random.uniform(1.8, 2.6)
        else:
            pct = random.uniform(-4.0, 2.0)
            headway = random.uniform(1.5, 2.2)

    cap_pct = speed_cap_pct_min(v, traffic_speed_cap_kmh)
    if cap_pct is not None:
        pct = max(pct, float(cap_pct))

    try:
        pct = float(clamp(pct, -10.0, 40.0))
        tm.vehicle_percentage_speed_difference(v, pct)
    except Exception:
        pass

    tm_set_headway(tm, v, headway)

    tm_try_call(tm, "auto_lane_change", v, False)
    tm_try_call(tm, "keep_right_rule_percentage", v, 80.0)
    tm_try_call(tm, "set_lane_change_safety_factor", v, 1.5)
    tm_try_call(tm, "ignore_lights_percentage", v, float(random.uniform(0.0, 3.0)))
    tm_try_call(tm, "ignore_signs_percentage", v, float(random.uniform(0.0, 3.0)))

def spawn_traffic(world, tm, n, seed, ego,
                  spawn_bias_close=False,
                  ring_min_m=25.0, ring_max_m=85.0,
                  risk_profile="aggressive",
                  traffic_speed_cap_kmh=None):
    random.seed(seed)
    bp_lib = world.get_blueprint_library()
    veh_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    ego_loc = ego.get_transform().location

    if spawn_bias_close:
        cands = []
        for sp in spawn_points:
            d = sp.location.distance(ego_loc)
            if d < float(ring_min_m):
                continue
            if d > float(ring_max_m):
                continue
            cands.append(sp)
        if len(cands) > 10:
            spawn_points = cands
        random.shuffle(spawn_points)

    vehicles = []
    for sp in spawn_points:
        if len(vehicles) >= int(n):
            break
        bp = random.choice(veh_bps)
        v = safe_spawn_actor(world, bp, sp, max_tries=7, jitter_xy=0.35, yaw_jitter=6.0)
        if v is None:
            continue
        v.set_autopilot(True, tm.get_port())
        apply_vehicle_profile(tm, v, risk_profile, traffic_speed_cap_kmh=traffic_speed_cap_kmh)
        vehicles.append(v)
    return vehicles


# -------------------- Scenario actors --------------------

def _vehicle_bp(world):
    bp_lib = world.get_blueprint_library()
    prefer = bp_lib.filter("vehicle.audi.a2")
    bps = bp_lib.filter("vehicle.*")
    return prefer[0] if prefer else random.choice(bps)

def spawn_stopped_vehicle_ahead(world, ego, distance_m):
    m = world.get_map()
    ego_wp = m.get_waypoint(ego.get_transform().location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    nxt = ego_wp.next(float(distance_m))
    if not nxt:
        return None
    tf = nxt[0].transform
    obs = safe_spawn_actor(world, _vehicle_bp(world), tf, max_tries=12, jitter_xy=0.25, yaw_jitter=4.0)
    if obs is None:
        return None
    obs.set_autopilot(False)
    try:
        obs.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    except Exception:
        pass
    return obs

def spawn_lead_vehicle_ahead(world, tm, ego, distance_m, risk_profile, traffic_speed_cap_kmh=None):
    """
    Spawn a lead vehicle in the SAME lane ahead of the ego.

    v3: lead drives at MODERATE speed (~10% slower than limit) so the ego,
    which is configured slightly faster than limit, naturally closes on it
    over ~5-8 seconds of warmup. The lead also ignores traffic lights to
    avoid mid-warmup stops that confuse the ego's autopilot.

    Earlier versions made the lead very slow (20-28% pct), which actually
    made things worse: the ego's autopilot saw the slow lead, slammed the
    brakes, and stayed stationary the entire episode.
    """
    m = world.get_map()
    ego_wp = m.get_waypoint(ego.get_transform().location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    nxt = ego_wp.next(float(distance_m))
    if not nxt:
        return None
    tf = nxt[0].transform
    lead = safe_spawn_actor(world, _vehicle_bp(world), tf, max_tries=12, jitter_xy=0.25, yaw_jitter=4.0)
    if lead is None:
        return None
    lead.set_autopilot(True, tm.get_port())

    # Lead drives moderately, NOT very slowly. Ego (configured -2 to -6 pct,
    # i.e. faster than limit) catches up gradually.
    try:
        pct = float(random.uniform(8.0, 14.0))
        cap_pct = speed_cap_pct_min(lead, traffic_speed_cap_kmh)
        if cap_pct is not None:
            pct = max(pct, float(cap_pct))
        tm.vehicle_percentage_speed_difference(lead, float(clamp(pct, 0.0, 25.0)))
    except Exception:
        pass

    # Tight headway, no auto-lane change, ignore lights to avoid mid-warmup stops.
    tm_set_headway(tm, lead, 1.5)
    tm_try_call(tm, "auto_lane_change", lead, False)
    tm_try_call(tm, "keep_right_rule_percentage", lead, 0.0)
    tm_try_call(tm, "ignore_lights_percentage", lead, 100.0)
    tm_try_call(tm, "ignore_signs_percentage", lead, 100.0)
    return lead

def spawn_cut_in_vehicle(world, tm, ego, ahead_m, risk_profile, traffic_speed_cap_kmh=None):
    """
    Spawn in adjacent lane slightly ahead of ego.

    Publication-grade fix: cut-in vehicle is set ~5 % faster than ego so it
    is actively passing during the trigger window. Auto-lane-change is
    DISABLED initially so the explicit `force_lane_change` call dominates.
    """
    m = world.get_map()
    ego_wp = m.get_waypoint(ego.get_transform().location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    nxt = ego_wp.next(float(ahead_m))
    if not nxt:
        return None
    wp = nxt[0]
    side = wp.get_left_lane() or wp.get_right_lane()
    if side is None or side.lane_type != carla.LaneType.Driving:
        return None

    v = safe_spawn_actor(world, _vehicle_bp(world), side.transform, max_tries=14, jitter_xy=0.25, yaw_jitter=5.0)
    if v is None:
        return None

    v.set_autopilot(True, tm.get_port())

    # Slightly faster than ego so the cut-in is geometrically meaningful.
    try:
        pct = float(random.uniform(-8.0, -3.0))
        cap_pct = speed_cap_pct_min(v, traffic_speed_cap_kmh)
        if cap_pct is not None:
            pct = max(pct, float(cap_pct))
        tm.vehicle_percentage_speed_difference(v, float(clamp(pct, -12.0, 0.0)))
    except Exception:
        pass

    tm_set_headway(tm, v, 1.0)
    tm_try_call(tm, "auto_lane_change", v, False)
    tm_try_call(tm, "ignore_lights_percentage", v, 0.0)
    tm_try_call(tm, "ignore_signs_percentage", v, 0.0)
    return v


def spawn_tailgater_behind(world, tm, ego, distance_m, risk_profile, traffic_speed_cap_kmh=None, speedup_pct=-8.0):
    """
    Spawn a vehicle behind ego in the same lane at distance_m.
    Make it slightly faster (negative pct speed difference) so it closes in.
    """
    m = world.get_map()
    ego_wp = m.get_waypoint(ego.get_transform().location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    prev = ego_wp.previous(float(distance_m))
    if not prev:
        return None
    tf = prev[0].transform
    tg = safe_spawn_actor(world, _vehicle_bp(world), tf, max_tries=14, jitter_xy=0.25, yaw_jitter=4.0)
    if tg is None:
        return None
    tg.set_autopilot(True, tm.get_port())
    apply_vehicle_profile(tm, tg, risk_profile, traffic_speed_cap_kmh=traffic_speed_cap_kmh)

    try:
        pct = float(max(-10.0, min(-1.0, speedup_pct)))
        cap_pct = speed_cap_pct_min(tg, traffic_speed_cap_kmh)
        if cap_pct is not None:
            pct = max(pct, float(cap_pct))
        pct = float(clamp(pct, -12.0, -1.0))
        tm.vehicle_percentage_speed_difference(tg, pct)
    except Exception:
        pass
    # Very small headway to encourage closing in
    tm_set_headway(tm, tg, float(clamp(distance_m * 0.10, 0.8, 1.5)))
    return tg





# -------------------- Metrics helpers --------------------------

def lane_offset_m(world, vehicle):
    m = world.get_map()
    loc = vehicle.get_transform().location
    wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    dx = loc.x - wp.transform.location.x
    dy = loc.y - wp.transform.location.y
    return math.sqrt(dx*dx + dy*dy)

def compute_ttc_headway_to_actor(world, ego, target, max_dist=150.0):
    if target is None:
        return None, None, None, False
    try:
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        tgt_loc = target.get_transform().location
    except Exception:
        return None, None, None, False

    d = tgt_loc.distance(ego_loc)
    if d <= 0.01 or d > float(max_dist):
        return None, None, None, False

    rel_vec = carla.Vector3D(tgt_loc.x - ego_loc.x, tgt_loc.y - ego_loc.y, tgt_loc.z - ego_loc.z)
    ego_fwd = ego_tf.get_forward_vector()
    ahead_dot = rel_vec.x*ego_fwd.x + rel_vec.y*ego_fwd.y + rel_vec.z*ego_fwd.z
    if ahead_dot <= 0.0:
        return None, None, None, False

    inv = 1.0 / max(1e-6, math.sqrt(rel_vec.x*rel_vec.x + rel_vec.y*rel_vec.y + rel_vec.z*rel_vec.z))
    los = carla.Vector3D(rel_vec.x*inv, rel_vec.y*inv, rel_vec.z*inv)

    v_ego = ego.get_velocity()
    try:
        v_tgt = target.get_velocity()
    except Exception:
        v_tgt = carla.Vector3D(0.0, 0.0, 0.0)

    v_ego_los = v_ego.x*los.x + v_ego.y*los.y + v_ego.z*los.z
    v_tgt_los = v_tgt.x*los.x + v_tgt.y*los.y + v_tgt.z*los.z
    rel_speed = float(v_ego_los - v_tgt_los)

    # Stationary-target synthesis: stopped vehicles are still threats if ego
    # is approaching. Use ego's projected speed as the closing rate.
    tgt_speed_sq = v_tgt.x*v_tgt.x + v_tgt.y*v_tgt.y + v_tgt.z*v_tgt.z
    if tgt_speed_sq < 0.25 and v_ego_los > 0.5:
        rel_speed = float(v_ego_los)

    headway = float(d)
    if rel_speed <= 0.05:
        return None, headway, rel_speed, True
    return headway / rel_speed, headway, rel_speed, True


def compute_ttc_to_nearest_ahead_same_lane(world, ego, vehicles, max_dist=150.0, max_ttc_s=10.0):
    """
    Robust forward-cone TTC computation (v2).

    The strict road_id+lane_id filter was rejecting valid in-front targets
    in roughly 60% of episodes (CARLA's lane IDs are unreliable around
    junctions, lane-change zones, and curved roads). This v2 replaces the
    waypoint-based lane filter with a GEOMETRIC CONE filter:

        - target must be ahead of ego (forward dot > 0)
        - target must be within `lateral_offset_max` m of the ego's
          forward axis (default 4.0 m -> covers ego lane + half of each
          adjacent lane in typical urban geometry)
        - target must be within `max_dist` m

    For stationary targets, ego approach speed is used as the closing rate.

    Returns:
        (ttc_s, headway_m, rel_speed_mps, target_actor, valid)
    """
    if ego is None or vehicles is None:
        return None, None, None, None, False

    try:
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
    except Exception:
        return None, None, None, None, False

    # Lateral half-width of the forward cone in metres. 4 m comfortably
    # spans the ego's lane (~3.5 m) plus a small safety margin, so cut-in
    # vehicles partway through a lane change are still captured. Targets
    # in non-adjacent lanes are excluded geometrically.
    lateral_offset_max = 4.0

    best_ttc = None
    best_hw = None
    best_rel = None
    best_tgt = None

    for v in vehicles:
        if v is None:
            continue
        try:
            if v.id == ego.id:
                continue
        except Exception:
            pass

        try:
            tgt_tf = v.get_transform()
            tgt_loc = tgt_tf.location
        except Exception:
            continue

        d = float(tgt_loc.distance(ego_loc))
        if d <= 0.01 or d > float(max_dist):
            continue

        # Geometric forward-cone filter.
        rel_vec = carla.Vector3D(tgt_loc.x - ego_loc.x, tgt_loc.y - ego_loc.y, tgt_loc.z - ego_loc.z)
        ahead_dot = rel_vec.x * ego_fwd.x + rel_vec.y * ego_fwd.y + rel_vec.z * ego_fwd.z
        if ahead_dot <= 0.5:  # at least 0.5 m ahead in forward axis
            continue

        # Lateral distance from ego's forward line. Compute using cross-product
        # magnitude in the horizontal plane (z is small for ground vehicles).
        lateral = abs(rel_vec.x * ego_fwd.y - rel_vec.y * ego_fwd.x)
        if lateral > lateral_offset_max:
            continue

        inv = 1.0 / max(1e-6, math.sqrt(rel_vec.x * rel_vec.x + rel_vec.y * rel_vec.y + rel_vec.z * rel_vec.z))
        los = carla.Vector3D(rel_vec.x * inv, rel_vec.y * inv, rel_vec.z * inv)

        try:
            v_ego = ego.get_velocity()
            v_tgt = v.get_velocity()
        except Exception:
            continue

        v_ego_los = v_ego.x * los.x + v_ego.y * los.y + v_ego.z * los.z
        v_tgt_los = v_tgt.x * los.x + v_tgt.y * los.y + v_tgt.z * los.z
        rel_speed = float(v_ego_los - v_tgt_los)

        # Stationary-target synthesis: if target is essentially still and
        # ego is approaching, use ego's projected speed as the closing rate.
        tgt_speed = math.sqrt(v_tgt.x**2 + v_tgt.y**2 + v_tgt.z**2)
        if tgt_speed < 0.5 and v_ego_los > 0.5:
            rel_speed = float(v_ego_los)

        if rel_speed <= 0.05:
            continue

        ttc = float(d / rel_speed)

        if (not math.isfinite(ttc)) or ttc <= 0.0 or ttc > float(max_ttc_s):
            continue

        if best_ttc is None or ttc < best_ttc:
            best_ttc = float(ttc)
            best_hw = float(d)
            best_rel = float(rel_speed)
            best_tgt = v

    if best_ttc is None:
        return None, None, None, None, False

    return best_ttc, best_hw, best_rel, best_tgt, True

def traffic_density(ego, vehicles, radius=35.0):
    ego_loc = ego.get_transform().location
    c = 0
    for v in vehicles:
        if v.id == ego.id:
            continue
        try:
            if v.get_transform().location.distance(ego_loc) <= float(radius):
                c += 1
        except Exception:
            pass
    return c


# -------------------- Controllers -----------------------

def apply_ttc_baseline(tm, ego, ttc, thresh_s, slow_pct, headway_m, ego_speed_cap_kmh=None):
    cap_pct = speed_cap_pct_min(ego, ego_speed_cap_kmh)
    cap_pct = 0.0 if cap_pct is None else float(cap_pct)

    if ttc is not None and (not math.isinf(ttc)) and float(ttc) < float(thresh_s):
        pct = max(float(slow_pct), cap_pct)
        try:
            tm.vehicle_percentage_speed_difference(ego, pct)
        except Exception:
            pass
        tm_set_headway(tm, ego, float(headway_m))
        return pct
    else:
        # normal, but still obey speed cap
        try:
            tm.vehicle_percentage_speed_difference(ego, cap_pct)
        except Exception:
            pass
        return cap_pct


# -------------------- Visual debug overlay -----------------------

def debug_draw_overlay(world, ego, lead, R, B, M, u, ttc, headway, mode):
    loc = ego.get_transform().location + carla.Location(z=2.2)
    near = ego.get_transform().location + carla.Location(z=1.0)

    danger = (R > B)
    col = carla.Color(255, 40, 40) if danger else carla.Color(40, 255, 40)

    ttc_txt = "inf" if (ttc is None or math.isinf(ttc)) else "{:.2f}".format(ttc)
    head_txt = "na" if headway is None else "{:.1f}".format(headway)
    text = "{} | R={:.3f} B={:.3f} M={:.3f} u={:.2f} | TTC={}s HW={}m".format(
        mode, R, B, M, u, ttc_txt, head_txt
    )
    world.debug.draw_string(loc, text, draw_shadow=True, color=col, life_time=0.12)

    if lead is not None and headway is not None:
        lead_loc = lead.get_transform().location + carla.Location(z=1.0)
        world.debug.draw_line(near, lead_loc, thickness=0.08,
                              color=carla.Color(80, 160, 255), life_time=0.12)


# ------------------------- Episode run ----------------------------

def run_episode(args):
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    ego = None
    traffic = []
    scenario_actors = []
    vehicles = []

    try:
        print("[LOGGER] {} fps={} hybrid_physics={} ego_cap_kmh={} traffic_cap_kmh={}".format(
            LOGGER_VERSION, args.fps, args.hybrid_physics, args.ego_speed_cap_kmh, args.traffic_speed_cap_kmh))
        client = carla.Client(args.host, args.port)
        client.set_timeout(float(args.client_timeout_s))

        t0 = now_s()
        while True:
            try:
                world, tm = setup_world(client, args.town, args.fps, args.tm_port, args.tm_seed, args.risk_profile, args.hybrid_physics)
                break
            except RuntimeError:
                if now_s() - t0 > float(args.connect_retry_s):
                    raise
                time.sleep(1.0)

        set_weather(world, args.weather)

        # global headway baseline
        tm_set_headway(tm, ego=None, headway_m=float(args.global_headway_m))

        # spawn ego
        ego, used_spawn_index = spawn_ego_with_retries(world, tm, seed=args.seed,
                                                       spawn_index=args.spawn_index, tries=26)
        args.spawn_index = int(used_spawn_index)

        # v3: configure ego TM properties to keep it moving through the
        # episode. Forward-collision evaluation does not depend on traffic
        # light compliance; allowing red-light stops only injects variance.
        try:
            tm_try_call(tm, "ignore_lights_percentage", ego, 100.0)
            tm_try_call(tm, "ignore_signs_percentage", ego, 100.0)
            tm_try_call(tm, "auto_lane_change", ego, False)
            tm_try_call(tm, "keep_right_rule_percentage", ego, 0.0)
        except Exception:
            pass

        vehicles = [ego]

        # spawn traffic (locally packed but stable defaults)
        traffic = spawn_traffic(
            world, tm, args.traffic, args.seed + 999, ego,
            spawn_bias_close=args.spawn_bias_close,
            ring_min_m=float(args.spawn_ring_min_m),
            ring_max_m=float(args.spawn_ring_max_m),
            risk_profile=args.risk_profile,
            traffic_speed_cap_kmh=args.traffic_speed_cap_kmh
        )
        vehicles.extend(traffic)

        # scenario setup
        lead_vehicle = None
        cutin_vehicle = None
        stopped_vehicle = None
        tailgater_vehicle = None
        scenario_valid = 0

        # If scenario=mixed, sample 1..n_events from a pool (deterministic by seed)
        scenario_plan = [args.scenario]
        if args.scenario == "mixed":
            random.seed(int(args.seed) + 777)

            # ===============================
            # FIXED SCENARIO SAMPLING
            # ===============================

            scenarios = ["stopped_vehicle","sudden_brake_lead","cut_in","tailgater"]

            # Equal probability (scientifically clean)
            p = [1.0 / len(scenarios)] * len(scenarios)

            scenario_plan = random.choices(
                scenarios,
                weights=p,
                k=int(args.n_events)
            )

            
            scenario_plan = [s for s in scenario_plan if s != "none"]

        if "stopped_vehicle" in scenario_plan:
            stopped_vehicle = spawn_stopped_vehicle_ahead(world, ego, distance_m=args.lead_ahead_m)
            if stopped_vehicle is not None:
                scenario_actors.append(stopped_vehicle)
                vehicles.append(stopped_vehicle)
                scenario_valid = 1

        if "sudden_brake_lead" in scenario_plan or ("cut_in" in scenario_plan) or ("tailgater" in scenario_plan):
            lead_vehicle = spawn_lead_vehicle_ahead(world, tm, ego, distance_m=args.lead_ahead_m,
                                                    risk_profile=args.risk_profile,
                                                    traffic_speed_cap_kmh=args.traffic_speed_cap_kmh)
            if lead_vehicle is not None:
                scenario_actors.append(lead_vehicle)
                vehicles.append(lead_vehicle)
                scenario_valid = 1

        if "cut_in" in scenario_plan:
            cutin_vehicle = spawn_cut_in_vehicle(world, tm, ego, ahead_m=max(4.0, args.lead_ahead_m),
                                                 risk_profile=args.risk_profile,
                                                 traffic_speed_cap_kmh=args.traffic_speed_cap_kmh)
            if cutin_vehicle is not None:
                scenario_actors.append(cutin_vehicle)
                vehicles.append(cutin_vehicle)
                scenario_valid = 1

        if "tailgater" in scenario_plan or args.scenario == "tailgater":
            tailgater_vehicle = spawn_tailgater_behind(world, tm, ego, distance_m=args.tailgate_dist_m,
                                                        risk_profile=args.risk_profile,
                                                        traffic_speed_cap_kmh=args.traffic_speed_cap_kmh,
                                                        speedup_pct=args.tailgate_speedup_pct)
            if tailgater_vehicle is not None:
                scenario_actors.append(tailgater_vehicle)
                vehicles.append(tailgater_vehicle)
                scenario_valid = 1

        cfg = RiskConfig()
        if args.softness_override is not None:
            cfg.softness = float(args.softness_override)

        B = float(args.B_override) if args.B_override is not None else (0.65 if args.profile == "sensitive" else 0.72)

        # u smoothing and speed% rate limit
        ema_u = 0.0
        ema_alpha_u = None
        if args.innov_ema_tau_s > 0:
            dt = 1.0 / float(args.fps)
            ema_alpha_u = dt / (float(args.innov_ema_tau_s) + dt)

        last_applied_pct = 0.0

        # Intervention accounting
        intervention_ticks = 0
        sum_u_smooth = 0.0
        max_u_smooth = 0.0
        sum_pct = 0.0
        # Rate-limit ego TM speed % changes (smooth acceleration / no jitter)
        max_delta_pct_per_tick_ego = float(args.ego_max_delta_pct_per_s) / max(1.0, float(args.fps))

        # Speed EMA for comfort derivatives
        v_ema = None
        ema_alpha_v = None
        if args.speed_ema_tau_s > 0:
            dt = 1.0 / float(args.fps)
            ema_alpha_v = dt / (float(args.speed_ema_tau_s) + dt)

        steer_hist = deque(maxlen=int(2.0 * args.fps))
        v_hist_raw = deque(maxlen=3)
        a_hist_raw = deque(maxlen=3)

        v_hist_f = deque(maxlen=3)
        a_hist_f = deque(maxlen=3)

        # accumulators
        min_ttc = float("inf")
        min_headway = float("inf")

        

        max_abs_accel = 0.0
        max_abs_jerk = 0.0
        max_abs_accel_f = 0.0
        max_abs_jerk_f = 0.0

        lane_invasion_total = 0
        lane_invasion_solid = 0
        lane_invasion_broken = 0

        collision_events = 0
        collision_any = 0
        collision_impulse_sum = 0.0
        last_collision_time = -1e9
        sim_time_now = 0.0
        # accumulators
        ttc_valid_ticks = 0
        headway_valid_ticks = 0
        ttc_lt_2_ticks = 0
        ttc_lt_3_ticks = 0
        ttc_any_lt_3_ticks = 0
        ttc_any_valid_ticks = 0
        min_ttc_any = float('inf')
        valid_ttc_count = 0
        invalid_ttc_count = 0

        # Mission-efficiency accumulators
        prev_ego_loc_eff = None
        distance_travelled_m = 0.0
        sum_speed_mps = 0.0
        sum_abs_speed_error_mps = 0.0

        # Sensors
        bp_lib = world.get_blueprint_library()
        collision_bp = bp_lib.find("sensor.other.collision")
        invasion_bp = bp_lib.find("sensor.other.lane_invasion")

        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        invasion_sensor = world.spawn_actor(invasion_bp, carla.Transform(), attach_to=ego)
        scenario_actors.extend([collision_sensor, invasion_sensor])

        def _on_collision(ev):
            nonlocal collision_events, collision_impulse_sum, last_collision_time, sim_time_now, collision_any
            if (sim_time_now - last_collision_time) > 0.5:
                collision_events += 1
                collision_any = 1
                last_collision_time = sim_time_now
            try:
                collision_impulse_sum += float(ev.normal_impulse.x*ev.normal_impulse.x +
                                               ev.normal_impulse.y*ev.normal_impulse.y +
                                               ev.normal_impulse.z*ev.normal_impulse.z) ** 0.5
            except Exception:
                pass

        def _on_invasion(ev):
            nonlocal lane_invasion_total, lane_invasion_solid, lane_invasion_broken
            lane_invasion_total += 1
            try:
                for m in ev.crossed_lane_markings:
                    tname = str(m.type)
                    if "Solid" in tname:
                        lane_invasion_solid += 1
                    else:
                        lane_invasion_broken += 1
            except Exception:
                pass

        collision_sensor.listen(_on_collision)
        invasion_sensor.listen(_on_invasion)

        # ---------- Warm-up ----------
        warmup_ticks = int(float(args.warmup_s) * float(args.fps))
        for _ in range(max(0, warmup_ticks)):
            world.tick()

        fieldnames = [
            "episode_id","mode","profile","scenario","seed","spawn_index",
            "traffic","weather","ego_speedup_pct","global_headway_m",
            "lead_ahead_m","lead_brake_time_s","lead_brake_hold_s","cut_in_time_s","cut_in_force","tailgate_dist_m","tailgate_speedup_pct","n_events","scenario_valid",
            "ttc_thresh","ttc_slow_pct","ttc_headway_m","near_miss_ttc_s","ttc_metric_max_s",
            "B","softness",
            "risk_profile","spawn_bias_close","spawn_ring_min_m","spawn_ring_max_m",
            "ego_speed_cap_kmh","traffic_speed_cap_kmh","warmup_s",
            "sim_time_s","tick",
            "speed_mps","speed_mps_f",
            "accel_mps2","jerk_mps3",
            "accel_mps2_f","jerk_mps3_f",
            "steer","lane_offset_m",
            "ttc_s","headway_m","rel_speed_mps",
            "ttc_focus_s","headway_focus_m","rel_speed_focus_mps",
            "ttc_any_s","headway_any_m","rel_speed_any_mps","ttc_used_s",
            # Perceived (noisy) values fed to the controller
            "ttc_perceived_s","headway_perceived_m","rel_speed_perceived_mps",
            # Realism diagnostics
            "sensor_pos_noise_m","sensor_vel_noise_mps","control_latency_s",
            "control_compute_ms",
            "density_count",
            "R","M","u","u_smooth","pct_speed_diff_applied","pct_actuated","intervention_tick",
            "x_speed","x_accel","x_jerk","x_steer_var","x_lane_offset","x_ttc","x_headway","x_density",
        ]

        world.tick()
        start_sim = float(world.get_snapshot().timestamp.elapsed_seconds)






        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()

            num_ticks = int(float(args.duration) * float(args.fps))

            # Reaction delay applied as ego control freeze for first N ticks.
            # brake_tick and cut_in_tick use the raw scenario timing; the freeze
            # block below independently models the 0.3 s ego response latency.
            brake_tick = max(1, int(float(args.lead_brake_time_s) * float(args.fps)))
            cut_in_tick = max(1, int(float(args.cut_in_time_s) * float(args.fps)))

            lead_brake_active_until = -1

            # reaction delay state
            reaction_delay_ticks = int(0.3 * float(args.fps))
            prev_control = carla.VehicleControl()

            # ----------------------------------------------------------
            # PUBLICATION REALISM PATCHES (initialised once per episode)
            # ----------------------------------------------------------
            perception_rng = np.random.default_rng(int(args.seed) + 4242)
            sensor_pos_sigma = float(args.sensor_pos_noise_m)
            sensor_vel_sigma = float(args.sensor_vel_noise_mps)

            latency_ticks = max(0, int(round(float(args.control_latency_s) * float(args.fps))))
            latency_buffer = deque(maxlen=latency_ticks + 1)

            control_compute_times_ms = []
            measure_compute = bool(int(args.measure_compute_time))

            for k in range(num_ticks):
                world.tick()
                snap = world.get_snapshot()
                sim_time = float(snap.timestamp.elapsed_seconds) - start_sim
                sim_time_now = sim_time

                # ----------------------------------------
                # Ego reaction delay
                # ----------------------------------------
                if k < reaction_delay_ticks:
                    try:
                        ego.apply_control(prev_control)
                    except Exception:
                        pass

                # ----------------------------------------
                # Trigger strong lead braking event
                # ----------------------------------------
                if (
                    args.scenario in ("sudden_brake_lead", "cut_in", "mixed")
                    and (lead_vehicle is not None)
                    and (k == brake_tick)
                ):
                    lead_brake_active_until = k + int(float(args.lead_brake_hold_s) * float(args.fps))

                # ----------------------------------------
                # Apply strong lead braking while active
                # ----------------------------------------
                if (
                    lead_vehicle is not None
                    and lead_brake_active_until >= 0
                    and k <= lead_brake_active_until
                ):
                    try:
                        lead_vehicle.set_autopilot(False)
                        lead_vehicle.apply_control(
                            carla.VehicleControl(
                                throttle=0.0,
                                brake=1.0,
                                hand_brake=bool(args.lead_brake_handbrake)
                            )
                        )
                        lead_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                        lead_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                    except Exception:
                        pass

                # ----------------------------------------
                # Force cut-in at chosen time
                # ----------------------------------------
                if (
                    args.scenario in ("cut_in", "mixed")
                    and (cutin_vehicle is not None)
                    and (k == cut_in_tick)
                    and bool(args.cut_in_force)
                ):
                    if not tm_try_call(tm, "force_lane_change", cutin_vehicle, True):
                        tm_try_call(tm, "force_lane_change", cutin_vehicle, False)

                # ----------------------------------------
                # Re-enable lead vehicle after brake window
                # ----------------------------------------
                if (
                    lead_vehicle is not None
                    and lead_brake_active_until >= 0
                    and k == (lead_brake_active_until + 1)
                ):
                    try:
                        lead_vehicle.set_autopilot(True, tm.get_port())
                        apply_vehicle_profile(
                            tm,
                            lead_vehicle,
                            args.risk_profile,
                            traffic_speed_cap_kmh=args.traffic_speed_cap_kmh
                        )
                    except Exception:
                        pass

                # ----------------------------------------
                # Save previous ego control
                # ----------------------------------------
                try:
                    prev_control = ego.get_control()
                except Exception:
                    pass

                v_raw = vec_norm(ego.get_velocity())

                # Mission-efficiency tracking
                try:
                    ego_loc_eff = ego.get_transform().location
                    if prev_ego_loc_eff is not None:
                        step_dist = float(ego_loc_eff.distance(prev_ego_loc_eff))
                        if math.isfinite(step_dist) and step_dist < 20.0:
                            distance_travelled_m += step_dist
                    prev_ego_loc_eff = ego_loc_eff
                except Exception:
                    pass
                sum_speed_mps += float(v_raw)
                target_speed_mps = (float(args.ego_speed_cap_kmh) / 3.6) if args.ego_speed_cap_kmh is not None else max(float(v_raw), 1e-6)
                sum_abs_speed_error_mps += abs(float(target_speed_mps) - float(v_raw))

                v_hist_raw.append(v_raw)
                a_raw = (v_hist_raw[-1] - v_hist_raw[-2]) * float(args.fps) if len(v_hist_raw) >= 2 else 0.0
                a_hist_raw.append(a_raw)
                j_raw = (a_hist_raw[-1] - a_hist_raw[-2]) * float(args.fps) if len(a_hist_raw) >= 2 else 0.0

                if v_ema is None:
                    v_ema = v_raw
                else:
                    v_ema = (1.0 - ema_alpha_v) * v_ema + ema_alpha_v * v_raw if ema_alpha_v is not None else v_raw

                v_hist_f.append(float(v_ema))
                a_f = (v_hist_f[-1] - v_hist_f[-2]) * float(args.fps) if len(v_hist_f) >= 2 else 0.0
                a_hist_f.append(float(a_f))
                if len(a_hist_f) >= 2:
                    j_f = (a_hist_f[-1] - a_hist_f[-2]) * float(args.fps)
                else:
                    j_f = 0.0
                j_f = clamp(j_f, -60.0, 60.0)  # m/s^3; 60 is a generous physical cap

                ctrl = ego.get_control()
                steer_hist.append(float(ctrl.steer))
                off = lane_offset_m(world, ego)

                # Focus actor for TTC
                focus = None
                if args.scenario == "stopped_vehicle":
                    focus = stopped_vehicle
                elif args.scenario == "sudden_brake_lead":
                    focus = lead_vehicle
                elif args.scenario == "cut_in":
                    focus = cutin_vehicle if cutin_vehicle is not None else lead_vehicle
                elif args.scenario == "tailgater":
                    focus = lead_vehicle  # TTC still forward-focused
                elif args.scenario == "mixed":
                    focus = cutin_vehicle if cutin_vehicle is not None else (lead_vehicle if lead_vehicle is not None else stopped_vehicle)

                ttc_focus, headway_focus, rel_focus, _ = compute_ttc_headway_to_actor(world, ego, focus, max_dist=150.0)

                # TTC to nearest ahead vehicle in the same lane among ALL vehicles (traffic + scenario actors)
                ttc_any, headway_any, rel_any, tgt_any, _ = compute_ttc_to_nearest_ahead_same_lane(world, ego, vehicles, max_dist=150.0, max_ttc_s=float(args.ttc_metric_max_s))





                # =========================================================
                # FINAL TTC COMPUTATION (physics-first and strict)
                # Use SAME-LANE nearest-ahead TTC only when the ego is closing.
                # No TTC is reported for non-closing, no-lead, infinite, or huge values.
                # =========================================================
                if (
                    ttc_any is not None
                    and rel_any is not None
                    and float(rel_any) > 0.05
                    and math.isfinite(float(ttc_any))
                    and 0.0 < float(ttc_any) <= float(args.ttc_metric_max_s)
                ):
                    ttc_used = float(ttc_any)
                    headway = headway_any
                    rel = rel_any
                else:
                    ttc_used = None
                    headway = None
                    rel = None

                lead_now = tgt_any if tgt_any is not None else focus

                ttc = ttc_used

                # ----------------------------------------------------------
                # PUBLICATION REALISM PATCH: sensor noise on perception inputs
                # ----------------------------------------------------------
                # Ground-truth values (ttc, headway, rel) stay unchanged for
                # safety-metric logging. The CONTROLLER sees only the perceived
                # noisy versions, modelling realistic perception sensors.
                if headway is not None:
                    headway_perc = add_gaussian_noise(headway, sensor_pos_sigma, perception_rng)
                    if headway_perc is not None:
                        headway_perc = max(0.05, float(headway_perc))
                else:
                    headway_perc = None

                if rel is not None:
                    rel_perc = add_gaussian_noise(rel, sensor_vel_sigma, perception_rng)
                else:
                    rel_perc = None

                if (headway_perc is not None) and (rel_perc is not None) and (float(rel_perc) > 0.05):
                    _ttc_p = float(headway_perc) / float(rel_perc)
                    if math.isfinite(_ttc_p) and 0.0 < _ttc_p <= float(args.ttc_metric_max_s):
                        ttc_perc = _ttc_p
                    else:
                        ttc_perc = None
                else:
                    ttc_perc = None

                ttc_for_ctrl = ttc_perc
                headway_for_ctrl = headway_perc

                dens = traffic_density(ego, vehicles, radius=35.0)

                # ---------------- TTC METRICS ----------------
                if headway is not None and headway > 0:
                    headway_valid_ticks += 1
                    min_headway = min(min_headway, float(headway))
                if ttc is not None and (not math.isinf(ttc)) and ttc > 0:
                    ttc_valid_ticks += 1
                    min_ttc = min(min_ttc, float(ttc))

                    if float(ttc) < 2.0:
                        ttc_lt_2_ticks += 1

                    if float(ttc) < float(args.near_miss_ttc_s):
                        ttc_lt_3_ticks += 1

                if ttc is None or math.isinf(ttc) or ttc <= 0 or ttc > 50:
                    invalid_ttc_count += 1
                else:
                    valid_ttc_count += 1
                

                # ---------------- TTC_ANY STRICT METRICS ----------------
                if (
                    ttc_any is not None
                    and rel_any is not None
                    and float(rel_any) > 0.05
                    and math.isfinite(float(ttc_any))
                    and 0.0 < float(ttc_any) <= float(args.ttc_metric_max_s)
                ):
                    ttc_any_valid_ticks += 1
                    min_ttc_any = min(min_ttc_any, float(ttc_any))
                    if float(ttc_any) < float(args.near_miss_ttc_s):
                        ttc_any_lt_3_ticks += 1

                max_abs_accel = max(max_abs_accel, abs(float(a_raw)))
                max_abs_jerk = max(max_abs_jerk, abs(float(j_raw)))
                max_abs_accel_f = max(max_abs_accel_f, abs(float(a_f)))
                max_abs_jerk_f = max(max_abs_jerk_f, abs(float(j_f)))

                # ----------------------------------------------------------
                # CONTROLLER DECISION (timed) — uses PERCEIVED inputs
                # ----------------------------------------------------------
                _t_ctrl_start = time.perf_counter() if measure_compute else None

                R, feats = compute_risk_score(float(v_raw), float(a_f), float(j_f),
                                              steer_hist, float(off),
                                              ttc_for_ctrl, headway_for_ctrl, dens, cfg)
                M = float(B - R)
                u = sigmoid((R - B) / float(cfg.softness))

                pct_target = 0.0
                u_smooth = float(u)

                # Ego nominal speed: apply near-normal speedup_pct but always obey cap.
                cap_pct = speed_cap_pct_min(ego, args.ego_speed_cap_kmh)
                cap_pct = 0.0 if cap_pct is None else float(cap_pct)

                if args.mode == "normal":
                    base_pct = float(args.ego_speedup_pct)
                    target_pct = max(base_pct, cap_pct)
                    pct_target = float(clamp(target_pct,
                                             last_applied_pct - max_delta_pct_per_tick_ego,
                                             last_applied_pct + max_delta_pct_per_tick_ego))

                elif args.mode == "ttc_baseline":
                    if ttc_for_ctrl is not None and (not math.isinf(ttc_for_ctrl)) and float(ttc_for_ctrl) < float(args.ttc_thresh):
                        target_pct = max(float(args.ttc_slow_pct), cap_pct)
                    else:
                        target_pct = cap_pct
                    pct_target = float(clamp(target_pct,
                                             last_applied_pct - max_delta_pct_per_tick_ego,
                                             last_applied_pct + max_delta_pct_per_tick_ego))

                elif args.mode == "innov":
                    u_s = float(u)
                    if ema_alpha_u is not None:
                        ema_u = (1.0 - ema_alpha_u) * ema_u + ema_alpha_u * float(u)
                        u_s = float(ema_u)
                    u_smooth = float(u_s)

                    u_eff = clamp01(u_s) ** float(max(0.25, args.innov_u_gamma))
                    target_pct = max(cap_pct, min(float(args.innov_max_slow_pct) * float(u_eff),
                                                  float(args.innov_max_slow_pct)))
                    max_delta = float(args.innov_max_delta_pct_per_s) / max(1.0, float(args.fps))
                    pct_target = float(clamp(target_pct,
                                             last_applied_pct - max_delta,
                                             last_applied_pct + max_delta))
                    pct_target = float(clamp(pct_target, 0.0, max(5.0, float(args.innov_max_slow_pct))))

                last_applied_pct = pct_target

                control_compute_ms = 0.0
                if measure_compute and (_t_ctrl_start is not None):
                    control_compute_ms = (time.perf_counter() - _t_ctrl_start) * 1000.0
                    control_compute_times_ms.append(float(control_compute_ms))

                # ----------------------------------------------------------
                # ACTUATION through control-latency buffer
                # ----------------------------------------------------------
                latency_buffer.append(float(pct_target))
                if latency_ticks <= 0 or len(latency_buffer) <= latency_ticks:
                    pct_actuated = float(pct_target)
                else:
                    pct_actuated = float(latency_buffer[0])
                pct_applied = float(pct_actuated)  # legacy field name

                if args.mode == "normal":
                    try:
                        tm.vehicle_percentage_speed_difference(ego, pct_actuated)
                    except Exception:
                        pass
                    tm_set_headway(tm, ego, float(args.global_headway_m))
                    tm_try_call(tm, "auto_lane_change", ego, False)

                elif args.mode == "ttc_baseline":
                    try:
                        tm.vehicle_percentage_speed_difference(ego, pct_actuated)
                    except Exception:
                        pass
                    if ttc_for_ctrl is not None and (not math.isinf(ttc_for_ctrl)) and float(ttc_for_ctrl) < float(args.ttc_thresh):
                        tm_set_headway(tm, ego, float(args.ttc_headway_m))
                    else:
                        tm_set_headway(tm, ego, float(args.global_headway_m))
                    tm_try_call(tm, "auto_lane_change", ego, False)

                elif args.mode == "innov":
                    try:
                        tm.vehicle_percentage_speed_difference(ego, pct_actuated)
                    except Exception:
                        pass
                    tm_set_headway(tm, ego, float(args.global_headway_m) + float(args.innov_headway_gain_m) * float(u_smooth))
                    tm_try_call(tm, "auto_lane_change", ego, False)

                # Intervention-tick definition (uses perceived TTC for ttc_baseline)
                intervention_tick = 0
                if args.mode == "innov":
                    if (float(u_smooth) >= float(args.intervene_u_thresh)) or (float(M) < 0.0):
                        intervention_tick = 1
                elif args.mode == "ttc_baseline":
                    if (ttc_for_ctrl is not None) and (not math.isinf(ttc_for_ctrl)) and (float(ttc_for_ctrl) < float(args.ttc_thresh)):
                        intervention_tick = 1

                intervention_ticks += int(intervention_tick)
                sum_u_smooth += float(u_smooth)
                max_u_smooth = max(float(max_u_smooth), float(u_smooth))
                sum_pct += float(pct_applied)

                if args.debug_overlay:
                    debug_draw_overlay(world, ego, lead_now, R=R, B=B, M=M, u=float(u),
                                       ttc=ttc, headway=headway, mode=args.mode)

                row = {
                    "episode_id": args.episode_id,
                    "mode": args.mode,
                    "profile": args.profile,
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "spawn_index": args.spawn_index,
                    "traffic": args.traffic,
                    "weather": args.weather,
                    "ego_speedup_pct": args.ego_speedup_pct,
                    "global_headway_m": args.global_headway_m,
                    "lead_ahead_m": args.lead_ahead_m,
                    "lead_brake_time_s": args.lead_brake_time_s,
                    "lead_brake_hold_s": float(args.lead_brake_hold_s),
                    "cut_in_time_s": float(args.cut_in_time_s),
                    "cut_in_force": int(bool(args.cut_in_force)),
                    "tailgate_dist_m": float(args.tailgate_dist_m),
                    "tailgate_speedup_pct": float(args.tailgate_speedup_pct),
                    "n_events": int(args.n_events),
                    "scenario_valid": int(scenario_valid),

                    "ttc_thresh": args.ttc_thresh,
                    "ttc_slow_pct": args.ttc_slow_pct,
                    "ttc_headway_m": args.ttc_headway_m,
                    "near_miss_ttc_s": float(args.near_miss_ttc_s),
                    "ttc_metric_max_s": float(args.ttc_metric_max_s),

                    "B": B,
                    "softness": cfg.softness,

                    "risk_profile": args.risk_profile,
                    "spawn_bias_close": int(args.spawn_bias_close),
                    "spawn_ring_min_m": float(args.spawn_ring_min_m),
                    "spawn_ring_max_m": float(args.spawn_ring_max_m),

                    "ego_speed_cap_kmh": (None if args.ego_speed_cap_kmh is None else float(args.ego_speed_cap_kmh)),
                    "traffic_speed_cap_kmh": (None if args.traffic_speed_cap_kmh is None else float(args.traffic_speed_cap_kmh)),
                    "warmup_s": float(args.warmup_s),

                    "sim_time_s": round(sim_time, 5),
                    "tick": k,

                    "speed_mps": float(v_raw),
                    "speed_mps_f": float(v_ema),

                    "accel_mps2": float(a_raw),
                    "jerk_mps3": float(j_raw),
                    "accel_mps2_f": float(a_f),
                    "jerk_mps3_f": float(j_f),

                    "steer": float(ctrl.steer),
                    "lane_offset_m": float(off),

                    "ttc_s": None if (ttc is None or math.isinf(ttc) or ttc > 100) else float(ttc),
                    "headway_m": None if headway is None else float(headway),
                    "rel_speed_mps": None if rel is None else float(rel),

                    "ttc_focus_s": None if (ttc_focus is None or math.isinf(ttc_focus)) else float(ttc_focus),
                    "headway_focus_m": None if headway_focus is None else float(headway_focus),
                    "rel_speed_focus_mps": None if rel_focus is None else float(rel_focus),

                    "ttc_any_s": None if (ttc_any is None or math.isinf(ttc_any)) else float(ttc_any),
                    "headway_any_m": None if headway_any is None else float(headway_any),
                    "rel_speed_any_mps": None if rel_any is None else float(rel_any),
                    "ttc_used_s": None if (ttc_used is None or math.isinf(ttc_used)) else float(ttc_used),

                    "ttc_perceived_s": None if ttc_perc is None else float(ttc_perc),
                    "headway_perceived_m": None if headway_perc is None else float(headway_perc),
                    "rel_speed_perceived_mps": None if rel_perc is None else float(rel_perc),

                    "sensor_pos_noise_m": float(sensor_pos_sigma),
                    "sensor_vel_noise_mps": float(sensor_vel_sigma),
                    "control_latency_s": float(args.control_latency_s),
                    "control_compute_ms": float(control_compute_ms),

                    "density_count": int(dens),

                    "R": float(R),
                    "M": float(M),
                    "u": float(u),
                    "u_smooth": float(u_smooth),
                    "pct_speed_diff_applied": float(pct_applied),
                    "pct_actuated": float(pct_actuated),
                    "intervention_tick": int(intervention_tick),
                }
                row.update(feats)
                wr.writerow(row)

        num_ticks = int(float(args.duration) * float(args.fps))
        ttc_valid_ratio = float(ttc_valid_ticks) / float(num_ticks) if num_ticks > 0 else 0.0
        headway_valid_ratio = float(headway_valid_ticks) / float(num_ticks) if num_ticks > 0 else 0.0
        intervention_ratio = float(intervention_ticks) / float(num_ticks) if num_ticks > 0 else 0.0
        mean_u_smooth = float(sum_u_smooth) / float(num_ticks) if num_ticks > 0 else 0.0
        mean_pct_speed_diff = float(sum_pct) / float(num_ticks) if num_ticks > 0 else 0.0
        # Near-miss ratios: denominator is total ticks (not just valid-TTC ticks).
        # This gives a true episode-level near-miss rate comparable across modes.
        ttc_lt_2_ratio = (float(ttc_lt_2_ticks) / float(num_ticks)) if num_ticks > 0 else 0.0
        ttc_lt_3_ratio = (float(ttc_lt_3_ticks) / float(num_ticks)) if num_ticks > 0 else 0.0
        ttc_any_lt_3_ratio = (float(ttc_any_lt_3_ticks) / float(num_ticks)) if num_ticks > 0 else 0.0
        ttc_any_valid_ratio = (float(ttc_any_valid_ticks) / float(num_ticks)) if num_ticks > 0 else 0.0
        ttc_used_valid_ratio = (float(ttc_valid_ticks) / float(num_ticks)) if num_ticks > 0 else 0.0
        low_interaction_flag = int(ttc_valid_ratio < 0.01)
        travel_time_s = float(args.duration)
        mean_speed_mps = float(sum_speed_mps) / float(num_ticks) if num_ticks > 0 else 0.0
        target_speed_mps = (float(args.ego_speed_cap_kmh) / 3.6) if args.ego_speed_cap_kmh is not None else max(mean_speed_mps, 1e-6)
        speed_efficiency_ratio = float(mean_speed_mps / max(target_speed_mps, 1e-6))
        velocity_tracking_error_mps = float(sum_abs_speed_error_mps) / float(num_ticks) if num_ticks > 0 else 0.0
        throughput_mps = float(distance_travelled_m / max(travel_time_s, 1e-6))

        if ttc_valid_ratio is not None and ttc_valid_ratio < 0.01:
            print(f"[WARNING] Low interaction episode: {int(args.episode_id)} | ratio={ttc_valid_ratio:.4f}")

        # Compute-time aggregates for deployment-feasibility evidence.
        if len(control_compute_times_ms) > 0:
            _ct = np.array(control_compute_times_ms, dtype=float)
            ctrl_compute_mean_ms = float(np.mean(_ct))
            ctrl_compute_p95_ms  = float(np.percentile(_ct, 95))
            ctrl_compute_max_ms  = float(np.max(_ct))
        else:
            ctrl_compute_mean_ms = 0.0
            ctrl_compute_p95_ms  = 0.0
            ctrl_compute_max_ms  = 0.0

        summary_path = os.path.splitext(args.out_csv)[0] + "_episode_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as fsum:
            wr = csv.DictWriter(fsum, fieldnames=[
                "episode_id","mode","scenario","seed","spawn_index","traffic","weather","scenario_valid","lead_brake_hold_s","cut_in_time_s","cut_in_force","tailgate_dist_m","tailgate_speedup_pct","n_events",
                "collision_any","intervention_ratio",
                "min_ttc","min_headway",
                "max_abs_accel","max_abs_jerk",
                "max_abs_accel_f","max_abs_jerk_f",
                "ttc_lt_2_ticks","ttc_lt_2_ratio","ttc_lt_3_ticks","ttc_lt_3_ratio","ttc_any_lt_3_ticks","ttc_any_lt_3_ratio","ttc_any_valid_ratio","ttc_used_valid_ratio","min_ttc_any",
                "lane_invasions_total","lane_invasions_solid","lane_invasions_broken",
                "collision_rate","collision_events","collision_impulse_sum","intervention_ticks","intervention_rate","mean_u_smooth","max_u_smooth","mean_pct_speed_diff",
                "ttc_valid_ratio","headway_valid_ratio","low_interaction_flag",
                "travel_time_s","distance_travelled_m","mean_speed_mps","speed_efficiency_ratio","velocity_tracking_error_mps","throughput_mps",
                "episode_ticks","fps","duration_s",
                "risk_profile","spawn_bias_close","spawn_ring_min_m","spawn_ring_max_m",
                "ego_speed_cap_kmh","traffic_speed_cap_kmh","warmup_s",
                "sensor_noise_level","sensor_pos_noise_m","sensor_vel_noise_mps",
                "control_latency_s",
                "ctrl_compute_mean_ms","ctrl_compute_p95_ms","ctrl_compute_max_ms"
            ])
            wr.writeheader()
            wr.writerow({
                "episode_id": int(args.episode_id),
                "mode": args.mode,
                "scenario": args.scenario,
                "seed": int(args.seed),
                "spawn_index": int(args.spawn_index),
                "traffic": int(args.traffic),
                "weather": args.weather,
                "scenario_valid": int(scenario_valid),
                "lead_brake_hold_s": float(args.lead_brake_hold_s),
                "cut_in_time_s": float(args.cut_in_time_s),
                "cut_in_force": int(bool(args.cut_in_force)),
                "tailgate_dist_m": float(args.tailgate_dist_m),
                "tailgate_speedup_pct": float(args.tailgate_speedup_pct),
                "n_events": int(args.n_events),

                "collision_any": int(collision_any),
                "intervention_ratio": float(intervention_ratio),

                "min_ttc": (None if math.isinf(min_ttc) else float(min_ttc)),
                "min_headway": (None if math.isinf(min_headway) else float(min_headway)),

                "max_abs_accel": float(max_abs_accel),
                "max_abs_jerk": float(max_abs_jerk),
                "max_abs_accel_f": float(max_abs_accel_f),
                "max_abs_jerk_f": float(max_abs_jerk_f),

                "ttc_lt_2_ticks": int(ttc_lt_2_ticks),
                "ttc_lt_2_ratio": float(ttc_lt_2_ratio),
                "ttc_lt_3_ticks": int(ttc_lt_3_ticks),
                "ttc_lt_3_ratio": float(ttc_lt_3_ratio),
                "ttc_any_lt_3_ticks": int(ttc_any_lt_3_ticks),
                "ttc_any_lt_3_ratio": float(ttc_any_lt_3_ratio),
                "ttc_any_valid_ratio": float(ttc_any_valid_ratio),
                "ttc_used_valid_ratio": float(ttc_used_valid_ratio),
               "min_ttc_any": (None if math.isinf(min_ttc_any) else float(min_ttc_any)),

                "lane_invasions_total": int(lane_invasion_total),
                "lane_invasions_solid": int(lane_invasion_solid),
                "lane_invasions_broken": int(lane_invasion_broken),

                "collision_rate": int(collision_any),
                "collision_events": int(collision_events),
                "collision_impulse_sum": float(collision_impulse_sum),

                "intervention_ticks": int(intervention_ticks),
                "intervention_rate": float(intervention_ratio),
                "mean_u_smooth": float(mean_u_smooth),
                "max_u_smooth": float(max_u_smooth),
                "mean_pct_speed_diff": float(mean_pct_speed_diff),

                "ttc_valid_ratio": float(ttc_valid_ratio),
                "headway_valid_ratio": float(headway_valid_ratio),
                "low_interaction_flag": int(low_interaction_flag),

                "travel_time_s": float(travel_time_s),
                "distance_travelled_m": float(distance_travelled_m),
                "mean_speed_mps": float(mean_speed_mps),
                "speed_efficiency_ratio": float(speed_efficiency_ratio),
                "velocity_tracking_error_mps": float(velocity_tracking_error_mps),
                "throughput_mps": float(throughput_mps),

                "episode_ticks": int(num_ticks),
                "fps": int(args.fps),
                "duration_s": float(args.duration),

                "risk_profile": args.risk_profile,
                "spawn_bias_close": int(args.spawn_bias_close),
                "spawn_ring_min_m": float(args.spawn_ring_min_m),
                "spawn_ring_max_m": float(args.spawn_ring_max_m),

                "ego_speed_cap_kmh": (None if args.ego_speed_cap_kmh is None else float(args.ego_speed_cap_kmh)),
                "traffic_speed_cap_kmh": (None if args.traffic_speed_cap_kmh is None else float(args.traffic_speed_cap_kmh)),
                "warmup_s": float(args.warmup_s),

                "sensor_noise_level": str(args.sensor_noise_level),
                "sensor_pos_noise_m": float(args.sensor_pos_noise_m),
                "sensor_vel_noise_mps": float(args.sensor_vel_noise_mps),
                "control_latency_s": float(args.control_latency_s),
                "ctrl_compute_mean_ms": float(ctrl_compute_mean_ms),
                "ctrl_compute_p95_ms": float(ctrl_compute_p95_ms),
                "ctrl_compute_max_ms": float(ctrl_compute_max_ms),
            })

    finally:
        # Ensure sensors stop listening before destroy
        for s in scenario_actors:
            try:
                if hasattr(s, "stop"):
                    s.stop()
            except Exception:
                pass
        cleanup(scenario_actors)
        cleanup(traffic)
        cleanup([ego])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal","ttc_baseline","innov"], required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)
    ap.add_argument("--client_timeout_s", type=float, default=60.0)
    ap.add_argument("--connect_retry_s", type=float, default=120.0)

    ap.add_argument("--town", default="Town03")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--warmup_s", type=float, default=3.0)  # 3 s ensures vehicles are moving before logging

    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tm_seed", type=int, default=0)
    ap.add_argument("--spawn_index", type=int, default=0)

    ap.add_argument("--traffic", type=int, default=50)
    ap.add_argument("--weather", choices=["clear","rain_fog"], default="rain_fog")

    ap.add_argument("--scenario", choices=["stopped_vehicle","sudden_brake_lead","cut_in","tailgater","mixed"], required=True)
    ap.add_argument("--lead_ahead_m", type=float, default=6.0)
    ap.add_argument("--lead_brake_time_s", type=float, default=0.4)
    ap.add_argument("--lead_brake_hold_s", type=float, default=1.2,
                    help="How long the scenario lead holds hard braking once triggered.")
    ap.add_argument("--lead_brake_handbrake", type=int, default=0,
                    help="If 1, apply hand_brake during forced braking (stronger, can increase collisions).")
    ap.add_argument("--cut_in_time_s", type=float, default=0.6)
    ap.add_argument("--cut_in_force", type=int, default=1)
    ap.add_argument("--tailgate_dist_m", type=float, default=6.0)
    ap.add_argument("--tailgate_speedup_pct", type=float, default=-10.0)
    ap.add_argument("--n_events", type=int, default=1)  # for scenario=mixed

    ap.add_argument("--profile", choices=["normal","sensitive"], default="normal")
    ap.add_argument("--B_override", type=float, default=0.32)
    ap.add_argument("--softness_override", type=float, default=None)

    # ego speed near-normal: percent diff from speed limit
    ap.add_argument("--ego_speedup_pct", type=float, default=0.0)

    # smoother default headway
    ap.add_argument("--global_headway_m", type=float, default=1.0)

    # Optional speed caps (km/h). If omitted, no cap is enforced.
    ap.add_argument("--ego_speed_cap_kmh", type=float, default=65.0)
    ap.add_argument("--traffic_speed_cap_kmh", type=float, default=60.0)
    ap.add_argument("--ego_max_delta_pct_per_s", type=float, default=25.0)

    ap.add_argument("--ttc_thresh", type=float, default=2.50)
    ap.add_argument("--ttc_slow_pct", type=float, default=28.0)
    ap.add_argument("--ttc_headway_m", type=float, default=12.0)

    ap.add_argument("--innov_ema_tau_s", type=float, default=0.55)
    ap.add_argument("--innov_max_delta_pct_per_s", type=float, default=40.0)
    ap.add_argument("--innov_max_slow_pct", type=float, default=38.0)
    ap.add_argument("--innov_headway_gain_m", type=float, default=8.0)
    ap.add_argument("--innov_lane_freeze_u", type=float, default=0.45)
    ap.add_argument("--innov_u_gamma", type=float, default=0.9)

    ap.add_argument("--speed_ema_tau_s", type=float, default=0.40)

    # Intervention logging thresholds
    ap.add_argument("--intervene_u_thresh", type=float, default=0.14)
    ap.add_argument("--intervene_pct_eps", type=float, default=1.0)
    ap.add_argument("--near_miss_ttc_s", type=float, default=3.0)
    ap.add_argument("--ttc_metric_max_s", type=float, default=6.0,
                    help="Discard TTC values above this threshold from safety metrics (6 s is physically meaningful).")

    ap.add_argument("--risk_profile", choices=["normal","aggressive"], default="normal")
    ap.add_argument("--hybrid_physics", type=int, default=0)

    # local packing — minimum 8 m to avoid spawn-time collisions
    ap.add_argument("--spawn_bias_close", type=int, default=1)
    ap.add_argument("--spawn_ring_min_m", type=float, default=8.0)
    ap.add_argument("--spawn_ring_max_m", type=float, default=30.0)

    ap.add_argument("--debug_overlay", type=int, default=0)

    # ----------------------------------------------------------
    # PUBLICATION REALISM PATCHES
    # ----------------------------------------------------------
    ap.add_argument("--sensor_noise_level",
                    choices=["none", "light", "medium", "heavy"],
                    default="none",
                    help="Preset sensor-noise level for perception inputs.")
    ap.add_argument("--sensor_pos_noise_m", type=float, default=-1.0,
                    help="Override sigma of headway noise (m). <0 means use preset.")
    ap.add_argument("--sensor_vel_noise_mps", type=float, default=-1.0,
                    help="Override sigma of relative-speed noise (m/s). <0 means use preset.")
    ap.add_argument("--control_latency_s", type=float, default=0.10,
                    help="End-to-end control latency in seconds.")
    ap.add_argument("--measure_compute_time", type=int, default=1,
                    help="If 1, measure controller compute time per tick.")

    args = ap.parse_args()
    args.debug_overlay = bool(args.debug_overlay)
    args.spawn_bias_close = bool(args.spawn_bias_close)

    # Resolve sensor-noise preset to concrete sigmas.
    _NOISE_PRESETS = {
        "none":   {"pos_m": 0.00, "vel_mps": 0.00},
        "light":  {"pos_m": 0.10, "vel_mps": 0.20},
        "medium": {"pos_m": 0.25, "vel_mps": 0.50},
        "heavy":  {"pos_m": 0.50, "vel_mps": 1.00},
    }
    _preset = _NOISE_PRESETS.get(str(args.sensor_noise_level), _NOISE_PRESETS["none"])
    if float(args.sensor_pos_noise_m) < 0:
        args.sensor_pos_noise_m = float(_preset["pos_m"])
    if float(args.sensor_vel_noise_mps) < 0:
        args.sensor_vel_noise_mps = float(_preset["vel_mps"])

    # Allow disabling caps by passing <=0
    if args.ego_speed_cap_kmh is not None and float(args.ego_speed_cap_kmh) <= 0:
        args.ego_speed_cap_kmh = None
    if args.traffic_speed_cap_kmh is not None and float(args.traffic_speed_cap_kmh) <= 0:
        args.traffic_speed_cap_kmh = None

    run_episode(args)

if __name__ == "__main__":
    main()

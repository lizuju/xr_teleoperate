import time
import argparse
import fcntl
import json
import math
import numpy as np
from multiprocessing import Value, Array, Lock
import threading
from pathlib import Path
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleimager.image_client import ImageClient
from teleop.utils.ipc import IPC_Server
from teleop.robot_control.r1_head_waist import R1HeadWaistFollower, compensate_wrist_for_waist
from teleop.robot_control.r1_hand_tracking import R1WristHold, hand_tracking_freshness
from sshkeyboard import listen_keyboard, stop_listening

def publish_reset_category(category: int, publisher): # Scene Reset signal
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
DRY_RUN_MODE = False
ARM_REQUEST_GENERATION = 0
R1_A7_DEFERRED_REAL_MODE = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, ARM_REQUEST_GENERATION
    if key == 'r':
        if R1_A7_DEFERRED_REAL_MODE and not READY:
            logger_mp.warning("[on_press] System is not ready; ignoring r.")
            return
        ARM_REQUEST_GENERATION += 1
        if not DRY_RUN_MODE:
            START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and (START == True or (DRY_RUN_MODE and READY)):
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

def is_fresh_motion_data(tele_data, timeout, now=None):
    if not tele_data.motion_data_ready or tele_data.motion_data_timestamp <= 0.0:
        return False
    if now is None:
        now = time.monotonic()
    age = now - tele_data.motion_data_timestamp
    return 0.0 <= age <= timeout

def wait_for_new_fresh_motion_data(tv_wrapper, after_timestamp, timeout):
    deadline = time.monotonic() + timeout
    while True:
        tele_data = tv_wrapper.get_tele_data()
        now = time.monotonic()
        if (
            tele_data.motion_data_timestamp > after_timestamp
            and is_fresh_motion_data(tele_data, timeout, now=now)
        ):
            return tele_data
        if now >= deadline:
            return None
        time.sleep(min(0.005, deadline - now))

def head_yaw_rotation(head_pose):
    x_axis = head_pose[:3, 0].copy()
    x_axis[2] = 0.0
    x_norm = np.linalg.norm(x_axis)
    if not np.isfinite(x_norm) or np.isclose(x_norm, 0.0, atol=1e-6):
        return np.eye(3)
    x_axis /= x_norm
    y_axis = np.cross(np.array([0.0, 0.0, 1.0]), x_axis)
    return np.column_stack((x_axis, y_axis, np.array([0.0, 0.0, 1.0])))

def relative_head_pitch_yaw(current_head_pose, reference_head_pose):
    current = np.asarray(current_head_pose, dtype=float)
    reference = np.asarray(reference_head_pose, dtype=float)
    if current.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("head poses must be 4x4 matrices")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("head poses must be finite")
    relative_rotation = reference[:3, :3].T @ current[:3, :3]
    forward = relative_rotation[:, 0]
    pitch = math.atan2(-forward[2], forward[0])
    yaw = math.atan2(forward[1], math.hypot(forward[0], forward[2]))
    return np.array([pitch, yaw])

def wrist_in_reference_head_yaw_frame(
    wrist_pose,
    current_head_pose,
    reference_head_yaw,
):
    current_to_reference = reference_head_yaw.T @ head_yaw_rotation(current_head_pose)
    waist_origin_offset = np.array([0.15, 0.0, 0.45])
    pose = wrist_pose.copy()
    pose[:3, :3] = current_to_reference @ wrist_pose[:3, :3]
    pose[:3, 3] = (
        current_to_reference @ (wrist_pose[:3, 3] - waist_origin_offset)
        + waist_origin_offset
    )
    return pose

def anchored_wrist_target(
    current_pose,
    vision_reference,
    robot_reference,
    waist_to_root,
    translation_scale,
):
    target = robot_reference.copy()
    target[:3, 3] += translation_scale * waist_to_root @ (
        current_pose[:3, 3] - vision_reference[:3, 3]
    )
    relative_rotation = (
        current_pose[:3, :3]
        @ vision_reference[:3, :3].T
    )
    target[:3, :3] = (
        waist_to_root
        @ relative_rotation
        @ waist_to_root.T
        @ robot_reference[:3, :3]
    )
    return target

def rotation_error_rad(actual_rotation, target_rotation):
    cosine = (np.trace(actual_rotation.T @ target_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

def write_json_line(file, payload, flush=False):
    file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    if flush:
        file.flush()

def write_atomic_json(path: Path, payload: dict):
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temp_path.chmod(0o600)
    temp_path.replace(path)

def acquire_live_writer_lock(live_state_path: Path):
    lock_path = live_state_path.parent / "writer.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    lock_path.chmod(0o600)
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"Another O6 live-state writer holds {lock_path}")
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} writer=teleop_hand_and_arm\n")
    lock_file.flush()
    return lock_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2', 'R1_A5', 'R1_A7'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco', 'linker_o6'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--hand-only', action='store_true', help='Skip robot arm, IK, and lowcmd initialization')
    parser.add_argument('--dry-run', action='store_true', help='Compute Linker O6 targets without DDS command publishers')
    parser.add_argument('--tracking-timeout', type=float, default=0.25, help='XR timeout in seconds; R1_A7 hand tracking holds each stale side independently')
    parser.add_argument('--linker-o6-urdf-root', type=str, default='/home/hnh/unitree_r1_dev/linkerhand-urdf/O6', help='Official Linker O6 URDF root')
    parser.add_argument('--linker-o6-calibration', type=str, default=None, help='Vision Pro input calibration for Linker O6')
    parser.add_argument('--linker-o6-live-state', type=str, default=None, help='Atomic JSON target snapshot for isolated Linker O6 simulation')
    parser.add_argument('--arm-translation-scale', type=float, default=1.0, help='R1_A7 Cartesian translation scale relative to Vision Pro motion')
    parser.add_argument('--arm-diagnostic-dir', type=str, default=None, help='Directory for R1_A7 alignment JSONL diagnostics')
    parser.add_argument('--waist-follow', action='store_true', help='R1_A7: sustained head turns drive waist yaw with feedback-based head and arm compensation')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    if args.ee == "dex1_internal" and args.motion:
        parser.error("--ee dex1_internal does not currently support --motion.")
    if args.dry_run and not (args.hand_only and args.ee == "linker_o6" and args.input_mode == "hand"):
        parser.error("--dry-run requires --hand-only --ee linker_o6 --input-mode hand.")
    if args.hand_only and not args.dry_run:
        parser.error("The first-stage --hand-only path requires --dry-run.")
    if args.dry_run and (args.sim or args.motion):
        parser.error("--dry-run cannot be combined with --sim or --motion.")
    if args.linker_o6_calibration and args.ee != "linker_o6":
        parser.error("--linker-o6-calibration requires --ee linker_o6.")
    if args.linker_o6_live_state and not args.dry_run:
        parser.error("--linker-o6-live-state requires --dry-run.")
    if args.ee == "linker_o6" and not args.dry_run:
        if args.arm != "R1_A7" or args.input_mode != "hand" or args.sim or args.motion:
            parser.error("Real Linker O6 control requires --arm R1_A7 --input-mode hand without --sim or --motion.")
        if not args.linker_o6_calibration:
            parser.error("Real Linker O6 control requires --linker-o6-calibration.")
    if not (math.isfinite(args.tracking_timeout) and args.tracking_timeout > 0.0):
        parser.error("--tracking-timeout must be positive.")
    if not (math.isfinite(args.frequency) and args.frequency > 0.0):
        parser.error("--frequency must be positive.")
    if not (math.isfinite(args.arm_translation_scale) and args.arm_translation_scale > 0.0):
        parser.error("--arm-translation-scale must be positive.")
    if args.waist_follow and (args.arm != "R1_A7" or args.hand_only or args.dry_run or args.motion):
        parser.error("--waist-follow requires R1_A7 full arm control without --motion.")

    DRY_RUN_MODE = args.dry_run
    r1_a7_deferred_real = (
        args.arm == "R1_A7"
        and not args.sim
        and not args.hand_only
        and not args.dry_run
    )
    r1_a7_anchored = r1_a7_deferred_real or args.waist_follow or (
        args.arm == "R1_A7" and args.sim and args.input_mode == "hand"
    )
    r1_independent_hands = r1_a7_anchored and args.input_mode == "hand"
    R1_A7_DEFERRED_REAL_MODE = r1_a7_anchored
    if args.arm_diagnostic_dir and not r1_a7_anchored:
        parser.error("--arm-diagnostic-dir requires real R1_A7, R1_A7 --sim --input-mode hand, or R1_A7 --sim --waist-follow.")

    arm_ctrl = None
    hand_ctrl = None
    motion_switcher = None
    r1_vision_left_reference = None
    r1_vision_right_reference = None
    r1_robot_left_reference = None
    r1_robot_right_reference = None
    r1_head_yaw_reference = None
    r1_head_pose_reference = None
    r1_waist_to_root = None
    r1_waist_yaw_reference = None
    waist_follower = None
    waist_diagnostic_next_time = 0.0
    arm_diagnostic_file = None
    arm_diagnostic_path = None
    arm_diagnostic_sequence = 0
    arm_diagnostic_next_time = 0.0
    arm_control_previous_time = None
    dry_run_record_file = None
    linker_o6_retargeter = None
    linker_o6_calibration = None
    img_client = None
    tv_wrapper = None
    recorder = None
    sim_state_subscriber = None
    ipc_server = None
    listen_keyboard_thread = None
    live_state_path = None
    live_writer_lock_file = None
    live_sequence = 0
    exit_code = 0

    try:
        if args.arm_diagnostic_dir:
            diagnostic_dir = Path(args.arm_diagnostic_dir).expanduser()
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            arm_diagnostic_path = diagnostic_dir / (
                f"r1_a7_alignment_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.jsonl"
            )
            arm_diagnostic_file = arm_diagnostic_path.open("x", encoding="utf-8")
            arm_diagnostic_path.chmod(0o600)
            logger_mp.info(f"R1_A7 alignment diagnostics: {arm_diagnostic_path}")

        # setup dds communication domains id
        if not args.dry_run:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            if args.sim:
                ChannelFactoryInitialize(1, networkInterface=args.network_interface)
            else:
                ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        if args.dry_run:
            camera_config = {
                "head_camera": {
                    "binocular": False,
                    "image_shape": [480, 640],
                    "enable_zmq": False,
                    "enable_webrtc": False,
                    "webrtc_port": 60001,
                },
                "left_wrist_camera": {"enable_zmq": False},
                "right_wrist_camera": {"enable_zmq": False},
            }
        else:
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        from televuer import TeleVuerWrapper
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw"
                                     )

        if args.ee == "linker_o6":
            from teleop.robot_control.linker_o6_retargeting import (
                DualLinkerO6Retargeter,
                LinkerO6Calibration,
            )

            linker_o6_retargeter = DualLinkerO6Retargeter(args.linker_o6_urdf_root)
            linker_o6_calibration = (
                LinkerO6Calibration(args.linker_o6_calibration)
                if args.linker_o6_calibration
                else None
            )
            mapping_name = linker_o6_calibration.name if linker_o6_calibration else "candidate_unvalidated"
            visionpro_input_calibrated = linker_o6_calibration is not None

        if args.dry_run:
            from teleop.robot_control.linker_o6_retargeting import is_tracking_fresh

            if args.linker_o6_live_state:
                live_state_path = Path(args.linker_o6_live_state).expanduser()
                live_state_path.parent.mkdir(parents=True, exist_ok=True)
                live_state_path.parent.chmod(0o700)
                live_writer_lock_file = acquire_live_writer_lock(live_state_path)
                logger_mp.info(f"Publishing isolated Linker O6 live state to: {live_state_path}")
            dry_run_status = None
            rearm_required = True
            fresh_seen_after_stale = False
            arm_request_floor = ARM_REQUEST_GENERATION
            if linker_o6_calibration:
                logger_mp.warning(f"Loaded Vision Pro input calibration: {linker_o6_calibration.name}. Linker O6 hardware direction is still unvalidated and no DDS command publisher exists in this path.")
            else:
                logger_mp.warning("Linker O6 dry-run uses an uncalibrated candidate geometric mapping; left/right thumb-yaw direction is not hardware-calibrated and no DDS command publisher exists in this path.")
            logger_mp.info("Linker O6 hardware axis order: thumb pitch, thumb yaw, index, middle, ring, pinky.")
            logger_mp.info("Press [r] to arm target output, [s] to toggle JSONL recording, or [q] to exit.")
            READY = True

            while not STOP:
                start_time = time.monotonic()
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    if head_img.bgr is not None:
                        tv_wrapper.render_to_xr(head_img.bgr)

                if args.record and RECORD_TOGGLE:
                    RECORD_TOGGLE = False
                    if not RECORD_RUNNING:
                        record_dir = Path(args.task_dir) / args.task_name
                        record_dir.mkdir(parents=True, exist_ok=True)
                        record_path = record_dir / f"linker_o6_dry_run_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.jsonl"
                        dry_run_record_file = record_path.open("x", encoding="utf-8")
                        RECORD_RUNNING = True
                        logger_mp.info(f"Started Linker O6 dry-run recording: {record_path}")
                    else:
                        RECORD_RUNNING = False
                        dry_run_record_file.close()
                        dry_run_record_file = None
                        logger_mp.info("Saved Linker O6 dry-run recording.")

                tele_data = tv_wrapper.get_tele_data()
                tracking_now = time.monotonic()
                tracking_age_s = (
                    max(0.0, tracking_now - tele_data.motion_data_timestamp)
                    if tele_data.motion_data_ready and tele_data.motion_data_timestamp > 0.0
                    else None
                )
                tracking_fresh = is_tracking_fresh(
                    tele_data.motion_data_ready,
                    tele_data.motion_data_timestamp,
                    args.tracking_timeout,
                    now=tracking_now,
                )
                targets_allowed = False
                if not tracking_fresh:
                    START = False
                    rearm_required = True
                    fresh_seen_after_stale = False
                    arm_request_floor = ARM_REQUEST_GENERATION
                    if dry_run_status != "tracking_stale":
                        status_payload = {
                            "schema": "linker_o6_target_v1",
                            "armed": False,
                            "reason": "tracking_stale",
                            "mapping": mapping_name,
                            "visionpro_input_calibrated": visionpro_input_calibrated,
                            "thumb_yaw_calibrated": False,
                            "thumb_yaw_hardware_direction_validated": False,
                            "tracking_age_s": tracking_age_s,
                        }
                        status_line = json.dumps(status_payload, separators=(",", ":"))
                        logger_mp.info(status_line)
                        if live_state_path:
                            live_sequence += 1
                            write_atomic_json(live_state_path, {
                                **status_payload,
                                "sequence": live_sequence,
                                "published_monotonic_ns": time.monotonic_ns(),
                            })
                        if RECORD_RUNNING:
                            dry_run_record_file.write(status_line + "\n")
                            dry_run_record_file.flush()
                    dry_run_status = "tracking_stale"
                elif rearm_required:
                    first_fresh_after_stale = not fresh_seen_after_stale
                    if first_fresh_after_stale:
                        fresh_seen_after_stale = True
                    elif ARM_REQUEST_GENERATION > arm_request_floor:
                        rearm_required = False
                        START = True

                    if not rearm_required:
                        targets_allowed = True
                    else:
                        START = False
                        if dry_run_status != "awaiting_arm":
                            status_payload = {
                                "schema": "linker_o6_target_v1",
                                "armed": False,
                                "reason": "awaiting_r_key",
                                "mapping": mapping_name,
                                "visionpro_input_calibrated": visionpro_input_calibrated,
                                "thumb_yaw_calibrated": False,
                                "thumb_yaw_hardware_direction_validated": False,
                                "tracking_age_s": tracking_age_s,
                            }
                            status_line = json.dumps(status_payload, separators=(",", ":"))
                            logger_mp.info(status_line)
                            if live_state_path:
                                live_sequence += 1
                                write_atomic_json(live_state_path, {
                                    **status_payload,
                                    "sequence": live_sequence,
                                    "published_monotonic_ns": time.monotonic_ns(),
                                })
                            if RECORD_RUNNING:
                                dry_run_record_file.write(status_line + "\n")
                                dry_run_record_file.flush()
                        dry_run_status = "awaiting_arm"
                        if first_fresh_after_stale:
                            arm_request_floor = ARM_REQUEST_GENERATION
                else:
                    targets_allowed = True

                if targets_allowed:
                    raw_left_target, raw_right_target = linker_o6_retargeter.retarget(
                        tele_data.left_hand_pos,
                        tele_data.right_hand_pos,
                    )
                    if linker_o6_calibration:
                        left_target, right_target = linker_o6_calibration.apply(
                            raw_left_target, raw_right_target
                        )
                    else:
                        left_target, right_target = raw_left_target, raw_right_target
                    payload = {
                        "schema": "linker_o6_target_v1",
                        "armed": True,
                        "mapping": mapping_name,
                        "visionpro_input_calibrated": visionpro_input_calibrated,
                        "thumb_yaw_calibrated": False,
                        "thumb_yaw_hardware_direction_validated": False,
                        "calibration_file": str(linker_o6_calibration.path) if linker_o6_calibration else None,
                        "raw_target_units": "normalized_0_1_geometric",
                        "target_units": "normalized_0_1",
                        "monotonic_timestamp": tele_data.motion_data_timestamp,
                        "tracking_age_s": tracking_age_s,
                        "hardware_axis_order": [
                            "thumb_pitch",
                            "thumb_yaw",
                            "index",
                            "middle",
                            "ring",
                            "pinky",
                        ],
                        "target_hand_order": ["left", "right"],
                        "raw_left_target": raw_left_target.tolist(),
                        "raw_right_target": raw_right_target.tolist(),
                        "raw_target_12": raw_left_target.tolist() + raw_right_target.tolist(),
                        "left_target": left_target.tolist(),
                        "right_target": right_target.tolist(),
                        "target_12": left_target.tolist() + right_target.tolist(),
                    }
                    line = json.dumps(payload, separators=(",", ":"))
                    if dry_run_status != "armed" or not live_state_path:
                        logger_mp.info(line)
                    if live_state_path:
                        live_sequence += 1
                        write_atomic_json(live_state_path, {
                            **payload,
                            "sequence": live_sequence,
                            "published_monotonic_ns": time.monotonic_ns(),
                        })
                    if RECORD_RUNNING:
                        dry_run_record_file.write(line + "\n")
                        dry_run_record_file.flush()
                    dry_run_status = "armed"

                time.sleep(max(0.0, 1.0 / args.frequency - (time.monotonic() - start_time)))

            raise SystemExit(0)
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if not args.hand_only:
            from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
            if args.motion:
                if args.input_mode == "controller":
                    loco_wrapper = LocoClientWrapper()
            elif not r1_a7_anchored:
                motion_switcher = MotionSwitcher()
                status, result = motion_switcher.Enter_Debug_Mode()
                logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        if not args.hand_only:
            xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived

        if args.ee == "dex1_internal":
            if args.arm != "G1_29":
                raise ValueError("dex1_internal is only supported with --arm G1_29.")
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.

        # arm
        if not args.hand_only:
            from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController, R1_A5_ArmController, R1_A7_ArmController
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK, R1_A5_ArmIK, R1_A7_ArmIK
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                if args.ee == "dex1_internal":
                    arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array,
                                                                  dual_gripper_action_array, motion_mode=args.motion, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
                else:
                    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "G1_23":
                arm_ik = G1_23_ArmIK()
                arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1_2":
                arm_ik = H1_2_ArmIK()
                arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1":
                arm_ik = H1_ArmIK()
                arm_ctrl = H1_ArmController(simulation_mode=args.sim)
            elif args.arm == "H2":
                arm_ik = H2_ArmIK()
                arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "R1_A5":
                arm_ik = R1_A5_ArmIK()
                arm_ctrl = R1_A5_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "R1_A7":
                if r1_a7_anchored:
                    arm_ctrl = R1_A7_ArmController(
                        motion_mode=args.motion,
                        simulation_mode=args.sim,
                        deferred_activation=True,
                    )
                    arm_ik = None
                else:
                    arm_ik = R1_A7_ArmIK()
                    arm_ctrl = R1_A7_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "linker_o6":
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock=False)
            dual_hand_action_array = Array('d', 12, lock=False)
        else:
            pass
        
        # simulation mode
        if args.sim:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            from teleop.utils.episode_writer import EpisodeWriter
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        if r1_a7_anchored:
            START = False
        r1_arm_request_floor = ARM_REQUEST_GENERATION
        READY = True                  # now ready to (1) enter START state
        while (
            not STOP
            and (
                not START
                or (r1_a7_anchored and r1_vision_left_reference is None)
            )
        ):
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)

            if (
                r1_a7_anchored
                and ARM_REQUEST_GENERATION > r1_arm_request_floor
            ):
                r1_arm_request_floor = ARM_REQUEST_GENERATION
                START = False
                reference_tele_data = tv_wrapper.get_tele_data()
                if not is_fresh_motion_data(
                    reference_tele_data,
                    args.tracking_timeout,
                ):
                    logger_mp.warning(
                        "R1_A7 activation rejected: Vision tracking is not fresh; press [r] again."
                    )
                    continue
                if args.ee == "linker_o6":
                    from teleop.robot_control.robot_hand_linker_o6 import LinkerO6Controller
                    hand_ctrl = LinkerO6Controller()
                    hand_ctrl.wait_until_ready(timeout=3.0)
                    if STOP:
                        continue
                if r1_a7_deferred_real:
                    motion_switcher = MotionSwitcher()
                    status, result = motion_switcher.Enter_Debug_Mode()
                    if status != 0:
                        raise RuntimeError(
                            f"R1_A7 failed to enter debug mode: status={status}, result={result}"
                        )
                arm_ctrl.activate()
                if STOP:
                    continue
                post_recenter_tele_data = tv_wrapper.get_tele_data()
                if not is_fresh_motion_data(
                    post_recenter_tele_data,
                    args.tracking_timeout,
                ):
                    raise RuntimeError(
                        "R1_A7 Vision tracking became stale during automatic recentering."
                    )
                post_recenter_timestamp = post_recenter_tele_data.motion_data_timestamp

                post_recenter_motor_q = arm_ctrl.get_current_motor_q()
                r1_waist_yaw_reference = float(post_recenter_motor_q[13])
                r1_waist_to_root = np.array([
                    [math.cos(r1_waist_yaw_reference), -math.sin(r1_waist_yaw_reference), 0.0],
                    [math.sin(r1_waist_yaw_reference), math.cos(r1_waist_yaw_reference), 0.0],
                    [0.0, 0.0, 1.0],
                ])
                arm_ik = R1_A7_ArmIK(waist_yaw=r1_waist_yaw_reference)
                if STOP:
                    continue

                reference_tele_data = wait_for_new_fresh_motion_data(
                    tv_wrapper,
                    post_recenter_timestamp,
                    args.tracking_timeout,
                )
                if reference_tele_data is None:
                    raise RuntimeError(
                        "R1_A7 Vision tracking did not provide a fresh post-activation sample."
                    )

                activation_motor_q = arm_ctrl.get_current_motor_q()
                if abs(float(activation_motor_q[13]) - r1_waist_yaw_reference) > 0.02:
                    raise RuntimeError(
                        "R1_A7 waist changed while IK was loading; command output has been stopped."
                    )
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                (
                    r1_robot_left_reference,
                    r1_robot_right_reference,
                ) = arm_ik.forward_wrist_poses(current_lr_arm_q)
                r1_vision_left_reference = reference_tele_data.left_wrist_pose.copy()
                r1_vision_right_reference = reference_tele_data.right_wrist_pose.copy()
                r1_head_yaw_reference = head_yaw_rotation(reference_tele_data.head_pose)
                r1_head_pose_reference = reference_tele_data.head_pose.copy()
                if args.waist_follow:
                    waist_follower = R1HeadWaistFollower(
                        r1_waist_yaw_reference, time.monotonic(), args.tracking_timeout,
                    )
                    logger_mp.info(
                        "R1_A7 head/waist following enabled: 20 deg for 0.4 s to engage, "
                        "0.35 rad/s waist speed, URDF waist range +/-2.618 rad."
                    )
                if STOP:
                    continue
                if args.ee == "linker_o6":
                    hand_ctrl.activate()
                if arm_diagnostic_file is not None:
                    write_json_line(arm_diagnostic_file, {
                        "schema": "r1_a7_alignment_v1",
                        "event": "activation",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_time_ns": time.monotonic_ns(),
                        "translation_scale": args.arm_translation_scale,
                        "official_head_waist_recenter": True,
                        "waist_follow": args.waist_follow,
                        "waist_yaw_rad": r1_waist_yaw_reference,
                        "waist_to_root": r1_waist_to_root.tolist(),
                        "head_pose_reference": reference_tele_data.head_pose.tolist(),
                        "head_yaw_reference": r1_head_yaw_reference.tolist(),
                        "vision_left_reference": r1_vision_left_reference.tolist(),
                        "vision_right_reference": r1_vision_right_reference.tolist(),
                        "robot_left_reference": r1_robot_left_reference.tolist(),
                        "robot_right_reference": r1_robot_right_reference.tolist(),
                        "motor_q_reference": activation_motor_q.tolist(),
                    }, flush=True)
                START = True
                logger_mp.info(
                    "R1_A7 activated from live posture; Vision heading and wrist references captured."
                )

        if STOP:
            raise SystemExit(0)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        last_fresh_tele_data = reference_tele_data if r1_a7_anchored else None
        tracking_hold_active = False
        if r1_independent_hands:
            wrist_holds = (R1WristHold(r1_robot_left_reference), R1WristHold(r1_robot_right_reference))
            held_head_q_target = arm_ctrl.get_current_head_q().copy()

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            loop_monotonic = time.monotonic()
            loop_period_ms = (
                None
                if arm_control_previous_time is None
                else 1000.0 * (loop_monotonic - arm_control_previous_time)
            )
            arm_control_previous_time = loop_monotonic
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if r1_independent_hands:
                hand_fresh = hand_tracking_freshness(tele_data, args.tracking_timeout, time.monotonic())
                for side, hold, fresh in zip(("left", "right"), wrist_holds, hand_fresh):
                    if fresh != hold.tracking:
                        logger_mp.info(f"[R1 HAND TRACKING] {side}: {'resumed; re-clutching at held target' if fresh else 'stale; holding target'}")
                    if not fresh:
                        hold.hold()
                if args.waist_follow:
                    if not all(hand_fresh):
                        arm_ctrl.hold_waist()
                    elif tracking_hold_active:
                        waist_follower.reset(arm_ctrl.get_current_waist_yaw(), time.monotonic())
                tracking_hold_active = not all(hand_fresh)
                if not any(hand_fresh):
                    time.sleep(1.0 / args.frequency)
                    continue
            elif r1_a7_anchored:
                if is_fresh_motion_data(tele_data, args.tracking_timeout):
                    last_fresh_tele_data = tele_data
                    if tracking_hold_active:
                        if args.waist_follow:
                            waist_follower.reset(arm_ctrl.get_current_waist_yaw(), time.monotonic())
                        logger_mp.info("R1_A7 Vision tracking resumed.")
                    tracking_hold_active = False
                else:
                    tele_data = last_fresh_tele_data
                    if not tracking_hold_active:
                        logger_mp.warning(
                            "R1_A7 Vision tracking is stale; holding the last tracked pose until tracking resumes."
                        )
                    tracking_hold_active = True
                if args.waist_follow and tracking_hold_active:
                    # Do not refresh the waist watchdog with a cached XR sample.
                    arm_ctrl.hold_waist()
                    time.sleep(1.0 / args.frequency)
                    continue
            if args.ee == "linker_o6":
                raw_left_target, raw_right_target = linker_o6_retargeter.retarget(
                    tele_data.left_hand_pos,
                    tele_data.right_hand_pos,
                )
                left_hand_target, right_hand_target = linker_o6_calibration.apply(
                    raw_left_target,
                    raw_right_target,
                )
                if r1_independent_hands:
                    previous_left, previous_right = hand_ctrl.get_action()
                    if not hand_fresh[0]:
                        left_hand_target = previous_left
                    if not hand_fresh[1]:
                        right_hand_target = previous_right
            elif args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            left_wrist_target = tele_data.left_wrist_pose
            right_wrist_target = tele_data.right_wrist_pose
            if r1_a7_anchored:
                head_q_target = relative_head_pitch_yaw(
                    tele_data.head_pose,
                    r1_head_pose_reference,
                )
                left_wrist_pose = wrist_in_reference_head_yaw_frame(
                    tele_data.left_wrist_pose,
                    tele_data.head_pose,
                    r1_head_yaw_reference,
                )
                right_wrist_pose = wrist_in_reference_head_yaw_frame(
                    tele_data.right_wrist_pose,
                    tele_data.head_pose,
                    r1_head_yaw_reference,
                )
                left_wrist_target = anchored_wrist_target(
                    left_wrist_pose,
                    r1_vision_left_reference,
                    r1_robot_left_reference,
                    r1_waist_to_root,
                    args.arm_translation_scale,
                )
                right_wrist_target = anchored_wrist_target(
                    right_wrist_pose,
                    r1_vision_right_reference,
                    r1_robot_right_reference,
                    r1_waist_to_root,
                    args.arm_translation_scale,
                )
            if r1_independent_hands:
                left_wrist_target = wrist_holds[0].prepare(left_wrist_target, hand_fresh[0])
                right_wrist_target = wrist_holds[1].prepare(right_wrist_target, hand_fresh[1])
                if not all(hand_fresh):
                    head_q_target = held_head_q_target.copy()
            left_ik_target = left_wrist_target
            right_ik_target = right_wrist_target
            waist_yaw_actual = None
            waist_yaw_target = None
            if args.waist_follow:
                waist_yaw_actual = arm_ctrl.get_current_waist_yaw()
                if not r1_independent_hands or all(hand_fresh):
                    head_q_target, waist_yaw_target = waist_follower.update(
                        tele_data.head_pose, r1_head_pose_reference,
                        waist_yaw_actual, time.monotonic(),
                    )
                left_ik_target = compensate_wrist_for_waist(
                    left_wrist_target, waist_yaw_actual, r1_waist_yaw_reference,
                )
                right_ik_target = compensate_wrist_for_waist(
                    right_wrist_target, waist_yaw_actual, r1_waist_yaw_reference,
                )
            if r1_a7_anchored:
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_ik_target,
                    right_ik_target,
                    current_lr_arm_q,
                    current_lr_arm_dq,
                    raise_on_failure=True,
                )
            else:
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_wrist_target,
                    right_wrist_target,
                    current_lr_arm_q,
                    current_lr_arm_dq,
                )
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            if STOP:
                break
            if r1_independent_hands:
                fresh_after_ik = hand_tracking_freshness(tele_data, args.tracking_timeout, time.monotonic())
                if any(before and not after for before, after in zip(hand_fresh, fresh_after_ik)):
                    for hold, fresh in zip(wrist_holds, fresh_after_ik):
                        if not fresh:
                            hold.hold()
                    if args.waist_follow:
                        arm_ctrl.hold_waist()
                    tracking_hold_active = True
                    time.sleep(1.0 / args.frequency)
                    continue
            elif args.waist_follow and not is_fresh_motion_data(tele_data, args.tracking_timeout):
                arm_ctrl.hold_waist()
                tracking_hold_active = True
                continue
            if args.ee == "linker_o6":
                hand_ctrl.update(left_hand_target, right_hand_target)
                with dual_hand_data_lock:
                    dual_hand_state_array[:] = np.concatenate(hand_ctrl.get_state())
                    dual_hand_action_array[:] = np.concatenate(hand_ctrl.get_action())
            if STOP:
                break
            if args.waist_follow:
                arm_ctrl.ctrl_dual_arm_and_head(
                    sol_q, sol_tauff, head_q_target, waist_yaw_target=waist_yaw_target,
                )
            elif r1_a7_anchored:
                arm_ctrl.ctrl_dual_arm_and_head(sol_q, sol_tauff, head_q_target)
            else:
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            if r1_independent_hands:
                wrist_holds[0].commit(left_wrist_target)
                wrist_holds[1].commit(right_wrist_target)
                held_head_q_target = head_q_target.copy()
            diagnostic_now = time.monotonic()
            if args.waist_follow and diagnostic_now >= waist_diagnostic_next_time:
                waist_diagnostic_next_time = diagnostic_now + 1.0
                logger_mp.info(
                    f"[R1 HEAD/WAIST] total_yaw={math.degrees(waist_follower.total_yaw):.1f} deg "
                    f"waist_actual={math.degrees(waist_yaw_actual):.1f} deg "
                    f"waist_target={math.degrees(waist_yaw_target if waist_yaw_target is not None else waist_yaw_actual):.1f} deg "
                    f"head_yaw={math.degrees(head_q_target[1]):.1f} deg "
                    f"following={waist_follower.following and not tracking_hold_active}"
                )
            if (
                arm_diagnostic_file is not None
                and diagnostic_now >= arm_diagnostic_next_time
            ):
                actual_left_pose, actual_right_pose = arm_ik.forward_wrist_poses(current_lr_arm_q)
                solved_left_pose, solved_right_pose = arm_ik.forward_wrist_poses(sol_q)
                if args.waist_follow:
                    # Fixed-waist IK FK must be returned to the actual robot root frame.
                    actual_left_pose, actual_right_pose, solved_left_pose, solved_right_pose = (
                        compensate_wrist_for_waist(pose, r1_waist_yaw_reference, waist_yaw_actual)
                        for pose in (actual_left_pose, actual_right_pose, solved_left_pose, solved_right_pose)
                    )
                arm_diagnostic_sequence += 1
                arm_diagnostic_next_time = diagnostic_now + 0.1
                write_json_line(arm_diagnostic_file, {
                    "schema": "r1_a7_alignment_v1",
                    "event": "sample",
                    "sequence": arm_diagnostic_sequence,
                    "wall_time_ns": time.time_ns(),
                    "monotonic_time_ns": time.monotonic_ns(),
                    "source_monotonic_time": tele_data.motion_data_timestamp,
                    "tracking_age_ms": 1000.0 * (diagnostic_now - tele_data.motion_data_timestamp),
                    "hand_tracking": {
                        side: {"fresh": fresh, "age_ms": 1000.0 * (diagnostic_now - timestamp)}
                        for side, fresh, timestamp in zip(
                            ("left", "right"), hand_fresh,
                            (tele_data.left_hand_timestamp, tele_data.right_hand_timestamp),
                        )
                    } if r1_independent_hands else None,
                    "loop_period_ms": loop_period_ms,
                    "ik_duration_ms": 1000.0 * (time_ik_end - time_ik_start),
                    "head_pose": tele_data.head_pose.tolist(),
                    "vision_left_processed_dynamic_heading": tele_data.left_wrist_pose.tolist(),
                    "vision_right_processed_dynamic_heading": tele_data.right_wrist_pose.tolist(),
                    "vision_left_fixed_heading": left_wrist_pose.tolist(),
                    "vision_right_fixed_heading": right_wrist_pose.tolist(),
                    "left_target": left_wrist_target.tolist(),
                    "right_target": right_wrist_target.tolist(),
                    "left_ik_target": left_ik_target.tolist(),
                    "right_ik_target": right_ik_target.tolist(),
                    "waist_yaw_actual_rad": waist_yaw_actual,
                    "waist_yaw_target_rad": waist_yaw_target,
                    "head_q_target": head_q_target.tolist(),
                    "q_actual": current_lr_arm_q.tolist(),
                    "dq_actual": current_lr_arm_dq.tolist(),
                    "q_ik_command": sol_q.tolist(),
                    "actual_left_pose": actual_left_pose.tolist(),
                    "actual_right_pose": actual_right_pose.tolist(),
                    "solved_left_pose": solved_left_pose.tolist(),
                    "solved_right_pose": solved_right_pose.tolist(),
                    "left_actual_position_error_m": float(np.linalg.norm(actual_left_pose[:3, 3] - left_wrist_target[:3, 3])),
                    "right_actual_position_error_m": float(np.linalg.norm(actual_right_pose[:3, 3] - right_wrist_target[:3, 3])),
                    "left_solved_position_error_m": float(np.linalg.norm(solved_left_pose[:3, 3] - left_wrist_target[:3, 3])),
                    "right_solved_position_error_m": float(np.linalg.norm(solved_right_pose[:3, 3] - right_wrist_target[:3, 3])),
                    "left_actual_rotation_error_rad": rotation_error_rad(actual_left_pose[:3, :3], left_wrist_target[:3, :3]),
                    "right_actual_rotation_error_rad": rotation_error_rad(actual_right_pose[:3, :3], right_wrist_target[:3, :3]),
                    "left_solved_rotation_error_rad": rotation_error_rad(solved_left_pose[:3, :3], left_wrist_target[:3, :3]),
                    "right_solved_rotation_error_rad": rotation_error_rad(solved_right_pose[:3, :3], right_wrist_target[:3, :3]),
                    "joint_following_error_max_rad": float(np.max(np.abs(current_lr_arm_q - sol_q))),
                })

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee in ("inspire_dfx", "inspire_ftp", "brainco", "linker_o6") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                if args.waist_follow:
                    # Body order: waist_yaw, head_pitch, head_yaw.
                    current_body_state = [waist_yaw_actual, *arm_ctrl.get_current_head_q().tolist()]
                    current_body_action = [
                        waist_yaw_target if waist_yaw_target is not None else waist_yaw_actual,
                        *head_q_target.tolist(),
                    ]

                # arm state and action (split into left/right halves by the arm's own DOF, so it works for any variant: H1/G1_23/R1_A5 = 4/5 per arm, G1_29/R1_A7 = 7)
                half = len(current_lr_arm_q) // 2
                left_arm_state,  right_arm_state  = current_lr_arm_q[:half], current_lr_arm_q[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        exit_code = 1
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        active_exception = sys.exc_info()[1]
        if isinstance(active_exception, SystemExit) and active_exception.code not in (None, 0):
            exit_code = active_exception.code if isinstance(active_exception.code, int) else 1
        try:
            if args.ee == "linker_o6" and hand_ctrl is not None:
                hand_ctrl.stop()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to stop Linker O6 controller: {e}")

        try:
            if arm_ctrl is not None:
                if r1_a7_anchored:
                    arm_ctrl.stop()
                else:
                    arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to stop arm controller: {e}")
        
        try:
            if args.ipc:
                if ipc_server is not None:
                    ipc_server.stop()
            else:
                stop_listening()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join(timeout=1.0)
                    if listen_keyboard_thread.is_alive():
                        logger_mp.warning("Keyboard listener did not stop within 1 second.")
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                if dry_run_record_file is not None:
                    dry_run_record_file.close()
                elif recorder is not None:
                    recorder.close()
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to close recorder: {e}")
        try:
            if arm_diagnostic_file is not None:
                write_json_line(arm_diagnostic_file, {
                    "schema": "r1_a7_alignment_v1",
                    "event": "exit",
                    "wall_time_ns": time.time_ns(),
                    "monotonic_time_ns": time.monotonic_ns(),
                    "exit_code": exit_code,
                }, flush=True)
                arm_diagnostic_file.close()
                logger_mp.info(f"Saved R1_A7 alignment diagnostics: {arm_diagnostic_path}")
        except Exception as e:
            exit_code = 1
            logger_mp.error(f"Failed to close R1_A7 alignment diagnostics: {e}")
        if live_writer_lock_file is not None:
            live_writer_lock_file.close()
        logger_mp.info("✅ Finally, exiting program.")
        raise SystemExit(exit_code)

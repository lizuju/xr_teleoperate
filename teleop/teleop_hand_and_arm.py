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

from teleop.utils.ipc import IPC_Server
from teleop.utils.xr_record_gate import recording_blocked_by_tracking, tracking_diagnostics
from teleop.robot_control.r1_head_waist import R1HeadWaistFollower, compensate_wrist_for_waist
from teleop.robot_control.r1_hand_tracking import (
    R1WristHold, hand_tracking_freshness, hand_tracking_present,
)
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
RECORD_OUTCOME = 'unspecified'
DRY_RUN_MODE = False
ARM_REQUEST_GENERATION = 0
R1_A7_DEFERRED_REAL_MODE = False
R1_PAUSE = None
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
    global STOP, START, RECORD_TOGGLE, RECORD_OUTCOME, ARM_REQUEST_GENERATION
    if key == 'r':
        if R1_PAUSE is not None and R1_PAUSE.paused:
            R1_PAUSE.request_resume()
            logger_mp.info("[R1 PAUSE] Resume requested; keep both hands stable to realign and continue.")
            return
        if R1_A7_DEFERRED_REAL_MODE and not READY:
            logger_mp.warning("[on_press] System is not ready; ignoring r.")
            return
        ARM_REQUEST_GENERATION += 1
        if not DRY_RUN_MODE:
            START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 'p' and R1_PAUSE is not None:
        R1_PAUSE.pause()
        logger_mp.info("[R1 PAUSE] Paused; arm, head and hand targets held. [r] requests realigned resume; [q] exits.")
    elif key in ('y', 'n', 'x') and RECORD_RUNNING:
        RECORD_OUTCOME = {'y': 'success', 'n': 'failure', 'x': 'discarded'}[key]
        RECORD_TOGGLE = True
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

def is_present_motion_data(tele_data):
    """True while Vision Pro still has a last sample (timestamp not zeroed)."""
    if not tele_data.motion_data_ready or tele_data.motion_data_timestamp <= 0.0:
        return False
    return math.isfinite(tele_data.motion_data_timestamp)


def is_fresh_motion_data(tele_data, timeout, now=None):
    # Keep this self-contained: several tests exec only this function.
    # Presence (timestamp > 0) is the arm-hold signal; age is startup/recording.
    if not tele_data.motion_data_ready or tele_data.motion_data_timestamp <= 0.0:
        return False
    if now is None:
        now = time.monotonic()
    age = now - tele_data.motion_data_timestamp
    return 0.0 <= age <= timeout

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
    reference_head_position,
):
    current_to_reference = reference_head_yaw.T @ head_yaw_rotation(current_head_pose)
    waist_origin_offset = np.array([0.15, 0.0, 0.45])
    pose = wrist_pose.copy()
    pose[:3, :3] = current_to_reference @ wrist_pose[:3, :3]
    pose[:3, 3] = (
        current_to_reference @ (wrist_pose[:3, 3] - waist_origin_offset)
        + waist_origin_offset
        # Undo the wrapper's current-head translation and anchor at activation.
        + reference_head_yaw.T @ (current_head_pose[:3, 3] - reference_head_position)
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
    relative_rotation = current_pose[:3, :3] @ vision_reference[:3, :3].T
    target[:3, :3] = (
        waist_to_root @ relative_rotation @ waist_to_root.T @ robot_reference[:3, :3]
    )
    return target


def waist_follower_from_args(args, waist_actual, now):
    """Build the waist follower from the launch settings.

    Shared by the activation and the resume paths so the two cannot drift apart.
    The getattr fallbacks are the argument defaults, so a namespace that predates
    one of these flags gets the intended behaviour rather than a silent zero.
    """
    return R1HeadWaistFollower(
        waist_actual,
        now,
        args.tracking_timeout,
        math.radians(getattr(args, "waist_follow_threshold_deg", 20.0)),
        getattr(args, "waist_follow_dwell", 0.2),
        math.radians(getattr(args, "waist_follow_speed_deg", 40.0)),
        math.radians(getattr(args, "waist_follow_accel_deg", 90.0)),
    )


class TeleImagerCameraClient:
    '''Camera access built on teleimager 2.x ``TeleImageClient``.

    teleimager 2 addressed one ``TeleImageClient`` per camera topic and dropped the
    old ``ImageClient`` convenience wrapper, so this adapts the 2.x API back onto the
    shape the main loop already uses. Topic names come from the server roster and are
    stable (``head_camera`` etc.), so callers keep indexing ``camera_config`` by topic.
    '''

    def __init__(self, server_host, request_bgr=True, request_port=60000):
        self.server_host = server_host
        self.request_bgr = request_bgr
        self.cameras = {}
        self.scanned_from_server = False
        self.camera_config = self._open(request_port=request_port)

    def _open(self, request_port=60000):
        '''Read the camera roster once and subscribe to every ZMQ-enabled camera.'''
        from teleimager.client import TeleImageClient

        config, from_server = TeleImageClient.scan(self.server_host, request_port=request_port)
        if not config:
            raise RuntimeError(
                f'teleimager returned no camera configuration from {self.server_host}:{request_port}.'
            )
        self.scanned_from_server = bool(from_server)
        self.cameras = {}
        for topic, entry in config.items():
            if not isinstance(entry, dict):
                raise RuntimeError(f'Camera entry {topic!r} is not a configuration mapping.')
            if not entry.get('enable_zmq'):
                continue
            port = entry.get('zmq_port')
            if not isinstance(port, int) or port <= 0:
                raise RuntimeError(f'Camera {topic!r} enables ZMQ without a usable zmq_port.')
            self.cameras[topic] = TeleImageClient(
                topic, server_host=self.server_host, zmq_port=port, request_bgr=self.request_bgr,
            )
        logger_mp.info(
            'Teleimager 2.x camera config from %s (%s); streaming %s',
            self.server_host,
            'server' if self.scanned_from_server else 'local cache',
            ', '.join(sorted(self.cameras)) or 'nothing',
        )
        return config

    def get_cam_config(self):
        '''Return the roster keyed by topic, matching the pre-2.x access pattern.'''
        return self.camera_config

    def get_head_frame(self):
        return self._frame('head_camera')

    def get_left_wrist_frame(self):
        return self._frame('left_wrist_camera')

    def get_right_wrist_frame(self):
        return self._frame('right_wrist_camera')

    def _frame(self, topic):
        client = self.cameras.get(topic)
        if client is None:
            return None
        return client.get_frame()

    def close(self):
        for client in self.cameras.values():
            try:
                client.close()
            except Exception as error:
                logger_mp.warning('Failed to close camera %s: %s', client, error)
        self.cameras = {}

def rotation_error_rad(actual_rotation, target_rotation):
    cosine = (np.trace(actual_rotation.T @ target_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

def hold_r1_published_targets(arm_ctrl):
    published = arm_ctrl.get_recording_snapshot()["published"]
    # Before the first successful publication there is nothing to replay, so the
    # controller's own hold path is the only safe way to keep the pose.
    if published is not None and published.get("arm_q") is not None:
        arm_ctrl.ctrl_dual_arm_and_head(
            np.array(published["arm_q"]),
            np.array(published["arm_tau"]),
            np.array(published["head_q"]),
        )
    else:
        arm_ctrl.hold_targets()
    arm_ctrl.hold_waist()

def r1_workspace_saturation(
    solved_left_pose,
    solved_right_pose,
    left_ik_target,
    right_ik_target,
    position_limit_m,
    rotation_limit_rad,
):
    '''Return per-side IK shortfall once a target leaves the reachable workspace.

    The solver treats position and orientation as soft costs, so an unreachable
    target is not an error: the arm silently follows as far as it can. This turns
    that silent shortfall into a reportable signal.
    '''
    sinks = (left_ik_target, right_ik_target)
    return {
        side: {
            "position_m": float(np.linalg.norm(solved[:3, 3] - target[:3, 3])),
            "rotation_rad": rotation_error_rad(solved[:3, :3], target[:3, :3]),
            "outside": bool(
                np.linalg.norm(solved[:3, 3] - target[:3, 3]) > position_limit_m
                or rotation_error_rad(solved[:3, :3], target[:3, :3]) > rotation_limit_rad
            ),
        }
        for side, solved, target in zip(
            ("left", "right"),
            (solved_left_pose, solved_right_pose),
            sinks,
        )
    }

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

class PalmFrame:
    """A palm camera frame with its red and blue channels put back.

    The two O6 palm cameras are UVC modules that hand us MJPG with red and blue
    transposed relative to every other JPEG in this pipeline, so red arrives as
    blue and yellow as green. The head stereo comes off a different path
    (GStreamer H.264) and is already correct, which is why the correction is
    applied here at the palm entry point rather than globally.

    A new object is returned instead of swapping the pixels in place: the image
    client's ring buffer hands back the same TeleImage until a new frame lands,
    so an in-place swap would be applied twice.
    """

    __slots__ = ("bgr", "sequence", "received_monotonic_ns")

    def __init__(self, frame):
        self.bgr = np.ascontiguousarray(frame.bgr[:, :, ::-1])
        self.sequence = frame.sequence
        self.received_monotonic_ns = frame.received_monotonic_ns


def correct_palm_frame(frame):
    """Wrap a palm frame so red and blue are the right way round."""
    if frame is None or frame.bgr is None:
        return frame
    return PalmFrame(frame)


def resolve_run_camera_calibration(args, root=None):
    """Resolve the camera calibration for this run, or stop before touching robot state.

    A calibration file that was asked for but cannot be trusted must end the run
    here rather than let a whole collection session record episodes whose
    geometry is silently wrong. A file found at the default location is held to
    the same standard: a corrupt one is a defect, not something to skip past.

    The default root is the xr_teleoperate repo (parent of teleop/), not
    unitree_r1_dev. Path(__file__).parents[2] from this file is the latter.
    """
    from teleop.utils.camera_calibration import (
        CalibrationError, default_camera_calibration_root, resolve_camera_calibration,
    )
    if root is None:
        root = default_camera_calibration_root(__file__)
    try:
        path, calibration = resolve_camera_calibration(args.camera_calibration, root=root)
    except CalibrationError as error:
        logger_mp.error(f"[CAMERA CALIBRATION] rejected: {error}")
        raise SystemExit(2)
    if calibration is None:
        logger_mp.info("[CAMERA CALIBRATION] none supplied; episodes will declare status=uncalibrated")
    else:
        logger_mp.info(
            f"[CAMERA CALIBRATION] {path} (sha256={calibration['sha256'][:12]}, "
            f"cameras={', '.join(sorted(calibration['cameras']))})"
        )
    return path, calibration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 40.0, help = 'control and record \'s frequency')
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
    parser.add_argument('--tracking-timeout', type=float, default=0.5, help="XR freshness window in seconds for startup and recording. "
                        "Arm hold no longer uses age: Vision Pro zeroes a lost hand's timestamp, "
                        "and only that freezes the arm. A Wi-Fi hole leaves the last timestamp in "
                        "place; following that last pose avoids the freeze-resume hitch.")
    parser.add_argument('--linker-o6-urdf-root', type=str, default='/home/hnh/unitree_r1_dev/linkerhand-urdf/O6', help='Official Linker O6 URDF root')
    parser.add_argument('--linker-o6-method', choices=['vector', 'position', 'dexpilot'], default='vector', help='dex-retargeting optimizer for Linker O6')
    parser.add_argument('--linker-o6-live-state', type=str, default=None, help='Atomic JSON target snapshot for isolated Linker O6 simulation')
    parser.add_argument('--arm-translation-scale', type=float, default=0.87, help='R1_A7 Cartesian translation scale for the hand displacement, measured from the '
                        'activation anchor. Rotations are never scaled. The robot arm is shorter than the operator arm, so a proportional mapping is the ratio of the '
                        'two: the R1_A7 shoulder-to-wrist chain is ~0.65 m against ~0.75 m for an adult arm, giving 0.87 (default). 1.0 asks the arm to reach 15%% '
                        'further than the operator and saturates the wrist more often; 0.7 under-reaches by 19%%, so the operator has to move further than the robot does.')
    parser.add_argument('--arm-diagnostic-dir', type=str, default=None, help='Directory for R1_A7 alignment JSONL diagnostics')
    parser.add_argument('--waist-follow', action='store_true', help='R1_A7: sustained head turns drive waist yaw with feedback-based head and arm compensation')
    parser.add_argument('--waist-follow-threshold-deg', type=float, default=20.0, help='Residual head yaw, i.e. the yaw the waist has not absorbed yet, needed to engage waist following. '
                        'Not the raw head yaw: the waist turns until the residual falls under the release threshold, so this is how far the operator can turn before the robot starts '
                        'following. Raising it stops small head movements from turning the torso at all. (degrees)')
    parser.add_argument('--waist-follow-dwell', type=float, default=0.2, help='Seconds the head yaw threshold must be held before waist following engages')
    parser.add_argument('--waist-follow-speed-deg', type=float, default=40.0, help='R1_A7: ceiling on how fast the waist target may slew while following, in degrees per second. '
                        'The old fixed ceiling was 20 deg/s, which took 2.2 s to absorb a 45 deg body turn and read as the robot lagging behind the operator. Must be positive.')
    parser.add_argument('--waist-follow-accel-deg', type=float, default=90.0, help='R1_A7: how fast the waist target may change speed while following, in degrees per second squared. '
                        'Together with --waist-follow-speed-deg this sets the ramp: 40 deg/s at 90 deg/s^2 reaches full speed in 0.44 s. Must be positive.')
    parser.add_argument('--waist-follow-compensation', choices=['torso', 'world'], default='torso', help='R1_A7: which frame the wrist target is held in while the waist turns. '
                        '"torso" (default) keeps the arm posture relative to the torso exactly as the operator commanded it, so turning the waist only rotates the whole arm with the '
                        'body and the joints do not move at all. "world" counter-rotates the target about the pelvis axis to hold the hand still in world space, which the arm can only '
                        'achieve by folding; measured on 2026-09-16 (r1-diag-15deg, 6494 samples) with the operator hand still to within 2 mm, the elbow moved 0.10 deg per tick while the '
                        'waist was still and 3.51 deg per tick once the waist turned more than 2 deg -- a 35x amplification of uncommanded arm motion, and the direct cause of the arm '
                        'twisting during waist following. The counter-rotation is only worth its cost when something must stay put in the world, which is not what a head-triggered waist '
                        'follow is doing.')
    parser.add_argument('--startup-wrist-align', action='store_true', help='R1_A7: align wrist orientation to the initial Vision Pro pose before following')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--record-max-tracking-age-ms', type=float, default=100.0, help='Refuse to start an episode while any tracked XR hand is older than this many milliseconds. 0 disables the gate. Measured 2026-09-18: a bad Wi-Fi evening was left_age_ms p50=106 ms with 7% of frames past 500 ms; a good morning was 24 ms.')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    parser.add_argument('--arm-diagnostic-hz', type=float, default=10.0, help='R1_A7 alignment JSONL sample rate; lower it to shrink long sessions')
    parser.add_argument('--workspace-position-tolerance-m', type=float, default=0.05, help='R1_A7: IK position shortfall above which a target counts as outside the reachable workspace')
    parser.add_argument('--workspace-rotation-tolerance-rad', type=float, default=0.15, help='R1_A7: IK rotation shortfall above which a target counts as outside the reachable workspace')
    parser.add_argument('--arm-limit-softness', type=float, default=0.1, help='R1_A7: weight of the soft joint-limit barrier that discourages the arm from riding its limits; 0 disables')
    parser.add_argument('--arm-posture-weight', type=float, default=0.02, help='R1_A7: weight pulling the arm posture toward the activation posture. This is the term that stops the elbow from drifting into a twisted pose over a session: with 7 DoF and only a wrist pose target there is a free null space and nothing else anchors it. Measured on the real URDF over a 120-frame reach sweep (validate_redundancy.py): 0.0 leaves the elbow travelling 0.685 rad, 0.02 cuts that to 0.629 with position tracking unchanged (+8%%) and wrist orientation error 0.089 -> 0.131 rad, 0.05 cuts it to 0.575 but costs 0.187 rad and starts riding joint limits. 0.02 is the knee of that curve; 0 disables')
    parser.add_argument('--arm-velocity-limit', type=float, default=30.0, help='R1_A7: safety cap on how fast the published arm POSITION target may move, in rad/s. 30 ~= off; set 3.0 to run in the conservative capped mode instead of the dq feed-forward.')
    parser.add_argument('--arm-dq-feedforward', choices=['on', 'off'], default='on', help='R1_A7: feed the target velocity into the servo dq field so the arm follows the commanded motion instead of chasing it with start-stop bursts (default on).')
    parser.add_argument('--arm-dq-limit', type=float, default=6.0, help='R1_A7: clamp on the feed-forward velocity in rad/s (safety net, not a tracking limit).')
    parser.add_argument('--arm-dq-filter', type=float, default=0.5, help='R1_A7: low-pass factor for the feed-forward velocity derivative, 1.0 = unfiltered.')
    parser.add_argument('--arm-target-velocity-limit', type=float, default=6.0, help='R1_A7: shape the IK joint target to this speed (rad/s) before dq feed-forward is taken from it. Fast XR/IK bursts otherwise become a step the servo chases. 0 disables. 6 matches --arm-dq-limit and the ~5 rad/s the arm delivered with feed-forward on 2026-09-16; this is not the revoked 3.0 position clip.')
    parser.add_argument('--arm-target-accel-limit', type=float, default=40.0, help='R1_A7: how fast the shaped reference may change speed, in rad/s^2. 40 reaches 6 rad/s in 0.15 s. 0 means no acceleration limit. This is the term that removes fast-motion jerk; the velocity ceiling only stretches catch-up after a stale sample.')
    parser.add_argument('--wrist-display', type=str, choices=['auto', 'both', 'left', 'right', 'off'], default='auto', help='Vision Pro wrist camera panels: auto/both show both sides, left/right show one, off disables them. Panels and recording both read the wrist JPEG over ZMQ; wrist WebRTC on PC2 is unused.')
    parser.add_argument('--wrist-panel-distance', type=float, default=1.2, help='Distance of the wrist panels in front of the eyes, in meters')
    parser.add_argument('--wrist-panel-offset', type=float, nargs=2, default=[0.40, 0.40], metavar=('X', 'Y'), help='Wrist panel centre offset in meters; X is mirrored per side, Y is downwards')
    parser.add_argument('--wrist-panel-height', type=float, default=0.26, help='Wrist panel height in meters at --wrist-panel-distance')
    parser.add_argument('--camera-calibration', type=str, default='', help='Camera calibration JSON to embed in every recorded episode. Defaults to assets/r1/camera_calibration.json when that file exists; an explicitly given path must exist and validate.')
    parser.add_argument('--camera-sync-tolerance-ms', type=float, default=25.0, help='R1_A7: cross-camera time spread above which a recorded sample is flagged camera_alignment.aligned=false. Samples are still recorded; filter on the flag downstream.')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    # Resolved before any robot state is touched; see resolve_run_camera_calibration.
    # Default root is xr_teleoperate (parent of teleop/), not unitree_r1_dev.
    camera_calibration_path, camera_calibration = resolve_run_camera_calibration(args)

    if args.ee == "dex1_internal" and args.motion:
        parser.error("--ee dex1_internal does not currently support --motion.")
    if args.dry_run and not (args.hand_only and args.ee == "linker_o6" and args.input_mode == "hand"):
        parser.error("--dry-run requires --hand-only --ee linker_o6 --input-mode hand.")
    if args.hand_only and not args.dry_run:
        parser.error("The first-stage --hand-only path requires --dry-run.")
    if args.dry_run and (args.sim or args.motion):
        parser.error("--dry-run cannot be combined with --sim or --motion.")
    if args.linker_o6_live_state and not args.dry_run:
        parser.error("--linker-o6-live-state requires --dry-run.")
    if args.ee == "linker_o6" and not args.dry_run:
        if args.arm != "R1_A7" or args.input_mode != "hand" or args.sim or args.motion:
            parser.error("Real Linker O6 control requires --arm R1_A7 --input-mode hand without --sim or --motion.")
    if not (math.isfinite(args.tracking_timeout) and args.tracking_timeout > 0.0):
        parser.error("--tracking-timeout must be positive.")
    if not (math.isfinite(args.frequency) and args.frequency > 0.0):
        parser.error("--frequency must be positive.")
    if not (math.isfinite(args.arm_translation_scale) and args.arm_translation_scale > 0.0):
        parser.error("--arm-translation-scale must be positive.")
    if not (math.isfinite(args.waist_follow_threshold_deg) and args.waist_follow_threshold_deg > 0.0):
        parser.error("--waist-follow-threshold-deg must be positive.")
    if not (math.isfinite(args.waist_follow_dwell) and args.waist_follow_dwell > 0.0):
        parser.error("--waist-follow-dwell must be positive.")
    if not (math.isfinite(args.waist_follow_speed_deg) and args.waist_follow_speed_deg > 0.0):
        parser.error("--waist-follow-speed-deg must be positive.")
    if not (math.isfinite(args.waist_follow_accel_deg) and args.waist_follow_accel_deg > 0.0):
        parser.error("--waist-follow-accel-deg must be positive.")
    if args.waist_follow and (args.arm != "R1_A7" or args.hand_only or args.dry_run or args.motion):
        parser.error("--waist-follow requires R1_A7 full arm control without --motion.")
    if not (math.isfinite(args.arm_diagnostic_hz) and args.arm_diagnostic_hz > 0.0):
        parser.error("--arm-diagnostic-hz must be positive.")
    if not (math.isfinite(args.workspace_position_tolerance_m) and args.workspace_position_tolerance_m >= 0.0):
        parser.error("--workspace-position-tolerance-m must be finite and non-negative.")
    if not (math.isfinite(args.workspace_rotation_tolerance_rad) and args.workspace_rotation_tolerance_rad >= 0.0):
        parser.error("--workspace-rotation-tolerance-rad must be finite and non-negative.")
    if not (math.isfinite(args.arm_limit_softness) and args.arm_limit_softness >= 0.0):
        parser.error("--arm-limit-softness must be finite and non-negative.")
    if not (math.isfinite(args.arm_posture_weight) and args.arm_posture_weight >= 0.0):
        parser.error("--arm-posture-weight must be finite and non-negative.")

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
    arm_ik = None
    hand_ctrl = None
    linker_o6_loop = None
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
    # Per-iteration loop-jitter accounting. The 10 Hz JSONL sample cannot show a
    # stall that lasts a few iterations, which is exactly what a stop-and-go arm
    # feels like, so every iteration feeds this and the summary reports the tail.
    loop_jitter = {"iterations": 0, "max_ms": 0.0, "over_40ms": 0, "over_80ms": 0,
                   "worst_at_s": None}
    loop_jitter_started = None
    workspace_diagnostic_next_time = 0.0
    workspace_saturation_events = 0
    workspace_warned_side = None
    tracking_hold_events = 0
    tracking_hold_previous = False
    arm_control_previous_time = None
    dry_run_record_file = None
    linker_o6_retargeter = None
    img_client = None
    tv_wrapper = None
    recorder = None
    r1_capture = None
    failure_reason = None
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
            img_client = TeleImagerCameraClient(server_host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # Wrist camera panels: one independent HUD panel per side, fed with the
        # teleimager JPEG frames the recording path already uses. A second WebRTC
        # element never negotiates on this headset client (measured 2026-09-15:
        # zero requests to 60002/60003 while the head's own stream connects), so
        # the panels ride the scene channel as ImageBackground overlays instead.
        wrist_panels = []
        if args.display_mode != 'pass-through' and args.wrist_display != 'off':
            for side, topic in (("left", "left_wrist_camera"), ("right", "right_wrist_camera")):
                if args.wrist_display not in ('auto', 'both', side):
                    continue
                wrist_cfg = camera_config.get(topic) or {}
                if wrist_cfg.get('enable_zmq'):
                    wrist_panels.append(side)
        wrist_image_shape = (camera_config.get('left_wrist_camera') or {}).get('image_shape') or [480, 640]
        # Both palm frames are quarter-turned on their way to the panels (see
        # TeleVuerWrapper.WRIST_PANEL_ROTATION), so the panel takes the camera's
        # portrait shape and aspect rather than its native landscape one.
        wrist_panel_shape = (int(wrist_image_shape[1] * 0.5), int(wrist_image_shape[0] * 0.5))
        wrist_panel_aspect = float(wrist_panel_shape[1]) / float(wrist_panel_shape[0])
        if wrist_panels:
            logger_mp.info(
                f"[XR WRIST PANELS] sides={wrist_panels} "
                f"(height={args.wrist_panel_height:.2f}m distance={args.wrist_panel_distance:.2f}m "
                f"offset={list(args.wrist_panel_offset)})"
            )
        else:
            logger_mp.info(
                f"[XR WRIST PANELS] disabled (display={args.display_mode}, wrist_display={args.wrist_display})"
            )

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
                                     wrist_panels=tuple(wrist_panels),
                                     wrist_panel_height=args.wrist_panel_height,
                                     wrist_panel_distance=args.wrist_panel_distance,
                                     wrist_panel_offset=tuple(args.wrist_panel_offset),
                                     wrist_panel_aspect=wrist_panel_aspect,
                                     wrist_panel_shape=wrist_panel_shape,
                                     arm_reference_mode="head_yaw"
                                     )

        def grab_wrist_frames():
            """Fetch the wrist frames the XR panels (and recording) need.

            Called from the pre-start waiting loop as well as the control loop, so
            the panels are live from the moment the headset page connects instead
            of only after the operator arms the robot.
            """
            left_frame = right_frame = None
            if camera_config['left_wrist_camera']['enable_zmq'] and (args.record or 'left' in wrist_panels):
                left_frame = correct_palm_frame(img_client.get_left_wrist_frame())
                if left_frame is not None and 'left' in wrist_panels:
                    tv_wrapper.render_wrist_to_xr('left', left_frame.bgr)
            if camera_config['right_wrist_camera']['enable_zmq'] and (args.record or 'right' in wrist_panels):
                right_frame = correct_palm_frame(img_client.get_right_wrist_frame())
                if right_frame is not None and 'right' in wrist_panels:
                    tv_wrapper.render_wrist_to_xr('right', right_frame.bgr)
            return left_frame, right_frame

        if args.ee == "linker_o6":
            from teleop.robot_control.linker_o6_retargeting import DualLinkerO6Retargeter

            linker_o6_retargeter = DualLinkerO6Retargeter(
                args.linker_o6_urdf_root, method=args.linker_o6_method
            )
            mapping_name = linker_o6_retargeter.mapping_name
            logger_mp.info(f"Linker O6 dex-retargeting: method={linker_o6_retargeter.method} mapping={mapping_name}")

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
            logger_mp.warning("Linker O6 dry-run uses dex-retargeting without the previous gesture calibration; no DDS command publisher exists in this path.")
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
                        reason = recording_blocked_by_tracking(
                            tracking_diagnostics(tv_wrapper),
                            getattr(args, "record_max_tracking_age_ms", 100.0),
                        )
                        if reason:
                            logger_mp.warning("[RECORD] not started: %s", reason)
                            continue
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
                        linker_o6_retargeter.reset()
                        status_payload = {
                            "schema": "linker_o6_target_v1",
                            "armed": False,
                            "reason": "tracking_stale",
                            "mapping": mapping_name,
                            "retargeting_method": linker_o6_retargeter.method,
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
                                "retargeting_method": linker_o6_retargeter.method,
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
                    left_target, right_target = linker_o6_retargeter.retarget(
                        tele_data.left_hand_pos,
                        tele_data.right_hand_pos,
                    )
                    payload = {
                        "schema": "linker_o6_target_v1",
                        "armed": True,
                        "mapping": mapping_name,
                        "retargeting_method": linker_o6_retargeter.method,
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
                        "hand_points_frame": "televuer_unitree_hand_wrist_local_meters",
                        "left_hand_points": tele_data.left_hand_pos.tolist(),
                        "right_hand_points": tele_data.right_hand_pos.tolist(),
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
                        arm_velocity_limit=args.arm_velocity_limit,
                        dq_feedforward=args.arm_dq_feedforward == 'on',
                        dq_feedforward_limit=args.arm_dq_limit,
                        dq_feedforward_filter=args.arm_dq_filter,
                        target_velocity_limit=args.arm_target_velocity_limit,
                        target_accel_limit=args.arm_target_accel_limit,
                    )
                    arm_ik = None
                else:
                    arm_ik = R1_A7_ArmIK()
                    arm_ctrl = R1_A7_ArmController(
                        motion_mode=args.motion, simulation_mode=args.sim,
                        arm_velocity_limit=args.arm_velocity_limit,
                        dq_feedforward=args.arm_dq_feedforward == 'on',
                        dq_feedforward_limit=args.arm_dq_limit,
                        dq_feedforward_filter=args.arm_dq_filter,
                        target_velocity_limit=args.arm_target_velocity_limit,
                        target_accel_limit=args.arm_target_accel_limit,
                    )

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
            metadata = None
            image_height, image_width = camera_config['head_camera']['image_shape']
            if camera_config['head_camera']['binocular']:
                image_width //= 2
            if r1_a7_deferred_real and args.ee == "linker_o6":
                from teleop.utils.r1_capture import R1Capture, capture_metadata
                metadata = capture_metadata(args, camera_config, linker_o6_retargeter, camera_calibration)
                calibration = metadata["camera_calibration"]
                logger_mp.info(
                    f"[RECORD] colour streams: {', '.join(sorted(metadata['images']))} "
                    f"(color_2/color_3 are the palm cameras)"
                )
                if calibration["status"] == "uncalibrated":
                    logger_mp.warning(
                        "[RECORD] camera_calibration status=uncalibrated; episodes will carry no intrinsics "
                        "and can never be undistorted or aligned afterwards."
                    )
                else:
                    logger_mp.info(
                        f"[RECORD] camera_calibration status={calibration['status']} "
                        f"cameras={', '.join(sorted(calibration['cameras']))}"
                    )
                for warning in calibration["warnings"]:
                    logger_mp.warning(f"[RECORD] camera_calibration: {warning}")
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     image_size = (image_width, image_height),
                                     metadata = metadata,
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        if r1_a7_anchored:
            logger_mp.info("🟢  Press [r] once to prepare R1_A7; following starts automatically when both hands are stable.")
        else:
            logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
            logger_mp.info("[RECORD] While recording: [y] save success; [n] save failure; [x] discard. [s] leaves outcome unspecified.")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        if r1_a7_anchored:
            START = False
        r1_arm_request_floor = ARM_REQUEST_GENERATION
        r1_start_requested = False
        r1_activation_prepared = False
        r1_startup_sample_floor = 0.0
        r1_startup_fresh_since = None
        r1_startup_last_timestamp = 0.0
        r1_startup_fresh_samples = 0
        r1_startup_tracking_ready = False
        tracking_diagnostic_next_time = 0.0
        READY = True                  # now ready to (1) enter START state
        while (
            not STOP
            and (
                not START
                or (r1_a7_anchored and r1_vision_left_reference is None)
            )
        ):
            time.sleep(0.033)
            if STOP:
                break
            if r1_a7_anchored:
                arm_ctrl.raise_if_failed()
            if r1_a7_deferred_real and time.monotonic() >= tracking_diagnostic_next_time:
                tracking_diagnostic_next_time = time.monotonic() + 2.0
                logger_mp.info(
                    "[XR TRACKING] " + json.dumps(tv_wrapper.tvuer.get_tracking_diagnostics())
                )
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            grab_wrist_frames()

            if r1_a7_anchored and ARM_REQUEST_GENERATION > r1_arm_request_floor:
                r1_arm_request_floor = ARM_REQUEST_GENERATION
                r1_start_requested = True
                START = False
                if not r1_activation_prepared:
                    logger_mp.info("[R1 STARTUP] Start requested; waiting for fresh hands before recentering. [q] cancels.")

            if r1_activation_prepared:
                # Waiting for XR is allowed; stale robot feedback is not.
                arm_ctrl.get_current_waist_yaw()
                startup_tele_data = tv_wrapper.get_tele_data()
                startup_now = time.monotonic()
                startup_timestamp = startup_tele_data.motion_data_timestamp
                if (
                    not is_fresh_motion_data(startup_tele_data, args.tracking_timeout, now=startup_now)
                    or startup_timestamp <= r1_startup_sample_floor
                ):
                    if r1_startup_tracking_ready:
                        logger_mp.warning("[R1 STARTUP] Tracking lost; holding posture and waiting for both hands. No recentering.")
                    r1_startup_fresh_since = None
                    r1_startup_last_timestamp = 0.0
                    r1_startup_fresh_samples = 0
                    r1_startup_tracking_ready = False
                    START = False
                else:
                    if startup_timestamp > r1_startup_last_timestamp:
                        if r1_startup_fresh_since is None:
                            r1_startup_fresh_since = startup_now
                        r1_startup_last_timestamp = startup_timestamp
                        r1_startup_fresh_samples += 1
                    if not r1_startup_tracking_ready:
                        START = False
                        if startup_now - r1_startup_fresh_since >= 0.35 and r1_startup_fresh_samples >= 5:
                            r1_startup_tracking_ready = True
                            logger_mp.info("[R1 STARTUP] Both hands stable; automatically starting following. No additional key or recentering.")

            if (
                r1_a7_anchored
                and r1_start_requested
                and (not r1_activation_prepared or r1_startup_tracking_ready)
            ):
                r1_arm_request_floor = ARM_REQUEST_GENERATION
                START = False
                reference_tele_data = tv_wrapper.get_tele_data()
                if not is_fresh_motion_data(
                    reference_tele_data,
                    args.tracking_timeout,
                ):
                    r1_startup_fresh_since = None
                    r1_startup_last_timestamp = 0.0
                    r1_startup_fresh_samples = 0
                    r1_startup_tracking_ready = False
                    continue
                if not r1_activation_prepared:
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
                    try:
                        arm_ctrl.defer_publishing()
                        arm_ctrl.activate(cancel_requested=lambda: STOP)
                    except InterruptedError:
                        if STOP:
                            continue
                        raise
                    if STOP:
                        continue
                    post_recenter_motor_q = arm_ctrl.get_current_motor_q()
                    r1_waist_yaw_reference = float(post_recenter_motor_q[13])
                    arm_ik = R1_A7_ArmIK(waist_yaw=r1_waist_yaw_reference)
                    # Prefer the posture the operator actually started from, and let the
                    # soft barrier keep the solver off the joint limits during long sessions.
                    arm_ik.set_redundancy_weights(
                        posture_weight=args.arm_posture_weight,
                        limit_weight=args.arm_limit_softness,
                        nominal_arm_q=post_recenter_motor_q[:14],
                    )
                    # Always reported: these two weights are what keep the elbow from
                    # drifting into a twisted pose, and a silent zero here is exactly
                    # how that protection went missing before.
                    logger_mp.info(
                        "[R1 IK] redundancy terms: limit_softness=%.3g posture_weight=%.3g "
                        "(%s) nominal=activation posture",
                        args.arm_limit_softness, args.arm_posture_weight,
                        "elbow drift protection on" if args.arm_posture_weight > 0.0
                        else "elbow drift protection OFF",
                    )
                    # Always reported while waist following is on: this one decides whether
                    # turning the waist rotates the whole arm with the body or makes it fold to
                    # hold the hand in world space, and the difference is felt immediately.
                    if args.waist_follow:
                        logger_mp.info(
                            "[R1 WAIST] follow=on compensation=%s (%s)",
                            getattr(args, "waist_follow_compensation", "torso"),
                            "the arm keeps its posture relative to the torso and rides the waist"
                            if getattr(args, "waist_follow_compensation", "torso") == 'torso'
                            else "the target is counter-rotated to hold the hand in world space",
                        )
                    arm_ctrl.start_publishing()
                    r1_waist_to_root = np.array([
                        [math.cos(r1_waist_yaw_reference), -math.sin(r1_waist_yaw_reference), 0.0],
                        [math.sin(r1_waist_yaw_reference), math.cos(r1_waist_yaw_reference), 0.0],
                        [0.0, 0.0, 1.0],
                    ])
                    initial_head_yaw_reference = head_yaw_rotation(reference_tele_data.head_pose)
                    initial_head_pose_reference = reference_tele_data.head_pose.copy()
                    vision_left_initial = reference_tele_data.left_wrist_pose.copy()
                    vision_right_initial = reference_tele_data.right_wrist_pose.copy()
                    if getattr(args, "startup_wrist_align", False):
                        current_q = arm_ctrl.get_current_dual_arm_q()
                        current_dq = arm_ctrl.get_current_dual_arm_dq()
                        robot_left_pose, robot_right_pose = arm_ik.forward_wrist_poses(current_q)
                        left_initial_pose = wrist_in_reference_head_yaw_frame(
                            vision_left_initial,
                            initial_head_pose_reference,
                            initial_head_yaw_reference,
                            initial_head_pose_reference[:3, 3],
                        )
                        right_initial_pose = wrist_in_reference_head_yaw_frame(
                            vision_right_initial,
                            initial_head_pose_reference,
                            initial_head_yaw_reference,
                            initial_head_pose_reference[:3, 3],
                        )
                        left_align_target = robot_left_pose.copy()
                        right_align_target = robot_right_pose.copy()
                        left_align_target[:3, :3] = left_initial_pose[:3, :3]
                        right_align_target[:3, :3] = right_initial_pose[:3, :3]
                        arm_ik.reset_smoothing(reference_q=current_q)
                        align_q, align_tau = arm_ik.solve_ik(
                            left_align_target,
                            right_align_target,
                            current_q,
                            current_dq,
                            raise_on_failure=True,
                        )
                        arm_ctrl.ctrl_dual_arm(align_q, align_tau)
                        align_deadline = time.monotonic() + 8.0
                        align_next_ik = time.monotonic() + 0.1
                        while time.monotonic() < align_deadline and not STOP:
                            now = time.monotonic()
                            if now >= align_next_ik:
                                current_align_q = arm_ctrl.get_current_dual_arm_q()
                                current_align_dq = arm_ctrl.get_current_dual_arm_dq()
                                align_q, align_tau = arm_ik.solve_ik(
                                    left_align_target,
                                    right_align_target,
                                    current_align_q,
                                    current_align_dq,
                                    raise_on_failure=True,
                                )
                                align_next_ik = now + 0.1
                            arm_ctrl.ctrl_dual_arm(align_q, align_tau)
                            current_align_q = arm_ctrl.get_current_dual_arm_q()
                            current_left_pose, current_right_pose = arm_ik.forward_wrist_poses(current_align_q)
                            left_position_error = np.linalg.norm(
                                current_left_pose[:3, 3] - left_align_target[:3, 3]
                            )
                            right_position_error = np.linalg.norm(
                                current_right_pose[:3, 3] - right_align_target[:3, 3]
                            )
                            left_rotation_error = rotation_error_rad(
                                current_left_pose[:3, :3], left_align_target[:3, :3]
                            )
                            right_rotation_error = rotation_error_rad(
                                current_right_pose[:3, :3], right_align_target[:3, :3]
                            )
                            if (
                                max(left_position_error, right_position_error) <= 0.02
                                and max(left_rotation_error, right_rotation_error) <= 0.12
                            ):
                                break
                            time.sleep(0.02)
                        if STOP:
                            continue
                        aligned_q = arm_ctrl.get_current_dual_arm_q()
                        aligned_left_pose, aligned_right_pose = arm_ik.forward_wrist_poses(aligned_q)
                        aligned_position_error = max(
                            np.linalg.norm(aligned_left_pose[:3, 3] - left_align_target[:3, 3]),
                            np.linalg.norm(aligned_right_pose[:3, 3] - right_align_target[:3, 3]),
                        )
                        aligned_rotation_error = max(
                            rotation_error_rad(aligned_left_pose[:3, :3], left_align_target[:3, :3]),
                            rotation_error_rad(aligned_right_pose[:3, :3], right_align_target[:3, :3]),
                        )
                        if aligned_position_error > 0.02 or aligned_rotation_error > 0.12:
                            logger_mp.warning(
                                "[R1 STARTUP] Absolute wrist alignment was not reachable; "
                                f"position_error_m={aligned_position_error:.3f} "
                                f"rotation_error_rad={aligned_rotation_error:.3f}; "
                                "continuing from the current robot posture."
                            )
                        else:
                            logger_mp.info("[R1 STARTUP] Wrist orientation aligned to initial Vision Pro pose.")
                    if STOP:
                        continue
                    r1_activation_prepared = True
                    r1_startup_sample_floor = time.monotonic()
                    r1_arm_request_floor = ARM_REQUEST_GENERATION
                    logger_mp.info(
                        "[R1 STARTUP] Recenter and IK ready. Holding posture; waiting for both hands, "
                        "following will start automatically once tracking is stable. [q] exits."
                    )
                    continue
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
                    waist_follower = waist_follower_from_args(
                        args, r1_waist_yaw_reference, time.monotonic(),
                    )
                    logger_mp.info(
                        "R1_A7 head/waist following enabled: %.4g deg residual for %.4g s to engage, "
                        "release at %.4g deg, %.4g deg/s with a %.4g deg/s^2 ramp, "
                        "compensation=%s, URDF waist range +/-2.618 rad.",
                        math.degrees(waist_follower.engage_threshold),
                        waist_follower.engage_duration,
                        math.degrees(waist_follower.release_threshold),
                        math.degrees(waist_follower.max_velocity),
                        math.degrees(waist_follower.max_acceleration),
                        getattr(args, "waist_follow_compensation", "torso"),
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
        r1_head_q_offset = np.zeros(2)
        r1_frozen_generation = -1
        if r1_independent_hands and args.ee in (None, "linker_o6"):
            from teleop.robot_control.r1_pause import R1PauseState
            R1_PAUSE = R1PauseState()
            logger_mp.info("[R1 CONTROL] [p] pauses and holds; [r] realigns and resumes after stable hands; [q] exits.")
        if r1_independent_hands:
            wrist_holds = (R1WristHold(r1_robot_left_reference), R1WristHold(r1_robot_right_reference))
            held_head_q_target = arm_ctrl.get_current_head_q().copy()

        if args.ee == "linker_o6":
            from teleop.robot_control.linker_o6_control_loop import LinkerO6ControlLoop

            def save_hand_sample(states, actions):
                with dual_hand_data_lock:
                    dual_hand_state_array[:] = np.concatenate(states)
                    dual_hand_action_array[:] = np.concatenate(actions)

            save_hand_sample(hand_ctrl.get_state(), hand_ctrl.get_action())
            linker_o6_loop = LinkerO6ControlLoop(
                tv_wrapper.get_tele_data, linker_o6_retargeter, hand_ctrl,
                args.frequency, args.tracking_timeout, lambda: STOP, save_hand_sample,
                should_pause=lambda: R1_PAUSE is not None and R1_PAUSE.paused,
            )
            linker_o6_loop.start()

        if args.record and r1_a7_deferred_real and args.ee == "linker_o6":
            # The palm cameras are recorded on the same samples as the head
            # stereo. Their streams are independent and run at ~28-29 fps
            # against this 30 Hz loop, so a repeated frame is normal; a stream
            # that never delivers fails the episode instead of silently
            # producing an episode without its wrist observation.
            wrist_image_shapes = {
                side: camera_config[f"{side}_wrist_camera"]["image_shape"]
                for side in ("left", "right")
                if (camera_config.get(f"{side}_wrist_camera") or {}).get("enable_zmq")
            }
            r1_capture = R1Capture(arm_ctrl, hand_ctrl, linker_o6_loop,
                                   args.tracking_timeout, camera_config['head_camera']['image_shape'],
                                   wrist_image_shapes=wrist_image_shapes,
                                   sync_tolerance_ms=args.camera_sync_tolerance_ms)
            logger_mp.info(
                f"[RECORD] camera sync: nearest frame to the head frame, "
                f"tolerance {args.camera_sync_tolerance_ms:.1f} ms"
            )

        # main loop. robot start to follow VR user's motion
        while not STOP:
            if r1_a7_anchored:
                arm_ctrl.raise_if_failed()
            if linker_o6_loop is not None:
                linker_o6_loop.raise_if_failed()
            if r1_a7_deferred_real and time.monotonic() >= tracking_diagnostic_next_time:
                tracking_diagnostic_next_time = time.monotonic() + 2.0
                logger_mp.info(
                    "[XR TRACKING] " + json.dumps(tv_wrapper.tvuer.get_tracking_diagnostics())
                )
                if RECORD_RUNNING:
                    reason = recording_blocked_by_tracking(
                        tracking_diagnostics(tv_wrapper),
                        getattr(args, "record_max_tracking_age_ms", 100.0),
                    )
                    if reason:
                        logger_mp.warning("[RECORD] XR tracking degraded while recording: %s", reason)
            start_time = time.time()
            loop_monotonic = time.monotonic()
            loop_period_ms = (
                None
                if arm_control_previous_time is None
                else 1000.0 * (loop_monotonic - arm_control_previous_time)
            )
            arm_control_previous_time = loop_monotonic
            if loop_period_ms is not None:
                loop_jitter["iterations"] += 1
                if loop_jitter_started is None:
                    loop_jitter_started = loop_monotonic
                if loop_period_ms > loop_jitter["max_ms"]:
                    loop_jitter["max_ms"] = loop_period_ms
                    loop_jitter["worst_at_s"] = loop_monotonic - loop_jitter_started
                if loop_period_ms > 40.0:
                    loop_jitter["over_40ms"] += 1
                if loop_period_ms > 80.0:
                    loop_jitter["over_80ms"] += 1
                    # Report a real stall while it happens instead of only at exit.
                    logger_mp.warning(
                        "[R1 LOOP STALL] iteration took %.0f ms (budget %.0f ms); "
                        "capture_mode=%s tracking_hold=%s",
                        loop_period_ms, 1000.0 / args.frequency,
                        capture_mode, tracking_hold_active,
                    )
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            # The panels need the frames even when recording is off; the fetch is a
            # cached ring-buffer read (measured 0.002 ms) plus one resize.
            left_wrist_img, right_wrist_img = grab_wrist_frames()
            if r1_capture is not None:
                # Keep the pairing history warm while not recording, so the first
                # sample of an episode already has frames to choose between.
                r1_capture.observe(head_img, {"left": left_wrist_img, "right": right_wrist_img})

            # record mode
            if args.record:
                recorder.raise_if_failed()
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    reason = recording_blocked_by_tracking(
                        tracking_diagnostics(tv_wrapper),
                        getattr(args, "record_max_tracking_age_ms", 100.0),
                    )
                    if reason:
                        logger_mp.warning("[RECORD] not started: %s", reason)
                    elif recorder.create_episode():
                        RECORD_RUNNING = True
                        RECORD_OUTCOME = 'unspecified'
                        if r1_capture is not None:
                            r1_capture.reset_episode()
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode(outcome=RECORD_OUTCOME)
                    RECORD_OUTCOME = 'unspecified'
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            capture_mode = "following"
            run_motion = True
            if R1_PAUSE is not None and R1_PAUSE.paused:
                capture_mode = "paused"
                run_motion = False
                pause_generation = R1_PAUSE.generation
                if r1_frozen_generation != pause_generation:
                    hold_r1_published_targets(arm_ctrl)
                    r1_frozen_generation = pause_generation
                arm_ctrl.hold_targets()
                resume_generation = R1_PAUSE.poll_resume(
                    tele_data, args.tracking_timeout, time.monotonic(),
                )
                if resume_generation is not None:
                    held = arm_ctrl.get_recording_snapshot()["requested"]
                    held_q = np.array(held["arm_q"])
                    r1_robot_left_reference, r1_robot_right_reference = arm_ik.forward_wrist_poses(held_q)
                    actual_waist_yaw = arm_ctrl.get_current_waist_yaw()
                    if args.waist_follow:
                        # world mode records the anchor in the robot root frame and lets the
                        # per-tick counter-rotation bring it back; torso mode keeps the FK
                        # result in the fixed-waist IK frame the solver actually works in.
                        if getattr(args, "waist_follow_compensation", "torso") == 'world':
                            r1_robot_left_reference, r1_robot_right_reference = (
                                compensate_wrist_for_waist(pose, r1_waist_yaw_reference, actual_waist_yaw)
                                for pose in (r1_robot_left_reference, r1_robot_right_reference)
                            )
                        waist_follower = waist_follower_from_args(
                            args, actual_waist_yaw, time.monotonic(),
                        )
                    r1_waist_to_root = np.array([
                        [math.cos(actual_waist_yaw), -math.sin(actual_waist_yaw), 0.0],
                        [math.sin(actual_waist_yaw), math.cos(actual_waist_yaw), 0.0],
                        [0.0, 0.0, 1.0],
                    ])
                    r1_vision_left_reference = tele_data.left_wrist_pose.copy()
                    r1_vision_right_reference = tele_data.right_wrist_pose.copy()
                    r1_head_pose_reference = tele_data.head_pose.copy()
                    r1_head_yaw_reference = head_yaw_rotation(tele_data.head_pose)
                    r1_head_q_offset = np.array(held["head_q"])
                    held_head_q_target = r1_head_q_offset.copy()
                    wrist_holds = (R1WristHold(r1_robot_left_reference), R1WristHold(r1_robot_right_reference))
                    arm_ik.reset_smoothing(reference_q=held_q)
                    last_fresh_tele_data = tele_data
                    tracking_hold_active = False
                    if R1_PAUSE.complete_resume(resume_generation):
                        logger_mp.info("[R1 PAUSE] References realigned at held targets; following resumes next frame.")
            if r1_independent_hands:
                raw_hand_present = hand_tracking_present(tele_data)
                hand_present = []
                for index, (side, hold, present) in enumerate(zip(("left", "right"), wrist_holds, raw_hand_present)):
                    hold.resume_streak = hold.resume_streak + 1 if present else 0
                    hand_present.append(present and (hold.tracking or hold.resume_streak >= 2))
                hand_present = tuple(hand_present)
                for side, hold, present in zip(("left", "right"), wrist_holds, hand_present):
                    if present != hold.tracking and capture_mode != "paused":
                        logger_mp.info(f"[R1 HAND TRACKING] {side}: {'resumed; position follows fixed reference via IK filter' if present else 'lost; holding target'}")
                    if not present:
                        hold.hold()
                if args.waist_follow:
                    if not all(hand_present):
                        arm_ctrl.hold_waist()
                    elif tracking_hold_active:
                        waist_follower.reset(arm_ctrl.get_current_waist_yaw(), time.monotonic())
                tracking_hold_active = not all(hand_present)
                if tracking_hold_active and capture_mode != "paused":
                    capture_mode = "tracking_hold"
                if not any(hand_present):
                    run_motion = False
            elif r1_a7_anchored:
                if is_present_motion_data(tele_data):
                    if is_fresh_motion_data(tele_data, args.tracking_timeout):
                        last_fresh_tele_data = tele_data
                    else:
                        tele_data = last_fresh_tele_data
                    if tracking_hold_active:
                        arm_ik.reset_smoothing()
                        if args.waist_follow:
                            waist_follower.reset(arm_ctrl.get_current_waist_yaw(), time.monotonic())
                        logger_mp.info("R1_A7 Vision tracking resumed.")
                    tracking_hold_active = False
                else:
                    tele_data = last_fresh_tele_data
                    if not tracking_hold_active:
                        arm_ik.reset_smoothing()
                        logger_mp.warning(
                            "R1_A7 Vision tracking is lost; holding the last tracked pose until tracking returns."
                        )
                    tracking_hold_active = True
                if args.waist_follow and tracking_hold_active:
                    # Do not refresh the waist watchdog with a cached XR sample.
                    arm_ctrl.hold_waist()
                    capture_mode = "tracking_hold"
                    run_motion = False
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
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
            # Measured torque, when the controller exposes it. A missing getter must
            # not break the loop, so this degrades to None.
            tau_getter = getattr(arm_ctrl, "get_current_dual_arm_tau", None)
            try:
                arm_tau_actual = None if tau_getter is None else tau_getter()
            except RuntimeError:
                arm_tau_actual = None

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            waist_yaw_actual = arm_ctrl.get_current_waist_yaw() if args.waist_follow else None
            waist_yaw_target = None
            if r1_independent_hands:
                left_wrist_target, right_wrist_target = (hold.target.copy() for hold in wrist_holds)
            if run_motion:
                left_wrist_target = tele_data.left_wrist_pose
                right_wrist_target = tele_data.right_wrist_pose
                if r1_a7_anchored:
                    head_q_target = r1_head_q_offset + relative_head_pitch_yaw(
                        tele_data.head_pose,
                        r1_head_pose_reference,
                    )
                    # torso mode takes the wrapper's pose as it comes. televuer already
                    # expresses the wrist in the CURRENT head-yaw frame with the origin moved
                    # from the head to the waist, so when the operator turns their body the
                    # hand, the head and that frame all rotate together and this pose does not
                    # move at all -- exactly the hand's pose relative to their own torso, which
                    # is what the arm should reproduce. Re-referencing it to the activation yaw
                    # below converts it to world coordinates, which is what world mode and the
                    # waist-off path need; feeding world coordinates to a fixed-waist solver
                    # while the real waist turns makes the hand travel twice the body rotation.
                    if args.waist_follow and getattr(
                        args, "waist_follow_compensation", "torso"
                    ) == 'torso':
                        left_wrist_pose = tele_data.left_wrist_pose
                        right_wrist_pose = tele_data.right_wrist_pose
                    else:
                        left_wrist_pose = wrist_in_reference_head_yaw_frame(
                            tele_data.left_wrist_pose,
                            tele_data.head_pose,
                            r1_head_yaw_reference,
                            r1_head_pose_reference[:3, 3],
                        )
                        right_wrist_pose = wrist_in_reference_head_yaw_frame(
                            tele_data.right_wrist_pose,
                            tele_data.head_pose,
                            r1_head_yaw_reference,
                            r1_head_pose_reference[:3, 3],
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
                    left_wrist_target = wrist_holds[0].prepare(left_wrist_target, hand_present[0])
                    right_wrist_target = wrist_holds[1].prepare(right_wrist_target, hand_present[1])
                    if not all(hand_present):
                        head_q_target = held_head_q_target.copy()
                left_ik_target = left_wrist_target
                right_ik_target = right_wrist_target
                waist_yaw_actual = None
                waist_yaw_target = None
                if args.waist_follow:
                    waist_yaw_actual = arm_ctrl.get_current_waist_yaw()
                    if not r1_independent_hands or all(hand_present):
                        head_q_target, waist_yaw_target = waist_follower.update(
                            tele_data.head_pose, r1_head_pose_reference,
                            waist_yaw_actual, time.monotonic(),
                        )
                        head_q_target = head_q_target + r1_head_q_offset
                    # torso mode feeds the anchored target straight to the solver: the IK
                    # model has the waist locked at r1_waist_yaw_reference, so the solved
                    # joints describe the arm relative to the torso, which is exactly the
                    # operator's own arm-to-torso relationship. The real waist then carries
                    # that whole posture with it. world mode counter-rotates the target about
                    # the pelvis axis instead, and the arm has to fold to hold the hand still.
                    if getattr(args, "waist_follow_compensation", "torso") == 'world':
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
            if r1_independent_hands and run_motion:
                present_after_ik = hand_tracking_present(tele_data)
                if any(before and not after for before, after in zip(hand_present, present_after_ik)):
                    arm_ik.reset_smoothing()
                    for hold, present in zip(wrist_holds, present_after_ik):
                        if not present:
                            hold.hold()
                    if args.waist_follow:
                        arm_ctrl.hold_waist()
                    tracking_hold_active = True
                    capture_mode = "tracking_hold"
                    run_motion = False
            elif run_motion and args.waist_follow and not is_present_motion_data(tele_data):
                arm_ik.reset_smoothing()
                arm_ctrl.hold_waist()
                tracking_hold_active = True
                capture_mode = "tracking_hold"
                run_motion = False
            # Count hold transitions once per event; a long single-side loss is one
            # event, not thousands of frames, so the exit summary stays readable.
            if tracking_hold_active and not tracking_hold_previous:
                tracking_hold_events += 1
            tracking_hold_previous = tracking_hold_active
            if linker_o6_loop is not None:
                linker_o6_loop.raise_if_failed()
            if STOP:
                break
            if R1_PAUSE is not None and R1_PAUSE.paused:
                capture_mode = "paused"
                run_motion = False
                pause_generation = R1_PAUSE.generation
                if r1_frozen_generation != pause_generation:
                    hold_r1_published_targets(arm_ctrl)
                    r1_frozen_generation = pause_generation
            if run_motion:
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
            else:
                arm_ctrl.hold_targets()
                held = arm_ctrl.get_recording_snapshot()["requested"]
                sol_q = np.array(held["arm_q"])
                sol_tauff = np.array(held["arm_tau"])
                head_q_target = np.array(held["head_q"])
                if r1_independent_hands:
                    left_wrist_target, right_wrist_target = (hold.target.copy() for hold in wrist_holds)
                else:
                    left_wrist_target = r1_robot_left_reference.copy()
                    right_wrist_target = r1_robot_right_reference.copy()
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
                if not run_motion:
                    left_ik_target, right_ik_target = solved_left_pose.copy(), solved_right_pose.copy()
                    left_wrist_pose = wrist_in_reference_head_yaw_frame(
                        tele_data.left_wrist_pose, tele_data.head_pose,
                        r1_head_yaw_reference, r1_head_pose_reference[:3, 3],
                    )
                    right_wrist_pose = wrist_in_reference_head_yaw_frame(
                        tele_data.right_wrist_pose, tele_data.head_pose,
                        r1_head_yaw_reference, r1_head_pose_reference[:3, 3],
                    )
                # The solver reports position/orientation as soft costs, so an
                # unreachable target silently under-follows. Surface it instead.
                #
                # This runs before the root-frame conversion below on purpose: the FK above
                # is in the fixed-waist IK frame and so are the *_ik_target values, while
                # the conversion rotates the FK to the actual root frame. Comparing across
                # the two charged the waist angle itself as solver error -- measured on the
                # 2026-09-16 waist run, 100% of samples read as "outside the workspace"
                # beyond 10 deg of waist while only about 36% actually were, which is how
                # waist following got blamed for destroying the workspace.
                workspace_saturation = r1_workspace_saturation(
                    solved_left_pose, solved_right_pose, left_ik_target, right_ik_target,
                    args.workspace_position_tolerance_m, args.workspace_rotation_tolerance_rad,
                )
                if args.waist_follow:
                    # Fixed-waist IK FK must be returned to the actual robot root frame
                    # before it is recorded.
                    actual_left_pose, actual_right_pose, solved_left_pose, solved_right_pose = (
                        compensate_wrist_for_waist(pose, r1_waist_yaw_reference, waist_yaw_actual)
                        for pose in (actual_left_pose, actual_right_pose, solved_left_pose, solved_right_pose)
                    )
                if not run_motion:
                    left_wrist_target, right_wrist_target = solved_left_pose.copy(), solved_right_pose.copy()
                workspace_outside = tuple(
                    side for side, item in workspace_saturation.items() if item["outside"]
                )
                if workspace_outside and diagnostic_now >= workspace_diagnostic_next_time:
                    workspace_diagnostic_next_time = diagnostic_now + 1.0
                    logger_mp.warning(
                        "[R1 WORKSPACE] %s target is outside the reachable workspace; "
                        "the arm is following as far as it can. shortfall position=%.3f m rotation=%.2f rad. "
                        "Move closer to the body or lower --arm-translation-scale.",
                        "/".join(workspace_outside),
                        max(workspace_saturation[side]["position_m"] for side in workspace_outside),
                        max(workspace_saturation[side]["rotation_rad"] for side in workspace_outside),
                    )
                    if workspace_warned_side != workspace_outside:
                        workspace_saturation_events += 1
                        workspace_warned_side = workspace_outside
                elif not workspace_outside:
                    workspace_warned_side = None
                arm_diagnostic_sequence += 1
                arm_diagnostic_next_time = diagnostic_now + 1.0 / args.arm_diagnostic_hz
                write_json_line(arm_diagnostic_file, {
                    "schema": "r1_a7_alignment_v1",
                    "event": "sample",
                    "sequence": arm_diagnostic_sequence,
                    "wall_time_ns": time.time_ns(),
                    "monotonic_time_ns": time.monotonic_ns(),
                    "source_monotonic_time": tele_data.motion_data_timestamp,
                    "tracking_age_ms": 1000.0 * (diagnostic_now - tele_data.motion_data_timestamp),
                    "hand_tracking": {
                        side: {
                            "fresh": age_fresh,
                            "present": present,
                            "age_ms": 1000.0 * (diagnostic_now - timestamp),
                        }
                        for side, age_fresh, present, timestamp in zip(
                            ("left", "right"),
                            hand_tracking_freshness(tele_data, args.tracking_timeout, diagnostic_now),
                            hand_present,
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
                    "q_reference_command": (
                        arm_ctrl.get_reference_q().tolist()
                        if hasattr(arm_ctrl, "get_reference_q") else sol_q.tolist()
                    ),
                    "arm_target_shaper": (
                        arm_ctrl.get_target_shaper_snapshot()
                        if hasattr(arm_ctrl, "get_target_shaper_snapshot") else None
                    ),
                    "tau_ik_command": sol_tauff.tolist(),
                    "tau_actual": arm_tau_actual.tolist() if arm_tau_actual is not None else None,
                    "workspace": workspace_saturation,
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
            if args.record and r1_capture is not None:
                if RECORD_RUNNING:
                    recorder.add_item(**r1_capture.frame(
                        tele_data, head_img, capture_mode,
                        {"left": left_wrist_img, "right": right_wrist_img}))
            elif args.record:
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
    except Exception as error:
        exit_code = 1
        failure_reason = str(error)
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        active_exception = sys.exc_info()[1]
        if isinstance(active_exception, SystemExit) and active_exception.code not in (None, 0):
            exit_code = active_exception.code if isinstance(active_exception.code, int) else 1
        try:
            # One readable line per session: a long single-side tracking loss is
            # otherwise only visible by digging through the alignment JSONL.
            if r1_a7_anchored and arm_ik is not None:
                logger_mp.info(
                    "[R1 SESSION SUMMARY] tracking_holding_events=%d workspace_saturation_events=%d "
                    "diagnostic_samples=%d. A hold event means Vision Pro zeroed a hand "
                    "(lost/missing), not a Wi-Fi gap; the arm kept its last pose until tracking returned.",
                    tracking_hold_events, workspace_saturation_events, arm_diagnostic_sequence,
                )
                # A stop-and-go arm shows up here as iterations that missed the loop
                # budget, not as a change in the 10 Hz diagnostic averages.
                logger_mp.info(
                    "[R1 LOOP JITTER] budget=%.1fms max=%.1fms over_40ms=%d over_80ms=%d "
                    "worst_at=%.1fs of %d iterations",
                    1000.0 / args.frequency, loop_jitter["max_ms"],
                    loop_jitter["over_40ms"], loop_jitter["over_80ms"],
                    loop_jitter["worst_at_s"] if loop_jitter["worst_at_s"] is not None else -1.0,
                    loop_jitter["iterations"],
                )
        except Exception as e:
            logger_mp.warning(f"Failed to write session summary: {e}")
        try:
            if linker_o6_loop is not None:
                linker_o6_loop.stop()
            elif args.ee == "linker_o6" and hand_ctrl is not None:
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
                    if exit_code:
                        recorder.abort(failure_reason or "Teleoperation cleanup failed")
                    else:
                        recorder.save_episode(outcome=RECORD_OUTCOME)
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

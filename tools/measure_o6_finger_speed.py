#!/usr/bin/env python3
"""Measure how fast Linker O6 fingers follow a step, and what changes that speed.

Moves the four fingers of one hand through repeated strokes for about 80 s while
comparing command write rates, a ramped target, the speed register and one finger
against four. Does not move the arms. Ctrl+C releases the hand but is not an e-stop:
use the robot button in an emergency.

Hardware axis order: thumb_pitch, thumb_yaw, index, middle, ring, pinky.
Normalised q: 0 open, 1 closed. PC2's bridge turns q, dq and tau into device bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

AXES = ("thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky")
FINGERS = (2, 3, 4, 5)
# Thumb away from the palm, so closing fingers cannot reach it.
THUMB_POSE = (0.15, 0.05)
# Short of both ends, so strokes never press a mechanical stop at full torque.
OPEN_Q = 0.10
CLOSED_Q = 0.85
TORQUE = 1.0
APPROACH_SPEED = 0.3
APPROACH_HZ = 40.0
SETTLE_S = 0.6
TICK_S = 0.005

GATE_REQUIRE_RELEASE = 0
GATE_READY = 1
STATE_TIMEOUT_S = 0.2
MAX_TEMPERATURE_C = 65.0
STALL_WINDOW_S = 0.5
STALL_PROGRESS = 0.01
STALL_MIN_ERROR = 0.15

MOVED = 0.02
# Differences smaller than this are treated as measurement spread.
SPEED_MARGIN = 1.15
DEAD_MARGIN_MS = 20.0
DRIFT_MARGIN = 1.10


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    rate_hz: float
    speed: float = 1.0
    ramp_s: float = 0.0
    axes: tuple = FINGERS
    hold_s: float = 1.3


CONDITIONS = (
    Condition("base", "40 Hz 阶跃（基准）", 40.0),
    Condition("w100", "100 Hz 阶跃", 100.0),
    Condition("w10", "10 Hz 阶跃", 10.0),
    # The bridge safe-stops after 250 ms without a command, so 6 Hz is near the slowest.
    Condition("w6", "6 Hz 阶跃", 6.0),
    Condition("ramp", "40 Hz 斜坡 200 ms（接近真人）", 40.0, ramp_s=0.2),
    Condition("one", "40 Hz 阶跃，只动食指", 40.0, axes=(2,)),
    Condition("base2", "40 Hz 阶跃（基准复测）", 40.0),
    # Last: if half speed is slow enough to look blocked, nothing else is lost.
    Condition("slow", "40 Hz 阶跃，速度寄存器减半", 40.0, speed=0.5, hold_s=3.0),
)


class Abort(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


class HandLink:
    def __init__(self, side, domain, interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        self.side = side
        self.rows = []
        self.malformed = 0
        self.writes = 0
        self.publisher = None
        self._publisher_type = ChannelPublisher
        self._subscriber_type = ChannelSubscriber
        self._cmds_type = MotorCmds_
        self._cmd_entry = unitree_go_msg_dds__MotorCmd_
        ChannelFactoryInitialize(domain, networkInterface=interface)
        self._state_sub = ChannelSubscriber(f"rt/linker/{side}/state", MotorStates_)
        self._state_sub.Init(self._on_state, 1)

    def _on_state(self, message):
        now = time.monotonic()
        if len(message.states) != 6:
            self.malformed += 1
            return
        self.rows.append((now, int(message.states[0].mode), tuple(float(s.q) for s in message.states),
                          max(float(s.temperature) for s in message.states)))

    def foreign_commands(self, seconds):
        """Count commands someone else sends. Runs before this process has a publisher."""
        seen = []
        subscriber = self._subscriber_type(f"rt/linker/{self.side}/cmd", self._cmds_type)
        subscriber.Init(lambda message: seen.append(1), 1)
        time.sleep(seconds)
        subscriber.Close()
        return len(seen)

    def open_publisher(self):
        self.publisher = self._publisher_type(f"rt/linker/{self.side}/cmd", self._cmds_type)
        self.publisher.Init()

    def write(self, q, mode, speed, torque):
        message = self._cmds_type()
        message.cmds = [self._cmd_entry() for _ in range(6)]
        for command, value in zip(message.cmds, q):
            command.mode = mode
            command.q = float(value)
            command.dq = speed
            command.tau = torque
        written = time.monotonic()
        if self.publisher.Write(message, timeout=0.1) is not True:
            raise Abort("手部指令发不出去：PC2 的 r1-linker-hand 没有在收")
        self.writes += 1
        return written

    def latest(self):
        if not self.rows:
            raise Abort("没有手部反馈")
        return self.rows[-1]

    def check(self, target, watch_stall):
        """Stop the run on stale feedback, a dropped gate, heat or a blocked finger."""
        now = time.monotonic()
        stamp, mode, q, temperature = self.latest()
        if now - stamp > STATE_TIMEOUT_S:
            raise Abort(f"手部反馈中断 {(now - stamp) * 1000:.0f} ms")
        if mode == GATE_REQUIRE_RELEASE:
            raise Abort("PC2 的指令闸门退回了待释放状态")
        if temperature >= MAX_TEMPERATURE_C:
            raise Abort(f"电机温度 {temperature:.0f}℃，达到 {MAX_TEMPERATURE_C:.0f}℃ 上限")
        if not watch_stall:
            return
        window = int(STALL_WINDOW_S * 100)
        if len(self.rows) <= window:
            return
        then_stamp, _, then_q, _ = self.rows[-1 - window]
        if now - then_stamp < 0.8 * STALL_WINDOW_S:
            return
        for index, axis in enumerate(AXES):
            if abs(q[index] - target[index]) > STALL_MIN_ERROR and abs(q[index] - then_q[index]) < STALL_PROGRESS:
                raise Abort(f"{axis} 停在 {q[index]:.2f}，目标 {target[index]:.2f}：手指可能被挡住")

    def release(self):
        q = self.rows[-1][2] if self.rows else (0.0,) * 6
        for _ in range(3):
            self.write(q, 0, 0.0, 0.0)
            time.sleep(0.02)


def stream(link, target_at, duration, rate_hz, speed, final_target, watch_stall=True, follow_limit=None):
    """Write commands at rate_hz for duration. Returns (first write time, writes sent)."""
    period = 1.0 / rate_hz
    started = time.monotonic()
    next_write = started
    first = None
    count = 0
    while True:
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= duration:
            return first, count
        if now >= next_write:
            commanded = target_at(elapsed)
            written = link.write(commanded, 1, speed, TORQUE)
            first = written if first is None else first
            count += 1
            next_write = max(next_write + period, now)
        link.check(final_target, watch_stall and elapsed > STALL_WINDOW_S + 0.2)
        if follow_limit is not None and elapsed > 0.3:
            error = np.abs(np.array(link.latest()[2]) - commanded)
            if error.max() > follow_limit:
                axis = AXES[int(error.argmax())]
                raise Abort(f"{axis} 跟不上慢速目标，差 {error.max():.2f}：手指可能被挡住")
        time.sleep(max(0.0, min(TICK_S, next_write - time.monotonic())))


def arm(link):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            link.release()
        except Abort:
            # Discovery may not have matched the bridge yet.
            time.sleep(0.1)
            continue
        before = len(link.rows)
        time.sleep(0.05)
        if len(link.rows) > before and link.latest()[1] == GATE_READY:
            return
    raise Abort("PC2 的指令闸门 3 秒内没有进入就绪状态")


def approach(link, target):
    """Move slowly from wherever the hand is to the start pose."""
    start = np.array(link.latest()[2])
    target = np.array(target)
    duration = max(float(np.abs(target - start).max()) / APPROACH_SPEED, 0.1)

    def target_at(elapsed):
        return start + (target - start) * min(elapsed / duration, 1.0)

    # Reaching the final target takes the whole ramp, so judge against the moving target.
    stream(link, target_at, duration, APPROACH_HZ, 1.0, target, watch_stall=False, follow_limit=0.2)
    stream(link, lambda _: target, SETTLE_S, APPROACH_HZ, 1.0, target)


def pose(fingers_q, axes):
    q = np.full(6, OPEN_Q)
    q[:2] = THUMB_POSE
    q[list(axes)] = fingers_q
    return q


def run_condition(link, condition, moves_per_direction, progress):
    moves = []
    for repeat in range(moves_per_direction):
        for closing in (True, False):
            begin, end = (OPEN_Q, CLOSED_Q) if closing else (CLOSED_Q, OPEN_Q)
            start_q, target_q = pose(begin, condition.axes), pose(end, condition.axes)

            def target_at(elapsed):
                if condition.ramp_s <= 0.0:
                    return target_q
                return start_q + (target_q - start_q) * min(elapsed / condition.ramp_s, 1.0)

            first, count = stream(link, target_at, condition.hold_s, condition.rate_hz, condition.speed, target_q)
            moves.append({"condition": condition.key, "repeat": repeat, "closing": closing, "t0": first,
                          "writes": count, "hold_s": condition.hold_s, "axes": list(condition.axes),
                          "target": float(end)})
            progress(condition, len(moves))
    return moves


def crossing(times, values, level):
    above = np.flatnonzero(values >= level)
    if not len(above):
        return None
    i = int(above[0])
    if i == 0:
        return float(times[0])
    span = values[i] - values[i - 1]
    fraction = (level - values[i - 1]) / span if span > 0 else 1.0
    return float(times[i - 1] + fraction * (times[i] - times[i - 1]))


def analyse_move(times, q, t0, hold_s, target):
    """One axis through one move: dead time, 10-90% rise, average and peak speed."""
    before = (times < t0) & (times >= t0 - 0.1)
    window = (times >= t0) & (times <= t0 + hold_s)
    if before.sum() < 3 or window.sum() < 10:
        return None
    start = float(np.median(q[before]))
    step = target - start
    if abs(step) < 0.2:
        return None
    sign = 1.0 if step > 0 else -1.0
    t = times[window] - t0
    moved = sign * (q[window] - start)
    dead = crossing(t, moved, MOVED)
    t10 = crossing(t, moved, 0.1 * abs(step))
    t90 = crossing(t, moved, 0.9 * abs(step))
    result = {"step": abs(step), "reached": t90 is not None, "final_error": abs(float(q[window][-1]) - target)}
    if dead is not None:
        result["dead_ms"] = dead * 1000.0
    if t10 is not None and t90 is not None and t90 > t10:
        result["rise_ms"] = (t90 - t10) * 1000.0
        result["speed"] = 0.8 * abs(step) / (t90 - t10)
    grid = np.arange(0.0, float(t[-1]), 0.005)
    if len(grid) > 12:
        sampled = np.interp(grid, t, moved)
        result["peak_speed"] = float(((sampled[12:] - sampled[:-12]) / 0.06).max())
    return result


def median_of(items, key):
    values = [item[key] for item in items if key in item]
    return float(np.median(values)) if values else None


def summarise(rows, moves):
    times = np.array([row[0] for row in rows])
    q = np.array([row[2] for row in rows])
    summary = {}
    for condition in CONDITIONS:
        taken = [move for move in moves if move["condition"] == condition.key and move["t0"] is not None]
        if not taken:
            continue
        entry = {"label": condition.label,
                 "rate_hz": round(sum(m["writes"] for m in taken) / sum(m["hold_s"] for m in taken), 1)}
        for name, closing in (("close", True), ("open", False)):
            results = []
            for move in taken:
                if move["closing"] != closing:
                    continue
                for axis in move["axes"]:
                    result = analyse_move(times, q[:, axis], move["t0"], move["hold_s"], move["target"])
                    if result is not None:
                        results.append(dict(result, axis=AXES[axis]))
            entry[name] = {
                "samples": len(results),
                "reached": sum(r["reached"] for r in results),
                **{key: median_of(results, key) for key in ("dead_ms", "rise_ms", "speed", "peak_speed", "final_error")},
                "per_axis_speed": {AXES[a]: median_of([r for r in results if r["axis"] == AXES[a]], "speed")
                                   for a in condition.axes},
            }
        summary[condition.key] = entry
    return summary


def both_directions(entry, key):
    values = [entry[name][key] for name in ("close", "open") if entry[name][key] is not None]
    return float(np.mean(values)) if values else None


def verdict(summary):
    """Plain conclusions, each stated against the 40 Hz baseline of the same run."""

    def measure(key, field="speed"):
        return both_directions(summary[key], field) if key in summary else None

    base_values = [v for v in (measure("base"), measure("base2")) if v is not None]
    dead_values = [v for v in (measure("base", "dead_ms"), measure("base2", "dead_ms")) if v is not None]
    if not base_values or not dead_values:
        return ["基准工况没有测完，无法比较。"]
    base_speed = float(np.mean(base_values))
    base_dead = float(np.mean(dead_values))
    lines = [f"基准：平均 {base_speed:.2f} 行程/秒（全行程约 {1.0 / base_speed:.2f} 秒），"
             f"指令到开始动约 {base_dead:.0f} ms。"]
    if len(base_values) == 2 and max(base_values) / min(base_values) > DRIFT_MARGIN:
        lines.append(f"注意：两次基准相差超过 10%（{base_values[0]:.2f} 和 {base_values[1]:.2f}），下面的比较不太可靠。")

    speeds = {k: measure(k) for k in ("w100", "w10", "w6") if measure(k) is not None}
    if speeds:
        fastest = max(speeds, key=speeds.get)
        if speeds[fastest] / base_speed >= SPEED_MARGIN:
            lines.append(f"写入频率有影响：{summary[fastest]['label']} 达到 {speeds[fastest]:.2f} 行程/秒，"
                         f"比基准快 {(speeds[fastest] / base_speed - 1) * 100:.0f}%，有改进空间。")
        else:
            lines.append(f"写入频率对速度影响不到 15%（最快 {speeds[fastest]:.2f} 行程/秒，"
                         f"来自 {summary[fastest]['label']}）。")
    deads = {k: measure(k, "dead_ms") for k in ("w100", "w10", "w6") if measure(k, "dead_ms") is not None}
    if deads:
        least = min(deads, key=deads.get)
        if base_dead - deads[least] >= DEAD_MARGIN_MS:
            lines.append(f"写入频率影响启动延迟：{summary[least]['label']} 约 {deads[least]:.0f} ms，"
                         f"比基准少 {base_dead - deads[least]:.0f} ms。")
        else:
            lines.append("写入频率对启动延迟影响不到 20 ms。")
    if measure("one") is not None:
        ratio = measure("one") / base_speed
        if ratio >= SPEED_MARGIN:
            lines.append(f"只动食指比四指一起快 {(ratio - 1) * 100:.0f}%：多指同动时可能受供电或电流限制。")
        else:
            lines.append("只动食指和四指一起差别不到 15%：没看到多指互相拖慢。")
    if measure("ramp") is not None:
        lines.append(f"200 ms 斜坡目标的速度是阶跃目标的 {measure('ramp') / base_speed * 100:.0f}%。")
    if measure("slow") is not None:
        ratio = measure("slow") / base_speed
        if ratio <= 1.0 / SPEED_MARGIN:
            lines.append(f"速度寄存器有效：减半后速度是基准的 {ratio * 100:.0f}%，所以满值确实比半值快。")
        else:
            lines.append(f"速度寄存器减半后速度仍是基准的 {ratio * 100:.0f}%，这个寄存器在高段可能不起作用。")
    return lines


def print_table(summary):
    print("闭合：启动ms / 10-90%用时ms / 行程每秒    张开：同左    实际写入Hz    工况")

    def cell(value, width, digits=0):
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    for entry in summary.values():
        close, opening = entry["close"], entry["open"]
        print(cell(close["dead_ms"], 7) + cell(close["rise_ms"], 9) + cell(close["speed"], 9, 2)
              + cell(opening["dead_ms"], 8) + cell(opening["rise_ms"], 9) + cell(opening["speed"], 9, 2)
              + cell(entry["rate_hz"], 8, 1) + f"  {entry['label']}")


def preflight(link, domain):
    if domain == 0:
        running = subprocess.run(["pgrep", "-af", r"teleop_hand_and_arm\.py|calibrate_o6_cup_grip\.py"],
                                 capture_output=True, text=True).stdout.strip()
        if running:
            raise Refused("遥操或标定程序还在运行，请先退出：\n" + running[:300])
    time.sleep(1.0)
    rows = list(link.rows)
    if link.malformed:
        raise Refused("手部反馈不是 6 轴")
    if len(rows) < 50:
        raise Refused(f"1 秒内只收到 {len(rows)} 条手部反馈，请检查 PC2 的 r1-linker-hand 和网卡")
    temperature = rows[-1][3]
    if temperature >= MAX_TEMPERATURE_C:
        raise Refused(f"电机温度 {temperature:.0f}℃，请冷却后再测")
    foreign = link.foreign_commands(1.0)
    if foreign:
        raise Refused(f"1 秒内收到别的程序发的 {foreign} 条手部指令，请先停掉它")
    return rows[-1]


def confirm(side, moves_per_direction):
    total = sum(c.hold_s * 2 * moves_per_direction for c in CONDITIONS)
    print(f"""
即将测试 {side} 手，四指全程开合约 {total + 5:.0f} 秒，手臂不动。
  - 手里不要有东西，手指和手掌周围留出空间，人的手别放在附近。
  - 拇指会先慢慢移到远离手掌的位置。
  - Ctrl+C 只是停止测试并释放手，不是急停。紧急情况请按机器人急停。
""")
    if not sys.stdin.isatty():
        raise Refused("需要在可交互的终端里运行（ssh -t）")
    if input(f"确认开始请输入 {side}：").strip() != side:
        raise Refused("没有确认，未动作。")


def default_output(side):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path.home() / "unitree_r1_dev" / "reports" / "o6-speed" / f"{stamp}_{side}.json"


def save(path, args, rows, moves, summary, lines, stopped_by):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"side": args.side, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stopped_by": stopped_by,
                "open_q": OPEN_Q, "closed_q": CLOSED_Q, "torque": TORQUE, "thumb_pose": THUMB_POSE,
                "conditions": [asdict(c) for c in CONDITIONS], "summary": summary, "verdict": lines,
                "moves": moves, "state_rows": rows}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--network-interface", default="eno1")
    parser.add_argument("--moves", type=int, default=3, help="closes and opens per condition")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--domain", type=int, default=0, help="DDS domain; anything but 0 is a test with no robot")
    args = parser.parse_args()
    if args.moves < 1:
        parser.error("--moves must be at least 1")

    def on_term(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    output = args.output or default_output(args.side)
    link = HandLink(args.side, args.domain, args.network_interface)
    moves = []
    armed = False
    stopped_by = None
    code = 0
    try:
        preflight(link, args.domain)
        if args.domain == 0:
            confirm(args.side, args.moves)
        link.open_publisher()
        armed = True
        arm(link)
        print("移到起始姿态……", flush=True)
        approach(link, pose(OPEN_Q, FINGERS))
        for condition in CONDITIONS:
            print(f"测试：{condition.label}", flush=True)
            moves.extend(run_condition(link, condition, args.moves, lambda *_: None))
    except Refused as error:
        print(f"未开始：{error}", flush=True)
        stopped_by = "refused"
        code = 2
    except Abort as error:
        print(f"已中止：{error}", flush=True)
        stopped_by = str(error)
        code = 1
    except KeyboardInterrupt:
        print("已中断。", flush=True)
        stopped_by = "interrupted"
        code = 1
    finally:
        if armed:
            try:
                link.release()
                print("已释放手部指令，手停在当前位置。", flush=True)
            except Exception as error:
                print(f"释放失败：{error}。PC2 会在 250 ms 没收到指令后自动停住。", flush=True)

    if stopped_by != "refused":
        # A short settle so the last move's feedback is in the record.
        time.sleep(0.2)
        rows = list(link.rows)
        try:
            summary = summarise(rows, moves)
            lines = verdict(summary)
        except Exception as error:
            summary, lines = {}, [f"分析失败：{error!r}，原始数据仍会保存。"]
        print()
        print_table(summary)
        print()
        for line in lines:
            print(line)
        save(output, args, rows, moves, summary, lines, stopped_by)
        print(f"\n结果已保存：{output}", flush=True)
    sys.stdout.flush()
    # DDS teardown can hang at interpreter exit.
    os._exit(code)


if __name__ == "__main__":
    main()

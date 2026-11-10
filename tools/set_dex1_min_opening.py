#!/usr/bin/env python3
"""Save the right Dex1-1 minimum target, optionally reading its current lowstate position."""
import argparse
from collections import deque
import math
from pathlib import Path
import statistics
import threading
import time


def capture_position(interface, slot):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from check_r1_teleop import validate_gripper_slot

    samples = deque(maxlen=512)
    lock = threading.Lock()
    first = threading.Event()
    error = None

    def receive(message):
        nonlocal error
        try:
            validate_gripper_slot(message, slot)
        except (ValueError, AttributeError, IndexError) as exc:
            with lock:
                error = str(exc)
                samples.clear()
            first.set()
            return
        with lock:
            samples.append((time.monotonic(), float(message.motor_state[slot].q)))
        first.set()

    ChannelFactoryInitialize(0, interface)
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    try:
        subscriber.Init(receive, 1)
        if not first.wait(2.0):
            raise ValueError('No gripper feedback received; no setting saved')
        time.sleep(0.3)
        with lock:
            if error is not None:
                raise ValueError(error)
            recent = [(t, q) for t, q in samples if time.monotonic() - t <= 0.3]
        if len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.2:
            raise ValueError('Gripper feedback is stale or too sparse; no setting saved')
        values = [q for _, q in recent]
        if max(values) - min(values) > 0.03:
            raise ValueError('Gripper is moving; hold the desired opening steady and retry')
        return statistics.median(values)
    finally:
        subscriber.Close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='保存右 Dex1-1 最小开口位置；只读电机反馈，不发送运动指令。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', action='store_true', help='将当前稳定的夹爪位置保存为最小开口（内走线 Dex1-1）')
    mode.add_argument('--q', type=float, help='直接保存目标角度，单位 rad；0 恢复完整闭合行程')
    parser.add_argument('--network-interface', default='eno1')
    parser.add_argument('--slot', type=int, default=33, choices=range(31, 35))
    args = parser.parse_args(argv)
    try:
        q = capture_position(args.network_interface, args.slot) if args.capture else args.q
        if not (math.isfinite(q) and 0.0 <= q < 5.4):
            raise ValueError('最小目标角度必须是 0 <= q < 5.4 rad；未保存')
        path = Path.home() / '.config/xr_teleoperate/dex1_min_q'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(f'{q!r}\n')
        temporary.replace(path)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'已保存最小目标角度 {q:.6f} rad：{path}')
    print('下次启动遥操或采集时生效；当前运行中的程序不会改变。')
    print('这是电机位置限位，不是毫米或夹持力标定。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

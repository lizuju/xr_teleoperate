"""Wrist camera panels: independent HUD overlays next to the head view.

The panels are fed with BGR frames from the main loop and published as
ImageBackground elements (the overlay type this headset renders next to the
head view). A second WebRTC element never negotiates on the client, which is
why the panels do not get their own offer URL.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGE_ROOT = Path(__file__).resolve().parents[1]
TELEVUER_PATH = STAGE_ROOT / "teleop" / "televuer" / "src" / "televuer" / "televuer.py"
MAIN_PATH = STAGE_ROOT / "teleop" / "teleop_hand_and_arm.py"


class FakePanelElement:
    """Records which vuer schema element TeleVuer asked for, and with what."""

    instances = []
    element_name = "?"

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.element_name = type(self).element_name
        FakePanelElement.instances.append(self)


def fake_panel_class(name):
    return type(f"Fake{name}", (FakePanelElement,), {"element_name": name})


def load_televuer_module():
    vuer_module = types.ModuleType("vuer")
    vuer_module.Vuer = object
    schemas_module = types.ModuleType("vuer.schemas")
    for name in ("Hands", "MotionControllers", "WebRTCVideoPlane", "WebRTCStereoVideoPlane"):
        setattr(schemas_module, name, fake_panel_class(name))
    schemas_module.ImageBackground = fake_panel_class("ImageBackground")
    cv2_module = types.ModuleType("cv2")
    cv2_module.INTER_AREA = 3

    def fake_resize(image, size, interpolation=None):
        """Nearest-neighbour stand-in: a flat input stays flat, sizes match."""
        height, width = int(size[1]), int(size[0])
        rows = np.clip((np.arange(height) * image.shape[0]) // height, 0, image.shape[0] - 1)
        cols = np.clip((np.arange(width) * image.shape[1]) // width, 0, image.shape[1] - 1)
        return image[rows][:, cols]

    cv2_module.resize = fake_resize
    with mock.patch.dict(
        sys.modules,
        {"vuer": vuer_module, "vuer.schemas": schemas_module, "cv2": cv2_module},
    ):
        spec = importlib.util.spec_from_file_location("televuer_wrist_panels_under_test", TELEVUER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def panel_viewer(sides, **overrides):
    """A TeleVuer instance with only the panel attributes initialised."""
    from multiprocessing import Value

    tele_vuer_class = load_televuer_module().TeleVuer
    tele_vuer = tele_vuer_class.__new__(tele_vuer_class)
    shape = overrides.get("shape", (240, 320, 3))
    tele_vuer.wrist_panel_sides = tuple(sides)
    tele_vuer.wrist_panel_height = overrides.get("height", 0.26)
    tele_vuer.wrist_panel_distance = overrides.get("distance", 1.2)
    tele_vuer.wrist_panel_offset = overrides.get("offset", (0.40, 0.40))
    tele_vuer.wrist_panel_aspect = overrides.get("aspect", 4.0 / 3.0)
    tele_vuer.wrist_panel_shape = shape
    tele_vuer.wrist_panel_frames = {side: np.zeros(shape, dtype=np.uint8) for side in sides}
    tele_vuer.wrist_panel_seq = {side: Value("L", 0, lock=True) for side in sides}
    tele_vuer.wrist_panel_shm = {}
    return tele_vuer


class WristPanelFrameTest(unittest.TestCase):
    def setUp(self):
        FakePanelElement.instances = []

    def test_frames_are_resized_into_shared_memory_and_marked_ready(self):
        viewer = panel_viewer(["left"])
        self.assertFalse(viewer.wrist_panel_ready("left"))
        frame = np.full((480, 640, 3), 7, dtype=np.uint8)
        viewer.render_wrist_to_xr("left", frame)
        self.assertTrue(viewer.wrist_panel_ready("left"))
        np.testing.assert_array_equal(viewer.wrist_panel_frames["left"],
                                      np.full((240, 320, 3), 7, dtype=np.uint8))

    def test_unknown_side_and_empty_frames_are_ignored(self):
        viewer = panel_viewer(["left"])
        viewer.render_wrist_to_xr("right", np.zeros((480, 640, 3), dtype=np.uint8))
        viewer.render_wrist_to_xr("left", None)
        self.assertFalse(viewer.wrist_panel_ready("left"))
        self.assertFalse(viewer.wrist_panel_ready("right"))

    def test_wrongly_shaped_frame_is_rejected(self):
        viewer = panel_viewer(["left"])
        viewer.render_wrist_to_xr("left", np.zeros((240, 320), dtype=np.uint8))
        self.assertFalse(viewer.wrist_panel_ready("left"))


class WristPanelGeometryTest(unittest.TestCase):
    def setUp(self):
        FakePanelElement.instances = []

    def test_one_mirrored_panel_per_side_that_has_a_frame(self):
        viewer = panel_viewer(["left", "right"])
        for side in ("left", "right"):
            viewer.render_wrist_to_xr(side, np.zeros((480, 640, 3), dtype=np.uint8))
        panels = viewer._wrist_panel_elements()
        self.assertEqual([p.element_name for p in panels], ["ImageBackground", "ImageBackground"])
        self.assertEqual([p.kwargs["key"] for p in panels], ["wrist-left", "wrist-right"])

        left, right = (panel.kwargs for panel in panels)
        self.assertAlmostEqual(left["position"][0], -0.40)
        self.assertAlmostEqual(right["position"][0], 0.40)
        for panel in (left, right):
            self.assertAlmostEqual(panel["position"][1], -0.40)   # below the eye line
            self.assertAlmostEqual(panel["position"][2], 0.0)     # depth comes from distanceToCamera
            self.assertAlmostEqual(panel["distanceToCamera"], 1.2)
            self.assertAlmostEqual(panel["height"], 0.26)
            self.assertAlmostEqual(panel["aspect"], 4.0 / 3.0)
            self.assertEqual(panel["format"], "jpeg")

    def test_panel_appears_only_after_the_first_frame(self):
        viewer = panel_viewer(["left", "right"])
        self.assertEqual(viewer._wrist_panel_elements(), [])
        viewer.render_wrist_to_xr("right", np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual([p.kwargs["key"] for p in viewer._wrist_panel_elements()], ["wrist-right"])

    def test_upsert_is_a_no_op_without_frames(self):
        viewer = panel_viewer(["left"])
        session = mock.Mock()
        viewer._upsert_wrist_panels(session)
        session.upsert.assert_not_called()

    def test_upsert_publishes_panels_into_the_background_group(self):
        viewer = panel_viewer(["left"])
        viewer.render_wrist_to_xr("left", np.zeros((480, 640, 3), dtype=np.uint8))
        session = mock.Mock()
        viewer._upsert_wrist_panels(session)
        session.upsert.assert_called_once()
        panels, kwargs = session.upsert.call_args
        self.assertEqual(kwargs["to"], "bgChildren")
        self.assertEqual([p.kwargs["key"] for p in panels[0]], ["wrist-left"])


class WristPanelWiringTest(unittest.TestCase):
    """The main program must derive the panels from the ZMQ wrist streams."""

    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_PATH.read_text(encoding="utf-8")

    def test_cli_switches_exist(self):
        self.assertIn("'--wrist-display'", self.source)
        self.assertIn("'--wrist-panel-height'", self.source)
        self.assertIn("'--wrist-panel-distance'", self.source)
        self.assertIn("'--wrist-panel-offset'", self.source)

    def test_panels_are_selected_from_zmq_enabled_wrist_cameras(self):
        self.assertIn('for side, topic in (("left", "left_wrist_camera"), ("right", "right_wrist_camera")):', self.source)
        self.assertIn("if wrist_cfg.get('enable_zmq'):", self.source)
        self.assertIn("wrist_panels.append(side)", self.source)

    def test_panels_are_disabled_in_pass_through_and_by_flag(self):
        self.assertIn("if args.display_mode != 'pass-through' and args.wrist_display != 'off':", self.source)

    def test_the_launch_scripts_leave_the_panels_off(self):
        """Off by default in both wrappers.

        The two panels are re-encoded and re-sent at 30 Hz while their content
        changes at about 12 Hz, measured at roughly 3.5 MB/s of base64 on the
        same Wi-Fi link the hand-tracking uplink uses. Recording is unaffected:
        grab_wrist_frames still fetches the palm frames whenever --record is set,
        so color_2/color_3 keep landing in the episode.
        """
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            script = (MAIN_PATH.parent / name).read_text(encoding="utf-8")
            self.assertIn('--wrist-display "${WRIST_DISPLAY:-off}"', script, name)

    def test_frames_are_pushed_every_iteration_not_only_while_recording(self):
        self.assertIn("def grab_wrist_frames():", self.source)
        self.assertIn("(args.record or 'left' in wrist_panels):", self.source)
        self.assertIn("(args.record or 'right' in wrist_panels):", self.source)
        self.assertIn("tv_wrapper.render_wrist_to_xr('left', left_frame.bgr)", self.source)
        self.assertIn("tv_wrapper.render_wrist_to_xr('right', right_frame.bgr)", self.source)

    def test_panels_are_fed_before_the_operator_arms_the_robot(self):
        """The pre-start waiting loop must publish frames too.

        Otherwise the panels stay empty until [r] is pressed, which is what the
        first field test showed.
        """
        waiting = self.source.index("while (\n            not STOP\n")
        waiting_loop = self.source[waiting:waiting + 2500]
        self.assertIn("grab_wrist_frames()", waiting_loop)
        control = self.source.index("# main loop. robot start to follow")
        control_loop = self.source[control:control + 4000]
        self.assertIn("left_wrist_img, right_wrist_img = grab_wrist_frames()", control_loop)

    def test_wrapper_receives_the_panel_configuration(self):
        for name in ("wrist_panels", "wrist_panel_height", "wrist_panel_distance",
                     "wrist_panel_offset", "wrist_panel_aspect"):
            self.assertIn(f"{name}=", self.source)

    def test_aspect_comes_from_the_wrist_image_shape(self):
        self.assertIn("wrist_image_shape = (camera_config.get('left_wrist_camera') or {}).get('image_shape')", self.source)


class WristPanelRenderLoopTest(unittest.TestCase):
    """Every image loop must publish the panels; pass-through must not."""

    @classmethod
    def setUpClass(cls):
        cls.source = TELEVUER_PATH.read_text(encoding="utf-8")

    def test_all_image_loops_publish_the_panels_inside_the_render_loop(self):
        """Per frame, not once at connect.

        The vuer client applies scene ops only after its store mounts, so an
        element published once when the render coroutine starts races the page
        load and is lost — the head background survives because it is re-sent
        every frame.
        """
        self.assertEqual(self.source.count("self._upsert_wrist_panels(session)"), 8)
        for loop in (
            "main_image_binocular_zmq",
            "main_image_monocular_zmq",
            "main_image_binocular_webrtc",
            "main_image_monocular_webrtc",
            "main_image_binocular_zmq_ego",
            "main_image_monocular_zmq_ego",
            "main_image_binocular_webrtc_ego",
            "main_image_monocular_webrtc_ego",
        ):
            start = self.source.index(f"async def {loop}(self, session):")
            end = self.source.index("\n    async def ", start)
            body = self.source[start:end]
            self.assertIn("while True:", body, f"{loop} has no render loop")
            loop_at = body.index("while True:")
            call_at = body.index("self._upsert_wrist_panels(session)")
            self.assertGreater(call_at, loop_at,
                               f"{loop} publishes the wrist panels outside its render loop")

    def test_pass_through_does_not_publish_panels(self):
        start = self.source.index("async def main_pass_through(self, session):")
        body = self.source[start:start + 1200]
        self.assertNotIn("_upsert_wrist_panels", body)

    def test_panels_do_not_open_their_own_webrtc_link(self):
        # A second WebRTC element never negotiates on the headset client.
        start = self.source.index("def _wrist_panel_elements(self):")
        body = self.source[start:start + 1600]
        self.assertNotIn("WebRTC", body)
        self.assertIn("ImageBackground", body)


if __name__ == "__main__":
    unittest.main()

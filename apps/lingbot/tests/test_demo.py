#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""CPU tests for ownership, non-interruptible model work and browser backpressure."""
import asyncio
import base64
import io
import json
import threading
import unittest

from PIL import Image

from apps.lingbot.demo import DemoApplication, start_arguments


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (832, 464), "#173c30").save(output, format="JPEG")
    return output.getvalue()


def payload():
    return {"type": "start", "prompt": "A quiet forest", "seed": 0,
            "image": "data:image/jpeg;base64," + base64.b64encode(image_bytes()).decode()}


class FakePool:
    """Explicit test fixture: no GPU or model latency claims."""

    config = {"gpus": 1}
    ready = False

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.chunk = 0

    def call(self, op, **kwargs):
        self.calls.append(op)
        if op == "start":
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test start was not released")
            self.chunk = 0
            self.ready = True
        if op == "step":
            result = {"frames": [image_bytes()], "chunk_index": self.chunk, "model_ms": 0,
                      "worker_ms": 0, "terminal": False, "width": 832, "height": 464}
            self.chunk += 1
            return result
        return {}

    def close(self):
        self.ready = False


class Socket:
    def __init__(self):
        self.messages = []

    async def send(self, value):
        self.messages.append(json.loads(value) if isinstance(value, str) else value)


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=4)


class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = FakePool()
        self.app = DemoApplication(self.pool, {}, idle_seconds=0.2)
        self.a, self.b = Socket(), Socket()
        self.app.waiting_room.connect("a", self.a)
        self.app.waiting_room.connect("b", self.b)

    async def asyncTearDown(self):
        self.pool.release.set()
        await self.app.shutdown()
        await self.app.shutdown()  # signal and context cleanup may both run

    async def start(self):
        await self.app.handle(self.a, "a", payload())
        await eventually(lambda: "step" in self.pool.calls and self.app.active.awaiting_display)

    async def test_second_player_cannot_start_or_move(self):
        await self.start()
        await self.app.handle(self.b, "b", payload())
        self.assertEqual(self.pool.calls.count("start"), 1)
        self.assertEqual(self.b.messages[-1]["code"], "busy")
        with self.assertRaises(ValueError):
            await self.app.handle(self.b, "b", {"type": "action", "action": "forward", "request_id": "b"})

    async def test_display_ack_controls_backpressure_and_latest_action(self):
        await self.start()
        for action in ("forward", "yaw_right", "yaw_left"):
            await self.app.handle(self.a, "a", {"type": "action", "action": action, "request_id": action})
        await asyncio.sleep(0.03)
        self.assertEqual(self.pool.calls.count("step"), 1)
        await self.app.handle(self.a, "a", {"type": "displayed", "chunk_index": 100})
        self.assertEqual(self.pool.calls.count("step"), 1)
        await self.app.handle(self.a, "a", {"type": "displayed", "chunk_index": 0})
        await eventually(lambda: self.pool.calls.count("step") == 2)
        await eventually(lambda: any(isinstance(m, dict) and m.get("request_id") == "yaw_left"
                                     for m in self.a.messages))

    async def test_disconnect_during_cold_start_drains_before_handoff(self):
        self.pool.release.clear()
        await self.app.handle(self.a, "a", payload())
        await eventually(self.pool.started.is_set)
        await self.app.handle(self.b, "b", {"type": "join_queue"})
        release = asyncio.create_task(self.app.waiting_room.disconnect("a"))
        await asyncio.sleep(0.03)
        self.assertEqual(self.app.waiting_room.owner, "a")
        self.assertEqual(self.pool.calls, ["start"])
        self.pool.release.set()
        await release
        self.assertEqual(self.pool.calls, ["start", "stop"])
        self.assertEqual(self.app.waiting_room.owner, "b")
        self.assertIsNone(self.app.active)  # promotion does not auto-start

    async def test_restart_retains_ownership_and_closes_old_world(self):
        await self.start()
        await self.app.handle(self.b, "b", {"type": "join_queue"})
        await self.app.handle(self.a, "a", payload())
        await eventually(lambda: self.pool.calls.count("start") == 2)
        self.assertEqual(self.app.waiting_room.owner, "a")
        self.assertEqual(self.pool.calls[:4], ["start", "step", "stop", "start"])

    async def test_previews_reach_the_browser_before_their_frames(self):
        call = self.pool.call

        def with_previews(op, on_preview=None, **kwargs):
            if op == "step" and on_preview is not None:
                for index in (1, 2):
                    on_preview({"jpeg": image_bytes(), "pass": index, "passes": 4, "chunk_index": self.pool.chunk})
            return call(op, **kwargs)
        self.pool.call = with_previews
        await self.start()
        await eventually(lambda: any(not isinstance(m, dict) for m in self.a.messages))
        kinds = [m.get("type") if isinstance(m, dict) else "frames" for m in self.a.messages]
        kinds = [k for k in kinds if k in ("preview", "chunk", "frames")]
        self.assertEqual(kinds, ["preview", "preview", "chunk", "frames"])
        previews = [m for m in self.a.messages if isinstance(m, dict) and m.get("type") == "preview"]
        self.assertEqual([m["pass"] for m in previews], [1, 2])
        self.assertTrue(previews[0]["image"].startswith("data:image/jpeg;base64,"))

    async def test_tilt_is_rejected(self):
        await self.start()
        for action in ("pitch_up", "pitch_down", {}, None):
            with self.assertRaises(ValueError):
                await self.app.handle(self.a, "a", {"type": "action", "action": action, "request_id": "x"})

    async def test_waiter_gets_idle_turn_only_after_safe_stop(self):
        await self.start()
        await asyncio.sleep(0.25)
        self.assertEqual(self.app.waiting_room.owner, "a")  # no waiters: unlimited idle
        await self.app.handle(self.b, "b", {"type": "join_queue"})
        await eventually(lambda: self.app.waiting_room.owner == "b")
        self.assertEqual(self.pool.calls[-1], "stop")
        self.assertIsNone(self.app.active)

    async def test_startup_failure_releases_turn(self):
        def fail(op, **kwargs):
            if op == "start":
                raise RuntimeError("fixture failure")
            return {}
        self.pool.call = fail
        with self.assertLogs("lingbot.demo", level="ERROR"):
            await self.app.handle(self.a, "a", payload())
            await eventually(lambda: self.app.waiting_room.start_task is None)
        await eventually(lambda: self.app.waiting_room.owner is None)
        self.assertIsNone(self.app.active)


class ValidationTests(unittest.TestCase):
    def test_upload_is_normalized_and_seed_is_validated(self):
        result = start_arguments(payload(), {})
        with Image.open(io.BytesIO(result["image"])) as image:
            self.assertEqual(image.size, (832, 464))
        for seed in (True, -1, 2**32, "0"):
            with self.assertRaises(ValueError):
                start_arguments({**payload(), "seed": seed}, {})

    def test_arbitrary_paths_and_invalid_uploads_rejected(self):
        for image in ("file:///etc/passwd", "data:image/jpeg;base64,!", [], ""):
            with self.assertRaises(ValueError):
                start_arguments({**payload(), "image": image}, {})
        with self.assertRaises(ValueError):
            start_arguments({"prompt": "test", "seed": 0, "example": "../../file"}, {})


if __name__ == "__main__":
    unittest.main()

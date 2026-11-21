#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""CPU tests for saving a demo session's video when it ends (needs ffmpeg)."""
import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from apps.lingbot.demo import DemoApplication
from apps.lingbot.demo_recording import VIDEO_NAME
from apps.lingbot.tests.test_demo import FakePool, Socket, eventually, payload


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.directory.name)
        self.pool = FakePool()
        self.app = DemoApplication(self.pool, {}, idle_seconds=0.2, save_dir=self.save_dir)
        self.socket = Socket()
        self.app.waiting_room.connect("a", self.socket)

    async def asyncTearDown(self):
        await self.app.shutdown()
        self.directory.cleanup()

    def recordings(self):
        return [m for m in self.socket.messages if isinstance(m, dict) and m.get("type") == "recording"]

    async def play_and_stop(self, chunks):
        await self.app.handle(self.socket, "a", payload())
        for chunk in range(chunks):
            await eventually(lambda: self.pool.calls.count("step") == chunk + 1 and self.app.active.awaiting_display)
            if chunk + 1 < chunks:
                await self.app.handle(self.socket, "a", {"type": "displayed", "chunk_index": chunk})
                await self.app.handle(self.socket, "a", {"type": "action", "action": "forward",
                                                         "request_id": f"r{chunk}"})
        await self.app.handle(self.socket, "a", {"type": "stop"})
        await eventually(lambda: self.recordings())
        return self.recordings()[-1]

    async def test_stop_saves_the_session_as_a_video(self):
        message = await self.play_and_stop(3)
        self.assertEqual(message["status"], "saved", message)
        self.assertEqual(message["frames"], 3)  # the fake pool sends one frame per chunk
        self.assertEqual((message["width"], message["height"]), (832, 464))
        self.assertTrue(VIDEO_NAME.match(message["name"]))
        video = self.save_dir / message["name"]
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                "stream=codec_name,width,height,r_frame_rate,nb_read_frames", "-of", "json",
                                str(video)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["codec_name"], stream["width"], stream["height"]), ("h264", 832, 464))
        self.assertEqual((stream["r_frame_rate"], stream["nb_read_frames"]), ("16/1", "3"))
        meta = json.loads(video.with_suffix(".json").read_text())
        self.assertEqual((meta["prompt"], meta["seed"], meta["chunks"]), ("A quiet forest", 0, 3))
        self.assertEqual(meta["actions"], ["idle", "forward", "forward"])
        self.assertTrue(video.with_suffix(".png").is_file())

        served = await self.app.http(None, SimpleNamespace(path=message["url"], headers={}))
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.body, video.read_bytes())
        for path in ("/videos/../demo.py", "/videos/notes.json", f"/videos/{video.with_suffix('.json').name}",
                     "/videos/20260101-000000-seed0-832x464.mp4"):
            self.assertEqual((await self.app.http(None, SimpleNamespace(path=path, headers={}))).status_code, 404)

    async def test_nothing_is_saved_without_frames_or_when_disabled(self):
        self.pool.release.clear()  # stop during the cold start: no chunk was generated
        await self.app.handle(self.socket, "a", payload())
        await eventually(self.pool.started.is_set)
        stop = asyncio.create_task(self.app.handle(self.socket, "a", {"type": "stop"}))
        self.pool.release.set()
        await stop
        await asyncio.sleep(0.05)
        self.assertEqual(self.recordings(), [])
        self.assertEqual(list(self.save_dir.iterdir()), [])

        app = DemoApplication(FakePool(), {}, idle_seconds=0.2, save_dir=None)
        try:
            self.assertEqual((await app.http(None, SimpleNamespace(path="/videos/x.mp4", headers={}))).status_code,
                             404)
        finally:
            await app.shutdown()


if __name__ == "__main__":
    unittest.main()

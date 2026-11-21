#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""CPU tests for ``conditioning.set_size``: every LingBot module sees the same frame size."""
import unittest

import torch

from apps.lingbot import conditioning
from apps.lingbot.demo import parse_size as demo_parse_size
from apps.lingbot.demo import start_arguments


class SizeTests(unittest.TestCase):
    def tearDown(self):
        conditioning.set_size(*conditioning.DEFAULT_SIZE)

    def test_default_matches_the_trained_size(self):
        from apps.lingbot import dit
        self.assertEqual(conditioning.DEFAULT_SIZE, (832, 464))
        conditioning.set_size(*conditioning.DEFAULT_SIZE)
        self.assertEqual((dit.GRID_W, dit.GRID_H, dit.FRAME_TOKENS, dit.TOKENS), (52, 29, 1508, 6032))

    def test_size_reaches_every_loaded_module(self):
        from apps.lingbot import decoder, dit, pipeline, refiner
        conditioning.set_size(576, 320)
        self.assertEqual((conditioning.WIDTH, conditioning.HEIGHT), (576, 320))
        self.assertEqual((conditioning.LATENT_WIDTH, conditioning.LATENT_HEIGHT), (72, 40))
        for module in (dit, refiner, pipeline):
            self.assertEqual((module.LATENT_WIDTH, module.LATENT_HEIGHT), (72, 40), module.__name__)
        self.assertEqual((dit.GRID_W, dit.GRID_H, dit.FRAME_TOKENS, dit.TOKENS), (36, 20, 720, 2880))
        self.assertEqual((decoder.GRID_W, decoder.GRID_H), (36, 20))
        self.assertEqual(refiner.FRAME_TOKENS, 720)

    def test_decoder_rows_follow_the_size(self):
        from apps.lingbot.decoder import StreamingDecoder
        from fastframes.models.pixel_decoder import PixelDecoder
        conditioning.set_size(576, 320)
        runtime = StreamingDecoder(PixelDecoder(64))
        self.assertEqual((runtime.rows, runtime.width), (20, 36))
        pixels = runtime(torch.zeros(1, runtime.tokens, 64), torch.zeros(1, conditioning.CHUNK, 1, 64))
        self.assertEqual(tuple(pixels.shape), (1, 16, 3, 320, 576))

    def test_invalid_sizes_are_rejected(self):
        for size in ((570, 320), (576, 300), (64, 64), (2048, 1024)):
            with self.assertRaises(ValueError):
                conditioning.set_size(*size)
        self.assertEqual(conditioning.parse_size("576x320"), (576, 320))
        with self.assertRaises(ValueError):
            conditioning.parse_size("576 by 320")

    def test_demo_fits_uploads_to_the_size(self):
        import base64
        import io

        from PIL import Image
        source = io.BytesIO()
        Image.new("RGB", (1000, 700), "#173c30").save(source, format="PNG")
        payload = {"prompt": "A quiet forest", "seed": 0,
                   "image": "data:image/png;base64," + base64.b64encode(source.getvalue()).decode()}
        with Image.open(io.BytesIO(start_arguments(payload, {}, (576, 320))["image"])) as image:
            self.assertEqual(image.size, (576, 320))
        self.assertEqual(demo_parse_size("416x240"), (416, 240))


if __name__ == "__main__":
    unittest.main()

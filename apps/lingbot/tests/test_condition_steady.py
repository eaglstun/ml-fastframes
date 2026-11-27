#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""CPU tests for LazyImageCondition's fixed-point shortcut, with a tiny fake causal VAE."""
import unittest

import torch
from PIL import Image

from apps.lingbot import conditioning


class FakeEncoderModel:
    """A causal 'encoder' whose state decays towards a fixed point on black input.

    The state lives in the feature cache like the Wan VAE's. ``settle`` bounds how many black
    steps the state keeps changing; ``cycle`` makes the output repeat once while the state still
    changes (the shortcut must not fire on an output repeat alone).
    """

    z_dim = 2

    def __init__(self, settle, cycle_at=None):
        self.settle, self.cycle_at, self.calls = settle, cycle_at, 0
        self._enc_feat_map, self._enc_conv_idx = [None, None], [0]

    def clear_cache(self):
        self._enc_feat_map, self._enc_conv_idx = [None, None], [0]

    def encoder(self, frames, feat_cache, feat_idx):
        self.calls += 1
        step = 0 if feat_cache[0] is None else int(feat_cache[0].item()) + 1
        feat_cache[0] = torch.tensor(float(min(step, self.settle)))
        feat_cache[1] = "Rep" if step == 0 else None  # non-tensor entries, as in the Wan VAE
        value = float(self.settle - min(step, self.settle))
        if self.cycle_at is not None and step == self.cycle_at:
            value = float(self.settle - (step - 1))  # same output as the previous step, different state
        h, w = conditioning.LATENT_HEIGHT, conditioning.LATENT_WIDTH
        return torch.full((1, 2 * self.z_dim, 1, h, w), value)

    def conv1(self, x):
        return x


class FakeVAE:
    dtype = torch.float32
    scale = (0.0, 1.0)

    def __init__(self, model):
        self.model = model


class ConditionShortcutTests(unittest.TestCase):
    def setUp(self):
        conditioning.set_size(128, 128)
        self.image = Image.new("RGB", (128, 128), "#336699")

    def tearDown(self):
        conditioning.set_size(*conditioning.DEFAULT_SIZE)

    def encode(self, model, chunks):
        condition = conditioning.LazyImageCondition(FakeVAE(model), self.image, 240, "cpu")
        return [condition.chunk(c) for c in range(chunks)], condition

    def test_shortcut_gives_identical_conditions_and_skips_the_encoder(self):
        reference, _ = self.encode(FakeEncoderModel(settle=10), 8)
        model = FakeEncoderModel(settle=10)
        shortcut, condition = self.encode(model, 8)
        no_shortcut = FakeEncoderModel(settle=10)
        condition_full = conditioning.LazyImageCondition(FakeVAE(no_shortcut), self.image, 240, "cpu")
        condition_full._same_cache = lambda a, b: False
        full = [condition_full.chunk(c) for c in range(8)]
        for a, b, c in zip(reference, shortcut, full):
            self.assertTrue(torch.equal(a, b) and torch.equal(b, c))
        self.assertTrue(condition.steady)
        self.assertLess(model.calls, no_shortcut.calls)  # 32 latents; the encoder stopped early
        self.assertEqual(no_shortcut.calls, 32)

    def test_repeated_output_alone_does_not_trigger_the_shortcut(self):
        model = FakeEncoderModel(settle=10, cycle_at=4)
        shortcut, condition = self.encode(model, 4)
        full_model = FakeEncoderModel(settle=10, cycle_at=4)
        full_condition = conditioning.LazyImageCondition(FakeVAE(full_model), self.image, 240, "cpu")
        full_condition._same_cache = lambda a, b: False
        full = [full_condition.chunk(c) for c in range(4)]
        for a, b in zip(shortcut, full):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(condition.steady)  # reached only after the state itself settled

    def test_reuse_after_repeats_the_last_encoded_latent(self):
        exact, _ = self.encode(FakeEncoderModel(settle=10), 4)
        model = FakeEncoderModel(settle=10)
        condition = conditioning.LazyImageCondition(FakeVAE(model), self.image, 240, "cpu",
                                                    reuse_after=conditioning.CHUNK)
        reused = [condition.chunk(c) for c in range(4)]
        self.assertEqual(model.calls, conditioning.CHUNK)  # only the first chunk was encoded
        self.assertTrue(torch.equal(reused[0], exact[0]))
        last = exact[0][4:, -1:]  # the first chunk's last latent, after the 4 mask channels
        for c in range(1, 4):
            self.assertTrue(torch.equal(reused[c][4:], last.expand_as(reused[c][4:])))
            self.assertFalse(torch.equal(reused[c], exact[c]))  # an approximation, unlike the fixed point
        self.assertIsNone(model._enc_feat_map[0])  # feature cache freed once encoding stopped


if __name__ == "__main__":
    unittest.main()

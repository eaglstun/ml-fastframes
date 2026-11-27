import pytest

from apps.lingbot.pipeline import denoising_passes


def test_default_runs_every_step():
    assert denoising_passes((0, 1, 2, 3), 4) == [0, 1, 2, 3]


@pytest.mark.parametrize("steps, passes", [((0, 2, 3), [0, 2, 3]), ((0, 1, 3), [0, 1, 3]), ((0, 3), [0, 3])])
def test_skipped_steps(steps, passes):
    assert denoising_passes(steps, 4) == passes


def test_structure_starts_at_its_step():
    assert denoising_passes((0, 1, 2, 3), 4, start=2) == [2, 3]
    assert denoising_passes((0, 1, 3), 4, start=2) == [2, 3]
    assert denoising_passes((0, 2, 3), 4, start=1) == [1, 2, 3]


@pytest.mark.parametrize("steps", [(1, 2, 3), (0, 1, 2), (0, 2, 1, 3), (0, 2, 2, 3), ()])
def test_rejects_bad_steps(steps):
    with pytest.raises(ValueError):
        denoising_passes(steps, 4)

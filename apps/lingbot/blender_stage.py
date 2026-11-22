#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Export a blocked-out Blender scene as a LingBot condition (run inside Blender).

    /Applications/Blender.app/Contents/MacOS/Blender -b scene.blend \\
        --python apps/lingbot/blender_stage.py -- --out outputs/stage --prompt "a rainy street at dusk"

Writes to ``--out``:

* ``staging.png`` -- the scene camera's view of the start frame at 832 x 480, as rendered;
* ``depth.png``   -- the same view as inverse depth (near white, far black) for a depth
  ControlNet or other image-to-image tool that turns the grey box into the real first frame
  (``depth.exr`` holds the raw camera-space Z);
* ``poses.npy``   -- ``[N, 4, 4]`` OpenCV camera-to-world poses (+x right, +y down, +z
  forward) sampled at LingBot's 16 fps from the start frame, ``N = 4k + 1``;
* ``intrinsics.npy`` -- ``[N, 4]`` fx, fy, cx, cy in pixels of a 480 x 832 frame;
* ``condition.json`` -- for ``generate.py --condition`` (its image is ``staging.png``;
  pass ``--image`` there to start from the restyled frame instead);
* ``structure/`` (with ``--structure``) -- every path frame as rendered, for
  ``generate.py --structure``, which steers each frame with them (SDEdit).

Without ``--structure`` only the start frame's look reaches LingBot; the camera path is
followed for every frame either way.
LingBot normalizes the path's scale, so scene units don't matter, only the shape of the move.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

FPS = 16  # LingBot's frame rate (wan_shared_cfg.sample_fps)
WIDTH, HEIGHT = 832, 480  # the frame the upstream intrinsics are given for
BLENDER_TO_OPENCV = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))  # flip y and z


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="blender_stage.py", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--camera", help="camera object (default: the scene camera)")
    parser.add_argument("--start", type=int, help="first frame (default: the scene's frame_start)")
    parser.add_argument("--end", type=int, help="last frame (default: the scene's frame_end)")
    parser.add_argument("--prompt", default="", help="LingBot prompt written to condition.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", help="export the camera only")
    parser.add_argument("--structure", action="store_true",
                        help="also render every path frame to structure/ (for generate.py --structure)")
    return parser.parse_args(argv)


def opencv_pose(camera):
    location, rotation, _ = camera.matrix_world.decompose()  # drop object scale
    c2w = Matrix.LocRotScale(location, rotation, None) @ BLENDER_TO_OPENCV
    return np.array(c2w, dtype=np.float64)


def intrinsics(camera):
    """fx, fy, cx, cy of a WIDTH x HEIGHT render with square pixels."""
    data = camera.data
    if data.type != "PERSP":
        raise SystemExit(f"camera {camera.name!r} is {data.type}; LingBot needs a perspective camera")
    fit = data.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if WIDTH >= HEIGHT else "VERTICAL"
    sensor = data.sensor_width if data.sensor_fit in ("AUTO", "HORIZONTAL") else data.sensor_height
    span = WIDTH if fit == "HORIZONTAL" else HEIGHT
    focal = data.lens / sensor * span
    longest = max(WIDTH, HEIGHT)  # Blender's shift is in units of the longer side
    return (focal, focal, WIDTH / 2 - data.shift_x * longest, HEIGHT / 2 + data.shift_y * longest)


def sample_path(scene, camera, start, end):
    """Poses and intrinsics at FPS from ``start``, ``4k + 1`` of them, within [start, end]."""
    fps = scene.render.fps / scene.render.fps_base
    seconds = (end - start) / fps
    count = (int(seconds * FPS + 1e-6) // 4) * 4 + 1
    if count < 13:  # LingBot generates in chunks of 4 latent frames = 13 RGB frames minimum
        raise SystemExit(f"frames {start}-{end} last {seconds:.2f} s; LingBot needs at least 0.75 s of camera path")
    poses, lenses = [], []
    for index in range(count):
        frame = start + index * fps / FPS
        whole = math.floor(frame)
        scene.frame_set(whole, subframe=frame - whole)
        poses.append(opencv_pose(camera))
        lenses.append(intrinsics(camera))
    scene.frame_set(start)
    return np.stack(poses).astype(np.float32), np.array(lenses, dtype=np.float32)


def depth_material():
    """Every surface emits its camera-space Z distance."""
    material = bpy.data.materials.new("lingbot_stage_depth")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    camera = nodes.new("ShaderNodeCameraData")
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(camera.outputs["View Z Depth"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def disparity_png(exr, png):
    """Z-depth EXR -> 8-bit inverse depth, MiDaS style: near white, far and sky black.

    Normalized over the 2nd-98th percentile of the covered pixels, so a huge ground plane
    or a far backdrop doesn't flatten the foreground.
    """
    image = bpy.data.images.load(str(exr))
    pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, 4)
    alpha = pixels[:, 3]
    covered = alpha > 0.5
    depth = pixels[:, 0] / np.maximum(alpha, 1e-6)  # premultiplied by coverage
    disparity = np.zeros_like(depth)
    disparity[covered] = 1.0 / np.maximum(depth[covered], 1e-6)
    if covered.any():
        low, high = np.percentile(disparity[covered], (2, 98))
        disparity = np.clip((disparity - low) / max(high - low, 1e-9), 0, 1) * covered
    gray = np.repeat(disparity[:, None], 4, axis=1)
    gray[:, 3] = 1
    out = bpy.data.images.new("lingbot_stage_disparity", image.size[0], image.size[1])
    out.pixels.foreach_set(gray.ravel())
    out.filepath_raw, out.file_format = str(png), "PNG"
    out.save()
    bpy.data.images.remove(image)
    bpy.data.images.remove(out)


def render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = bpy.context.scene
    camera = bpy.data.objects[args.camera] if args.camera else scene.camera
    if camera is None or camera.type != "CAMERA":
        raise SystemExit("no camera: set the scene camera or pass --camera")
    scene.camera = camera
    start = scene.frame_start if args.start is None else args.start
    end = scene.frame_end if args.end is None else args.end
    args.out.mkdir(parents=True, exist_ok=True)

    poses, lenses = sample_path(scene, camera, start, end)
    np.save(args.out / "poses.npy", poses)
    np.save(args.out / "intrinsics.npy", lenses)
    condition = {"image": "staging.png", "prompt": args.prompt, "poses": "poses.npy",
                 "intrinsics": "intrinsics.npy", "frames": len(poses), "seed": args.seed}
    (args.out / "condition.json").write_text(json.dumps(condition, indent=2) + "\n")
    print(f"blender_stage: {len(poses)} poses ({len(poses) / FPS:.2f} s at {FPS} fps) from frame {start}", flush=True)

    if args.no_render:
        return
    scene.frame_set(start)
    settings = scene.render
    settings.resolution_x, settings.resolution_y, settings.resolution_percentage = WIDTH, HEIGHT, 100
    settings.pixel_aspect_x = settings.pixel_aspect_y = 1.0
    settings.image_settings.file_format = "PNG"
    settings.image_settings.color_mode = "RGB"
    render(scene, args.out / "staging.png")
    if args.structure:  # the path's frames at 16 fps, as seen by the scene's own renderer
        fps = scene.render.fps / scene.render.fps_base
        for index in range(len(poses)):
            frame = start + index * fps / FPS
            whole = math.floor(frame)
            scene.frame_set(whole, subframe=frame - whole)
            render(scene, args.out / "structure" / f"{index:04d}.png")
        scene.frame_set(start)

    # Depth: raw Z into a float EXR (transparent film = sky), then inverse depth as a PNG.
    view_layer, image = bpy.context.view_layer, settings.image_settings
    saved = (view_layer.material_override, settings.film_transparent, scene.view_settings.view_transform,
             scene.view_settings.look, settings.engine, scene.cycles.samples, image.file_format,
             image.color_mode, image.color_depth)
    # Cycles: EEVEE Next renders a view layer's material override as the scene's own shading.
    settings.engine, scene.cycles.samples = "CYCLES", 4  # emission only: a few samples for antialiasing
    view_layer.material_override = depth_material()
    settings.film_transparent = True
    scene.view_settings.view_transform, scene.view_settings.look = "Raw", "None"
    image.file_format, image.color_mode, image.color_depth = "OPEN_EXR", "RGBA", "32"
    exr = args.out / "depth.exr"
    try:
        render(scene, exr)
    finally:
        (view_layer.material_override, settings.film_transparent, scene.view_settings.view_transform,
         scene.view_settings.look, settings.engine, scene.cycles.samples, image.file_format,
         image.color_mode, image.color_depth) = saved
    disparity_png(exr, args.out / "depth.png")
    print(f"blender_stage: wrote {args.out}/staging.png, depth.png, poses.npy, intrinsics.npy, condition.json",
          flush=True)


if __name__ == "__main__":
    main()

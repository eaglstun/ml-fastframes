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
* ``structure/`` (with ``--structure``) -- every path frame, for ``generate.py --structure``,
  which steers each frame with them (SDEdit). Rendered shadowless by default (Workbench,
  studio light, one grey, transparent sky): cast-shadow edges come out of the generation as
  painted walls, and a fixed sky colour skews the tone matching. With ``--project IMAGE``
  (the restyled first frame), the frames show that image projected onto the scene from the
  start camera instead: the generation keeps its look rather than a grey box's.

Without ``--structure`` only the start frame's look reaches LingBot; the camera path is
followed for every frame either way.
LingBot normalizes the path's scale, so scene units don't matter, only the shape of the move.
"""
import argparse
import json
import math
import sys
from contextlib import contextmanager
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
    parser.add_argument("--structure-lighting", choices=("studio", "scene"), default="studio",
                        help="structure frames: shadowless Workbench studio light (default) or the scene's renderer")
    parser.add_argument("--project", type=Path,
                        help="structure frames: this image (the restyled first frame) projected onto the scene "
                        "from the start camera instead of grey (implies --structure)")
    args = parser.parse_args(argv)
    args.structure = args.structure or args.project is not None
    return args


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


def projection_material(image_path, camera_matrix, lens):
    """Every surface emits ``image`` as seen from a fixed camera (``camera_matrix``, ``lens`` =
    ``(fx, fy, cx, cy)`` in pixels of a WIDTH x HEIGHT frame), projected per pixel; surfaces
    outside that view, or behind it, emit flat grey. Hidden surfaces get whatever covers them in
    the image (there is no occlusion test)."""
    material = bpy.data.materials.new("lingbot_stage_projection")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    # The shaded point in the projector's space (an empty at the start camera's pose; Blender
    # camera axes: +x right, +y up, looking down -z), via a Texture Coordinate node's Object output.
    projector = bpy.data.objects.new("lingbot_stage_projector", None)
    projector.matrix_world = camera_matrix
    bpy.context.scene.collection.objects.link(projector)
    coords = nodes.new("ShaderNodeTexCoord")
    coords.object = projector
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Object"], xyz.inputs["Vector"])
    fx, fy, cx, cy = lens

    def op(operation, a, b):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        for index, value in enumerate((a, b)):
            if isinstance(value, (int, float)):
                node.inputs[index].default_value = value
            else:
                links.new(value, node.inputs[index])
        return node.outputs[0]

    depth = op("MULTIPLY", xyz.outputs["Z"], -1.0)  # distance in front of the projector
    u = op("DIVIDE", op("ADD", op("MULTIPLY", op("DIVIDE", xyz.outputs["X"], depth), fx), cx), WIDTH)
    v = op("DIVIDE", op("SUBTRACT", cy, op("MULTIPLY", op("DIVIDE", xyz.outputs["Y"], depth), fy)), HEIGHT)
    uv = nodes.new("ShaderNodeCombineXYZ")
    links.new(u, uv.inputs["X"])
    links.new(op("SUBTRACT", 1.0, v), uv.inputs["Y"])  # image rows run top-down, texture v bottom-up
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(image_path))
    texture.extension = "CLIP"  # alpha 0 outside the image
    links.new(uv.outputs["Vector"], texture.inputs["Vector"])
    in_front = op("GREATER_THAN", depth, 0.0)
    visible = op("MULTIPLY", texture.outputs["Alpha"], in_front)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    # Mix has float, vector and colour sockets that share names: take the colour ones by type
    a, b = [socket for socket in mix.inputs if socket.type == "RGBA"][:2]
    a.default_value = (0.5, 0.5, 0.5, 1.0)
    links.new(visible, mix.inputs[0])  # Factor
    links.new(texture.outputs["Color"], b)
    emission = nodes.new("ShaderNodeEmission")
    links.new(next(socket for socket in mix.outputs if socket.type == "RGBA"), emission.inputs["Color"])
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material, projector


@contextmanager
def projected(scene, image_path, camera, start):
    """Render with ``image_path`` projected from ``camera`` at frame ``start`` (Cycles emission,
    transparent sky, colours passed through unchanged)."""
    scene.frame_set(start)
    material, projector = projection_material(image_path, camera.matrix_world.copy(), intrinsics(camera))
    view_layer, settings, image = bpy.context.view_layer, scene.render, scene.render.image_settings
    saved = (view_layer.material_override, settings.engine, scene.cycles.samples, settings.film_transparent,
             scene.view_settings.view_transform, scene.view_settings.look, image.color_mode)
    view_layer.material_override = material
    settings.engine, scene.cycles.samples = "CYCLES", 8  # emission only: a few samples for antialiasing
    settings.film_transparent = True
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    image.color_mode = "RGBA"
    try:
        yield
    finally:
        (view_layer.material_override, settings.engine, scene.cycles.samples, settings.film_transparent,
         scene.view_settings.view_transform, scene.view_settings.look, image.color_mode) = saved
        bpy.data.objects.remove(projector)


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


@contextmanager
def studio_lighting(scene, enabled):
    """Workbench, studio light, one grey, no shadows: surfaces shaded by their orientation only.

    Structure frames steer every generated frame, and the generation reads any edge as geometry:
    with the scene's sun, cast-shadow boundaries came out as painted walls. The sky is left
    transparent (RGBA): ``generate.py --structure`` fills it with the first image's sky colour.
    """
    if not enabled:
        yield
        return
    shading, view = scene.display.shading, scene.view_settings
    names = ("light", "color_type", "single_color", "show_shadows", "show_cavity", "show_object_outline",
             "show_specular_highlight")
    image = scene.render.image_settings
    saved = ({name: getattr(shading, name) for name in names}, scene.render.engine, view.view_transform,
             view.look, scene.render.film_transparent, image.color_mode)
    scene.render.engine = "BLENDER_WORKBENCH"
    shading.light, shading.color_type, shading.single_color = "STUDIO", "SINGLE", (0.8, 0.8, 0.8)
    shading.show_shadows = shading.show_cavity = shading.show_object_outline = False
    shading.show_specular_highlight = False
    view.view_transform, view.look = "Standard", "None"
    scene.render.film_transparent, image.color_mode = True, "RGBA"
    try:
        yield
    finally:
        for name, value in saved[0].items():
            setattr(shading, name, value)
        (scene.render.engine, view.view_transform, view.look, scene.render.film_transparent,
         image.color_mode) = saved[1:]


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
    if args.structure:  # the path's frames at 16 fps
        look = (projected(scene, args.project, camera, start) if args.project is not None
                else studio_lighting(scene, args.structure_lighting == "studio"))
        with look:
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

"""Build a detailed grey-box test scene for apps/lingbot/blender_stage.py and save it as a .blend.

A village street opening onto a plaza: houses with recessed windows and doors, shutters, sills,
balconies and awnings, parked cars with wheels, street lamps, trees, a fountain and a church.
The camera walks up the road for 5 s at eye height (frames 1-121 at 24 fps).

    /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/greybox_village.py -- village.blend
"""
import math
import random
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
random.seed(7)


def box(name, loc, size, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.object
    o.name, o.scale = name, (size[0] / 2, size[1] / 2, size[2] / 2)
    return o


def cyl(name, loc, r, h, verts=16, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=loc, rotation=rot)
    bpy.context.object.name = name
    return bpy.context.object


def blob(name, loc, r):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=r, location=loc)
    o = bpy.context.object
    o.name = name
    for v in o.data.vertices:  # lumpy canopy
        v.co *= random.uniform(0.85, 1.15)
    return o


def recess(target, cutters):
    """Cut windows and doors 0.3 m into the wall, so their reveals catch the light."""
    if not cutters:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for c in cutters:
        c.select_set(True)
    bpy.context.view_layer.objects.active = cutters[0]
    bpy.ops.object.join()
    cutter = bpy.context.object
    mod = target.modifiers.new("cut", "BOOLEAN")
    mod.operation, mod.object = "DIFFERENCE", cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter)


# Ground: street (x in [-3.5, 3.5]) between raised sidewalks, plaza from y = 34.
box("ground", (0, 40, -0.05), (200, 200, 0.1))
for side in (-1, 1):
    box(f"sidewalk{side}", (side * 5, 14, 0.075), (3, 44, 0.15))
    box(f"curb{side}", (side * 3.55, 14, 0.09), (0.1, 44, 0.18))
box("plaza", (0, 50, 0.075), (36, 32, 0.15))

# Facades along x = +-6.5 from y = -8 to 34: varied height, windows with sills, doors with steps,
# balconies, awnings over shopfronts.
for side in (-1, 1):
    y = -8.0
    i = 0
    while y < 34:
        width = random.uniform(5, 8)
        floors = random.randint(2, 4)
        height = floors * 3.2 + 0.6
        face = side * 6.5  # street-facing plane
        cx = face + side * 5
        house = box(f"house{side}_{i}", (cx, y + width / 2, height / 2), (10, width - 0.15, height))
        cutters = []
        box(f"cornice{side}_{i}", (face - side * 0.15, y + width / 2, height - 0.2), (0.3, width - 0.15, 0.4))
        box(f"roof{side}_{i}", (cx, y + width / 2, height + 0.9), (10.6, width + 0.2, 0.25),
            rot=(0, side * math.radians(14), 0))
        cols = max(2, int(width // 2.2))
        for f in range(floors):
            z = 1.6 + f * 3.2
            for c in range(cols):
                wy = y + (c + 0.5) * width / cols
                if f == 0 and c == cols // 2:  # door, step, awning
                    cutters.append(box(f"door{side}_{i}", (face, wy, 1.15), (0.8, 1.2, 2.3)))
                    box(f"step{side}_{i}", (face - side * 0.3, wy, 0.22), (0.6, 1.6, 0.15))
                    if random.random() < 0.6:
                        box(f"awning{side}_{i}", (face - side * 0.8, wy, 2.9), (1.6, width * 0.7, 0.08),
                            rot=(0, -side * math.radians(20), 0))
                    continue
                cutters.append(box(f"win{side}_{i}_{f}_{c}", (face, wy, z + 0.3), (0.6, 0.9, 1.5)))
                box(f"sill{side}_{i}_{f}_{c}", (face - side * 0.12, wy, z - 0.5), (0.25, 1.1, 0.1))
                for s in (-1, 1):  # shutters
                    box(f"shutter{side}_{i}_{f}_{c}{s}", (face - side * 0.05, wy + s * 0.72, z + 0.3),
                        (0.06, 0.45, 1.5))
            if f > 0 and random.random() < 0.35:  # balcony with railing
                by = y + width / 2
                box(f"balc{side}_{i}_{f}", (face - side * 0.6, by, z - 0.75), (1.2, 2.6, 0.12))
                box(f"rail{side}_{i}_{f}", (face - side * 1.15, by, z - 0.25), (0.05, 2.6, 0.06))
                for p in range(7):
                    box(f"post{side}_{i}_{f}_{p}", (face - side * 1.15, by - 1.25 + p * 0.42, z - 0.47),
                        (0.04, 0.04, 0.5))
        recess(house, cutters)
        y += width
        i += 1

# Plaza: church-like building at the far end, a fountain, benches, trees.
box("church", (0, 72, 7), (16, 10, 14))
box("tower", (6, 70, 12), (4, 4, 24))
box("tower_cap", (6, 70, 24.8), (4.6, 4.6, 1.6))
box("church_door", (0, 66.95, 2), (2.4, 0.15, 4))
cyl("fountain_basin", (0, 48, 0.5), 3.2, 0.7, 32)
cyl("fountain_water", (0, 48, 0.75), 2.9, 0.1, 32)
cyl("fountain_column", (0, 48, 1.6), 0.35, 2.2)
cyl("fountain_bowl", (0, 48, 2.6), 1.1, 0.25, 24)
for k in range(6):
    a = k * math.pi / 3 + 0.3
    x, y = 9 * math.cos(a), 48 + 7 * math.sin(a)
    if y < 42:
        continue
    cyl(f"trunk{k}", (x, y, 1.6), 0.18, 3.2)
    blob(f"canopy{k}", (x, y, 4.3), 2.0)
for k, (x, y, rot) in enumerate(((-5, 44, 0), (5, 44, 0), (-6, 53, 90), (6, 53, 90))):
    r = math.radians(rot)
    box(f"bench_seat{k}", (x, y, 0.6), (1.8, 0.5, 0.08), rot=(0, 0, r))
    box(f"bench_back{k}", (x - 0.25 * math.sin(r), y + 0.25 * math.cos(r), 0.95), (1.8, 0.06, 0.6), rot=(0, 0, r))
    for s in (-1, 1):
        box(f"bench_leg{k}{s}", (x + s * 0.8 * math.cos(r), y + s * 0.8 * math.sin(r), 0.38), (0.08, 0.45, 0.45),
            rot=(0, 0, r))

# Street furniture: lamps on both sidewalks, small trees in planters, parked cars on the left.
for k in range(5):
    side = 1 if k % 2 else -1
    y = -2 + k * 8
    x = side * 4.0
    cyl(f"lamp_pole{k}", (x, y, 2.1), 0.07, 4.2)
    box(f"lamp_arm{k}", (x - side * 0.4, y, 4.15), (0.8, 0.06, 0.06))
    box(f"lamp_head{k}", (x - side * 0.8, y, 4.0), (0.35, 0.35, 0.3))
for k in range(3):
    y = 6 + k * 10
    box(f"planter{k}", (5.4, y, 0.55), (1.0, 1.0, 0.8))
    cyl(f"ptrunk{k}", (5.4, y, 1.8), 0.08, 2.0)
    blob(f"pcanopy{k}", (5.4, y, 3.0), 0.9)


def car(name, x, y):
    box(f"{name}_body", (x, y, 0.65), (1.8, 4.2, 0.7))
    box(f"{name}_cabin", (x, y - 0.2, 1.25), (1.6, 2.2, 0.55))
    box(f"{name}_hood", (x, y + 1.6, 0.98), (1.7, 0.9, 0.06), rot=(math.radians(-6), 0, 0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(f"{name}_wheel{sx}{sy}", (x + sx * 0.85, y + sy * 1.35, 0.35), 0.35, 0.25, 20,
                rot=(0, math.radians(90), 0))


for k in range(4):
    car(f"car{k}", -2.4, 2 + k * 6.5 + random.uniform(-0.6, 0.6))
car("car_moving", 1.5, 30)

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 4
so = bpy.data.objects.new("sun", sun)
scene.collection.objects.link(so)
so.rotation_euler = (math.radians(40), 0, math.radians(-35))
world = bpy.data.worlds.new("w")
world.color = (0.45, 0.6, 0.85)
scene.world = world

# Camera: walking up the road's right lane at eye height, looking right at the shopfronts, then
# panning left past the parked cars. 5 s.
cam = bpy.data.cameras.new("cam")
cam.lens = 22
co = bpy.data.objects.new("cam", cam)
scene.collection.objects.link(co)
scene.camera = co
scene.render.fps = 24
scene.frame_start, scene.frame_end = 1, 121
for f, x, y, yaw in ((1, 1.6, -6, -6), (61, 1.3, 4, 4), (121, 0.9, 14, 12)):
    co.location = (x, y, 1.65)
    co.rotation_euler = (math.radians(87), 0, math.radians(yaw))
    co.keyframe_insert("location", frame=f)
    co.keyframe_insert("rotation_euler", frame=f)
scene.render.engine = "BLENDER_EEVEE_NEXT"
bpy.ops.wm.save_as_mainfile(filepath=out)

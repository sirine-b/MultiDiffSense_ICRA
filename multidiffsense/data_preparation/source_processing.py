"""
MultiDiffSense — Source (Depth Map) Processing

Generates aligned depth maps from STL files using pyrender, then applies
pose-based transformations (scaling, rotation, translation, depth modulation)
to align each depth map with the corresponding tactile sensor image.

Directory structure expected:
    <tactile_dir>/<obj_id>/<sensor_type>/target/   — tactile images
    <stl_dir>/<obj_id>.stl                         — STL mesh files
    <csv_dir>/<obj_id>.csv                         — pose CSV files

Output written to:
    <tactile_dir>/<obj_id>/<sensor_type>/source/   — aligned depth maps

Usage:
    python -m multidiffsense.data_preparation.source_processing \
        --stl_dir data/example/stl \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 \
        --sensor_type ViTacTip
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Object name lookup table (from dataset) ──────────────────────────
object_names = {
    1: "edge", 2: "flat slab", 3: "pacman", 4: "dot", 5: "cylinder",
    6: "hollow cylinder", 7: "ring", 8: "sphere", 9: "moon", 10: "dot in",
    11: "curved_surface", 12: "wave", 13: "dots", 14: "cross lines",
    15: "parallel lines", 16: "cone", 17: "cylinder side", 18: "cuboid",
    19: "hexagon", 20: "triangle", 21: "random",
}


# ── Coordinate System Analyser ────────────────────────────────────────
class CoordinateSystemAnalyzer:
    """Analyse and debug coordinate system transformations between
    the robot work frame and the image canvas."""

    def __init__(self, work_frame, home_pose, canvas_size=512, workspace_mm=60):
        self.work_frame = np.array(work_frame)
        self.home_pose = np.array(home_pose)
        self.canvas_size = canvas_size
        self.workspace_mm = workspace_mm
        self.pixels_per_mm = canvas_size / workspace_mm

    def analyze_csv_data(self, df, max_frames=10):
        """Analyse CSV data to understand coordinate conventions."""
        poses = []
        for i in range(min(max_frames, len(df))):
            pose = [
                df["pose_1"][i], df["pose_2"][i], df["pose_3"][i],
                df["pose_4"][i] if "pose_4" in df else 0,
                df["pose_5"][i] if "pose_5" in df else 0,
                df["pose_6"][i],
            ]
            poses.append(pose)
        poses = np.array(poses)

        labels = ["X", "Y", "Z", "RX", "RY", "YAW"]
        print(f"Pose ranges over {len(poses)} frames:")
        for i, label in enumerate(labels):
            if i < poses.shape[1]:
                print(f"  {label}: [{np.min(poses[:, i]):.3f}, {np.max(poses[:, i]):.3f}]")

        pos_ranges = [np.max(poses[:, i]) - np.min(poses[:, i]) for i in range(3)]
        if all(r < 50 for r in pos_ranges):
            coord_type = "RELATIVE_SMALL"
        elif all(r > 100 for r in pos_ranges):
            coord_type = "ABSOLUTE_LARGE"
        else:
            coord_type = "MIXED_UNCERTAIN"
        print(f"Coordinate system type: {coord_type}")
        return coord_type, poses

    def test_transformation_strategies(self, pose_sample):
        """Test different coordinate transformation strategies."""
        strategies = {
            "DIRECT": pose_sample[:3],
            "RELATIVE_TO_HOME": pose_sample[:3] - self.home_pose[:3],
            "RELATIVE_TO_WORK": pose_sample[:3] - self.work_frame[:3],
        }
        for name, translation in strategies.items():
            pixels = np.array(translation[:2]) * self.pixels_per_mm
            status = ("OK" if abs(pixels[0]) < self.canvas_size // 2
                      and abs(pixels[1]) < self.canvas_size // 2
                      else "OUT_OF_BOUNDS")
            print(f"  {name:20}: {translation} -> pixels {pixels} [{status}]")

    def calibrate_with_reference_image(self, ref_image, home_pose, cad_object):
        """Calibrate using a reference init image and known home pose."""
        ref_mask = extract_object_mask(ref_image)
        ref_bbox = find_bounding_box(ref_mask)
        ref_center = (ref_bbox[0] + ref_bbox[2] // 2, ref_bbox[1] + ref_bbox[3] // 2)

        cad_mask = extract_object_mask(cad_object)
        cad_bbox = find_bounding_box(cad_mask)
        cad_center = (cad_bbox[0] + cad_bbox[2] // 2, cad_bbox[1] + cad_bbox[3] // 2)

        center_offset_pixels = np.array(ref_center) - np.array(
            [self.canvas_size // 2, self.canvas_size // 2]
        )
        center_offset_mm = center_offset_pixels / self.pixels_per_mm
        print(f"Reference calibration offset: {center_offset_pixels} px = {center_offset_mm} mm")
        return center_offset_pixels, center_offset_mm


# ── Image processing helpers ──────────────────────────────────────────

def create_depth_map_pyrender(stl_file, resolution=512):
    """Create depth map from STL file using pyrender (orthographic top-down)."""
    import trimesh
    import pyrender

    mesh_trimesh = trimesh.load(stl_file)
    mesh_trimesh.apply_translation(-mesh_trimesh.centroid)

    # Rotate to match sensor view — adjust index for your objects
    orientation = trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1])
    mesh_trimesh.apply_transform(orientation)

    bounds = mesh_trimesh.bounds
    size = bounds[1] - bounds[0]

    render_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene = pyrender.Scene()
    scene.add(render_mesh)

    camera = pyrender.OrthographicCamera(xmag=size[0] / 2, ymag=size[1] / 2)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, size[2] + 10]
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    _, depth = renderer.render(scene)
    renderer.delete()

    depth[depth == 0] = np.nan
    depth_min, depth_max = np.nanmin(depth), np.nanmax(depth)
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_img = (255 * depth_norm).astype(np.uint8)
    depth_img = 255 - depth_img
    return depth_img


def apply_depth_modulation(depth_image, z_offset_mm, pixels_per_mm, depth_scale_factor=0.3):
    """Modulate depth values inside the object mask based on contact depth."""
    object_mask = depth_image > 0
    depth_offset = z_offset_mm * depth_scale_factor * pixels_per_mm
    modulated = depth_image.copy().astype(np.float32)
    modulated[object_mask] += depth_offset
    modulated[~object_mask] = depth_image[~object_mask]
    return np.clip(modulated, 0, 255).astype(np.uint8)


def apply_scale_based_on_depth(obj_img, z_offset_mm, scale_factor=0.02):
    """Scale object image based on contact depth."""
    scale = np.clip(1.0 + z_offset_mm * scale_factor, 0.8, 1.2)
    if abs(scale - 1.0) < 0.01:
        return obj_img
    h, w = obj_img.shape
    scaled = cv2.resize(obj_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return scaled


def load_image(path):
    """Load image and convert to grayscale."""
    img_color = plt.imread(path)[..., :3]
    return cv2.cvtColor((img_color * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def extract_object_mask(image, threshold=220, min_area_ratio=0.01):
    """Extract binary mask of the object via thresholding + morphology."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    mask = np.zeros_like(image, dtype=np.uint8)
    image_area = image.shape[0] * image.shape[1]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_ratio * image_area:
            mask[labels == i] = 255
    return mask


def viridis_map(image_gray):
    """Convert grayscale to viridis colourmap (BGR for cv2.imwrite)."""
    norm = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def find_bounding_box(mask, visualize=False, original_image=None, name=""):
    """Get bounding box (x, y, w, h) of the largest connected component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No object found in mask ({name})")
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def resize_object(obj_crop, target_size):
    """Resize object crop to target (w, h)."""
    return cv2.resize(obj_crop, target_size, interpolation=cv2.INTER_AREA)


def rotate_image(image, angle, center=None, scale=1.0):
    """Rotate image around centre. Adjust angle convention per object."""
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    # NOTE: Adjust angle convention per object geometry:
    #   final_angle = -angle + 90   # objects 1, 18
    #   final_angle = angle          # objects 6, 8
    #   final_angle = -angle + 180   # object 3
    #   final_angle = -angle         # object 9
    final_angle = -angle+90

    M = cv2.getRotationMatrix2D(center, final_angle, scale)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)


def embed_object_in_canvas_and_reposition(obj_img, canvas_size=(512, 512), position=None):
    """Embed object in a blank canvas at a given position (with bounds clamping)."""
    obj_h, obj_w = obj_img.shape
    canvas_h, canvas_w = canvas_size

    if obj_h > canvas_h or obj_w > canvas_w:
        sf = min(canvas_h / obj_h, canvas_w / obj_w) * 0.9
        obj_img = cv2.resize(obj_img, (int(obj_w * sf), int(obj_h * sf)), interpolation=cv2.INTER_AREA)
        obj_h, obj_w = obj_img.shape

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    if position is None:
        x = (canvas_w - obj_w) // 2
        y = (canvas_h - obj_h) // 2
    else:
        x = max(0, min(canvas_w - obj_w, position[0]))
        y = max(0, min(canvas_h - obj_h, position[1]))

    canvas[y:y + obj_h, x:x + obj_w] = obj_img
    return canvas


# ── Main processing function ─────────────────────────────────────────

def source_processing(stl_dir, csv_dir, tactile_dir, obj_id, sensor_type, visualise=False):
    """Full source (depth map) processing pipeline for one object + sensor.

    Args:
        stl_dir:      Directory containing <obj_id>.stl files
        csv_dir:      Directory containing <obj_id>.csv files
        tactile_dir:  Root tactile directory: <tactile_dir>/<obj_id>/<sensor_type>/target/
        obj_id:       Object ID (string)
        sensor_type:  One of TacTip, ViTac, ViTacTip
    """
    stl_file = os.path.join(stl_dir, f"{obj_id}.stl")
    csv_file = os.path.join(csv_dir, f"{obj_id}.csv")
    target_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "target")
    source_path = os.path.join(tactile_dir, str(obj_id), sensor_type, "source")
    os.makedirs(source_path, exist_ok=True)

    print(f"STL:    {stl_file}")
    print(f"CSV:    {csv_file}")
    print(f"Target: {target_dir}")
    print(f"Output: {source_path}")

    df = pd.read_csv(csv_file)

    # Build pose data dict
    pose_data = {}
    for dof in ["pose_1", "pose_2", "pose_3", "pose_6"]:
        pose_data[dof] = {i: df[dof][i] for i in range(len(df[dof]))}

    # Initialise coordinate analyser
    work_frame = [327.46, 16.59, -141.1, 0, 0, -33.61]
    home_pose = [300, 0, -20, 0, 0, -19]
    analyzer = CoordinateSystemAnalyzer(work_frame, home_pose)
    analyzer.analyze_csv_data(df, max_frames=10)

    # Render base depth map from STL
    print("Creating base depth map from STL...")
    depth_image = create_depth_map_pyrender(stl_file)

    # Reference calibration (optional)
    ref_offset_pixels = np.array([0, 0])
    init_path = os.path.join(target_dir, "init_0.png")
    if os.path.exists(init_path):
        init_image = load_image(init_path)
        ref_offset_pixels, _ = analyzer.calibrate_with_reference_image(
            init_image, home_pose, depth_image
        )

    # Extract CAD bounding box
    cad_mask = extract_object_mask(depth_image)
    x, y, w, h = find_bounding_box(cad_mask)
    cx, cy = x + w // 2, y + h // 2
    crop_w, crop_h = 2 * w, 2 * h
    img_h, img_w = depth_image.shape
    x1, x2 = max(0, cx - crop_w // 2), min(img_w, cx + crop_w // 2)
    y1, y2 = max(0, cy - crop_h // 2), min(img_h, cy + crop_h // 2)
    cad_object = depth_image[y1:y2, x1:x2]

    target_w_h = None

    for frame in range(len(df)):
        print(f"\n=== Frame {frame} ===")
        target_path = os.path.join(target_dir, f"{obj_id}_{sensor_type}_{frame}.png")
        if not os.path.exists(target_path):
            print(f"  Target not found: {target_path}")
            continue

        target_gray = load_image(target_path)
        target_mask = extract_object_mask(target_gray)
        _, _, tar_w, tar_h = find_bounding_box(target_mask)
        if frame == 0:
            target_w_h = [tar_w, tar_h]

        # Pose values
        dx_mm = pose_data["pose_1"][frame]
        dy_mm = pose_data["pose_2"][frame]
        dz_mm = pose_data["pose_3"][frame]
        rotation_angle = pose_data["pose_6"][frame]

        # Apply transformations
        scaled = apply_scale_based_on_depth(cad_object, dz_mm)
        resized = resize_object(scaled, (target_w_h[0], target_w_h[1]))
        depth_mod = apply_depth_modulation(resized, dz_mm, analyzer.pixels_per_mm)
        rotated = rotate_image(depth_mod, rotation_angle)

        # Calculate position
        canvas_center = analyzer.canvas_size // 2
        dx_px = int(dx_mm * analyzer.pixels_per_mm) + ref_offset_pixels[0]
        dy_px = int(dy_mm * analyzer.pixels_per_mm) + ref_offset_pixels[1]
        x_px = canvas_center - rotated.shape[1] // 2 + dx_px
        y_px = canvas_center - rotated.shape[0] // 2 + dy_px

        cad_final = embed_object_in_canvas_and_reposition(
            rotated, canvas_size=(analyzer.canvas_size, analyzer.canvas_size),
            position=(x_px, y_px),
        )

        # Error correction — align CAD centre to target centre
        try:
            target_bbox = find_bounding_box(extract_object_mask(target_gray))
            target_center = (target_bbox[0] + target_bbox[2] // 2,
                             target_bbox[1] + target_bbox[3] // 2)
            cad_bbox_final = find_bounding_box(extract_object_mask(cad_final))
            cad_center = (cad_bbox_final[0] + cad_bbox_final[2] // 2,
                          cad_bbox_final[1] + cad_bbox_final[3] // 2)
            error = np.array(target_center) - np.array(cad_center)
            x_px += error[0]
            y_px += error[1]

            cad_final = embed_object_in_canvas_and_reposition(
                rotated, canvas_size=(analyzer.canvas_size, analyzer.canvas_size),
                position=(x_px, y_px),
            )
        except Exception as e:
            print(f"  Centre correction failed: {e}")

        # Save
        out_path = os.path.join(source_path, f"{obj_id}_{frame}.png")
        cv2.imwrite(out_path, viridis_map(cad_final))
        print(f"  Saved: {out_path}")

    print(f"\nSource processing complete for object {obj_id} / {sensor_type}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate aligned depth maps from STL")
    parser.add_argument("--stl_dir", type=str, required=True,
                        help="Directory containing <obj_id>.stl files")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing <obj_id>.csv files")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/target/")
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--sensor_type", type=str, default="ViTacTip",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    parser.add_argument("--visualise", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_processing(
        stl_dir=args.stl_dir,
        csv_dir=args.csv_dir,
        tactile_dir=args.tactile_dir,
        obj_id=args.obj_id,
        sensor_type=args.sensor_type,
        visualise=args.visualise,
    )

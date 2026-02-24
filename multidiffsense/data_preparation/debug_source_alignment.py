"""
Debug script to diagnose depth map alignment issues.

Visualizes intermediate processing stages and compares CAD-rendered depth maps
with actual target tactile images. Allows testing multiple scale factors to diagnose
oversizing/undersizing issues.

By default, object center is FIXED at canvas center while testing different scales,
so you can clearly see SIZE CHANGES. Use --no-fix-position to see the current behavior
where position shifts with size.

Usage (default: fix position, show size changes):
    python -m multidiffsense.data_preparation.debug_source_alignment \
        --stl_dir data/example/stl \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 \
        --sensor_type ViTac \
        --frame_id 0 \
        --test_scales 0.0 0.01 0.02

Usage (show positioning behavior: let position shift):
    python -m multidiffsense.data_preparation.debug_source_alignment \
        --stl_dir data/example/stl \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 \
        --sensor_type ViTac \
        --no-fix-position \
        --test_scales 0.0 0.005 0.01 0.015 0.02
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from multidiffsense.data_preparation.source_processing import (
    create_depth_map_pyrender,
    extract_object_mask,
    find_bounding_box,
    load_image,
    apply_scale_based_on_depth,
    resize_object,
    apply_depth_modulation,
    rotate_image,
    embed_object_in_canvas_and_reposition,
    CoordinateSystemAnalyzer,
    viridis_map,
)
from multidiffsense.data_preparation.target_processing import get_frame_indices


def debug_source_alignment(stl_dir, csv_dir, tactile_dir, obj_id, sensor_type, frame_id=0, test_scale_factors=None, fix_position=True):
    """Debug alignment between CAD depth map and target image.
    
    Args:
        test_scale_factors: List of scale factors to test (e.g., [0.0, 0.01, 0.02, 0.03])
                           If None, uses default [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        fix_position: If True, keeps object center fixed at canvas center while scaling
                     If False, lets position shift with size changes (current behavior)
    """
    if test_scale_factors is None:
        test_scale_factors = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

    stl_file = os.path.join(stl_dir, f"{obj_id}.stl")
    csv_file = os.path.join(csv_dir, f"{obj_id}.csv")
    target_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "target")

    print("=" * 80)
    print(f"DEBUG: Source Alignment for Object {obj_id} / {sensor_type}")
    print("=" * 80)

    # ─── Load data ───────────────────────────────────────────────────
    print("\n[1] Loading input files...")
    if not os.path.exists(stl_file):
        print(f"ERROR: STL file not found: {stl_file}")
        return
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        return
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory not found: {target_dir}")
        return

    df = pd.read_csv(csv_file)
    print(f"  ✓ CSV loaded: {len(df)} rows")

    frame_indices = get_frame_indices(target_dir, obj_id, sensor_type)
    print(f"  ✓ Found {len(frame_indices)} target frames: {frame_indices[:5]}...")

    if len(frame_indices) == 0:
        print("ERROR: No target frames found")
        return

    # Use first frame or specified frame_id
    frame = frame_indices[min(frame_id, len(frame_indices) - 1)]
    print(f"  ✓ Using frame index: {frame}")

    # ─── Load target image ───────────────────────────────────────────
    print("\n[2] Loading target image...")
    target_path = os.path.join(target_dir, f"{obj_id}_{sensor_type}_{frame}.png")
    if not os.path.exists(target_path):
        print(f"WARNING: Expected target at {target_path}")
        # Try alternative naming
        target_path = os.path.join(target_dir, f"{frame}.png")
        if not os.path.exists(target_path):
            print(f"ERROR: Target image not found")
            return

    target_gray = load_image(target_path)
    print(f"  ✓ Target image shape: {target_gray.shape}")
    print(f"  ✓ Target image range: [{target_gray.min()}, {target_gray.max()}]")

    # ─── Create base depth map ───────────────────────────────────────
    print("\n[3] Rendering depth map from STL...")
    depth_image = create_depth_map_pyrender(stl_file)
    print(f"  ✓ Depth map shape: {depth_image.shape}")
    print(f"  ✓ Depth map range: [{depth_image.min()}, {depth_image.max()}]")

    # ─── Analyse coordinate frame ────────────────────────────────────
    print("\n[4] Coordinate system analysis...")
    work_frame = [327.46, 16.59, -141.1, 0, 0, -33.61]
    home_pose = [300, 0, -20, 0, 0, -19]
    analyzer = CoordinateSystemAnalyzer(work_frame, home_pose)
    coord_type, poses = analyzer.analyze_csv_data(df, max_frames=10)

    # ─── Extract CAD bounding box ────────────────────────────────────
    print("\n[5] Extracting CAD object region...")
    cad_mask = extract_object_mask(depth_image)
    print(f"  ✓ CAD mask created, non-zero pixels: {np.count_nonzero(cad_mask)}")

    try:
        x, y, w, h = find_bounding_box(cad_mask)
        print(f"  ✓ CAD bounding box: x={x}, y={y}, w={w}, h={h}")
        print(f"    Center: ({x + w // 2}, {y + h // 2})")
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    cx, cy = x + w // 2, y + h // 2
    crop_w, crop_h = 2 * w, 2 * h
    img_h, img_w = depth_image.shape
    x1 = max(0, cx - crop_w // 2)
    x2 = min(img_w, cx + crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    y2 = min(img_h, cy + crop_h // 2)

    cad_object = depth_image[y1:y2, x1:x2]
    print(f"  ✓ CAD crop region: [{y1}:{y2}, {x1}:{x2}] → shape {cad_object.shape}")
    print(f"  ✓ CAD crop range: [{cad_object.min()}, {cad_object.max()}]")

    # ─── Extract target bounding box ────────────────────────────────
    print("\n[6] Extracting target object region...")
    target_mask = extract_object_mask(target_gray)
    print(f"  ✓ Target mask created, non-zero pixels: {np.count_nonzero(target_mask)}")

    try:
        tx, ty, tw, th = find_bounding_box(target_mask)
        print(f"  ✓ Target bounding box: x={tx}, y={ty}, w={tw}, h={th}")
        print(f"    Center: ({tx + tw // 2}, {ty + th // 2})")
        target_center = (tx + tw // 2, ty + th // 2)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # ─── Lookup pose data ────────────────────────────────────────────
    print("\n[7] Looking up pose data for frame...")
    if frame >= len(df):
        print(f"ERROR: Frame {frame} out of CSV range (0-{len(df) - 1})")
        return

    try:
        dx_mm = df["pose_1"][frame]
        dy_mm = df["pose_2"][frame]
        dz_mm = df["pose_3"][frame]
        rotation_angle = df["pose_6"][frame]
        print(f"  ✓ dx_mm={dx_mm:.3f}, dy_mm={dy_mm:.3f}, dz_mm={dz_mm:.3f}")
        print(f"  ✓ rotation_angle={rotation_angle:.3f}°")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # ─── Apply transformations step-by-step ──────────────────────────
    print("\n[8] Testing different scale factors...")
    if fix_position:
        print(f"  [MODE] Position fix: ENABLED (center stays at canvas center)")
    else:
        print(f"  [MODE] Position fix: DISABLED (position recalculated per scale)")
    
    # Create dict to store results for each scale factor
    scale_test_results = {}
    
    # Calculate reference position from first scale factor
    ref_x_px = None
    ref_y_px = None
    canvas_center = analyzer.canvas_size // 2
    dx_px = int(dx_mm * analyzer.pixels_per_mm)
    dy_px = int(dy_mm * analyzer.pixels_per_mm)
    
    for test_sf in test_scale_factors:
        print(f"\n  Testing scale_factor = {test_sf}...")
        
        # Apply scale
        scaled = apply_scale_based_on_depth(cad_object, dz_mm, scale_factor=test_sf, debug=False)
        print(f"    Input cad_object size:  {cad_object.shape}")
        print(f"    After scale:            {scaled.shape}")
        
        # Continue with rest of pipeline
        depth_mod = apply_depth_modulation(scaled, dz_mm, analyzer.pixels_per_mm)
        rotated = rotate_image(depth_mod, rotation_angle)
        print(f"    After rotation:         {rotated.shape}")
        
        # Position calculation
        if fix_position and ref_x_px is None:
            # First iteration: save reference position
            ref_x_px = canvas_center - rotated.shape[1] // 2 + dx_px
            ref_y_px = canvas_center - rotated.shape[0] // 2 + dy_px
            x_px = ref_x_px
            y_px = ref_y_px
            print(f"    Reference position:     ({ref_x_px}, {ref_y_px})")
        elif fix_position:
            # Use reference position (don't recalculate)
            x_px = ref_x_px
            y_px = ref_y_px
            print(f"    Using fixed position:   ({ref_x_px}, {ref_y_px})")
        else:
            # Recalculate position for each scale (current behavior)
            x_px = canvas_center - rotated.shape[1] // 2 + dx_px
            y_px = canvas_center - rotated.shape[0] // 2 + dy_px
            print(f"    Calculated position:    ({x_px}, {y_px})")
        
        # Embed in canvas
        cad_final_test = embed_object_in_canvas_and_reposition(
            rotated, canvas_size=(analyzer.canvas_size, analyzer.canvas_size),
            position=(x_px, y_px),
        )
        
        scale_test_results[test_sf] = {
            'scaled': scaled,
            'rotated': rotated,
            'final': cad_final_test,
        }
        print(f"    ✓ Completed for scale_factor={test_sf}")
    
    # Use the default (first) scale factor for remaining analysis
    default_sf = test_scale_factors[0]
    print(f"\n  Using scale_factor={default_sf} for main analysis...")
    
    scaled = apply_scale_based_on_depth(cad_object, dz_mm, scale_factor=default_sf, debug=False)
    print(f"    → shape: {scaled.shape}, range: [{scaled.min()}, {scaled.max()}]")

    depth_mod = apply_depth_modulation(scaled, dz_mm, analyzer.pixels_per_mm)
    rotated = rotate_image(depth_mod, rotation_angle)

    # ─── Position calculation ────────────────────────────────────────
    print("\n[9] Position calculation...")
    canvas_center = analyzer.canvas_size // 2
    dx_px = int(dx_mm * analyzer.pixels_per_mm)
    dy_px = int(dy_mm * analyzer.pixels_per_mm)
    print(f"  dx_mm={dx_mm} → {dx_px} px")
    print(f"  dy_mm={dy_mm} → {dy_px} px")
    print(f"  pixels_per_mm = {analyzer.pixels_per_mm}")

    x_px = canvas_center - rotated.shape[1] // 2 + dx_px
    y_px = canvas_center - rotated.shape[0] // 2 + dy_px
    print(f"  ✓ Initial position: ({x_px}, {y_px})")
    print(f"    (canvas_center={canvas_center}, rotated shape={rotated.shape})")

    # ─── Embed in canvas ────────────────────────────────────────────
    print("\n[10] Embedding in canvas...")
    cad_final = embed_object_in_canvas_and_reposition(
        rotated, canvas_size=(analyzer.canvas_size, analyzer.canvas_size),
        position=(x_px, y_px),
    )
    print(f"  ✓ Final image shape: {cad_final.shape}")

    # ─── Error correction ────────────────────────────────────────────
    print("\n[11] Comparing alignment...")
    try:
        cad_bbox_final = find_bounding_box(extract_object_mask(cad_final))
        cad_center = (cad_bbox_final[0] + cad_bbox_final[2] // 2,
                      cad_bbox_final[1] + cad_bbox_final[3] // 2)
        error = np.array(target_center) - np.array(cad_center)
        print(f"  Target center: {target_center}")
        print(f"  CAD center:    {cad_center}")
        print(f"  ⚠ CENTER OFFSET: {error} pixels")
        print(f"  Distance: {np.linalg.norm(error):.1f} pixels")

        if np.linalg.norm(error) > 10:
            print(f"\n  ⚠⚠⚠ SIGNIFICANT MISALIGNMENT DETECTED ⚠⚠⚠")
            print(f"  This suggests issues with:")
            print(f"    • Coordinate frame transformation")
            print(f"    • Rotation angle convention")
            print(f"    • STL orientation in 3D")

    except Exception as e:
        print(f"  ERROR during alignment check: {e}")

    # ─── Save visualizations ────────────────────────────────────────
    print("\n[12] Saving diagnostic images...")
    output_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "debug")
    os.makedirs(output_dir, exist_ok=True)

    # Raw depths
    cv2.imwrite(os.path.join(output_dir, "01_raw_depth.png"), viridis_map(depth_image))
    cv2.imwrite(os.path.join(output_dir, "02_cad_crop.png"), viridis_map(cad_object))
    cv2.imwrite(os.path.join(output_dir, "03_depth_modulated.png"), viridis_map(depth_mod))
    cv2.imwrite(os.path.join(output_dir, "04_rotated.png"), viridis_map(rotated))
    cv2.imwrite(os.path.join(output_dir, "05_final_cad.png"), viridis_map(cad_final))
    cv2.imwrite(os.path.join(output_dir, "06_target.png"), target_gray)

    # Save test scale results with side-by-side comparisons
    print(f"\n  Saving scale factor test images (with target comparison)...")
    for sf in sorted(scale_test_results.keys()):
        # Clean up scale factor for filename (e.g., 0.005 -> 0005)
        sf_str = f"{sf:.4f}".replace(".", "")
        final_img = scale_test_results[sf]['final']
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(final_img, cmap='viridis')
        axes[0].set_title(f"CAD (scale_factor={sf})")
        axes[0].axis('off')
        axes[1].imshow(target_gray, cmap='gray')
        axes[1].set_title(f"Target (Frame {frame})")
        axes[1].axis('off')
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"comparison_scale_factor_{sf_str}.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ comparison_scale_factor_{sf_str}.png")

    # Create a grid showing first few scale factors for quick overview
    num_to_compare = min(4, len(scale_test_results))
    sf_keys = sorted(scale_test_results.keys())[:num_to_compare]
    
    fig, axes = plt.subplots(1, num_to_compare + 1, figsize=(5 * (num_to_compare + 1), 5))
    if num_to_compare == 1:
        axes = [axes]
    
    for idx, sf in enumerate(sf_keys):
        final_img = scale_test_results[sf]['final']
        axes[idx].imshow(final_img, cmap='viridis')
        axes[idx].set_title(f"scale_factor={sf}")
        axes[idx].axis('off')
    
    axes[-1].imshow(target_gray, cmap='gray')
    axes[-1].set_title(f"Target (Frame {frame})")
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "00_scale_comparison.png"), dpi=100, bbox_inches='tight')
    print(f"    ✓ 00_scale_comparison.png (grid of scale factors)")
    
    print(f"  ✓ Saved all diagnostic images to: {output_dir}")

    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    print(f"Object: {obj_id}, Sensor: {sensor_type}, Frame: {frame}")
    print(f"Target size: {tw}×{th}, CAD final size: {cad_final.shape}")
    print(f"Coordinate offset: {error} px (distance: {np.linalg.norm(error):.1f} px)")
    print(f"\nSCALE FACTOR TESTING:")
    print(f"  Tested {len(scale_test_results)} scale factors: {sorted(scale_test_results.keys())}")
    print(f"  Results saved as: test_scale_factor_XXXX.png (in debug dir)")
    print(f"  Grid comparison saved as: 00_scale_comparison.png")
    print(f"\nNEXT STEPS:")
    print(f"1. View output images in: {output_dir}")
    print(f"2. Check 00_scale_comparison.png to see which scale factor looks best")
    print(f"3. Once you find ideal scale_factor, update source_processing.py")
    print(f"4. Check if CAD object orientation matches target (check 04_rotated.png)")
    print(f"5. Verify rotation_angle conversion (should match your sensor frame)")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug source depth map alignment")
    parser.add_argument("--stl_dir", type=str, required=True)
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--tactile_dir", type=str, required=True)
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--sensor_type", type=str, default="ViTac",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    parser.add_argument("--frame_id", type=int, default=0,
                        help="Index into frame_indices list (0-based)")
    parser.add_argument("--test_scales", type=float, nargs="+", default=None,
                        help="Space-separated scale factors to test (e.g., 0.0 0.01 0.02 0.03)")
    parser.add_argument("--no-fix-position", action="store_true",
                        help="Disable position fixing (let position shift with size changes)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debug_source_alignment(
        stl_dir=args.stl_dir,
        csv_dir=args.csv_dir,
        tactile_dir=args.tactile_dir,
        obj_id=args.obj_id,
        sensor_type=args.sensor_type,
        frame_id=args.frame_id,
        test_scale_factors=args.test_scales,
        fix_position=not args.no_fix_position,  # Invert the flag
    )

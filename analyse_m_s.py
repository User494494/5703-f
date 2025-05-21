#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import os
import sys
import time
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import shutil


# In[ ]:


def detect_scene_changes(video_path, threshold=30.0, min_scene_duration=10):
    """
    Detect scene changes in a video using frame differencing.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection (higher = fewer scenes)
        min_scene_duration: Minimum number of frames for a scene
        
    Returns:
        List of scene boundaries [(start_frame, end_frame), ...]
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Scene detection: analyzing {total_frames} frames, FPS: {fps}")
    
    # Variables for scene detection
    scene_boundaries = []
    scene_scores = []
    current_scene_start = 0
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame")
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 1
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            # Add the final scene boundary
            if frame_count - current_scene_start >= min_scene_duration:
                scene_boundaries.append((current_scene_start, frame_count - 1))
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and mean
        frame_diff = cv2.absdiff(gray, prev_gray)
        mean_diff = np.mean(frame_diff)
        scene_scores.append(mean_diff)
        
        # Detect scene change
        if mean_diff > threshold and frame_count - current_scene_start >= min_scene_duration:
            scene_boundaries.append((current_scene_start, frame_count - 1))
            current_scene_start = frame_count
            
        # Update for next iteration
        prev_gray = gray
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"Scene detection: processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    
    print(f"Scene detection complete. Found {len(scene_boundaries)} scenes.")
    
    # Plot the scene scores for visualization
    plt.figure(figsize=(12, 4))
    plt.plot(scene_scores)
    plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold})')
    plt.title('Scene Change Detection Scores')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference Score')
    plt.legend()
    
    # Mark scene boundaries
    for start, end in scene_boundaries:
        plt.axvline(x=start, color='g', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    return scene_boundaries

# =======================
# Helper Functions for Scene Processing
# =======================
def scene_initialize_video(video_path):
    """Initialize the video capture object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None
    return cap


def scene_get_video_properties(cap):
    """Get video frame width, height, and other properties."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_width, frame_height, frame_rate, total_frames


def scene_detect_stable_frame(prev_gray, gray_frame, motion_threshold):
    """Detect stable frames using optical flow."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 
                                      0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_motion = np.mean(magnitude)
    return avg_motion


def scene_process_frame_difference(background_frame, gray_frame, fgbg):
    """Compute foreground mask by comparing the background and current frame."""
    frame_diff = cv2.absdiff(background_frame, gray_frame)
    fg_mask = fgbg.apply(frame_diff)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
    return fg_mask_thresh


def scene_find_valid_contours(contours, min_area=100):
    """Filter contours based on minimum area."""
    return [contour for contour in contours if cv2.contourArea(contour) > min_area]


def scene_draw_bounding_boxes(frame, contours, heatmap_data):
    """Draw bounding boxes around contours and update heatmap."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        heatmap_data[y:y+h, x:x+w] += 1  # Accumulate motion in heatmap

def process_video_with_scene_detection(video_path, output_base_dir, video_name=None, 
                                      scene_threshold=30.0, min_scene_duration=10):
    """
    Process a video with scene change detection, generating per-scene heatmaps
    
    Args:
        video_path: Path to the input video
        output_base_dir: Base directory for outputs
        video_name: Name to use for the video folders (if None, will extract from filename)
        scene_threshold: Threshold for scene change detection
        min_scene_duration: Minimum frame count for a scene
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Extract video name from path if not provided
    if video_name is None:
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]
    
    print(f"Processing video with scene detection: {video_path}")
    
    # Step 1: Detect scene changes
    print("Step 1: Detecting scene changes...")
    scene_boundaries = detect_scene_changes(video_path, threshold=scene_threshold, 
                                           min_scene_duration=min_scene_duration)
    
    # Save scene boundaries as JSON
    scenes_dir = os.path.join(output_base_dir, f"{video_name}_scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    
    scenes_data_path = os.path.join(scenes_dir, "scene_boundaries.json")
    with open(scenes_data_path, 'w') as f:
        json.dump({
            'video_info': {
                'path': video_path,
                'total_scenes': len(scene_boundaries)
            },
            'scene_boundaries': [{"start_frame": start, "end_frame": end} for start, end in scene_boundaries]
        }, f, indent=2)
    
    # Get video info for timing calculations
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Step 2: Process each scene separately
    print(f"Step 2: Processing {len(scene_boundaries)} scenes...")
    for i, (start_frame, end_frame) in enumerate(scene_boundaries):
        # Calculate time boundaries
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        print(f"Processing scene {i+1}/{len(scene_boundaries)}: frames {start_frame}-{end_frame} (duration: {duration:.2f}s)")
        
        # Create scene-specific directories
        scene_output_dir = os.path.join(scenes_dir, f"scene_{i+1}")
        os.makedirs(scene_output_dir, exist_ok=True)
        
        # Process the scene with existing methods, but only for the frame range
        process_scene(video_path, scene_output_dir, f"{video_name}_scene{i+1}", 
                     start_frame, end_frame)
    
    print(f"Scene-based processing complete!")
    print(f"Scene results saved to: {scenes_dir}")
    
    return {
        'scenes_dir': scenes_dir,
        'scene_boundaries': scene_boundaries,
        'scenes_data': scenes_data_path
    }

def process_scene(video_path, output_dir, scene_name, start_frame, end_frame):
    """
    Process a specific scene (frame range) from a video
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save outputs
        scene_name: Name for this scene
        start_frame: First frame of the scene
        end_frame: Last frame of the scene
    """
    # Create mouse and keyboard subdirectories
    mouse_dir = os.path.join(output_dir, "mouse")
    keyboard_dir = os.path.join(output_dir, "keyboard")
    
    os.makedirs(mouse_dir, exist_ok=True)
    os.makedirs(keyboard_dir, exist_ok=True)
    
    # Process mouse cursor tracking for this scene
    process_scene_mouse_cursor(video_path, mouse_dir, start_frame, end_frame)
    
    # Process screen reader focus for this scene
    process_scene_screen_reader(video_path, keyboard_dir, start_frame, end_frame)
    
    return {
        'mouse_dir': mouse_dir,
        'keyboard_dir': keyboard_dir
    }

def process_scene_mouse_cursor(video_path, output_dir, start_frame, end_frame):
    """
    Track mouse cursor in a specific scene (frame range) of a video
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save outputs
        start_frame: First frame of the scene
        end_frame: Last frame of the scene
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize variables for cursor tracking
    cursor_positions = []
    heat_map = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Parameters
    diff_threshold = 20
    heat_decay = 0.85
    blur_kernel = (21, 21)
    
    # Create video writer for visualization
    output_video_path = os.path.join(output_dir, 'cursor_tracking.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Skip to start frame
    frame_count = 0
    while frame_count < start_frame:
        ret = cap.grab()  # Faster than read() for skipping
        if not ret:
            print(f"Error: Could not skip to frame {start_frame}")
            cap.release()
            return None
        frame_count += 1
    
    # Read first frame of scene
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame of scene")
        cap.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    # Previous cursor position for continuity
    prev_cursor = None
    scene_frame_count = 0
    
    print(f"Processing scene frames {start_frame}-{end_frame} for mouse cursor...")
    
    # Process frames in scene
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Update heat map with decay
        heat_map = heat_map * heat_decay + thresh.astype(np.float32)
        
        # Apply Gaussian blur to consolidate cursor activity
        heat_map_blurred = cv2.GaussianBlur(heat_map, blur_kernel, 0)
        
        # Find the hottest point
        _, max_val, _, max_loc = cv2.minMaxLoc(heat_map_blurred)
        
        # Apply temporal filtering with previous cursor
        current_cursor = max_loc
        if prev_cursor is not None:
            # Calculate distance
            dist = np.sqrt((current_cursor[0] - prev_cursor[0])**2 + 
                          (current_cursor[1] - prev_cursor[1])**2)
            
            # If movement is too large, smooth it
            if dist > 50:
                alpha = 0.7  # Weight for previous position
                current_cursor = (
                    int(alpha * prev_cursor[0] + (1-alpha) * current_cursor[0]),
                    int(alpha * prev_cursor[1] + (1-alpha) * current_cursor[1])
                )
        
        # Store cursor position
        cursor_positions.append({
            'global_frame': frame_count,
            'scene_frame': scene_frame_count,
            'time': frame_count / fps,
            'x': current_cursor[0],
            'y': current_cursor[1],
            'intensity': float(max_val)
        })
        
        # Update previous cursor
        prev_cursor = current_cursor
        
        # Draw cursor position on frame
        cv2.circle(frame, current_cursor, 5, (0, 255, 0), -1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count} (Scene: {scene_frame_count})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output video
        out.write(frame)
        
        # Update for next iteration
        prev_gray = gray
        frame_count += 1
        scene_frame_count += 1
        
        # Show progress
        if scene_frame_count % 100 == 0:
            total_scene_frames = end_frame - start_frame + 1
            print(f"Processed {scene_frame_count}/{total_scene_frames} scene frames ({scene_frame_count/total_scene_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Generate heatmap
    heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Add cursor positions to heatmap
    for pos in cursor_positions:
        x, y = int(pos['x']), int(pos['y'])
        # Skip if outside bounds
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            continue
            
        # Add weighted point to heatmap
        intensity = max(1.0, pos['intensity'] / 50.0)  # Normalize intensity
        cv2.circle(heatmap_data, (x, y), 10, intensity, -1)
    
    # Apply Gaussian blur
    heatmap_data = cv2.GaussianBlur(heatmap_data, (31, 31), 0)
    
    # Save cursor data to JSON
    cursor_data_path = os.path.join(output_dir, "cursor_data.json")
    with open(cursor_data_path, 'w') as f:
        json.dump({
            'video_info': {
                'path': video_path,
                'scene_start_frame': start_frame,
                'scene_end_frame': end_frame,
                'scene_frames': scene_frame_count
            },
            'cursor_positions': cursor_positions
        }, f, indent=2)
    
    # Save heatmap
    plt.figure(figsize=(frame_width/100, frame_height/100))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(heatmap_data, cmap='jet', interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label('Cursor Intensity')
    
    # Save as image
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # Also save a heatmap overlay
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    cap.release()
    
    if ret:
        # Convert heatmap to color
        heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Convert first frame to RGB (from BGR)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        # Blend images
        alpha = 0.7
        overlay = cv2.addWeighted(first_frame_rgb, 1-alpha, heatmap_color, alpha, 0)
        
        # Save the overlay
        overlay_path = os.path.join(output_dir, "heatmap_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def process_scene_screen_reader(video_path, output_dir, start_frame, end_frame, 
                               motion_threshold=2.0, region_count_threshold=5, 
                               total_area_threshold_ratio=0.5):
    """
    Track screen reader focus in a specific scene (frame range) of a video
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save outputs
        start_frame: First frame of the scene
        end_frame: Last frame of the scene
    """
    # Initialize video
    cap = scene_initialize_video(video_path)
    frame_width, frame_height, frame_rate, _ = scene_get_video_properties(cap)
    total_area_threshold = frame_width * frame_height * total_area_threshold_ratio

    # Heatmap and background initialization
    heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    background_initialized = False
    first_window_background = None
    prev_gray = None
    frame_count = 0
    scene_frame_count = 0
    selected_frame_number = None

    # Skip to start frame
    while frame_count < start_frame:
        ret = cap.grab()  # Faster than read() for skipping
        if not ret:
            print(f"Error: Could not skip to frame {start_frame}")
            cap.release()
            return None
        frame_count += 1

    print(f"Processing scene frames {start_frame}-{end_frame} for screen reader focus...")

    # Process frames in scene
    while frame_count <= end_frame:
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        scene_frame_count += 1

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect stable frame using optical flow
        if prev_gray is not None and not background_initialized:
            avg_motion = scene_detect_stable_frame(prev_gray, gray_frame, motion_threshold)
            if avg_motion < motion_threshold:
                first_window_background = gray_frame.copy()
                background_initialized = True
                selected_frame_number = frame_count
                print(f"Background frame selected: Frame {selected_frame_number} (scene frame {scene_frame_count}), avg_motion={avg_motion:.2f}.")

        prev_gray = gray_frame  # Update for next optical flow calculation

        # Skip processing until background is ready
        if not background_initialized:
            frame_count += 1
            continue

        # Process frame difference
        fg_mask_thresh = scene_process_frame_difference(first_window_background, gray_frame, fgbg)

        # Find contours
        contours, _ = cv2.findContours(fg_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid contours
        valid_contours = scene_find_valid_contours(contours)

        # Count regions and compute total area
        region_count = len(valid_contours)
        total_area = sum(cv2.contourArea(contour) for contour in valid_contours)

        # Skip frame if it exceeds thresholds
        if region_count > region_count_threshold or total_area > total_area_threshold:
            frame_count += 1
            continue

        # Draw bounding boxes and update heatmap
        scene_draw_bounding_boxes(frame, valid_contours, heatmap_data)
        
        frame_count += 1

    # Release resources
    cap.release()

    # Save heatmap
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_scene_heatmap(heatmap_data, first_window_background, output_dir, 
                      f"{video_name}_scene_{start_frame}_to_{end_frame}", selected_frame_number)

def save_scene_heatmap(heatmap_data, background_frame, output_dir, scene_name, frame_number):
    """Save heatmap for a specific scene."""
    if background_frame is None:
        print("Warning: No stable background frame found for this scene")
        # Create a blank background
        background_frame = np.zeros_like(heatmap_data, dtype=np.uint8)
        
    # Normalize heatmap
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = np.uint8(heatmap_norm)

    # Plot and save heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_uint8, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=255)
    plt.colorbar()
    plt.title(f"Frequency Heatmap for {scene_name}")

    # Overlay initial background for context
    plt.imshow(background_frame, cmap='gray', alpha=0.3)
    plt.axis('off')

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"heatmap.png"))
    plt.close()


# In[ ]:

def process_video_with_scenes(video_path, output_dir, timestamp, user, 
                            scene_threshold=22.0, min_scene_duration=10):
    """
    Process a video with scene detection or as a single scene
    
    Args:
        video_path: Path to the video
        output_dir: Directory to save outputs
        timestamp: Current timestamp string
        user: Current username
        scene_threshold: Threshold for scene detection
        min_scene_duration: Minimum scene duration in frames
    
    Returns:
        True if processing was successful
    """
    try:
        # Try to detect scenes
        print("Attempting scene detection...")
        scenes = detect_scenes(video_path, threshold=scene_threshold, min_duration=min_scene_duration)
        
        if not scenes or len(scenes) == 0:
            print("No scenes detected, processing as a single scene")
            # Get total frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            scenes = [(0, total_frames-1)]
    
    except Exception as e:
        print(f"Scene detection failed: {str(e)}. Processing as a single scene.")
        # Get total frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        scenes = [(0, total_frames-1)]
    
    # Process each scene
    print(f"Processing {len(scenes)} scene(s)...")
    for i, (start_frame, end_frame) in enumerate(scenes):
        scene_number = i + 1
        scene_folder = os.path.join(output_dir, f"scene{scene_number}")
        os.makedirs(scene_folder, exist_ok=True)
        
        print(f"Processing scene {scene_number}/{len(scenes)}: frames {start_frame}-{end_frame}")
        
        # Calculate the middle frame of this scene
        middle_frame = start_frame + (end_frame - start_frame) // 2
        print(f"  Using middle frame {middle_frame} as background")
        
        # Generate heatmaps for this scene
        try:
            # Generate mouse cursor heatmap (using original method)
            print(f"  Generating mouse cursor heatmap...")
            generate_mouse_cursor_heatmap_original(
                video_path, scene_folder, start_frame, end_frame, timestamp, user, middle_frame
            )
            
            # Generate keyboard focus heatmap
            print(f"  Generating keyboard focus heatmap...")
            generate_keyboard_focus_heatmap(
                video_path, scene_folder, start_frame, end_frame, timestamp, user, middle_frame
            )
        except Exception as e:
            print(f"Error processing scene {scene_number}: {str(e)}")
            # Continue with next scene
    
    return True

def detect_scenes(video_path, threshold=22.0, min_duration=10):
    """
    Basic scene detection based on frame differences
    
    Args:
        video_path: Path to the video
        threshold: Threshold for scene change detection
        min_duration: Minimum scene duration in frames
    
    Returns:
        List of (start_frame, end_frame) tuples for each scene
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Could not read first frame")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    scene_boundaries = []
    current_scene_start = 0
    frame_count = 1
    
    print(f"Analyzing {total_frames} frames for scene detection...")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        avg_diff = np.mean(frame_diff)
        
        # Detect scene change
        if avg_diff > threshold and (frame_count - current_scene_start) >= min_duration:
            scene_boundaries.append((current_scene_start, frame_count - 1))
            current_scene_start = frame_count
            
            # Show progress periodically
            if len(scene_boundaries) % 5 == 0:
                print(f"  Detected {len(scene_boundaries)} scenes so far at frame {frame_count}/{total_frames}")
        
        prev_gray = gray
        frame_count += 1
        
        # Show progress periodically
        if frame_count % 500 == 0:
            print(f"  Analyzed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Add the final scene
    if current_scene_start < frame_count - 1:
        scene_boundaries.append((current_scene_start, frame_count - 1))
    
    cap.release()
    print(f"Detected {len(scene_boundaries)} scenes")
    
    return scene_boundaries

def generate_mouse_cursor_heatmap_original(video_path, scene_folder, start_frame, end_frame, timestamp, user, background_frame=None):
    """
    Generate mouse cursor heatmap using the original approach from 5703combined.ipynb
    
    Args:
        video_path: Path to the video
        scene_folder: Folder to save the heatmap
        start_frame: Starting frame of the scene
        end_frame: Ending frame of the scene
        timestamp: Current timestamp string
        user: Current username
        background_frame: Frame to use as background (if None, will be calculated)
    
    Returns:
        Path to the generated heatmap
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize variables for cursor tracking
    cursor_positions = []
    heat_map = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Parameters - from original code
    diff_threshold = 20
    heat_decay = 0.85
    blur_kernel = (21, 21)
    
    # Set the default background frame if not provided
    if background_frame is None:
        background_frame = start_frame + (end_frame - start_frame) // 2
    
    # Get the background frame first
    background_img = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)
    ret, background_img = cap.read()
    if not ret:
        print(f"Error: Could not read background frame {background_frame}")
        cap.release()
        return None
    
    # Save background image
    first_frame = background_img.copy()
    
    # Now reset and skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    
    # Read first frame of scene
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame of scene")
        cap.release()
        return None
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    # Previous cursor position for continuity
    prev_cursor = None
    
    # Process frames in scene
    print(f"  Processing scene frames {start_frame}-{end_frame} for mouse cursor...")
    scene_frame_count = 0
    
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Update heat map with decay
        heat_map = heat_map * heat_decay + thresh.astype(np.float32)
        
        # Apply Gaussian blur to consolidate cursor activity
        heat_map_blurred = cv2.GaussianBlur(heat_map, blur_kernel, 0)
        
        # Find the hottest point
        _, max_val, _, max_loc = cv2.minMaxLoc(heat_map_blurred)
        
        # Apply temporal filtering with previous cursor
        current_cursor = max_loc
        if prev_cursor is not None:
            # Calculate distance
            dist = np.sqrt((current_cursor[0] - prev_cursor[0])**2 + 
                          (current_cursor[1] - prev_cursor[1])**2)
            
            # If movement is too large, smooth it
            if dist > 50:
                alpha = 0.7  # Weight for previous position
                current_cursor = (
                    int(alpha * prev_cursor[0] + (1-alpha) * current_cursor[0]),
                    int(alpha * prev_cursor[1] + (1-alpha) * current_cursor[1])
                )
        
        # Store cursor position
        cursor_positions.append({
            'frame': frame_count,
            'time': (frame_count - start_frame) / fps,
            'x': current_cursor[0],
            'y': current_cursor[1],
            'intensity': float(max_val)
        })
        
        # Update previous cursor
        prev_cursor = current_cursor
        
        # Update for next iteration
        prev_gray = gray
        frame_count += 1
        scene_frame_count += 1
        
        # Show progress for long scenes
        if scene_frame_count % 100 == 0:
            total_scene_frames = end_frame - start_frame + 1
            print(f"    Processed {scene_frame_count}/{total_scene_frames} scene frames ({scene_frame_count/total_scene_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    
    print("  Generating mouse cursor heatmap visualization...")
    # Generate heatmap data from cursor positions
    heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Add cursor positions to heatmap
    for pos in cursor_positions:
        x, y = int(pos['x']), int(pos['y'])
        # Skip if outside bounds
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            continue
            
        # Add weighted point to heatmap - original implementation
        intensity = max(1.0, pos['intensity'] / 50.0)  # Normalize intensity
        cv2.circle(heatmap_data, (x, y), 10, intensity, -1)
    
    # Apply Gaussian blur to smooth - original implementation
    heatmap_data = cv2.GaussianBlur(heatmap_data, (31, 31), 0)
    
    # Save a heatmap overlay on the first frame - exactly as in original code
    # Convert heatmap to color
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Convert first frame to RGB (from BGR)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Blend images
    alpha = 0.7
    overlay = cv2.addWeighted(first_frame_rgb, 1-alpha, heatmap_color, alpha, 0)
    
    # Add timestamp and user info
    cv2.putText(overlay, f"Generated: {timestamp}", (10, frame_height - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"User: {user}", (10, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save the overlay
    mousecursor_path = os.path.join(scene_folder, "mousecursor.png")
    cv2.imwrite(mousecursor_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return mousecursor_path

def generate_keyboard_focus_heatmap(video_path, scene_folder, start_frame, end_frame, timestamp, user, background_frame=None):
    """
    Generate screen reader (keyboard) focus heatmap for a specific scene
    
    Args:
        video_path: Path to the video
        scene_folder: Folder to save the heatmap
        start_frame: Starting frame of the scene
        end_frame: Ending frame of the scene
        timestamp: Current timestamp
        user: Current username
        background_frame: Frame to use as background (if None, will use middle frame)
    
    Returns:
        Path to the generated heatmap
    """
    # Initialize video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set the default background frame if not provided
    if background_frame is None:
        background_frame = start_frame + (end_frame - start_frame) // 2
    
    print(f"  Using frame {background_frame} as background for keyboard focus")
    
    # Get the background frame first
    first_window_background = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)
    ret, background_img = cap.read()
    if not ret:
        print(f"Error: Could not read background frame {background_frame}")
        cap.release()
        return None
    
    # Convert background to grayscale
    first_window_background = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize for frame processing
    heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    total_area_threshold_ratio = 0.5
    total_area_threshold = frame_width * frame_height * total_area_threshold_ratio
    region_count_threshold = 5
    
    # Now reset and skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    
    print(f"  Processing scene frames {start_frame}-{end_frame} for keyboard focus...")
    scene_frame_count = 0
    
    # Process frames in scene
    while frame_count <= end_frame:
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        scene_frame_count += 1

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process frame difference with our fixed background
        frame_diff = cv2.absdiff(first_window_background, gray_frame)
        fg_mask = fgbg.apply(frame_diff)
        _, fg_mask_thresh = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(fg_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid contours
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

        # Count regions and compute total area
        region_count = len(valid_contours)
        total_area = sum(cv2.contourArea(contour) for contour in valid_contours)

        # Skip frame if it exceeds thresholds
        if region_count > region_count_threshold or total_area > total_area_threshold:
            frame_count += 1
            continue

        # Update heatmap with detected regions
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            heatmap_data[y:y+h, x:x+w] += 1
        
        frame_count += 1
        
        # Show progress periodically
        if scene_frame_count % 100 == 0:
            total_scene_frames = end_frame - start_frame + 1
            print(f"    Processed {scene_frame_count}/{total_scene_frames} scene frames ({scene_frame_count/total_scene_frames*100:.1f}%)")

    # Release resources
    cap.release()
    
    # Normalize heatmap
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = np.uint8(heatmap_norm)

    # Plot and save heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_uint8, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=255)  # Use coolwarm color map
    plt.colorbar()
    plt.title(f"Keyboard Focus Heatmap (Scene frames {start_frame}-{end_frame})")

    # Overlay initial background for context
    plt.imshow(first_window_background, cmap='gray', alpha=0.3)  # Reduce alpha to prevent hiding buttons
    plt.axis('off')
    
    # Add timestamp and user info
    plt.annotate(f"Generated: {timestamp}\nUser: {user}", xy=(0.01, 0.01), xycoords='figure fraction', 
                color='white', backgroundcolor='black', fontsize=8)

    # Save to file
    keyboard_path = os.path.join(scene_folder, "keyboard.png")
    plt.savefig(keyboard_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    
    return keyboard_path

def analyze_screen(video_path, output_dir, timestamp, user):
    process_video_with_scenes(
        video_path, output_dir, timestamp, user,
        scene_threshold=22.0, min_scene_duration=10
    )
    return output_dir



# In[ ]:





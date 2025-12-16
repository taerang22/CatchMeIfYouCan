from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
import pyrealsense2 as rs
import os
import yaml

# =========================
# ARGUMENTS
# =========================
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("--serial", type=str, default=None, help="RealSense device serial to use")
ap.add_argument("--calib", type=str, default="camera_calib_cd.yaml", help="optional yaml path")
ap.add_argument("--session_id", type=int, default=1, help="session id for file naming")
args = vars(ap.parse_args())

# =========================
# CONFIGURATION
# =========================
MARKER_ID = 7
MARKER_LENGTH = 0.100     # 100 mm marker
MAKE_Y_UP = False
USE_YAML_FOR_ARUCO = False # Set to False to use Factory Intrinsics (Recommended)
MAX_DEPTH_M = 4.5
MISS_FRAMES_KEEP = 25    # Used to clear the visual trail if ball is lost
RECORD_MISS_LIMIT = 30   # Frames to wait before cutting the file after ball leaves
CALIBRATION_FRAMES = 60   # How many frames to average for the fixed anchor

# # Tennis ball mask (Yellow)
# yellowLower = (21, 79, 54)
# yellowUpper = (57, 204, 245)
# # Tennis ball mask (Yellow) in lab
# yellowLower = (21, 63, 137)
# yellowUpper = (67, 202, 255)
# # Tennis ball mask (Yellow) in door lab
# yellowLower = (13, 74, 81)
# yellowUpper = (49, 167, 255)
# # yellow ball mask (real yellow) in door lab
# yellowLower = (20, 111, 121)
# yellowUpper = (31, 180, 239)
# bee mask (Yellow) in lab
yellowLower = (19, 120, 112)
yellowUpper = (34, 234, 255)

# =========================
# 1. RealSense Setup
# =========================
pipe = rs.pipeline()
cfg = rs.config()

if args["serial"]:
    cfg.enable_device(args["serial"])

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

try:
    profile = pipe.start(cfg)
except RuntimeError as e:
    print("Error: Could not start RealSense camera.")
    print(e)
    raise SystemExit

dev = profile.get_device()
name = dev.get_info(rs.camera_info.name)
serial = dev.get_info(rs.camera_info.serial_number)
align = rs.align(rs.stream.color)

# Get Factory Intrinsics
color_stream = profile.get_stream(rs.stream.color)
color_intr = color_stream.as_video_stream_profile().get_intrinsics()

rs_K = np.array([
    [color_intr.fx, 0,             color_intr.ppx],
    [0,             color_intr.fy, color_intr.ppy],
    [0,             0,             1]
], dtype=np.float32)
rs_dist = np.array(color_intr.coeffs, dtype=np.float32).reshape(-1, 1)

# =========================
# 2. Intrinsics Selection
# =========================
aruco_K = rs_K.copy()
aruco_dist = rs_dist.copy()

calib_path = args["calib"]
if USE_YAML_FOR_ARUCO and calib_path and os.path.exists(calib_path):
    try:
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)
        if "color" in calib:
            K_y = np.array(calib["color"]["K"], dtype=np.float32)
            d_y = np.array(calib["color"]["dist"], dtype=np.float32).reshape(-1, 1)
            fx_y = float(K_y[0, 0])
            if 200.0 < fx_y < 2000.0:
                aruco_K = K_y
                aruco_dist = d_y
                print(f"[INFO] Using YAML intrinsics (fx={fx_y:.1f})")
            else:
                print(f"[WARN] YAML fx={fx_y:.1f} suspicious. Using Factory.")
    except Exception as e:
        print(f"[WARN] YAML load failed: {e}")

print("[INFO] ArUco K:\n", aruco_K)

# =========================
# 3. Globals & Helpers
# =========================
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 

pts = deque(maxlen=args["buffer"])
miss_count = 0 # For visual trail logic

# Recording State Machine
recording_mode = False        # Is the 'r' key toggled on?
recording_has_started = False # Have we seen the first ball yet?
record_file = None
record_miss_count = 0         # For file cut-off logic
record_counter = 0

# FIXED ANCHOR VARIABLES
world_calibrated = False
calib_rvecs = []
calib_tvecs = []
fixed_R_cm = None # Fixed Rotation Matrix
fixed_t_cm = None # Fixed Translation Vector
fixed_rvec = None # For visualization
fixed_tvec = None # For visualization

def stop_recording():
    global recording_mode, recording_has_started, record_file, record_miss_count
    if recording_mode and record_file is not None:
        record_file.close()
        record_file = None
        print(f"[INFO] File closed.")
    recording_mode = False
    recording_has_started = False
    record_miss_count = 0

print("[INFO] Starting... Please allow camera to see marker for calibration.")

# =========================
# MAIN LOOP
# =========================
while True:
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    
    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    display_frame = frame.copy() # Copy for visualization

    # =====================================
    # PHASE 1: CALIBRATION (First N frames)
    # =====================================
    if not world_calibrated:
        # We need to find the marker to build our coordinate system
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        if ids is not None and len(ids) > 0:
            for i, m_id in enumerate(ids.flatten()):
                if m_id == MARKER_ID:
                    # Calculate pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], MARKER_LENGTH, aruco_K, aruco_dist
                    )
                    rvec = rvecs[0, 0, :]
                    tvec = tvecs[0, 0, :]
                    
                    # Store for averaging
                    calib_rvecs.append(rvec)
                    calib_tvecs.append(tvec)
                    
                    # Visual feedback
                    cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    cv2.drawFrameAxes(display_frame, aruco_K, aruco_dist, rvec, tvec, MARKER_LENGTH * 1.5)
                    break
        
        # Progress Bar logic
        current_samples = len(calib_rvecs)
        progress = current_samples / CALIBRATION_FRAMES
        
        # Draw loading bar
        bar_x, bar_y, bar_w, bar_h = 160, 240, 320, 30
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.rectangle(display_frame, (bar_x + 5, bar_y + 5), (bar_x + 5 + int((bar_w-10)*progress), bar_y + bar_h - 5), (0, 255, 0), -1)
        
        status_text = f"CALIBRATING WORLD: {current_samples}/{CALIBRATION_FRAMES}"
        cv2.putText(display_frame, status_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_samples >= CALIBRATION_FRAMES:
            # --- FINISH CALIBRATION ---
            print("[INFO] Calibration complete. Calculating average pose...")
            
            avg_rvec = np.mean(calib_rvecs, axis=0)
            avg_tvec = np.mean(calib_tvecs, axis=0)
            
            fixed_R_cm, _ = cv2.Rodrigues(avg_rvec)
            fixed_t_cm = avg_tvec.reshape(3, 1)
            
            fixed_rvec = avg_rvec
            fixed_tvec = avg_tvec
            
            world_calibrated = True
            print(f"[INFO] World Fixed at: {fixed_t_cm.flatten()}")

    # =====================================
    # PHASE 2: TRACKING (Using Fixed World)
    # =====================================
    else:
        # 1. Visualize the "Ghost" Marker (Fixed World Origin)
        try:
            cv2.drawFrameAxes(display_frame, aruco_K, aruco_dist, fixed_rvec, fixed_tvec, MARKER_LENGTH * 1.5)
            cv2.putText(display_frame, "WORLD LOCKED", (450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except:
            pass

        # 2. Detect Ball
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        center = None
        valid_detection = False
        ball_marker_cm = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if radius > 3 and M["m00"] != 0:
                cx_px = int(M["m10"] / M["m00"])
                cy_px = int(M["m01"] / M["m00"])
                center = (cx_px, cy_px)

                depth_m = depth_frame.get_distance(cx_px, cy_px)
                
                if 0 < depth_m < MAX_DEPTH_M:
                    valid_detection = True
                    miss_count = 0 # Reset visual trail miss count

                    # Draw Ball
                    cv2.circle(display_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(display_frame, center, 5, (0, 0, 255), -1)

                    # --- COORDINATE TRANSFORM ---
                    Pc = rs.rs2_deproject_pixel_to_point(color_intr, [cx_px, cy_px], depth_m)
                    Pc = np.array(Pc, dtype=np.float32).reshape(3, 1)

                    # Camera -> Fixed World
                    Pm = fixed_R_cm.T @ (Pc - fixed_t_cm)

                    if MAKE_Y_UP:
                        Pm[1, 0] = -Pm[1, 0]

                    ball_marker_cm = Pm.reshape(3) * 100.0
                    
                    # Display text
                    txt = f"World: {ball_marker_cm[0]:.1f}, {ball_marker_cm[1]:.1f}, {ball_marker_cm[2]:.1f}"
                    cv2.putText(display_frame, txt, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3. Trail Logic (Uses MISS_FRAMES_KEEP)
        if not valid_detection:
            miss_count += 1
            if miss_count > MISS_FRAMES_KEEP:
                center = None # This breaks the red visual line

        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(display_frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # 4. Smart Recording Logic
        if recording_mode:
            if valid_detection and ball_marker_cm is not None:
                # Ball FOUND: Start/Continue Writing
                recording_has_started = True 
                record_miss_count = 0 # Reset the "stop" timer
                
                ts = time.time()
                record_file.write(f"{ts:.6f},{ball_marker_cm[0]:.5f},{ball_marker_cm[1]:.5f},{ball_marker_cm[2]:.5f}\n")
                record_file.flush()
                
                cv2.putText(display_frame, "REC (WRITING)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            elif recording_has_started:
                # Ball LOST (but we had started): Countdown to stop
                record_miss_count += 1
                
                cv2.putText(display_frame, f"REC (LOST: {record_miss_count}/{RECORD_MISS_LIMIT})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if record_miss_count > RECORD_MISS_LIMIT:
                    print(f"[INFO] Ball gone for {RECORD_MISS_LIMIT} frames. Saving & Stopping.")
                    stop_recording()
            
            else:
                # Waiting for FIRST ball
                cv2.putText(display_frame, "REC (WAITING FOR BALL...)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # =========================
    # UI & CONTROLS
    # =========================
    cv2.putText(display_frame, f"{name} | S/N: {serial}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Frame", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        if world_calibrated:
            if recording_mode:
                # Manually stop if already recording
                stop_recording()
                print("[INFO] Manual stop.")
            else:
                # Start "Waiting" mode
                record_counter += 1
                fname = f"ball_track_{args['session_id']}_{record_counter}.txt"
                record_file = open(fname, "w")
                recording_mode = True
                recording_has_started = False # Wait for ball
                record_miss_count = 0
                print(f"[INFO] Arming Recorder... Waiting for ball to appear in frame.")
        else:
            print("[WARN] Cannot record yet - Calibration incomplete!")
    elif key == ord("e"):
        stop_recording()
        print("[INFO] Recording stopped manually.")
    elif key == ord("c"):
        # Force recalibrate
        world_calibrated = False
        calib_rvecs = []
        calib_tvecs = []
        print("[INFO] Restarting calibration...")

stop_recording()
pipe.stop()
cv2.destroyAllWindows()
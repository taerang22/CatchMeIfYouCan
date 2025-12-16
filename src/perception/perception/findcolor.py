import cv2
import numpy as np
import pyrealsense2 as rs # Import RealSense library

# A dummy function required by the createTrackbar function
def empty(a):
    pass

# --- 1. RealSense Camera Setup ---
pipe = rs.pipeline()
cfg  = rs.config()

# Configure the color stream for the RealSense camera (640x480)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 

try:
    pipe.start(cfg)
except RuntimeError:
    print("Error: Could not start RealSense camera. Check USB connection and drivers.")
    exit()

# --- 2. Trackbar Setup ---
# Create a window for the trackbars
cv2.namedWindow("HSV Trackbars")
cv2.resizeWindow("HSV Trackbars", 640, 240)

# Create trackbars for H, S, V min and max values
# Starting with a common range for Yellow (H: 20-40, S/V: 100-255)
cv2.createTrackbar("H Min", "HSV Trackbars", 20, 179, empty)
cv2.createTrackbar("S Min", "HSV Trackbars", 100, 255, empty)
cv2.createTrackbar("V Min", "HSV Trackbars", 100, 255, empty)
cv2.createTrackbar("H Max", "HSV Trackbars", 40, 179, empty)
cv2.createTrackbar("S Max", "HSV Trackbars", 255, 255, empty)
cv2.createTrackbar("V Max", "HSV Trackbars", 255, 255, empty)


# --- 3. Main Loop: Read Camera and Apply Mask ---
print("Adjust the trackbars until the ball is white in the 'Mask' window.")
print("Press 'q' to print the final HSV values and quit.")

while True:
    # Get RealSense frames
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue
        
    img = np.asanyarray(color_frame.get_data())
    
    # Convert to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Read current positions of trackbars
    h_min = cv2.getTrackbarPos("H Min", "HSV Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "HSV Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "HSV Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "HSV Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "HSV Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "HSV Trackbars")
    
    # Define the lower and upper HSV boundaries (the color filter)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create the mask (isolates colors within the range)
    mask = cv2.inRange(imgHSV, lower, upper)
    
    # Display the live video and the resulting mask
    cv2.imshow("RealSense Feed", img)
    cv2.imshow("Mask", mask)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Print Final Values and Cleanup ---

# Print the final values you need for your main tracking script
print("\n--- FINAL HSV VALUES FOR YOUR YELLOW BALL ---")
print(f"hmin: {h_min}, smin: {s_min}, vmin: {v_min}")
print(f"hmax: {h_max}, smax: {s_max}, vmax: {v_max}")
print("-" * 40)
print(f"Update your main script's hsvVals dictionary to:")
print("{'hmin': " + str(h_min) + ", 'smin': " + str(s_min) + ", 'vmin': " + str(v_min) + 
      ", 'hmax': " + str(h_max) + ", 'smax': " + str(s_max) + ", 'vmax': " + str(v_max) + "}")

# Stop RealSense streaming and close all OpenCV windows
pipe.stop()
cv2.destroyAllWindows()
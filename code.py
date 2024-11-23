import cv2 
import numpy as np
from niryo_one_tcp_client import *
from niryo_one_camera import *

# Niryo One robot's IP address
robot_ip_address = "10.10.10.10"

# Set Observation Pose. This is the position the robot will be placed for streaming
inter_pose = PoseObject(
    x=0.2, y=0.0, z=0.34,
    roll=0, pitch=1.57, yaw=-0.2,
)
gripper_used = RobotTool.GRIPPER_2  # Tool used for picking
inter=[0.266, -0.034, -0.071, 0.035, -0.57, 1.326]
pos2=[1.914,-0.618,-0.127,0.006,-1.044,1.367]
pos1_3=[1.910,-0.467,-0.055,-0.003,-1.176,1.377]
pos4=[0.094,-0.861,-0.173,0.055,-0.680,-2.035]

# Placeholder for robot's Z-axis height above the workspace
workspace_z_height = 0.01  # Adjust based on your workspace configuration
#inter=[0.295, -0.044, -0.022, 0.017, -0.131, -2.045]

def detect_empty_hole_in_frame(image, workspace_width_cm, workspace_height_cm, niryo_one_client):
    # Step 1: Detect the white box (command box) in the image using HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])  # Range for detecting white color
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find the largest contour, which should correspond to the command box
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # Take the largest contour (the command box)

    # Create a mask for isolating the command box area
    mask_box = np.zeros_like(mask_white)
    cv2.drawContours(mask_box, [contour], -1, 255, thickness=cv2.FILLED)
    roi = cv2.bitwise_and(image, image, mask=mask_box)  # Restrict to command box

    # Step 3: Convert the region to grayscale and apply Gaussian blur
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Step 4: Detect circles (holes) within the command box using Hough Circle Transform
    min_radius = 20  # Adjust as per the size of the holes
    max_radius = 25

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    found_empty_hole = False
    image_with_hole = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Image dimensions for pixel-to-real-world conversion
        image_height, image_width = mask_white.shape[:2]

        # Calculate pixel-to-centimeter conversion factors
        x_factor = workspace_width_cm / image_width
        y_factor = workspace_height_cm / image_height

        # Sort the circles (holes) based on their position (top to bottom, left to right)
        sorted_circles = sorted(circles[0, :], key=lambda c: (c[1], c[0]))

        for i in sorted_circles:
            x, y, r = i[0], i[1], i[2]

            # Check if the circle is within the command box (ROI)
            if mask_box[y, x] == 0:  # Ignore circles outside the command box
                continue

            # Create a mask for analyzing the content of the circle
            mask = np.zeros_like(gray_roi)
            cv2.circle(mask, (x, y), r, 255, -1)  # White circle on black background

            # Calculate the average intensity in the circle (empty or filled hole check)
            mean_intensity = cv2.mean(gray_roi, mask=mask)[0]

            # If the average intensity is low, the hole is likely empty
            if mean_intensity < 100:  # Adjust this threshold based on your image conditions
                found_empty_hole = True

                # Convert pixel coordinates to real-world coordinates
                real_x = x * x_factor
                real_y = y * y_factor

                # Highlight the empty hole with a green circle
                cv2.circle(image_with_hole, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image_with_hole, (x, y), 3, (0, 0, 255), -1)  # Red center
                cv2.putText(image_with_hole, f"({real_x:.2f}cm, {real_y:.2f}cm)", (x - 30, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                print(f"Empty hole detected at pixel (x, y): ({x}, {y}) with radius {r} pixels.")
                print(f"Converted to real-world coordinates: ({real_x:.2f} cm, {real_y:.2f} cm).")
                
                # Command the robot to move to the detected coordinates
                target_pose = PoseObject(
                    x=real_x / 100,  # Convert cm to meters
                    y=real_y / 100,  # Convert cm to meters
                    z=workspace_z_height,
                    roll=0.912, pitch=1.424, yaw=2.996
                )
                niryo_one_client.move_joints(*inter)
                #niryo_one_client.move_pose(*target_pose.to_list())
                print(f"Robot moved to: {target_pose}")
                break

    if not found_empty_hole:
        print("No empty hole detected in the command box.")

    return image_with_hole
def pick_n_place_version_0(niryo_one_client):
    gripper_speed = 400
    niryo_one_client.move_joints(*inter)
    niryo_one_client.close_gripper(gripper_used, gripper_speed)

    niryo_one_client.move_joints(*pos1_3)
    niryo_one_client.open_gripper(gripper_used, gripper_speed)
    niryo_one_client.move_joints(*pos2) 
  
    niryo_one_client.close_gripper(gripper_used, gripper_speed)
    niryo_one_client.move_joints(*pos1_3)
   

def video_stream(niryo_one_client, workspace_width_cm, workspace_height_cm):
    # Getting calibration parameters
    _, mtx, dist = niryo_one_client.get_calibration_object()
    
    # Moving robot to observation pose for camera view
    #niryo_one_client.move_pose(*observation_pose.to_list())

    while True:
        # Getting compressed image from the robot
        status, img_compressed = niryo_one_client.get_img_compressed()
        if status is not True:
            print("Error: Unable to retrieve image from Niryo One.")
            break

        # Uncompress the image
        img_raw = uncompress_image(img_compressed)

        # Undistort the image using camera calibration parameters
        img_undistort = undistort_image(img_raw, mtx, dist)

        # Detect the first empty hole in the video frame
        img_with_empty_hole = detect_empty_hole_in_frame(
            img_undistort, workspace_width_cm, workspace_height_cm, niryo_one_client
        )

        # Display the result with the detected hole
        cv2.imshow("First Empty Hole Detected", img_with_empty_hole)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'Esc'
            break

    niryo_one_client.set_learning_mode(True)
    cv2.destroyAllWindows()


if _name_ == "_main_":
    # Workspace dimensions in centimeters
    workspace_width_cm = 22.5
    workspace_height_cm = 12.7

    # Connect to robot
    client = NiryoOneClient()
    client.connect(robot_ip_address)
    pick_n_place_version_0(client)

    # Start video streaming and detection
    video_stream(client, workspace_width_cm, workspace_height_cm)

    # Releasing connection after stream ends
    client.quit()
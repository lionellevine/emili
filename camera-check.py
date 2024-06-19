import cv2
import sys

def get_camera_info(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Camera {camera_id} failed to initialize.")
        return None
    info = {}
    info['id'] = camera_id
    info['backend_name'] = cap.getBackendName()

    # Try common resolutions
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    supported_resolutions = []
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) == (actual_width, actual_height):
            supported_resolutions.append((actual_width, actual_height))

    info['supported_resolutions'] = supported_resolutions
    cap.release()
    return info

def list_cameras():
    max_cameras = 10  # Adjust based on expected number of cameras
    camera_details = []
    for i in range(max_cameras):
        info = get_camera_info(i)
        if info is not None:
            camera_details.append(info)
        else:
            break
    return camera_details

if __name__ == "__main__":
    # List available cameras
    cameras = list_cameras()
    if len(cameras) == 0:
        print("No camera devices found.")
        sys.exit(1)

    print("Available cameras:")
    for camera in cameras:
        print(f"Camera ID: {camera['id']}")
        print(f"Backend: {camera['backend_name']}")
        print(f"Supported Resolutions: {camera['supported_resolutions']}")
        print("-" * 40)
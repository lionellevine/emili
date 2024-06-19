# List available cameras on macOS using system_profiler
# (But it's unclear how they match up with the cameras detected by cv2.VideoCapture)

import subprocess
import re

def list_cameras_mac():
    try:
        result = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True, check=True)
        output = result.stdout
        cameras = re.findall(r'(\w+ Camera)', output)
        return cameras
    except subprocess.CalledProcessError as e:
        print(f"Error listing cameras: {e}")
        return []

if __name__ == "__main__":
    cameras = list_cameras_mac()
    print("Available cameras:", cameras)
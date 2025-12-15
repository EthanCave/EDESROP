import time
import os
from datetime import datetime
import cv2
from picamera2 import Picamera2

output_folder = "captured_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("STARTING CAMERA")
picam2 = Picamera2()

# Configure the camera for dual streams: low-res preview and high-res capture
preview_config = picam2.create_preview_configuration(
    main={"format": 'BGR888', "size": (640, 480)},  # Low resolution for preview
    lores={"format": 'YUV420', "size": (320, 240)},  # Optional lower-res stream
    display="lores"
)
capture_config = picam2.create_still_configuration(
    main={"format": 'BGR888', "size": (4056, 3040)}  # Maximum resolution for photos
)

picam2.configure(preview_config)
picam2.start()

# Show the low-resolution preview
print("Press 'c' to capture a photo or 'q' to quit.")
while True:
    frame = picam2.capture_array()
    cv2.imshow("Preview", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Capture photo on pressing 'c'
        print("Capturing photo...")
        picam2.switch_mode_and_capture_file(capture_config, os.path.join(output_folder, f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"))
        print("Photo captured!")
    elif key == ord('q'):  # Quit on pressing 'q'
        break

cv2.destroyAllWindows()
picam2.stop()
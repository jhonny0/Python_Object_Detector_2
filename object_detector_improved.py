import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

# Load the trained model
model_dir = '/Users/jonathanfernandez/Downloads/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]

# Load the label map and create a category index
PATH_TO_LABELS = '/Users/jonathanfernandez/Downloads/centernet_mobilenetv2_fpn_od/label_map.txt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def detect_objects(frame: np.ndarray) -> np.ndarray:
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (320, 320))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform object detection
    detections = infer(input_tensor)

    # Visualize the detections on the frame
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,  # Only display detections with confidence above 0.5
        agnostic_mode=False
    )

    return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width (optional)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height (optional)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        frame_with_detections = detect_objects(frame)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame_with_detections)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

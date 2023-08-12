import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

# Load the trained model
model = tf.saved_model.load('/Users/jonathanfernandez/Downloads/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')
infer = model.signatures["serving_default"]

# Load the label map and create a category index
PATH_TO_LABELS = '/Users/jonathanfernandez/Downloads/centernet_mobilenetv2_fpn_od/label_map.txt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def detect_objects(frame: np.ndarray) -> np.ndarray:
    """
    Detects objects in the given frame using object detection model.

    Args:
        frame: The input frame to detect objects in.

    Returns:
        The frame with object detections.
    """
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (320, 320))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform object detection
    detections = infer(input_tensor)
    print(detections.keys())

    # TODO: Map the outputs to their corresponding values based on the model's documentation or your understanding.
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    # Visualize the detections on the frame
    frame_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        boxes,
        classes.astype(int),
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        agnostic_mode=False
    )

    return frame_with_detections


def main() -> None:
    """
    Captures video from the default camera, detects objects in each frame using the detect_objects function,
    and displays the frames with object detections. The loop continues until the user presses 'q' to quit.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera

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

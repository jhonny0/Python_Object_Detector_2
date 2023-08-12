import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from keras.models import load_model

# Load the trained model
MODEL_PATHS = {
    "ssd_mobilenet": '/Users/jonathanfernandez/Downloads/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model',
    "custom_model": '/Users/jonathanfernandez/Downloads/converted_savedmodel-3/model.savedmodel'
}
model = {name: tf.saved_model.load(path) for name, path in MODEL_PATHS.items()}

# Load the label map and create a category index
PATH_TO_LABELS = {
    "ssd_mobilenet": '/Users/jonathanfernandez/Downloads/centernet_mobilenetv2_fpn_od/label_map.txt',
    "custom_model": '/Users/jonathanfernandez/Downloads/converted_savedmodel-3/labels.txt'
}
category_index = {name: label_map_util.create_category_index_from_labelmap(path, use_display_name=True) for name,
path in PATH_TO_LABELS.items()}


def detect_objects(frame: np.ndarray) -> np.ndarray:
    # Process with custom_model
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction_function = model["custom_model"].__call__  # Get the callable function for the custom model
    prediction = prediction_function(tf.constant(image))
    prediction = prediction.numpy()  # Convert tensor to numpy array
    index = np.argmax(prediction) + 1
    print("Predicted class ID:", index)

    class_name = category_index["custom_model"][index]['name']
    confidence_score = prediction[0][index-1]

    # Display the class label and confidence score from the custom model
    label_text = f"{class_name}: {confidence_score:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Process with ssd_mobilenet
    resized_frame = cv2.resize(frame, (320, 320))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    infer = model["ssd_mobilenet"].signatures["serving_default"]
    detections = infer(input_tensor)

    # Visualize the detections on the frame
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(int),
        detections['detection_scores'][0].numpy(),
        category_index["ssd_mobilenet"],
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
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

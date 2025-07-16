import onnxruntime as ort
import numpy as np
import cv2

CONF_THRESHOLD = 0.4
INPUT_SIZE = 640

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Load ONNX model once
session = ort.InferenceSession("yolo_nas_s_fp16.onnx")
input_name = session.get_inputs()[0].name


def detect_objects(image_path_or_np):
    # If PIL.Image, convert to NumPy
    if hasattr(image_path_or_np, 'convert'):  # it's a PIL image
        img = np.array(image_path_or_np.convert("RGB"))
    elif isinstance(image_path_or_np, str):  # path
        img = cv2.imread(image_path_or_np)
    else:
        img = image_path_or_np  # assume numpy

    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    input_image = np.transpose(img_resized, (2, 0, 1))[None, ...].astype(np.uint8)

    output = session.run(None, {input_name: input_image})
    num_dets = int(output[0][0][0])
    bboxes = output[1][0][:num_dets]
    confs = output[2][0][:num_dets]
    class_ids = output[3][0][:num_dets].astype(int)

    results = []

    for i in range(num_dets):
        x1, y1, x2, y2 = bboxes[i]
        conf = confs[i]
        class_id = class_ids[i]
        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

        # Scale to original image size
        x1 = int(x1 * w0 / INPUT_SIZE)
        y1 = int(y1 * h0 / INPUT_SIZE)
        x2 = int(x2 * w0 / INPUT_SIZE)
        y2 = int(y2 * h0 / INPUT_SIZE)

        results.append({
            "label": label,
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img, results

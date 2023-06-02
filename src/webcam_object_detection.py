# import cv2

# from ultralytics import YOLO

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# # Initialize YOLOv7 object detector
from ultralytics import YOLO
import supervision as sv
import cv2
model_path = "/Users/Admin/Documents/personal/ONNX-YOLOv8-Object-Detection/runs/detect/train11/weights/best.pt"
# yolov8_detector = YOLO(model_path)
# # yolov8_detector = YOLO(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# while cap.isOpened():

#     # Read frame from the video
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Update object localizer
#     boxes, scores, class_ids = yolov8_detector.predict(frame)

#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.imshow("Detected Objects", combined_img)

#     # Press key q to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


def main():

    # # to save the video
    # writer = cv2.VideoWriter('webcam_yolo.mp4',
    #                          cv2.VideoWriter_fourcc(*'DIVX'),
    #                          7,
    #                          (1280, 720))

    # define resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # specify the model
    model = YOLO(
        "/Users/Admin/Documents/personal/ONNX-YOLOv8-Object-Detection/runs/detect/train13/weights/best.pt")
    # model = YOLO(
    #     "/Users/Admin/Documents/personal/ONNX-YOLOv8-Object-Detection/models/yolov8n.pt")
    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        print(result)
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.1f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # writer.write(frame)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):  # break with escape key
            break

    cap.release()
    # writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

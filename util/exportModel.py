from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict(source="https://ultralytics.com/images/bus.jpg")[0]
print(results)
# export the model to ONNX format
model.export(format="onnx", imgsz=[640, 640], opset=12)

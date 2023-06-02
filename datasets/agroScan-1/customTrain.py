from ultralytics import YOLO

# Load a model
model = YOLO("/models/yolov8n.pt")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/Users/Admin/Documents/personal/ONNX-YOLOv8-Object-Detection/datasets/agroScan-1/data.yaml",
            epochs=10)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
model.export(format="pt")  # export the model to ONNX format
# C:\Users\Admin\Documents\personal\ONNX-YOLOv8-Object-Detection\datasets\agroScan-1

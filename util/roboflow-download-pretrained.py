from roboflow import Roboflow
rf = Roboflow(api_key="o638K18mmUlsYpQHLmn7")
project = rf.workspace(
    "dr-ambedkar-institute-of-technology-vlluz").project("agroscan")
dataset = project.version(1).download("yolov8")

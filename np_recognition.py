import torch
import cv2

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # your own model
# OR for pretrained coco:
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Run detection
img = 'output_dataset(1)/output_dataset/test/images/Cars77.png'  # path to image
results = model(img)

# Show results
results.print()
results.show()  # display in a pop-up window
results.save(save_dir='output_yolov5')  # saves to output_yolov5


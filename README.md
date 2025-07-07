# Handwritten-Form-Recognition
# 📝 DeHaDo-AI Challenge – [YourTeamName]

This repository contains our submission to the **DeHaDo-AI Challenge 2025**, focused on handwritten document understanding. Our solution performs layout-independent detection and recognition of handwritten fields using YOLOv8 for form structure detection and a CNN+BiLSTM+CTC model for text recognition.

---------------------------------------------------

## 📁 Folder Structure
submission/
│
├── model/
│ └── handwritten_model.pth # Trained PyTorch model file
│ └── best_yolov8s.pt # YOLOv8 trained detector
│
├── src/
│ ├── detect_and_read.py # Inference script (YOLO + OCR)
│ ├── model.py # OCR model architecture
│ ├── loader.py # Data loader for training
│ ├── train.py # Training script for OCR model
│ ├── evaluation.py # Evaluation script for metrics
│ └── utils.py # (Optional) helper functions
│
├── data/
│ └── sample_input/ # Sample images for testing
│
├── outputs/ # Folder where prediction JSONs are saved
│
├── evaluation_metrics.txt # Final evaluation summary
├── requirements.txt # List of Python dependencies
└── README.md # This file

-----------------------------------------------------



## 🚀 Setup Instructions

### 1. Python Environment

Ensure you are using Python 3.8 or later. Install dependencies:

pip install -r requirements.txt

Main dependencies:

torch

torchvision

ultralytics

opencv-python

editdistance

pillow

psutil

--------------------------------------------------------


📄 Input
✏️ Mandatory Step Before Running
In all scripts (inference.py, evaluation.py, loader.py, etc.), you must replace the placeholder paths like:

image_folder = r"Path to your image folder"
ocr_model_path = r"Path to your trained OCR .pth file"
yolo_model_path = r"Path to your YOLOv8 model"
ground_truth_dir = r"Path to ground truth JSONs"
with actual valid paths on your local machine before running.

Use YOLOv8 to detect fields on forms.

Apply the OCR model to recognize handwritten content.

Save structured JSONs with bounding boxes and predicted text to outputs/.

--------------------------------------------------------------

📊 Evaluation
Run evaluation on the outputs:

python src/evaluation.py \
  --ground_truth_dir data/ground_truth_labels \
  --prediction_dir outputs/
This will print:

Field-level accuracy

Average CER and WER

Document-level accuracy

Final weighted score

Computational metrics (time and memory)

Output saved to evaluation_metrics.txt


--------------------------------------------------------------------------

📄 Output Format
The OCR output is saved per image in JSON format as:

json
[
  {
    "bounding_box": [x1, y1, x2, y2],
    "predicted_text": "Recognized Text"
  },
  ...
]
You can convert JSONs to CSV using the script:\

python src/json_to_csv.py --input_dir outputs/ --output_file final_output.csv

-----------------------------------------------------------------------------




#Inference script with YOLO+OCR
import os
import cv2
import json
import torch
import re
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from model import CNN_BiLSTM_CTC
from loader import char_to_idx, idx_to_char


image_dir = r"C:\Users\sripa\OneDrive\Desktop\DehadoComp\TESTDATASET\IMAGES"  #INPUT IMAGE DIRECTORY
ocr_model_path = r"C:\Users\sripa\OneDrive\Desktop\DehadoComp\DeHaDoAI-TCE_DeHaDo-AI_Challenge\submission\model\best_model.pth"        # OCR PATH
yolo_model_path = r"C:\Users\sripa\OneDrive\Desktop\DehadoComp\DeHaDoAI-TCE_DeHaDo-AI_Challenge\submission\src\best_yolov8s.pt"      #ADD YOLO FILE PATH
output_dir = r"C:\Users\sripa\OneDrive\Desktop\DehadoComp\TESTDATASET\IMAGES\outputs" 
os.makedirs(output_dir, exist_ok=True)

# ==== DEVICE ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== IMAGE PREPROCESSING ====
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def is_blank_region(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray.std() < 8  # Low variation = likely empty

def ctc_decode(output):
    output = output.argmax(2).permute(1, 0)  # (batch, seq)
    decoded_texts = []
    for seq in output:
        pred = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                pred.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded_texts.append(''.join(pred))
    return decoded_texts

import re

def fix_case(text):
    return ' '.join(w.capitalize() for w in text.split())

def fix_gender(text):
    t = text.lower()
    if "male" in t or "ma1e" in t:
        return "Male"
    elif "female" in t or "fema" in t:
        return "Female"
    return text

def fix_blood_group(text):
    bg = re.sub(r'[^AOB+-]', '', text.upper())
    return bg if re.match(r'^(A|B|AB|O)[+-]$', bg) else text

def fix_indian(text):
    if re.search(r'\b(indian|1ndian|indan|inadin)\b', text.lower()):
        return "Indian"
    return text

def clean_text(text):
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9.+\-/'():,& ]", '', text)
    text = re.sub(r"\s+", ' ', text)

    text = fix_gender(text)
    text = fix_blood_group(text)
    text = fix_indian(text)

    if len(text.split()) <= 4 and any(c.isalpha() for c in text):
        text = fix_case(text)

    return text.strip()


# ==== LOAD MODELS ====
print("🔍 Loading YOLOv8 detector...")
detector = YOLO(yolo_model_path)

print("🔤 Loading OCR model...")
ocr_model = CNN_BiLSTM_CTC(num_classes=len(char_to_idx)).to(device)
ocr_model.load_state_dict(torch.load(ocr_model_path, map_location=device))
ocr_model.eval()

# ==== RUN DETECTION + RECOGNITION ====
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for img_file in image_files:
    image_path = os.path.join(image_dir, img_file)
    print(f"\n📷 Processing: {img_file}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Image not found: {image_path}")
        continue

    results = detector(image_path)[0]
    predicted_fields = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0 or is_blank_region(cropped):
            continue  # Skip blank or invalid crops

        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(cropped)
        input_tensor = transform(pil_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = ocr_model(input_tensor)
            decoded_raw = ctc_decode(output)[0]
            decoded = clean_text(decoded_raw)


        predicted_fields.append({
            "bounding_box": [x1, y1, x2, y2],
            "predicted_text": decoded
        })

    # Save output
    out_path = os.path.join(output_dir, img_file.replace(".jpg", "_output.json"))
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(predicted_fields, f, indent=4)

    print(f"✅ Saved: {out_path}")

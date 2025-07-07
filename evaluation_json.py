#Evaluation Script
import os
import time
import psutil
import re
import json
import editdistance
# === CONFIGURATION ===
ground_truth_dir = r"ADD GROUND TRUTH JSON LABELS FILE PATH" #GROUND TRUTH JSON LABELS
prediction_dir = r"ADD PREDICTED JSON LABELS FILE PATH" #PREDICTED JSON LABELS

# === IoU Helper ===
def box_iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    union_area = (xa2 - xa1 + 1) * (ya2 - ya1 + 1) + (xb2 - xb1 + 1) * (yb2 - yb1 + 1) - inter_area
    return inter_area / union_area if union_area != 0 else 0

def cer(pred, gt):
    return editdistance.eval(pred, gt) / max(1, len(gt))

def wer(pred, gt):
    return editdistance.eval(pred.split(), gt.split()) / max(1, len(gt.split()))

# === Evaluation Accumulators ===
total_matches = 0
total_correct = 0
total_cer = 0
total_wer = 0
total_gt_fields = 0
doc_correct = 0
total_docs = 0

start_time = time.time()

# === Loop through prediction files ===
for file in os.listdir(prediction_dir):
    if not file.endswith("_output.json"):
        continue

    base_name = file.replace("_output.json", "")
    gt_path = os.path.join(ground_truth_dir, base_name + ".json")
    pred_path = os.path.join(prediction_dir, file)

    if not os.path.exists(gt_path):
        continue

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_fields = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_fields = json.load(f)

    total_gt_fields += len(gt_fields)
    total_docs += 1
    matches = []
    correct_fields_in_doc = 0

    # === Match predictions to GT ===
    for gt in gt_fields:
        best_match = None
        best_iou = 0
        for pred in pred_fields:
            iou = box_iou(gt["Coordinate"], pred["bounding_box"])
            if iou > best_iou and iou > 0.5:
                best_iou = iou
                best_match = pred
        if best_match:
            matches.append((gt, best_match))

    total_matches += len(matches)

    # === Evaluate matched pairs ===
    for gt, pred in matches:
        gt_val = re.sub(r'\s+', ' ', gt["Field value"].lower().strip())
        pred_val = re.sub(r'\s+', ' ', pred["predicted_text"].lower().strip())

        total_cer += cer(pred_val, gt_val)
        total_wer += wer(pred_val, gt_val)

        if gt_val == pred_val:
            total_correct += 1
            correct_fields_in_doc += 1

    if correct_fields_in_doc == len(gt_fields):
        doc_correct += 1

# === Compute Metrics ===
elapsed_time = time.time() - start_time
avg_cer = total_cer / total_matches if total_matches else 1.0
avg_wer = total_wer / total_matches if total_matches else 1.0
field_accuracy = total_correct / total_matches if total_matches else 0.0
recall = total_matches / total_gt_fields if total_gt_fields else 0.0
doc_accuracy = doc_correct / total_docs if total_docs else 0.0
final_score = 0.15 * field_accuracy + 0.35 * (1 - avg_cer) + 0.35 * (1 - avg_wer) + 0.15 * doc_accuracy
avg_time_per_doc = elapsed_time / total_docs if total_docs else 0.0
mem_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

# === Print Report ===
# === Prepare Report String ===
report = f"""
📊 Evaluation Summary
🧾 Files evaluated: {total_docs}
✅ Matches found: {total_matches}
📝 Field-level Accuracy: {total_correct}/{total_matches} = {field_accuracy:.2f}
🔡 Average CER: {avg_cer:.4f}
🔤 Average WER: {avg_wer:.4f}
📦 Total GT fields: {total_gt_fields}
📉 Recall: {total_matches}/{total_gt_fields} = {recall:.2f}
📑 Document-Level Accuracy: {doc_correct}/{total_docs} = {doc_accuracy:.2f}
⚖️ Final Score (weighted): {final_score:.4f}
⏱️ Avg Time per Doc: {avg_time_per_doc:.2f}s
💾 Approx. RAM Usage: {mem_usage_mb:.2f} MB
"""

print(report)

# === Save to TXT File ===
output_path = os.path.join(prediction_dir, "evaluation_metrics.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(report.strip())

print(f"📄 Evaluation report saved to: {output_path}")


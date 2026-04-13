import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from detection import detect_targets
from data_loader.loader import load_images
from metrics.metrics import compute_metrics
# PATHS
folder = r"D:/IRSTD-1k/IRSTD-1k/IRSTD1k_Img"
label_folder = r"D:/IRSTD-1k/IRSTD-1k/IRSTD1k_Label"
output_folder = "results/outputs"

# CREATE OUTPUT DIR (IMPORTANT FIX)
os.makedirs(output_folder, exist_ok=True)

# LOAD IMAGES
files = load_images(folder)
print("Total images:", len(files))

total_PD = []
total_FA = []

for image_name, img in files:

    try:
        print(f"\nProcessing: {image_name}")

        # DETECTION
        final_mask, mask, peaks, rg_mask = detect_targets(img)

        # LOAD GT
        label_path = os.path.join(label_folder, image_name)
        gt = cv2.imread(label_path, 0)

        if gt is None:
            print(f"[WARNING] GT not found for {image_name}")
            continue

        # BINARIZE GT
        _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

        # METRICS
        PD, FA = compute_metrics(final_mask, gt)
        total_PD.append(PD)
        total_FA.append(FA)

        print(f"PD: {PD:.4f} | FA: {FA:.6f}")

        # DRAW BOUNDING BOXES
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if 5 < area < 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 1)

        # SAVE OUTPUT
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, output)

    except Exception as e:
        print(f"[ERROR] {image_name}: {e}")
        continue
    cv2.imshow("Detected", output)
    cv2.imshow("Final Mask", final_mask)

# speed control (IMPORTANT)
    key = cv2.waitKey(1)

# press 'q' to stop
    if key == ord('q'):
        break
# FINAL AVERAGE METRICS
if total_PD and total_FA:
    print("\n===== FINAL RESULTS =====")
    print(f"Average PD: {sum(total_PD)/len(total_PD):.4f}")
    print(f"Average FA: {sum(total_FA)/len(total_FA):.6f}")

# CLEANUP
cv2.destroyAllWindows()
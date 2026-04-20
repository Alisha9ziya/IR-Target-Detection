import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from detection import detect_targets
from data_loader.loader import load_images
from metrics.metrics import compute_metrics

# PATHS
folder = "D:/IR_Folder/sample_img"
label_folder = r"D:/IRSTD-1k/IRSTD-1k/IRSTD1k_Label"
output_folder = "results/outputs"
results_folder = "results"

# CREATE OUTPUT DIRS
os.makedirs(output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# LOAD IMAGES
files = load_images(folder)
print("Total images:", len(files))

total_PD = []
total_FA = []
results_log = []

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
        results_log.append((image_name, PD, FA))

        print(f"PD: {PD:.4f} | FA: {FA:.6f}")

        # DRAW BOUNDING BOXES
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 1)

        # SAVE OUTPUT IMAGE
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, output)

        # OPTIONAL DISPLAY (safe)
        if os.environ.get("DISPLAY"):
            cv2.imshow("Detected", output)
            cv2.imshow("Final Mask", final_mask)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {image_name}: {e}")
        continue


# ===== FINAL RESULTS =====
if total_PD and total_FA:
    avg_PD = sum(total_PD) / len(total_PD)
    avg_FA = sum(total_FA) / len(total_FA)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print(f"Average PD: {avg_PD:.4f}")
    print(f"Average FA: {avg_FA:.6f}")
    print("="*50)

    # SAVE FINAL METRICS
    with open(os.path.join(results_folder, "final_metrics.txt"), "w") as f:
        f.write("FINAL RESULTS\n")
        f.write(f"Average PD: {avg_PD:.4f}\n")
        f.write(f"Average FA: {avg_FA:.6f}\n")

    # SAVE PER-IMAGE METRICS (TXT)
    with open(os.path.join(results_folder, "per_image_metrics.txt"), "w") as f:
        for name, pd, fa in results_log:
            f.write(f"{name} -> PD: {pd:.4f}, FA: {fa:.6f}\n")

    # SAVE CSV (WITHOUT pandas)
    with open(os.path.join(results_folder, "metrics.csv"), "w") as f:
        f.write("Image,PD,FA\n")
        for name, pd, fa in results_log:
            f.write(f"{name},{pd:.4f},{fa:.6f}\n")


# CLEANUP
cv2.destroyAllWindows()
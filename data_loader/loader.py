import os
import cv2

def load_images(folder):
    files = []

    if not os.path.exists(folder):
        raise ValueError(f"Folder not found: {folder}")

    # SORTED for consistency
    for f in sorted(os.listdir(folder)):

        # skip hidden/system files
        if f.startswith('.'):
            continue

        if f.lower().endswith(('.png', '.jpg', '.bmp')):

            path = os.path.join(folder, f)
            img = cv2.imread(path)

            if img is None:
                print(f"[WARNING] Could not read: {f}")
                continue

            files.append((f, img))

    print(f"[INFO] Loaded {len(files)} valid images from {folder}")
    return files
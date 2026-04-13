import os
import cv2
import random

input_folder = "D:/IR_Folder/results/outputs"
output_folder = "D:/IR_Folder/results/sample_outputs"

os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.endswith(('.png','.jpg'))]

selected = []

for img_name in images:
    path = os.path.join(input_folder, img_name)
    img = cv2.imread(path)

    if img is None:
        continue

    # detect green pixels (bbox)
    green_pixels = ((img[:,:,1] > 200) & (img[:,:,0] < 50) & (img[:,:,2] < 50)).sum()

    if green_pixels > 20:
        selected.append(img_name)

# fallback random
if len(selected) < 10:
    remaining = list(set(images) - set(selected))
    selected += random.sample(remaining, min(10-len(selected), len(remaining)))

selected = selected[:10]

# save selected
for img_name in selected:
    src = os.path.join(input_folder, img_name)
    dst = os.path.join(output_folder, img_name)
    img = cv2.imread(src)
    cv2.imwrite(dst, img)

print("Selected images:", selected)
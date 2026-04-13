import cv2
import numpy as np


# DENSITY 
def compute_density(gray):
    kernel = np.ones((5, 5), np.float32)
    density = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return density


# PEAK DETECTION
def find_density_peaks(density):
    norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # adaptive-ish threshold (better than fixed 230)
    thresh_val = np.percentile(norm, 99)  # top 1% values
    _, peaks = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)

    return peaks


# REGION GROWING 
def region_growing(gray, peaks):
    mask = np.zeros_like(gray, dtype=np.uint8)

    ys, xs = np.where(peaks == 255)

    for (y, x) in zip(ys[:50], xs[:50]):

        seed_val = int(gray[y, x])
        threshold = max(seed_val - 15, 0)

        stack = [(y, x)]
        visited = set()

        while stack:
            cy, cx = stack.pop()

            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))

            if len(visited) > 150:   # slightly larger region
                break

            if cy < 0 or cx < 0 or cy >= gray.shape[0] or cx >= gray.shape[1]:
                continue

            if gray[cy, cx] >= threshold:
                mask[cy, cx] = 255

                stack.extend([
                    (cy+1, cx), (cy-1, cx),
                    (cy, cx+1), (cy, cx-1)
                ])

    return mask


#  MAIN DETECTION PIPELINE 
def detect_targets(img):

    # STEP 1: grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 2: TopHat (IMPORTANT ADD)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm, 100, 255, cv2.THRESH_BINARY)

    # STEP 3: density + peaks
    density = compute_density(gray)
    peaks = find_density_peaks(density)

    # STEP 4: region growing
    rg_mask = region_growing(gray, peaks)

    # STEP 5: combine
    final_mask = cv2.bitwise_or(mask, rg_mask)

    return final_mask, mask, peaks, rg_mask
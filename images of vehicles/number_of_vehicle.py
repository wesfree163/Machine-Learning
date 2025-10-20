import os
import cv2
from time import sleep as halt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# =============================
# PATHS (adjust if needed)
# =============================
cars_only = r"C:\Users\Wes\Desktop\Vehicle Type\All Vehicles"   # cropped / close-up robots
image_dir   = r"C:\Users\Wes\Desktop\Vehicle Type\Sample Set"      # room screenshots

# =============================
# CONFIGURATION
# =============================
patch_size = 64        # patch width & height (64–128 works well)
neg_per_image = 41     # how many background patches per room
confidence_threshold = 0.97  # SVM decision margin for detection
merge_distance = 51    # pixels to merge nearby detections
visual_delay_ms = 300  # delay between images when showing results

# =============================
# BUILD TRAINING DATA
# =============================
X, y = [], []

# --- positive samples: robot crops ---
for f in os.listdir(cars_only):
    if f.lower().endswith(('.jpg', '.png')):
        im = cv2.imread(os.path.join(cars_only, f))
        if im is None:
            continue
        im = cv2.resize(im, (patch_size, patch_size))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        X.append(im_gray.flatten())
        y.append(1)

# --- negative samples: random background patches from room images ---
for f in os.listdir(image_dir):
    if f.lower().endswith(('.jpg', '.png')):
        im = cv2.imread(os.path.join(image_dir, f))
        if im is None:
            continue
        h, w = im.shape[:2]
        for _ in range(neg_per_image):
            x = np.random.randint(0, w - patch_size)
            y0 = np.random.randint(0, h - patch_size)
            patch = im[y0:y0 + patch_size, x:x + patch_size]
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            X.append(patch_gray.flatten())
            y.append(0)

X = np.array(X)
y = np.array(y)
print(f"Total samples: {len(X)} (robots: {sum(y)}, background: {len(y)-sum(y)})")

# =============================
# TRAIN LINEAR SVM
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LinearSVC(max_iter=3000)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Training accuracy: {acc:.3f}")

# =============================
# SCAN ROOM IMAGES
# =============================
for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(image_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    step = patch_size // 2  # slide by half-patch for coverage

    # collect all boxes + confidence scores
    boxes, scores = [], []

    for y0 in range(0, h - patch_size, step):
        for x0 in range(0, w - patch_size, step):
            patch = gray[y0:y0 + patch_size, x0:x0 + patch_size]
            score = clf.decision_function([patch.flatten()])[0]
            if score > confidence_threshold:
                boxes.append([x0, y0, x0 + patch_size, y0 + patch_size])
                scores.append(score)

    # =============================
    # MERGE OVERLAPPING / NEARBY BOXES
    # =============================
    boxes = np.array(boxes)
    if len(boxes) == 0:
        print(f"{fname}: 0 car(s)/truck(s) in jpg/png")
        continue

    used = np.zeros(len(boxes), dtype=bool)
    merged = []
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = b
        group = [b]
        for j, b2 in enumerate(boxes):
            if i == j or used[j]:
                continue
            gx1, gy1, gx2, gy2 = b2
            # distance between box centers
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            cx2, cy2 = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < merge_distance:
                group.append(b2)
                used[j] = True
        gx1 = min(g[0] for g in group)
        gy1 = min(g[1] for g in group)
        gx2 = max(g[2] for g in group)
        gy2 = max(g[3] for g in group)
        merged.append([gx1, gy1, gx2, gy2])

    count = len(merged)
    print(f"{fname}: {count} car(s)/truck(s) in jpg/png")

    # =============================
    # VISUALIZE
    # =============================
    for (x1, y1, x2, y2) in merged:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{count} robots",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Robots", img)
    halt(15)
    key = cv2.waitKey(visual_delay_ms) & 0xFF
    if key == 27:
        break

while True:
    pass
cv2.destroyAllWindows()

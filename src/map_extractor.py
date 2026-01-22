import cv2
import numpy as np

# CONFIGURATION
# Try with both images to verify
IMAGE_PATH = "work\\bakersfield-californian-jun-22-1964-p-30_clean.png" 
OUTPUT_PATH = "outputs\\maps\\full_map_with_title.png"

# 1. Load Image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# 2. Threshold (Invert)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# ==========================================================
# KEY FIX: FUSE TEXT CHARACTERS FIRST
# ==========================================================
# Before filtering, we fuse nearby letters.
# The title "ZONING MAP" will become one long "bar" of pixels.
# Regular text words will become small "bars".
fuse_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) 
fused = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, fuse_kernel)

# 3. COMPONENT FILTERING
# Now we look at the fused blobs.
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused, connectivity=8)

# Mask for keeping the "Good Stuff" (Map lines + Big Titles)
map_layer_mask = np.zeros_like(binary)

print(f"Filtering {num_labels} elements...")

for i in range(1, num_labels): # Skip background
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    # --- SMART FILTER ---
    
    # 1. Keep Structure: Any long line (grid lines)
    is_line = (w > 100) or (h > 100)
    
    # 2. Keep Titles: Any blob that is wide (fused words) AND tall enough 
    # (Titles are usually taller than 20px, body text is usually ~10-15px)
    is_title = (w > 50) and (h > 25) 
    
    # 3. Keep Graphics: Any large solid block
    is_graphic = (area > 500)

    if is_line or is_title or is_graphic:
        map_layer_mask[labels == i] = 255

# Debug: See what survived the filter
cv2.imwrite("outputs/debug_filtered_layer.png", map_layer_mask)

# 4. VERTICAL CONNECTION
# Now we have the Grid and the Title floating separately.
# We dilate VERTICALLY to bridge the gap between "Title" and "Map".
# (30px vertical reach should be enough to grab the title above)
connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 40))
connected = cv2.dilate(map_layer_mask, connect_kernel, iterations=1)

# 5. FIND THE BEST REGION
contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_box = None
max_area = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    
    # Filter: Must be significant size
    if area < (w_img * h_img * 0.05):
        continue

    # Filter: Reject Skyscraper Text Columns
    # (If it's 3x taller than it is wide, it's likely a text column we accidentally kept)
    if h > (w * 3):
        continue
        
    if area > max_area:
        max_area = area
        best_box = (x, y, w, h)

# 6. CROP & SAVE
if best_box:
    x, y, w, h = best_box
    
    # Add Padding (Important so we don't cut the edges of the border)
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + (pad*2))
    h = min(h_img - y, h + (pad*2))
    
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(OUTPUT_PATH, crop)
    print(f"SUCCESS: Captured Map + Title. Saved to {OUTPUT_PATH}")
    
    # Visual check
    vis = img.copy()
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 5)
    cv2.imwrite("outputs/debug_final_selection.png", vis)
else:
    print("No map candidate found.")

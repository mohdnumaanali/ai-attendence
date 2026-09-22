import cv2
from ultralytics import YOLO

# 1. Load model trained on your real dataset
model = YOLO('runs/classify/train-3/weights/best.pt')

# 2. Bin Destination Mapping
BIN_MAPPING = {
    "gauze": "Yellow Bin (Infectious Cotton Waste)",
    "glove_pair_latex": "Red Bin (Contaminated Gloves)",
    "glove_pair_nitrile": "Red Bin (Contaminated Gloves)",
    "glove_pair_surgery": "Red Bin (Contaminated Gloves)",
    "glove_single_latex": "Red Bin (Contaminated Gloves)",
    "glove_single_nitrile": "Red Bin (Contaminated Gloves)",
    "glove_single_surgery": "Red Bin (Contaminated Gloves)",
    "medical_cap": "Yellow Bin (Infectious Fabric)",
    "medical_filter": "Yellow Bin (Infectious Filter)",
    "medical_glasses": "Blue Bin (Recyclable Plastic/Goggles)",
    "shoe_cover_pair": "Yellow Bin (Infectious Covers)",
    "shoe_cover_single": "Yellow Bin (Infectious Covers)",
    "test_tube": "Blue Bin (Glassware / Lab Tube)",
    "urine_bag": "Red Bin (Contaminated Fluid Bag)"
}

# Require high confidence (70%) to avoid background/shirt triggers
CONFIDENCE_THRESHOLD = 0.70

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Webcam feed running... Place medical waste inside the box guide.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define bounding box region in the center
    box_w, box_h = 280, 280
    x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
    x2, y2 = x1 + box_w, y1 + box_h

    # Extract ROI inside the box
    roi = frame[y1:y2, x1:x2]

    # Run inference on ROI
    results = model(roi, verbose=False)[0]
    
    top1_index = results.probs.top1
    class_name = results.names[top1_index]
    confidence = results.probs.top1conf.item()

    # Check if confidence meets threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        destination = BIN_MAPPING.get(class_name, "General Bin")
        status_text = f"Detected: {class_name} ({confidence * 100:.1f}%)"
        bin_text = f"Bin: {destination}"
        box_color = (0, 255, 0)  # Green box
    else:
        status_text = "No Valid Medical Waste Detected"
        bin_text = "Bin: None"
        box_color = (0, 0, 255)  # Red box

    # Draw target box and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
    cv2.putText(frame, "HOLD WASTE HERE", (x1 + 35, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Render separate lines for status and bin destination (no overlapping text)
    cv2.putText(frame, status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    cv2.putText(frame, bin_text, (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

    cv2.imshow("Smart Medical Waste Hardware Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
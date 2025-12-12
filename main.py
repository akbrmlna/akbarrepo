import cv2
import numpy as np

# -----------------------------
# Fungsi filter
# -----------------------------
def cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=250, sigmaSpace=250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def sketch_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
    edges_inv = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

def grayscale_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# -----------------------------
# Setup webcam
# -----------------------------
cap = cv2.VideoCapture(0)
filter_index = 0
filters = [cartoon_filter, sketch_filter, grayscale_filter]
filter_names = ["Cartoon", "Sketch", "Grayscale"]

print("Tekan 'n' untuk ganti filter, 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    filtered_frame = filters[filter_index](frame)

    # Tampilkan nama filter
    cv2.putText(filtered_frame, filter_names[filter_index], (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Instagram-style Filters", filtered_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        filter_index = (filter_index + 1) % len(filters)

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
from pathlib import Path

def process_otsu_clean(image_path, output_dir, C0=0.6, alpha_A0=1.2):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if np.mean(binary_otsu) > 127:
        binary_otsu = cv2.bitwise_not(binary_otsu)
    contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    initial_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 3]
    
    if not initial_areas:
        cv2.imwrite(str(output_dir / image_path.name), np.zeros_like(binary_otsu))
        return
    A0 = np.mean(initial_areas) * alpha_A0
    clean_binary = binary_otsu.copy()

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        if area <= 3:  # Hapus noise mikro
            cv2.drawContours(clean_binary, [c], -1, 0, -1)
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        is_adhesive = (area >= A0) and (circularity < C0)
        
        if is_adhesive:
            cv2.drawContours(clean_binary, [c], -1, 0, -1)
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), clean_binary)

def run_batch_processing():
    current_dir = Path(__file__).resolve().parent  
    base_dir = current_dir.parent                 
    input_bilateral = base_dir / "preprocessing" / "hasil_bilateral"
    input_clahe = base_dir / "preprocessing" / "hasil_clahe"
    output_bilateral = current_dir / "hasil_otsu_bilateral"
    output_clahe = current_dir / "hasil_otsu_clahe"
    output_bilateral.mkdir(parents=True, exist_ok=True)
    output_clahe.mkdir(parents=True, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    print("Memproses Otsu Tanpa Gumpalan untuk HASIL BILATERAL...")
    files_bilateral = [f for f in input_bilateral.rglob('*') if f.suffix.lower() in valid_extensions]

    for file_path in files_bilateral:
        process_otsu_clean(file_path, output_bilateral)
    print(f"Selesai! {len(files_bilateral)} citra disimpan di: {output_bilateral}\n")

    print("Memproses Otsu Tanpa Gumpalan untuk HASIL CLAHE...")
    files_clahe = [f for f in input_clahe.rglob('*') if f.suffix.lower() in valid_extensions]

    for file_path in files_clahe:
        process_otsu_clean(file_path, output_clahe)
    print(f"Selesai! {len(files_clahe)} citra disimpan di: {output_clahe}\n")
    print("PROSES SELESAI. DATASET CITRA SIAP LABELING BERHASIL DIBUAT")

if __name__ == "__main__":
    run_batch_processing()
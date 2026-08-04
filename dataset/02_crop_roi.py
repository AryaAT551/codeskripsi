import os
import cv2
from pathlib import Path

def process_roi_for_dataset():
    current_dir = Path(__file__).resolve().parent
    input_folder = current_dir / "frames"
    output_folder = current_dir / "roi_frames"
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Memeriksa seluruh isi folder di: {input_folder}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif'}
    image_paths = [
        f for f in input_folder.rglob('*') 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        print(f"\nTIDAK ADA GAMBAR DITEMUKAN DI KELOMPOK FOLDER: {input_folder}")
        return
    print(f"Ditemukan {len(image_paths)} gambar. Memulai Crop ROI\n")

    CROP_RATIO = 0.20  
    success_count = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[-] Gagal membaca file: {img_path.name}")
            continue
        h, w = img.shape[:2]
        crop_w = int(w * CROP_RATIO)
        roi_center = img[0:h, crop_w : w - crop_w]
        relative_path = img_path.relative_to(input_folder)
        save_path = output_folder / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), roi_center)
        print(f"Saved to roi_frames: {relative_path} ({w}x{h} -> {roi_center.shape[1]}x{roi_center.shape[0]})")
        success_count += 1

    print(f"\nSelesai! {success_count} gambar berhasil diproses & disimpan di: {output_folder}")

if __name__ == "__main__":
    process_roi_for_dataset()
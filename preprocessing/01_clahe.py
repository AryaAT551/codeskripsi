import cv2
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def apply_clahe_to_roi_frame(
    input_dir: Path, 
    output_dir: Path, 
    clip_limit: float = 2.0, 
    tile_grid_size: tuple = (8, 8)
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Folder tidak ditemukan: {input_path.resolve()}")
        return
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"Tidak ada gambar yang ditemukan di: {input_path.resolve()}")
        return
    print(f"Menemukan {len(image_files)} gambar di '{input_path.resolve()}' Memulai CLAHE...")

    for img_file in sorted(image_files):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Gagal membaca gambar: {img_file.name}")
            continue
        clahe_img = clahe.apply(img)
        relative_path = img_file.relative_to(input_path)
        save_file_path = output_path / relative_path
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_file_path), clahe_img)
    print(f"Selesai! gambar berhasil diproses dan disimpan di: {output_path.resolve()}")

def generate_intensity_mask(clahe_img: np.ndarray) -> np.ndarray:
    normalized_img = clahe_img.astype(np.float32) / 255.0
    intensity_mask = 1.0 - normalized_img
    return intensity_mask

if __name__ == "__main__":
    roi_folder = BASE_DIR / "dataset" / "roi_frames"
    output_folder = BASE_DIR / "preprocessing" / "hasil_clahe"
    apply_clahe_to_roi_frame(
        input_dir=roi_folder,
        output_dir=output_folder,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    )
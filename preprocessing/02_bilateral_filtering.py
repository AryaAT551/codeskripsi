import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def improved_bilateral_filter(img: np.ndarray, d: int = 5, sigma_s: float = 3.0, sigma_r: float = 30.0) -> np.ndarray:
    img_float = img.astype(np.float32)
    h, w = img_float.shape
    pad = d // 2
    padded_img = np.pad(img_float, pad, mode='reflect')
    output_img = np.zeros_like(img_float)
    y_grid, x_grid = np.mgrid[-pad:pad+1, -pad:pad+1]
    omega_s = np.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma_s ** 2)))
    total_window_pixels = d * d
    threshold_similar = sigma_r / 3.0
    
    for y in range(h):
        for x in range(w):
            window = padded_img[y:y+d, x:x+d]
            center_val = window[pad, pad]
            diff = np.abs(window - center_val)
            is_similar = diff < threshold_similar
            non_similar_count = np.sum(~is_similar)
            tau = 0.0

            if non_similar_count >= (total_window_pixels / 3.0):
                valid_pixels = window[is_similar]

                if len(valid_pixels) > 0:
                    mean_val = np.mean(valid_pixels)
                    max_val = np.max(valid_pixels)
                    min_val = np.min(valid_pixels)
                    phi = center_val - mean_val

                    if phi > 0:
                        tau = max_val - center_val
                    elif phi < 0:
                        tau = min_val - center_val
                    else:
                        tau = 0.0
            
            omega_r = np.exp(-((window - center_val - tau) ** 2) / (2 * (sigma_r ** 2)))
            weights = omega_s * omega_r
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                output_img[y, x] = np.sum(weights * window) / weight_sum
            else:
                output_img[y, x] = center_val
    return np.clip(output_img, 0, 255).astype(np.uint8)


def process_bilateral_pipeline(input_dir: Path, output_dir: Path):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Folder sumber tidak ditemukan: {input_path.resolve()}")
        return
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"Tidak ada gambar ditemukan di: {input_path.resolve()}")
        return
    print(f"Menemukan {len(image_files)} gambar di '{input_path.resolve()}'. Memulai Bilateral Filter...")

    for img_file in sorted(image_files):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Gagal membaca gambar: {img_file.name}")
            continue
        filtered_img = improved_bilateral_filter(img, d=5, sigma_s=3.0, sigma_r=30.0)
        relative_path = img_file.relative_to(input_path)
        save_file_path = output_path / relative_path
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_file_path), filtered_img)
    print(f"Selesai! Hasil Bilateral Filter disimpan di: {output_path.resolve()}")

if __name__ == "__main__":
    roi_folder = BASE_DIR / "dataset" / "roi_frames"
    output_folder = BASE_DIR / "preprocessing" / "hasil_bilateral"
    process_bilateral_pipeline(roi_folder, output_folder)
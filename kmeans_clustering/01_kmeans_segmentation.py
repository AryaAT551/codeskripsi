import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 1. Path Konfigurasi Utama
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_INPUT_CLAHE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "preprocessing", "hasil_clahe"))
DIR_INPUT_BILATERAL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "preprocessing", "hasil_bilateral"))

DIR_OUTPUT_CLAHE = os.path.join(SCRIPT_DIR, "hasil_kmeans_clahe")
DIR_OUTPUT_BILATERAL = os.path.join(SCRIPT_DIR, "hasil_kmeans_bilateral")

# Sub-folder terpisah untuk Masker dan Segmentasi
DIR_CLAHE_MASK = os.path.join(DIR_OUTPUT_CLAHE, "mask")
DIR_CLAHE_SEGMENTED = os.path.join(DIR_OUTPUT_CLAHE, "segmented")

DIR_BILATERAL_MASK = os.path.join(DIR_OUTPUT_BILATERAL, "mask")
DIR_BILATERAL_SEGMENTED = os.path.join(DIR_OUTPUT_BILATERAL, "segmented")

# Buat folder & sub-folder otomatis
for folder in [DIR_CLAHE_MASK, DIR_CLAHE_SEGMENTED, DIR_BILATERAL_MASK, DIR_BILATERAL_SEGMENTED]:
    os.makedirs(folder, exist_ok=True)


def apply_kmeans_paper(image, k_values=[6, 7, 8]):
    """
    K-Means Clustering pixel-based.
    Mengembalikan 2 dictionary terpisah: segmented dan mask.
    """
    h, w = image.shape[:2]
    pixel_values = image.reshape((-1, 1)).astype(np.float32)
    
    results_segmented = {}
    results_mask = {}
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixel_values)
        centers = kmeans.cluster_centers_
        
        # Urutkan dari gelap ke terang
        sorted_indices = np.argsort(centers.flatten())
        
        # Citra terklasterisasi (Segmentasi)
        clustered_pixels = centers[labels].reshape((h, w)).astype(np.uint8)
        
        # Binary mask pada cluster berdensitas paling terang
        brightest_cluster_idx = sorted_indices[-1]
        binary_mask = (labels == brightest_cluster_idx).reshape((h, w)).astype(np.uint8) * 255
        
        results_segmented[f"K{k}_segmented"] = clustered_pixels
        results_mask[f"K{k}_mask"] = binary_mask
        
    return results_segmented, results_mask


def process_images_recursive(input_dir, mask_out_dir, seg_out_dir, label):
    print(f"Memproses K-Means untuk: {label}")
    print(f"Mencari gambar di dalam : {input_dir}")
    print(f"Menyimpan MASK ke       : {mask_out_dir}")
    print(f"Menyimpan SEGMENTED ke  : {seg_out_dir}")
    
    if not os.path.exists(input_dir):
        print(f"Folder input tidak ditemukan: {input_dir}\n")
        return

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"Tidak ada berkas gambar ditemukan di '{input_dir}' maupun sub-foldernya!\n")
        return

    print(f"Ditemukan {len(image_paths)} gambar. Memulai proses K-Means...\n")
    success_count = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"File corrupt/salah format: {filename}")
            continue
            
        base_name = os.path.splitext(filename)[0]
        results_seg, results_mask = apply_kmeans_paper(image, k_values=[6, 7, 8])
        
        # 1. Simpan file MASK ke folder mask
        for key, res_img in results_mask.items():
            save_path = os.path.join(mask_out_dir, f"{base_name}_{key}.png")
            cv2.imwrite(save_path, res_img)

        # 2. Simpan file SEGMENTED ke folder segmented
        for key, res_img in results_seg.items():
            save_path = os.path.join(seg_out_dir, f"{base_name}_{key}.png")
            cv2.imwrite(save_path, res_img)

        success_count += 1
        print(f"BERHASIL {filename} -> Dipisah ke sub-folder mask & segmented")

    print(f"\nSelesai! Total {success_count} gambar berhasil diproses untuk '{label}'.\n")


if __name__ == "__main__":
    # 1. Proses gambar CLAHE
    process_images_recursive(DIR_INPUT_CLAHE, DIR_CLAHE_MASK, DIR_CLAHE_SEGMENTED, "Hasil CLAHE")
    
    print("-" * 60 + "\n")
    
    # 2. Proses gambar Bilateral Filtering
    process_images_recursive(DIR_INPUT_BILATERAL, DIR_BILATERAL_MASK, DIR_BILATERAL_SEGMENTED, "Hasil Bilateral Filtering")
    
    print("Seluruh proses K-Means Clustering Selesai!")
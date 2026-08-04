import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_mse(image_ref, image_proc):
    ref = image_ref.astype(np.float64)
    proc = image_proc.astype(np.float64)
    err = np.sum((ref - proc) ** 2)
    total_pixels = ref.size
    mse = err / float(total_pixels)
    return mse

def calculate_psnr(image_ref, image_proc, max_pixel=255.0):
    mse = calculate_mse(image_ref, image_proc)
    if mse == 0:
        return float('inf')
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return psnr

def get_sorted_images(folder_path):
    if not os.path.exists(folder_path):
        return []
    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.PNG')
    files = []
    
    for root, _, _ in os.walk(folder_path):
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(root, ext)))
    files.sort()
    return files

def evaluate_psnr(ref_dir, proc_dir):
    ref_files = get_sorted_images(ref_dir)
    proc_files = get_sorted_images(proc_dir)

    if not ref_files or not proc_files:
        return {}
    ref_dict = {os.path.basename(f): f for f in ref_files}
    results = {}

    for proc_path in proc_files:
        fname = os.path.basename(proc_path)
        
        if fname in ref_dict:
            ref_path = ref_dict[fname]
            img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            img_proc = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)

            if img_ref is not None and img_proc is not None:
                if img_ref.shape != img_proc.shape:
                    img_proc = cv2.resize(img_proc, (img_ref.shape[1], img_ref.shape[0]))
                psnr_val = calculate_psnr(img_ref, img_proc)
                results[fname] = psnr_val

    if not results:
        min_len = min(len(ref_files), len(proc_files))
        for i in range(min_len):
            ref_path = ref_files[i]
            proc_path = proc_files[i]
            fname = os.path.basename(proc_path)
            img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            img_proc = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)

            if img_ref is not None and img_proc is not None:
                if img_ref.shape != img_proc.shape:
                    img_proc = cv2.resize(img_proc, (img_ref.shape[1], img_ref.shape[0]))
                psnr_val = calculate_psnr(img_ref, img_proc)
                results[fname] = psnr_val
    return results

def save_txt_report(output_dir, method_name, results):
    os.makedirs(output_dir, exist_ok=True)
    avg_psnr = np.mean(list(results.values())) if results else 0
    txt_path = os.path.join(output_dir, f"hasil_psnr_{method_name.lower()}.txt")
    
    with open(txt_path, 'w') as f:
        f.write(f"  HASIL EVALUASI PSNR - {method_name.upper()}\n")
        f.write("==================================================\n\n")
        f.write(f"Total Frame Evaluasi : {len(results)}\n")
        f.write(f"Rata-rata PSNR       : {avg_psnr:.4f} dB\n\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{'No.':<5} | {'Nama File Frame':<25} | {'PSNR (dB)':<10}\n")
        f.write("--------------------------------------------------\n")
        for idx, (fname, val) in enumerate(results.items(), start=1):
            f.write(f"{idx:<5} | {fname:<25} | {val:.4f}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"RATA-RATA PSNR: {avg_psnr:.4f} dB\n")
        f.write("--------------------------------------------------\n")
    print(f"Laporan TXT {method_name} tersimpan di: {txt_path}")

def plot_combined_bar_chart(clahe_results, bilateral_results, output_file):
    common_files = sorted(list(set(clahe_results.keys()) & set(bilateral_results.keys())))
    
    if not common_files:
        print("Tidak ada nama frame yang sama untuk grafik perbandingan.")
        return
    clahe_vals = [clahe_results[f] for f in common_files]
    bilateral_vals = [bilateral_results[f] for f in common_files]
    x = np.arange(len(common_files))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, clahe_vals, width, label='CLAHE', color='#1f77b4')
    ax.bar(x + width/2, bilateral_vals, width, label='Bilateral Filter', color='#ff7f0e')
    avg_clahe = np.mean(clahe_vals)
    avg_bilateral = np.mean(bilateral_vals)
    ax.axhline(y=avg_clahe, color='#1f77b4', linestyle='--', alpha=0.8, 
               label=f'Rata-rata CLAHE ({avg_clahe:.2f} dB)')
    ax.axhline(y=avg_bilateral, color='#ff7f0e', linestyle='--', alpha=0.8, 
               label=f'Rata-rata Bilateral ({avg_bilateral:.2f} dB)')
    ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_title('Perbandingan Nilai PSNR Per Frame: CLAHE vs Bilateral Filter', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}" for i in range(len(common_files))], rotation=0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Grafik Batang Perbandingan tersimpan di: {output_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    ref_dir = os.path.join(project_root, "dataset", "roi_frames")
    clahe_dir = os.path.join(project_root, "preprocessing", "hasil_clahe")
    bilateral_dir = os.path.join(project_root, "preprocessing", "hasil_bilateral")
    out_psnr_root = os.path.join(current_dir, "output_psnr")
    out_psnr_clahe = os.path.join(out_psnr_root, "psnr_clahe")
    out_psnr_bilateral = os.path.join(out_psnr_root, "psnr_bilateral")

    print("\nMEMULAI EVALUASI PSNR")
    print(f"Folder Referensi (ROI) : {ref_dir}")
    print(f"Folder Hasil CLAHE      : {clahe_dir}")
    print(f"Folder Hasil Bilateral  : {bilateral_dir}")

    clahe_results = evaluate_psnr(ref_dir, clahe_dir)
    print(f"Frame CLAHE terproses     : {len(clahe_results)}")

    bilateral_results = evaluate_psnr(ref_dir, bilateral_dir)
    print(f"Frame Bilateral terproses : {len(bilateral_results)}")

    if clahe_results:
        save_txt_report(out_psnr_clahe, "CLAHE", clahe_results)
        
    if bilateral_results:
        save_txt_report(out_psnr_bilateral, "Bilateral", bilateral_results)

    if clahe_results and bilateral_results:
        chart_combined_path = os.path.join(out_psnr_root, "grafik_perbandingan_psnr_batang.png")
        plot_combined_bar_chart(clahe_results, bilateral_results, chart_combined_path)
    print("\nEvaluasi PSNR Selesai!")
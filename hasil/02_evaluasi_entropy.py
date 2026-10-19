import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_entropy(image):
    """
    Menghitung Entropi Citra sesuai rumus proposal:
    ENT(i) = - sum_{l=0}^{L-1} I(l) * log2(I(l))
    
    di mana:
    - L   : 256 (tingkat keabuan 0-255)
    - I(l): probabilitas kemunculan intensitas l (p)
    """
    if image is None:
        return 0.0

    # Menghitung histogram untuk L = 256 tingkat keabuan (0 hingga 255)
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    
    # I(l) = Probabilitas kemunculan piksel dengan intensitas l
    i_l = hist / float(total_pixels)
    
    # Hanya mengambil probabilitas > 0 untuk menghindari log2(0)
    i_l = i_l[i_l > 0]
    
    # Menghitung ENT(i) = - sum( I(l) * log2(I(l)) )
    entropy = -np.sum(i_l * np.log2(i_l))
    return float(entropy)


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


def evaluate_entropy(proc_dir):
    proc_files = get_sorted_images(proc_dir)

    if not proc_files:
        return {}
    results = {}

    for proc_path in proc_files:
        fname = os.path.basename(proc_path)
        img_proc = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)

        if img_proc is not None:
            entropy_val = calculate_entropy(img_proc)
            results[fname] = entropy_val
            
    return results


def save_txt_report(output_dir, method_name, results):
    os.makedirs(output_dir, exist_ok=True)
    avg_entropy = np.mean(list(results.values())) if results else 0
    txt_path = os.path.join(output_dir, f"hasil_entropy_{method_name.lower()}.txt")
    
    with open(txt_path, 'w') as f:
        f.write(f"   HASIL EVALUASI ENTROPY - {method_name.upper()}\n")
        f.write("==================================================\n\n")
        f.write(f"Total Frame Evaluasi : {len(results)}\n")
        f.write(f"Rata-rata Entropy    : {avg_entropy:.4f}\n\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{'No.':<5} | {'Nama File Frame':<25} | {'Entropy':<10}\n")
        f.write("--------------------------------------------------\n")
        for idx, (fname, val) in enumerate(results.items(), start=1):
            f.write(f"{idx:<5} | {fname:<25} | {val:.4f}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"RATA-RATA ENTROPY: {avg_entropy:.4f}\n")
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
               label=f'Rata-rata CLAHE ({avg_clahe:.2f})')
    ax.axhline(y=avg_bilateral, color='#ff7f0e', linestyle='--', alpha=0.8, 
               label=f'Rata-rata Bilateral ({avg_bilateral:.2f})')
    
    ax.set_ylabel('Entropy', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_title('Perbandingan Nilai Entropy Per Frame: CLAHE vs Bilateral Filter', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}" for i in range(len(common_files))], rotation=0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Grafik Perbandingan tersimpan di: {output_file}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    clahe_dir = os.path.join(project_root, "preprocessing", "hasil_clahe")
    bilateral_dir = os.path.join(project_root, "preprocessing", "hasil_bilateral")
    
    out_entropy_root = os.path.join(current_dir, "output_entropy")
    out_entropy_clahe = os.path.join(out_entropy_root, "entropy_clahe")
    out_entropy_bilateral = os.path.join(out_entropy_root, "entropy_bilateral")

    print("\nMEMULAI EVALUASI ENTROPY")
    print(f"Folder Hasil CLAHE      : {clahe_dir}")
    print(f"Folder Hasil Bilateral  : {bilateral_dir}")

    # Process CLAHE
    clahe_results = evaluate_entropy(clahe_dir)
    print(f"Frame CLAHE terproses     : {len(clahe_results)}")

    # Process Bilateral
    bilateral_results = evaluate_entropy(bilateral_dir)
    print(f"Frame Bilateral terproses : {len(bilateral_results)}")

    # Save Reports
    if clahe_results:
        save_txt_report(out_entropy_clahe, "CLAHE", clahe_results)
        
    if bilateral_results:
        save_txt_report(out_entropy_bilateral, "Bilateral", bilateral_results)

    # Save Combined Plot
    if clahe_results and bilateral_results:
        chart_combined_path = os.path.join(out_entropy_root, "grafik_perbandingan entropy batang.png")
        plot_combined_bar_chart(clahe_results, bilateral_results, chart_combined_path)

    print("\nEvaluasi Entropy Selesai!")
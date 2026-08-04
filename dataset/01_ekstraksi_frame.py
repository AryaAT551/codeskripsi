import cv2
import os
import glob

def extract_frames():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(current_dir, "video")
    output_base_folder = os.path.join(current_dir, "frames")
    video_paths = glob.glob(os.path.join(video_folder, "*.avi")) + glob.glob(os.path.join(video_folder, "*.mp4"))

    if not video_paths:
        print(f"[ERROR] Tidak ditemukan file video (.avi/.mp4) di dalam folder: {video_folder}")
        return
    print(f"Ditemukan {len(video_paths)} video. Memulai ekstraksi ke folder 'frames'...\n")

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_base_folder, video_name)

        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Gagal membuka file video: {video_path}")
            continue

        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(video_output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()
        print(f"Video '{video_name}': {frame_count} frame disimpan di '{video_output_folder}'")

if __name__ == "__main__":
    extract_frames()
import os
import glob
import cv2
import cinereader as cr


def extract_frames():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(current_dir, "video")
    output_base_folder = os.path.join(current_dir, "frames")

    video_paths = glob.glob(os.path.join(video_folder, "*.cine"))

    if not video_paths:
        print(
            f"[ERROR] Tidak ditemukan file video (.cine) di dalam folder: {video_folder}"
        )
        return

    print(f"Ditemukan {len(video_paths)} file .cine.")
    print("Memulai ekstraksi dan pengecekan dimensi citra\n")

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_base_folder, video_name)

        os.makedirs(video_output_folder, exist_ok=True)

        try:
            metadata = cr.read_metadata(video_path)
            _, images, _ = cr.read(
                video_path,
                start=metadata.FirstImageNo,
                count=metadata.ImageCount,
            )
        except Exception as e:
            print(
                f"[ERROR] Gagal membuka atau membaca file .cine: {video_path}. Error: {e}"
            )
            continue

        frame_count = 0
        dimension_reference = None
        dimension_consistent = True
        converted_to_grayscale = False

        for frame in images:

            height, width = frame.shape[:2]
            orig_channels = 1 if frame.ndim == 2 else frame.shape[2]

            if orig_channels == 3:
                channel_info = "3 Kanal"
            elif orig_channels == 1:
                channel_info = "1 Kanal"
            else:
                channel_info = f"{orig_channels} Kanal"

            if orig_channels == 3:
                frame_to_save = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                converted_to_grayscale = True
            else:
                frame_to_save = frame

            post_height, post_width = frame_to_save.shape[:2]
            post_channels = (
                1 if frame_to_save.ndim == 2 else frame_to_save.shape[2]
            )
            current_dimension = (post_width, post_height, post_channels)

            if dimension_reference is None:
                dimension_reference = current_dimension
                print(f"\nVideo Cine '{video_name}':")
                print(f"  - Dimensi citra : {width} x {height} piksel")
                print(f"  - Format awal   : {channel_info}")
                if converted_to_grayscale:
                    print(
                        f"  - Format hasil  : 1 Kanal (Grayscale - {post_width} x {post_height})"
                    )

            elif current_dimension != dimension_reference:
                dimension_consistent = False
                print(
                    f"[PERINGATAN] Frame ke-{frame_count} pada video "
                    f"'{video_name}' memiliki dimensi berbeda: "
                    f"{post_width} x {post_height}, kanal = {post_channels}"
                )

            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(video_output_folder, frame_name)

            if cv2.imwrite(frame_path, frame_to_save):
                frame_count += 1
            else:
                print(f"[ERROR] Gagal menyimpan frame: {frame_path}")

        if converted_to_grayscale:
            print(
                "Input RGB terdeteksi. Berhasil dikonversi ke grayscale dan ekstraksi frame berhasil."
            )
        else:
            print("Input Grayscale terdeteksi. Ekstraksi frame berhasil.")

        print(
            f"Video '{video_name}': {frame_count} frame berhasil disimpan."
        )

        if dimension_consistent:
            print("Status dimensi: Konsisten.")
        else:
            print("Status dimensi: Tidak konsisten.")

        print("-" * 50)


if __name__ == "__main__":
    extract_frames()
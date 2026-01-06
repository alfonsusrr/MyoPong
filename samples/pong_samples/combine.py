import os
import imageio
import numpy as np
from PIL import Image

def combine_samples_to_gif():
    input_dir = "./"
    output_path = "./combined_samples.gif"
    
    # Get all mp4 files in the directory and sort them
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mp4") and f.startswith("sample_")])
    
    if not video_files:
        print(f"No sample videos found in {os.path.abspath(input_dir)}")
        return

    print(f"Combining {len(video_files)} videos...")

    # Load all videos
    video_readers = [imageio.get_reader(os.path.join(input_dir, f)) for f in video_files]
    
    # Determine the maximum number of frames among all videos
    max_frames = 0
    all_video_frames = []
    for reader in video_readers:
        frames = [frame for frame in reader]
        all_video_frames.append(frames)
        max_frames = max(max_frames, len(frames))
        reader.close()

    # Padding shorter videos with their last frame
    for i in range(len(all_video_frames)):
        last_frame = all_video_frames[i][-1]
        while len(all_video_frames[i]) < max_frames:
            all_video_frames[i].append(last_frame)

    combined_frames = []
    rows = 1
    cols = 4
    
    # Ensure we have enough videos for the grid (using first 4 as requested)
    videos_to_use = all_video_frames[:rows * cols]
    
    for f_idx in range(max_frames):
        row_images = []
        for r in range(rows):
            col_images = []
            for c in range(cols):
                v_idx = r * cols + c
                if v_idx < len(videos_to_use):
                    frame = videos_to_use[v_idx][f_idx]
                    # Resize for better GIF performance and visibility
                    img = Image.fromarray(frame)
                    img = img.resize((320, 240), Image.Resampling.LANCZOS)
                    col_images.append(np.array(img))
                else:
                    # Black placeholder if less than 10 videos
                    col_images.append(np.zeros((240, 320, 3), dtype=np.uint8))
            
            row_images.append(np.concatenate(col_images, axis=1))
        
        full_frame = np.concatenate(row_images, axis=0)
        combined_frames.append(full_frame)

    print(f"Saving GIF to {output_path}...")
    # loop=0 means loop forever in imageio
    imageio.mimsave(output_path, combined_frames, fps=20, loop=0)
    print("Done.")

if __name__ == "__main__":
    combine_samples_to_gif()


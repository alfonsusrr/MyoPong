import imageio
import os
from pathlib import Path
from PIL import Image
import numpy as np

def create_gif_from_folder(folder_path, output_gif_path, scale=0.5, target_fps=10):
    video_files = sorted(list(folder_path.glob("*.mp4")), key=lambda x: x.name)
    if not video_files:
        print(f"No mp4 files found in {folder_path}")
        return

    print(f"Processing folder: {folder_path.name}")
    all_frames = []
    original_fps = 30 # Default
    
    for i, video_file in enumerate(video_files):
        print(f"  Reading {video_file.name}...")
        try:
            reader = imageio.get_reader(str(video_file))
            meta = reader.get_meta_data()
            if i == 0 and 'fps' in meta:
                original_fps = meta['fps']
                print(f"  Detected original FPS: {original_fps}")
            
            # Calculate frame skipping to reach target_fps
            skip = max(1, int(original_fps / target_fps))
            
            for j, frame in enumerate(reader):
                if j % skip == 0:
                    # Resize frame using Pillow
                    img = Image.fromarray(frame)
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    all_frames.append(np.array(img))
            reader.close()
        except Exception as e:
            print(f"  Error reading {video_file}: {e}")

    if all_frames:
        actual_fps = original_fps / skip
        print(f"  Writing GIF to {output_gif_path} (Actual FPS: {actual_fps:.2f}, Scale: {scale})...")
        try:
            # optimize=True and palettesize help reduce GIF size
            # loop=0 ensures the GIF loops forever
            imageio.mimsave(output_gif_path, all_frames, fps=actual_fps, opt=True, loop=0)
            print(f"  Done: {output_gif_path} (Size: {os.path.getsize(output_gif_path) / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  Error writing GIF: {e}")

def main():
    root_dir = Path("/home/alfonsusrr/workspace/MyoPong/selected_samples")
    subdirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        # Save GIF in the root selected_samples directory
        output_gif = root_dir / f"{subdir.name}.gif"
        create_gif_from_folder(subdir, output_gif, scale=0.5, target_fps=12)

if __name__ == "__main__":
    main()


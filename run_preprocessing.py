import cv2
import glob
import os
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import warnings

# --- 1. SCRIPT SETTINGS (MUST CONFIGURE) ---

# This MUST point to the folder you unzipped
# os.path.expanduser('~') automatically finds your home folder (e.g., /Users/abimcharly)
RAW_VIDEO_DIR = os.path.expanduser('~/Desktop/Celeb-DF-v2')

# This is the new folder where processed videos will be saved
PROCESSED_DATA_DIR = 'Processed_Data'

# Model parameters
SEQUENCE_LENGTH = 150  # How many frames to process per video
IMAGE_SIZE = 112       # The final image size (112x112)

# --- END OF SETTINGS ---


def setup_device():
    """Checks for M2 (MPS), CUDA, or CPU"""
    
    # --- START FIX ---
    # We are forcing the script to use the CPU.
    # This avoids the "Adaptive pool MPS" bug on the M2 GPU.
    # This will be MUCH SLOWE, but it will work reliably.
    device = torch.device("cpu")
    print("Forcing CPU to avoid MPS bug. This will be slow but reliable.")
    # --- END FIX ---
    
    return device


def process_videos(device):
    """
    Main function to find videos, detect faces, crop, and save new videos.
    (Robust version that processes frame-by-frame)
    """
    # Suppress warnings from the face detector
    warnings.filterwarnings('ignore', category=UserWarning)

    # Initialize the MTCNN face detector
    # We keep it on the CPU to avoid the MPS bug
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, 
        margin=0, 
        min_face_size=20, 
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709, 
        post_process=True,
        device=device
    )

    # Find all subfolders within the raw video directory
    input_subfolders = glob.glob(os.path.join(RAW_VIDEO_DIR, '*/'))

    if not input_subfolders:
        print(f"Error: No subfolders found in {RAW_VIDEO_DIR}")
        print("Please check the RAW_VIDEO_DIR path in the script.")
        return

    print(f"Found {len(input_subfolders)} subfolders.")

    # Loop through each subfolder (e.g., 'Celeb-real')
    for folder_path in input_subfolders:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # Create a matching folder in our output directory
        output_folder_path = os.path.join(PROCESSED_DATA_DIR, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Get a list of all videos in this subfolder
        video_files = glob.glob(os.path.join(folder_path, '*.mp4'))
        
        print(f"\n--- Processing folder: {folder_name} ({len(video_files)} videos) ---")

        # Loop through each video file, with a progress bar
        for video_path in tqdm(video_files):
            video_name = os.path.basename(video_path)
            output_video_path = os.path.join(output_folder_path, video_name)

            # --- Skip if this video has already been processed ---
            if os.path.exists(output_video_path):
                continue
                
            try:
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                
                # --- Create the output video file ---
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (IMAGE_SIZE, IMAGE_SIZE))

                frame_count = 0
                saved_frames = 0
                
                # --- Process frame-by-frame (Robust Fix) ---
                while frame_count < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video
                    
                    # Convert frame from BGR (OpenCV) to RGB (for face detector)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # --- Run Face Detection (Single Frame) ---
                    # This avoids the batching error
                    face_tensor = mtcnn(frame_rgb)
                    
                    # Check if a face was detected in this frame
                    if face_tensor is not None:
                        # Convert tensor from (C, H, W) to (H, W, C)
                        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                        # Denormalize (MTCNN outputs -1 to 1, OpenCV needs 0-255)
                        face_np = (face_np * 127.5 + 127.5).astype('uint8') 
                        # Convert back from RGB to BGR for saving
                        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                        
                        out.write(face_bgr)
                        saved_frames += 1
                    
                    frame_count += 1
                
                cap.release()
                out.release()
                
                if saved_frames == 0:
                    # No faces were found, so delete the empty 0-byte file
                    os.remove(output_video_path) 

            except Exception as e:
                print(f"\nError processing {video_path}: {e}")

    print("\n--- Preprocessing Complete! ---")
    print(f"Your processed, face-cropped videos are saved in: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    device = setup_device()
    process_videos(device)
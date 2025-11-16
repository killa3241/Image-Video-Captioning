import argparse
import os
import cv2
import torch
import math 
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

# ---------------------------------------------
#  MODELS SETUP
# ---------------------------------------------
# 1. Image Captioning Model (BLIP)
BLIP_MODEL_PATH = "./blip"

if not os.path.isdir(BLIP_MODEL_PATH):
    raise FileNotFoundError(
        f"BLIP model folder not found at {BLIP_MODEL_PATH}\n"
        f"Make sure it contains: config.json, pytorch_model.bin, tokenizer files, etc."
    )

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH, local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH, local_files_only=True)

# 2. Text Summarization Model (DistilBART)
SUM_MODEL_PATH = "./distilbart-summarizer" 
SUM_MODEL_NAME = "sshleifer/distilbart-cnn-12-6" 

try:
    sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_PATH, local_files_only=True)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_PATH, local_files_only=True)
except Exception:
    print(f"⚠ Could not load summarization model locally from {SUM_MODEL_PATH}. Attempting online download.")
    sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)


device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)
sum_model.to(device)

print(f"Models loaded successfully. Running on device: {device}")
# ---------------------------------------------
#  CAPTION A SINGLE FRAME
# ---------------------------------------------
def generate_caption_for_frame(frame_image: Image.Image) -> str:
    """Generates a caption for a single PIL Image frame using the BLIP model."""
    inputs = blip_processor(frame_image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_length=50)

    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# ---------------------------------------------
#  FINAL SUMMARY GENERATION (NEW FUNCTION)
# ---------------------------------------------
def generate_summary(caption_list: list) -> str:
    """Uses a summarization model to create a concise summary from a list of captions."""
    if not caption_list:
        return "No unique scenes detected."
    
    text_to_summarize = " ".join(caption_list)
    
    inputs = sum_tokenizer(
        text_to_summarize, 
        max_length=1024, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)

    # --- REFINED PARAMETERS HERE ---
    with torch.no_grad():
        summary_ids = sum_model.generate(
            inputs['input_ids'], 
            num_beams=3, 
            max_length=35, 
            min_length=10, 
            early_stopping=True
        )
    # -------------------------------

    summary_text = sum_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    
    # Ensure capitalization and period (existing logic)
    if summary_text and summary_text[0].islower():
        summary_text = summary_text[0].upper() + summary_text[1:]
    if summary_text and not summary_text.endswith('.'):
        summary_text += '.'

    return summary_text.strip()


# ---------------------------------------------
#  PROCESS FULL VIDEO (UPDATED LOGIC)
# ---------------------------------------------
# Define the maximum acceptable time interval (in seconds) between sampled frames
MAX_INTERVAL_SECONDS = 60 

# Removed 'num_frames=10' from the function signature. The default is handled by ARGPARSE if no argument is passed.
def caption_video(video_path: str, num_frames: int): 
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at '{video_path}'")
        return

    print(f"\n--- Processing video: {os.path.basename(video_path)} ---")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration_s = total_frames / (fps if fps else 1)
    
    print(f"Total frames: {total_frames}, FPS: {fps}, Duration: {total_duration_s:.2f}s")

    # --- DYNAMIC FRAME CALCULATION LOGIC ---
    default_num_frames = 3 # Use a sensible minimum default if dynamic calculation isn't relevant

    if total_duration_s > 0:
        # Calculate the minimum number of frames needed to ensure the interval is <= MAX_INTERVAL_SECONDS
        required_segments = total_duration_s / MAX_INTERVAL_SECONDS
        
        # We need N frames, so N = (Total Duration / MAX_INTERVAL_SECONDS) - 1
        min_required_frames = max(default_num_frames, math.ceil(required_segments) - 1)
        
        # If the user provided a number (num_frames), respect it only if it's greater than the required minimum.
        # Otherwise, use the dynamically calculated minimum.
        num_frames_to_extract = max(num_frames if num_frames is not None else default_num_frames, min_required_frames)
        
        if num_frames_to_extract > (num_frames if num_frames is not None else default_num_frames):
             print(f"ℹ️ Video is long. Dynamically setting sampled frames to {num_frames_to_extract} (Max {MAX_INTERVAL_SECONDS}s interval).")
    else:
         num_frames_to_extract = num_frames if num_frames is not None else default_num_frames


    # Determine which frames to capture 
    if total_frames <= num_frames_to_extract:
        actual_num_frames = total_frames
        print(f"⚠ Video too short or sampling maximum. Capturing all {actual_num_frames} frames.")
        frames_to_extract = list(range(0, total_frames))
    else:
        # Calculate interval based on the chosen num_frames_to_extract
        interval = total_frames / (num_frames_to_extract + 1)
        frames_to_extract = [int(i * interval) for i in range(1, num_frames_to_extract + 1)]
        actual_num_frames = num_frames_to_extract

    print(f"Frames to check: {frames_to_extract}")

    frame_captions = []
    raw_captions_for_summary = [] 
    last_caption = "" 
    unique_frame_count = 0 

    # Extract + caption frames
    for frame_no in frames_to_extract:
        if frame_no >= total_frames:
             continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠ Could not read frame {frame_no}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        current_caption = generate_caption_for_frame(pil_image)
        
        # --- LOGIC: Check for unique caption ---
        if current_caption.lower() != last_caption.lower():
            unique_frame_count += 1
            timestamp = frame_no / (fps if fps else 1)
            
            frame_entry = f"Frame {unique_frame_count} @ {timestamp:.2f}s: {current_caption}"
            frame_captions.append(frame_entry)
            
            raw_captions_for_summary.append(current_caption.capitalize() + ".") 
            
            print(f"✅ Unique Caption Found: {current_caption} (Frame {frame_no})")
            
            last_caption = current_caption
        else:
            print(f"⏭️ Skipping duplicate caption: {current_caption} (Frame {frame_no})")

    cap.release()

    # ---------------------------------------------
    # Summary Generation (NEW)
    # ---------------------------------------------
    final_summary = generate_summary(raw_captions_for_summary)


    # ---------------------------------------------
    # Print results
    # ---------------------------------------------
    print("\nUNIQUE FRAME CAPTIONS:")
    for c in frame_captions:
        print(f"- {c}")

    print("\nFINAL VIDEO CAPTION (Summarized):")
    print(final_summary)

    # Save output
    os.makedirs("outputs", exist_ok=True)

    output_file = os.path.splitext(os.path.basename(video_path))[0] + "_summary.txt"
    output_path = os.path.join("outputs", output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("UNIQUE FRAME CAPTIONS:\n")
        for c in frame_captions:
            f.write(f"- {c}\n")

        f.write("\nFINAL VIDEO CAPTION (Summarized):\n")
        f.write(final_summary)

    print(f"\n✅ Summary saved to: {output_path}")


# ---------------------------------------------
#  MAIN
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Video Captioning using Local BLIP model and DistilBART summarization.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    # Set default to None to indicate the value is dynamically calculated.
    # The minimum is set to 3 for compliance, but the dynamic calculation takes precedence.
    parser.add_argument("--num_frames", type=int, default=None, help=f"Number of frames to extract. Defaults dynamically based on video length (Max {MAX_INTERVAL_SECONDS}s interval).") 

    args = parser.parse_args()
    caption_video(args.video_path, args.num_frames)
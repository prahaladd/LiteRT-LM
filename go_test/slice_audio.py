import subprocess
import re
import os

def slice_audio(input_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run ffmpeg to detect silences
    cmd = [
        "ffmpeg", "-i", input_file,
        "-af", "silencedetect=noise=-40dB:d=1.5",
        "-f", "null", "-"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # Parse silences
    starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    
    # Match starts and ends
    splits = []
    for start, end in zip(starts, ends):
        start_t = float(start)
        end_t = float(end)
        midpoint = start_t + (end_t - start_t) / 2.0
        splits.append(midpoint)
    
    print(f"Detected {len(splits)} silence midpoints: {splits}")
    
    # Filter splits so chunks are at least 15 seconds long
    filtered_splits = []
    last_split = 0.0
    for split in splits:
        if split - last_split >= 15.0:
            filtered_splits.append(split)
            last_split = split
            
    print(f"Filtered split points: {filtered_splits}")
    
    # Now execute slicing
    start_t = 0.0
    chunk_index = 1
    for split_t in filtered_splits:
        output_file = os.path.join(output_dir, f"chunk{chunk_index}.mp3")
        slice_cmd = [
            "ffmpeg", "-y", "-ss", str(start_t), "-to", str(split_t),
            "-i", input_file, "-c", "copy", output_file
        ]
        print(f"Slicing chunk {chunk_index}: {start_t:.2f} -> {split_t:.2f}")
        subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start_t = split_t
        chunk_index += 1
        
    # Slice the final chunk
    output_file = os.path.join(output_dir, f"chunk{chunk_index}.mp3")
    slice_cmd = [
        "ffmpeg", "-y", "-ss", str(start_t),
        "-i", input_file, "-c", "copy", output_file
    ]
    print(f"Slicing final chunk {chunk_index}: {start_t:.2f} -> EOF")
    subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    input_f = "/Users/prahaladd/Projects/realtime-voice-browser/explainer_video_ultra_long.mp3"
    output_d = "/Users/prahaladd/Projects/litelmrt/go_test/chunks"
    slice_audio(input_f, output_d)

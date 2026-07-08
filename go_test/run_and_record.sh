#!/bin/bash
set -e

# Configuration
SCREEN_INDEX="4"
AUDIO_SRC="/Users/prahaladd/Projects/realtime-voice-browser/explainer_video_ultra_long.mp3"
WAV_SRC="explainer_video_ultra_long_16k.wav"
TMP_VIDEO="/tmp/screen_video_only.mp4"
FINAL_VIDEO="/tmp/explainer_recording.mp4"

echo "=== 1. Starting Screen-Only Recording in Background ==="
# Launch FFmpeg silently capturing only the screen
ffmpeg -y -f avfoundation -framerate 30 -i "$SCREEN_INDEX" -vf "scale=1280:-2" -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$TMP_VIDEO" > /tmp/ffmpeg_record.log 2>&1 &
FFMPEG_PID=$!

echo "FFmpeg started with PID $FFMPEG_PID. Allowing 2 seconds to initialize..."
sleep 2

echo "=== 2. Running Local VAD Browser Operator ==="
# Run the local operator loop
if go run vad_operator.go "$WAV_SRC"; then
    echo "Operator execution completed successfully."
else
    echo "Operator execution failed."
fi

echo "=== 3. Finalizing Screen Capture ==="
# Send SIGINT to FFmpeg to write container headers gracefully
kill -INT $FFMPEG_PID
wait $FFMPEG_PID || true
echo "Screen capture finalized."

echo "=== 4. Multiplexing Pristine Audio ==="
# Instant copy muxing of pristine audio file over the screen track
ffmpeg -y -i "$TMP_VIDEO" -i "$AUDIO_SRC" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "$FINAL_VIDEO" > /tmp/ffmpeg_mux.log 2>&1

echo "=== Muxing Complete! ==="
echo "Final recording saved to: $FINAL_VIDEO"
rm -f "$TMP_VIDEO"

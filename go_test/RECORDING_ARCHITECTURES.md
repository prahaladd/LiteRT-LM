# Voice Browser Recording Architectures

This document preserves the two design architectures for recording voice browser walkthroughs natively or post-run.

---

## Option A: Stream-Driven Local Assistant (True Real-Time)

In this architecture, the local operator acts as a live, streaming client, receiving audio chunks dynamically from the recording server and processing them in real time.

```text
[recorder-mcp] (FFmpeg)
  │  1. Capture screen & read audio file in background
  │  2. Pipe audio to MP4 video multiplexer
  │  3. Broadcast audio chunks on ws://localhost:9001/audio
  ▼
[audio-listener-mcp]
  │  4. Connect to WebSocket & buffer incoming PCM bytes
  ▼
[vad_operator] (Go client)
     5. Periodically poll read_audio() tool call
     6. Feed PCM chunks to local Silero VAD (detect silence)
     7. Feed speech buffer to local Whisper (transcribe text)
     8. Query local Gemma LLM for target action
     9. Call cdp-runner tool to click/type in Chrome
```

### Required Code Modifications
* Launch `bin/audio-listener-mcp` as a subprocess at startup.
* Replace static WAV file loading with a polling loop querying `audio-listener-mcp.read_audio()`.
* Feed incoming float32 slices to `sd.Detect()` dynamically.
* Trigger Whisper and Gemma execution only when VAD registers a silence threshold boundary.

---

## Option B: Parallel Post-Muxing (Silent Execution + Instant Merge)

In this architecture, the browser automation is driven by the local operator reading a static WAV file, while a video-only screen capture runs in the background. Once the run completes, the original pristine audio is multiplexed with the video in a single, instant command.

```text
1. Start screen-only capture (no audio overhead, no speaker leak):
   ffmpeg -y -f avfoundation -framerate 30 -i "4" -vf "scale=1280:-2" -c:v libx264 -preset ultrafast -pix_fmt yuv420p /tmp/screen_video_only.mp4

2. Run local operator (automation runs in parallel):
   go run vad_operator.go explainer_video_ultra_long_16k.wav

3. Gracefully stop screen capture once operator exits (SIGINT/Ctrl+C).

4. Instant merge of pristine audio file over recorded screen track:
   ffmpeg -y -i /tmp/screen_video_only.mp4 -i /Users/prahaladd/Projects/realtime-voice-browser/explainer_video_ultra_long.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 /tmp/explainer_recording.mp4
```

### Required Code Modifications
* **None**. The Go script does not need to be rewritten to support real-time streaming buffers.

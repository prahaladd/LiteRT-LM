package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-audio/wav"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/streamer45/silero-vad-go/speech"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"

	"github.com/chromedp/cdproto/page"
	cdpRuntime "github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
)

type MCPRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params"`
	ID      int         `json:"id"`
}

type CallToolParams struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type Content struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type ToolCallResponse struct {
	JSONRPC string `json:"jsonrpc"`
	Result  struct {
		Content []Content `json:"content"`
	} `json:"result"`
	Error interface{} `json:"error"`
	ID    int         `json:"id"`
}

type LLMDecision struct {
	Action string                 `json:"action"`
	Args   map[string]interface{} `json:"args"`
}

type CalibrationMapping struct {
	Transcript   string `json:"transcript"`
	Action       string `json:"action"`
	Selector     string `json:"selector"`
	Text         string `json:"text,omitempty"`
	TimeOffsetMs int    `json:"time_offset_ms"`
	Intent       string `json:"intent,omitempty"`
}

var chromeUserDataDir string
var chromeProfileDir string

func main() {
	ctx := context.Background()

	// Set up signal handling for clean exit on interrupt
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	var cleanup func()
	cleanup = func() {}
	go func() {
		sig := <-sigChan
		log.Printf("\n[SYSTEM] Received signal %v. Cleaning up...", sig)
		cleanup()
		os.Exit(0)
	}()
	defer func() {
		cleanup()
	}()

	// Silence loader and engine initialization logs
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	libDir := "/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries"
	modelPath := "/Users/prahaladd/Projects/litelmrt/LiteRT-LM/models/gemma-4-E2B-it.litertlm"
	rulesPath := "prompts/canva_rules.txt"
	onnxModelPath := "staging/silero_vad.onnx"
	whisperModelPath := "staging/ggml-tiny.bin"

	calibrateFlag := false
	fineTuneFlag := false
	var outputVideoPath string
	var cleanArgs []string
	argsList := os.Args[1:]
	for i := 0; i < len(argsList); i++ {
		arg := argsList[i]
		if arg == "--calibrate" {
			calibrateFlag = true
		} else if arg == "--fine-tune" {
			fineTuneFlag = true
		} else if arg == "--output" && i+1 < len(argsList) {
			outputVideoPath = argsList[i+1]
			i++
		} else if arg == "--chrome-user-data-dir" && i+1 < len(argsList) {
			chromeUserDataDir = argsList[i+1]
			i++
		} else if arg == "--chrome-profile-dir" && i+1 < len(argsList) {
			chromeProfileDir = argsList[i+1]
			i++
		} else {
			cleanArgs = append(cleanArgs, arg)
		}
	}

	audioPath := "explainer_audio_16k.wav"
	if len(cleanArgs) > 0 {
		audioPath = cleanArgs[0]
	}

	if outputVideoPath == "" {
		outputVideoPath = "output/explainer_recording.mp4"
	}

	if calibrateFlag {
		runCalibration(audioPath, onnxModelPath, whisperModelPath, modelPath, libDir)
		return
	}

	if fineTuneFlag {
		runFineTuning(audioPath, onnxModelPath, whisperModelPath, modelPath, libDir)
		return
	}

	// 1. Read System Prompt Rules
	rulesBytes, err := os.ReadFile(rulesPath)
	if err != nil {
		log.Fatalf("Failed to read system prompt rules: %v", err)
	}
	systemInstructions := string(rulesBytes)

	// 1b. Read Calibrated Mappings
	var mappings []CalibrationMapping
	mapData, err := os.ReadFile("canva_mappings.json")
	if err == nil {
		if err := json.Unmarshal(mapData, &mappings); err != nil {
			log.Printf("Warning: Failed to parse canva_mappings.json: %v", err)
		} else {
			log.Printf("Loaded %d calibrated mappings from canva_mappings.json", len(mappings))
		}
	} else {
		log.Printf("Warning: canva_mappings.json not found: %v", err)
	}

	// 2. Initialize VAD Detector
	log.Printf("Initializing Silero VAD (Model: %s)...", onnxModelPath)
	sd, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:            onnxModelPath,
		SampleRate:           16000,
		Threshold:            0.5,
		MinSilenceDurationMs: 1500, // Trigger boundary after 1.5 seconds of silence
		SpeechPadMs:          100,
		LogLevel:             speech.LogLevelWarn,
	})
	if err != nil {
		log.Fatalf("Failed to create Silero VAD detector: %v", err)
	}
	defer sd.Destroy()

	// 3. Initialize Whisper Model
	log.Printf("Initializing Whisper.cpp (Model: %s)...", whisperModelPath)
	wModel, err := whisper.New(whisperModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize Whisper model: %v", err)
	}
	defer wModel.Close()

	// 4. Initialize local Gemma-4 Client
	log.Printf("Initializing LiteRT-LM Client...")
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithAudioBackend("cpu"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize Gemma client: %v", err)
	}
	defer client.Close()

	// 5. Read Wav Audio File into memory
	log.Printf("Loading audio file: %s...", audioPath)
	audioFile, err := os.Open(audioPath)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer audioFile.Close()

	dec := wav.NewDecoder(audioFile)
	if !dec.IsValidFile() {
		log.Fatalf("Invalid WAV file format")
	}
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("Failed to read PCM data: %v", err)
	}
	pcmBuf := buf.AsFloat32Buffer()
	pcmData := pcmBuf.Data
	log.Printf("Loaded %d float32 samples (%0.2f seconds of audio)", len(pcmData), float64(len(pcmData))/16000.0)

	// 6. Run Silero VAD detection
	log.Printf("Running speech activity detection (VAD)...")
	segments, err := sd.Detect(pcmData)
	if err != nil {
		log.Fatalf("VAD detection failed: %v", err)
	}
	log.Printf("Speech segments identified: %d", len(segments))

	// 7. Spawn cdp-runner MCP server for ARIA page snapshots
	log.Printf("Launching cdp-runner MCP server...")
	runnerPath := "/Users/prahaladd/Projects/realtime-voice-browser/bin/cdp-runner"
	cmd := exec.Command(runnerPath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatalf("StdinPipe failed: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("StdoutPipe failed: %v", err)
	}

	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start cdp-runner: %v", err)
	}
	cleanup = func() {
		if cmd != nil && cmd.Process != nil {
			log.Printf("[SYSTEM] Killing cdp-runner MCP subprocess...")
			stdin.Close()
			cmd.Process.Kill()
		}
	}

	reader := bufio.NewReader(stdout)

	// Initialize MCP Session
	initReq := MCPRequest{
		JSONRPC: "2.0",
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]string{
				"name":    "vad-operator-client",
				"version": "1.0.0",
			},
		},
		ID: 1,
	}
	sendRequest(stdin, initReq)
	readResponseByID(reader, 1)

	mcpRequestID := 2

	// 7b. Initialize Recording State Variables
	var (
		playbackMu       sync.Mutex
		playbackPaused   bool
		playbackProgress float64
		playbackIndex    int
	)

	log.Printf("[SYSTEM] Starting screen capture recording dynamically...")
	var videoInputArgs []string
	switch runtime.GOOS {
	case "darwin":
		screenIdx := discoverScreenIndex()
		log.Printf("[SYSTEM] Discovered macOS screen capture index: %s", screenIdx)
		videoInputArgs = []string{
			"-f", "avfoundation",
			"-thread_queue_size", "4096",
			"-framerate", "30",
			"-pixel_format", "nv12",
			"-i", screenIdx + ":",
		}
	case "windows":
		videoInputArgs = []string{
			"-f", "gdigrab",
			"-thread_queue_size", "4096",
			"-framerate", "30",
			"-i", "desktop",
		}
	case "linux":
		videoInputArgs = []string{
			"-f", "x11grab",
			"-thread_queue_size", "4096",
			"-framerate", "30",
			"-i", ":0.0",
		}
	default:
		log.Fatalf("Unsupported platform: %s", runtime.GOOS)
	}

	args := []string{"-y"}
	args = append(args, videoInputArgs...)
	args = append(args,
		"-thread_queue_size", "4096",
		"-f", "s16le",
		"-ar", "16000",
		"-ac", "1",
		"-i", "pipe:0",
		"-map", "0:v",
		"-map", "1:a",
		"-vf", "scale=1280:-2",
		"-r", "30",
		"-c:v", "libx264",
		"-preset", "ultrafast",
		"-pix_fmt", "yuv420p",
		"-c:a", "aac",
		"-af", "aresample=async=1",
		"-max_interleave_delta", "0",
		"-movflags", "frag_keyframe+empty_moov+default_base_moof",
		outputVideoPath,
	)

	ffmpegPath := "ffmpeg"
	if _, err := os.Stat("/Users/prahaladd/.voicebrowser/ffmpeg"); err == nil {
		ffmpegPath = "/Users/prahaladd/.voicebrowser/ffmpeg"
	}
	recCmd := exec.Command(ffmpegPath, args...)
	ffLog, ffErr := os.Create("/tmp/ffmpeg_record.log")
	if ffErr == nil {
		recCmd.Stdout = ffLog
		recCmd.Stderr = ffLog
	}
	stdinPipe, pipeErr := recCmd.StdinPipe()
	if pipeErr != nil {
		log.Fatalf("Failed to create StdinPipe for recorder: %v", pipeErr)
	}

	// 7a. Generic Pre-Flight Preparation Check
	var startURL string
	for _, m := range mappings {
		if m.Action == "navigate" && m.Text != "" {
			startURL = m.Text
			break
		}
	}

	fmt.Println("\n=======================================================================")
	fmt.Println("=== PRE-FLIGHT RECORDING PREPARATION                                ===")
	fmt.Println("=======================================================================")
	if startURL != "" {
		log.Printf("[SYSTEM] Pre-flight: Navigating Chrome to starting URL: %s", startURL)
		actionReq := MCPRequest{
			JSONRPC: "2.0",
			Method:  "resources/call",
			Params: map[string]interface{}{
				"name": "execute_action",
				"arguments": map[string]interface{}{
					"action": "navigate",
					"url":    startURL,
				},
			},
			ID: mcpRequestID,
		}
		mcpRequestID++
		sendRequest(stdin, actionReq)
		readResponseByID(reader, actionReq.ID)
		fmt.Printf("1. Automatically navigated Chrome to starting URL: %s\n", startURL)
	} else {
		fmt.Println("1. Starting URL not found in mappings. Please navigate manually if needed.")
	}
	fmt.Println("2. Please log in or perform any necessary setup in the browser window.")
	fmt.Println("3. Set up your screen capture tool and share the correct browser window.")
	fmt.Println("\nOnce you are completely logged in, aligned, and ready to record:")
	fmt.Print("Press [ENTER] to start the walkthrough...")
	fmt.Println("\n=======================================================================")
	_, _ = bufio.NewReader(os.Stdin).ReadString('\n')

	if err := recCmd.Start(); err != nil {
		log.Printf("[WARNING] Failed to start screen recording subprocess: %v", err)
	} else {
		// Wait 2 seconds to make sure FFmpeg has fully opened device and is receiving frames
		time.Sleep(2 * time.Second)
		log.Printf("[SYSTEM] Screen recording process started successfully.")

		oldCleanup := cleanup
		cleanup = func() {
			log.Printf("[SYSTEM] Stopping screen recording gracefully (Closing audio pipe)...")
			stdinPipe.Close() // EOF signals FFmpeg to finish encoding and write trailer
			
			// Wait up to 5 seconds for FFmpeg to finalize container and exit
			done := make(chan error, 1)
			go func() {
				done <- recCmd.Wait()
			}()
			select {
			case err := <-done:
				if err != nil {
					log.Printf("[SYSTEM] FFmpeg exited with: %v", err)
				}
			case <-time.After(5 * time.Second):
				log.Printf("[WARNING] FFmpeg finalization timed out. Killing process...")
				if recCmd.Process != nil {
					recCmd.Process.Kill()
				}
			}

			if ffLog != nil {
				ffLog.Close()
			}
			log.Printf("[SYSTEM] Screen recording finalized and saved directly to %s", outputVideoPath)
			oldCleanup()
		}
	}

	// 7c. Start background real-time audio streaming goroutine
	go func() {
		sampleRate := 16000
		chunkSize := 800 // 50ms of audio
		interval := 50 * time.Millisecond

		for {
			playbackMu.Lock()
			paused := playbackPaused
			currIdx := playbackIndex
			playbackMu.Unlock()

			if currIdx >= len(pcmData) {
				break
			}

			var chunk []float32
			if paused {
				// Generate silence to keep FFmpeg audio stream alive and recording going
				chunk = make([]float32, chunkSize)
			} else {
				limit := currIdx + chunkSize
				if limit > len(pcmData) {
					limit = len(pcmData)
				}
				chunk = pcmData[currIdx:limit]

				playbackMu.Lock()
				playbackIndex = limit
				playbackProgress = float64(limit) / float64(sampleRate)
				playbackMu.Unlock()
			}

			// Convert to int16 bytes
			byteBuf := make([]byte, len(chunk)*2)
			for idx, s := range chunk {
				if s > 1.0 {
					s = 1.0
				} else if s < -1.0 {
					s = -1.0
				}
				val := int16(s * 32767.0)
				byteBuf[idx*2] = byte(val & 0xff)
				byteBuf[idx*2+1] = byte((val >> 8) & 0xff)
			}

			// Write to FFmpeg
			if stdinPipe != nil {
				stdinPipe.Write(byteBuf)
			}

			time.Sleep(interval)
		}
	}()

	// 8. Processing Loop: Segment -> Transcribe -> Get ARIA -> Gemma Dry Run
	for idx, s := range segments {
		// Wait until playbackProgress reaches the segment speech start
		for {
			playbackMu.Lock()
			prog := playbackProgress
			playbackMu.Unlock()
			if prog >= s.SpeechStartAt {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		log.Printf("\n======================================================================")
		log.Printf("PROCESSING VOICE SEGMENT %d/%d (Start: %0.2fs, End: %0.2fs)", idx+1, len(segments), s.SpeechStartAt, s.SpeechEndAt)
		log.Printf("======================================================================")



		// Slice samples
		startSample := int(s.SpeechStartAt * 16000)
		endSample := int(s.SpeechEndAt * 16000)
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(pcmData) || endSample <= startSample {
			endSample = len(pcmData)
		}
		segmentSamples := pcmData[startSample:endSample]

		if len(segmentSamples) == 0 {
			log.Printf("Segment is empty, skipping.")
			continue
		}

		// A. Transcribe sliced PCM segment with Whisper
		log.Printf("[WHISPER] Transcribing audio segment (%d samples)...", len(segmentSamples))
		wContext, err := wModel.NewContext()
		if err != nil {
			log.Printf("[WHISPER ERROR] Failed to create context: %v", err)
			continue
		}
		if err := wContext.Process(segmentSamples, nil, nil, nil); err != nil {
			log.Printf("[WHISPER ERROR] Processing failed: %v", err)
			continue
		}

		transcriptParts := []string{}
		for {
			seg, err := wContext.NextSegment()
			if err != nil {
				break
			}
			transcriptParts = append(transcriptParts, seg.Text)
		}
		transcript := strings.TrimSpace(strings.Join(transcriptParts, " "))
		log.Printf("[TRANSCRIPT] %q", transcript)

		if transcript == "" {
			log.Printf("[SYSTEM] Empty transcript, skipping segment.")
			continue
		}

		// B. Fetch Chrome ARIA accessibility snapshot
		log.Printf("[CDP-RUNNER] Capturing current ARIA page snapshot...")
		mcpRequestID++
		snapshotReq := MCPRequest{
			JSONRPC: "2.0",
			Method:  "tools/call",
			Params: CallToolParams{
				Name: "aria_snapshot",
				Arguments: map[string]interface{}{
					"format": "llm-text",
					"focus":  "interactive",
				},
			},
			ID: mcpRequestID,
		}
		sendRequest(stdin, snapshotReq)
		snapshotRespStr := readResponseByID(reader, mcpRequestID)

		var snapshotResp ToolCallResponse
		if err := json.Unmarshal([]byte(snapshotRespStr), &snapshotResp); err != nil {
			log.Fatalf("Failed to parse ARIA snapshot response: %v", err)
		}

		snapshotText := ""
		if len(snapshotResp.Result.Content) > 0 {
			snapshotText = snapshotResp.Result.Content[0].Text
		}

		// Format the snapshot into a compact single-line representation: • Type 'Name' (CSS: selector)
		snapshotText = compactSnapshot(snapshotText, transcript)

		if snapshotText == "" {
			snapshotText = "(No interactive elements found on the current page)"
		}

		// Truncate to remain safely within gemma limits
		originalLen := len(snapshotText)
		if len(snapshotText) > 6000 {
			snapshotText = snapshotText[:6000]
		}
		log.Printf("[CDP-RUNNER] Captured snapshot (original: %d, kept: %d characters).", originalLen, len(snapshotText))

		// C. Construct the exact prompt structure matching voice_operator
		prompt := fmt.Sprintf(
			"%s\n\n"+
				"CURRENT PAGE ACCESSIBILITY SNAPSHOT:\n"+
				"---\n"+
				"%s\n"+
				"---\n\n"+
				"USER COMMAND:\n"+
				"%s\n\n"+
				"CRITICAL ENFORCED STEPS & RULES:\n"+
				"1. STEP 1 (Intent & Actionability): If the user command is only describing a page element, placeholder, or location (e.g. 'the placeholder says \"Describe the image in your mind\"'), but does not explicitly request typing or clicking right now, you MUST output: {\"action\": \"none\", \"args\": {}}.\n"+
				"2. STEP 2 (Target Visibility): If the user command requests an action (click/type), but the target element is NOT present in the snapshot, you MUST output: {\"action\": \"none\", \"args\": {}}.\n"+
				"3. STEP 3 (Dropdown/Combo Boxes): For custom/React combo box dropdown elements in Canva, do NOT use \"select_dropdown\". Instead, use \"click\" to expand the menu, and then "+
				"\"click\" to select the option when it appears.\n"+
				"4. General: Always specify \"clear\": true when using the \"type_text\" action.\n"+
				"5. General: Analyze the actual ARIA snapshot structure; do not blindly copy selectors from the examples.\n"+
				"6. Selector Rule: In the arguments of click/type_text tools, you MUST copy the exact CSS Selector string from the \"(CSS: ...)\" value of the target element. Do NOT copy the element name/text, and do NOT make up custom selectors.\n"+
				"7. Format: Output decision ONLY as a valid JSON object matching: {\"action\": \"...\", \"args\": {...}}",
			systemInstructions,
			snapshotText,
			transcript,
		)

		var decision LLMDecision
		var matchedMappings []CalibrationMapping
		tClean := cleanString(transcript)
		for _, m := range mappings {
			mClean := cleanString(m.Transcript)
			if strings.Contains(tClean, mClean) || strings.Contains(mClean, tClean) {
				alreadyAdded := false
				for _, existing := range matchedMappings {
					if existing.Transcript == m.Transcript && existing.Action == m.Action && existing.Text == m.Text && existing.Selector == m.Selector {
						alreadyAdded = true
						break
					}
				}
				if !alreadyAdded {
					matchedMappings = append(matchedMappings, m)
				}
			}
		}

		if len(matchedMappings) > 0 {
			log.Printf("[SYSTEM] Found %d matching calibrated step(s) for transcript segment:", len(matchedMappings))
			for sIdx, mm := range matchedMappings {
				log.Printf("  [%d] Action: %s, Selector: %s, Text: %s", sIdx+1, mm.Action, mm.Selector, mm.Text)
			}

			var lastActionTime time.Time
			for sIdx, mm := range matchedMappings {
				// 1. Wait until playhead reaches the target time
				targetTime := s.SpeechStartAt + (float64(mm.TimeOffsetMs) / 1000.0)
				log.Printf("[SYSTEM] Waiting for playhead to reach %0.2fs (Segment Start: %0.2fs + Offset: %0.2fs) for action %d...", targetTime, s.SpeechStartAt, float64(mm.TimeOffsetMs)/1000.0, sIdx+1)
				for {
					playbackMu.Lock()
					prog := playbackProgress
					playbackMu.Unlock()
					if prog >= targetTime {
						break
					}
					time.Sleep(10 * time.Millisecond)
				}

				// 2. Safety check: Ensure at least 1.5s has elapsed since the last action
				if sIdx > 0 && !lastActionTime.IsZero() {
					elapsed := time.Since(lastActionTime)
					if elapsed < 1500*time.Millisecond {
						waitDuration := (1500 * time.Millisecond) - elapsed
						log.Printf("[SYSTEM] Safety wait: Sleeping %0.2fs to let DOM settle after previous action...", waitDuration.Seconds())
						time.Sleep(waitDuration)
					}
				}
				lastActionTime = time.Now()

				if mm.Action == "wait_for_user" {
					// 1. Pause playback
					playbackMu.Lock()
					playbackPaused = true
					playbackMu.Unlock()

					log.Printf("\n!!! USER ACTION REQUIRED !!!\n%s\nPress ENTER to continue...", mm.Text)
					bufio.NewReader(os.Stdin).ReadBytes('\n')

					// 2. Resume playback
					playbackMu.Lock()
					playbackPaused = false
					playbackMu.Unlock()
					continue
				}

				mcpArgs := make(map[string]interface{})
				switch mm.Action {
				case "navigate":
					mcpArgs["url"] = mm.Text
				case "click":
					mcpArgs["selector"] = mm.Selector
				case "type_text":
					mcpArgs["selector"] = mm.Selector
					mcpArgs["text"] = mm.Text
					mcpArgs["clear"] = true
				default:
					mcpArgs["selector"] = mm.Selector
					mcpArgs["text"] = mm.Text
				}

				mcpRequestID++
				actionReq := MCPRequest{
					JSONRPC: "2.0",
					Method:  "tools/call",
					Params: CallToolParams{
						Name:      mm.Action,
						Arguments: mcpArgs,
					},
					ID: mcpRequestID,
				}

				log.Printf("[SYSTEM] Executing sequential mapping tool call: %s on %q...", mm.Action, mm.Selector)
				sendRequest(stdin, actionReq)
				respLine := readResponseByID(reader, mcpRequestID)
				log.Printf("[SYSTEM] Response: %s", strings.TrimSpace(respLine))

				time.Sleep(3 * time.Second) // wait between sequence steps to allow DOM changes to settle
			}
			continue // Skip local model query and single action block, proceed directly to next voice segment
		} else {
			log.Printf("[SYSTEM] No calibrated step matches transcript. Skipping local LLM query.")
			continue
		}
			// D. Call Gemma-4 model for dry-run decision (free-form generation with step-by-step reasoning)
			log.Printf("[GEMMA] Querying local model...")
			startGen := time.Now()
			modelResponse, err := client.Generate(ctx, prompt)
			if err != nil {
				log.Fatalf("Gemma generation failed: %v", err)
			}
			log.Printf("[GEMMA] Response generated in %v", time.Since(startGen))
			log.Printf("[GEMMA RAW RESPONSE]\n%s\n", strings.TrimSpace(modelResponse))

			// E. Parse the last valid JSON decision block
			cleanJSON := extractLastJSON(modelResponse)
			if err := json.Unmarshal([]byte(cleanJSON), &decision); err != nil {
				log.Printf("[PARSING WARNING] Failed to parse JSON decision: %v. Raw text was: %s", err, modelResponse)
				log.Printf("Skipping this segment action.")
				continue
			}

		if decision.Args == nil {
			decision.Args = make(map[string]interface{})
		}

		// Format arguments strictly according to the MCP schemas to avoid schema validation errors
		mcpArgs := make(map[string]interface{})
		switch decision.Action {
		case "navigate":
			urlVal, ok := decision.Args["text"].(string)
			if !ok || urlVal == "" {
				urlVal, _ = decision.Args["url"].(string)
			}
			mcpArgs["url"] = urlVal
		case "click":
			mcpArgs["selector"] = decision.Args["selector"]
		case "type_text":
			mcpArgs["selector"] = decision.Args["selector"]
			mcpArgs["text"] = decision.Args["text"]
			mcpArgs["clear"] = true
		default:
			for k, v := range decision.Args {
				mcpArgs[k] = v
			}
		}

		log.Printf("[DECISION] Action: %s, Args: %v", decision.Action, mcpArgs)

		if decision.Action == "none" || decision.Action == "" {
			log.Printf("[SYSTEM] No browser action required. Proceeding to next segment.")
			continue
		}

		if decision.Action == "wait_for_user" {
			message, _ := decision.Args["message"].(string)
			fmt.Printf("\n==========================================\n")
			fmt.Printf("!!! USER INTERACTION REQUIRED !!!\n")
			fmt.Printf("Instruction: %s\n", message)
			fmt.Printf("Press ENTER in the terminal to resume the loop...\n")
			fmt.Printf("==========================================\n")

			inReader := bufio.NewReader(os.Stdin)
			_, _ = inReader.ReadString('\n')
			log.Printf("[SYSTEM] Manual step confirmed. Resuming.")
			continue
		}

		// Execute standard tool actions
		mcpRequestID++
		actionReq := MCPRequest{
			JSONRPC: "2.0",
			Method:  "tools/call",
			Params: CallToolParams{
				Name:      decision.Action,
				Arguments: mcpArgs,
			},
			ID: mcpRequestID,
		}

		log.Printf("[SYSTEM] Executing tool call: %s...", decision.Action)
		sendRequest(stdin, actionReq)
		respLine := readResponseByID(reader, mcpRequestID)
		log.Printf("[SYSTEM] Tool execution complete. Response: %s", strings.TrimSpace(respLine))

		// Wait for page updates/navigation to settle
		time.Sleep(4 * time.Second)
	}

	log.Printf("\nVoice browser dry-run operator loop complete.")
}

func sendRequest(w io.Writer, req interface{}) {
	data, _ := json.Marshal(req)
	fmt.Fprintf(w, "%s\n", data)
}

func readResponseByID(r *bufio.Reader, targetID int) string {
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			log.Fatalf("ReadString failed: %v", err)
		}

		var msg struct {
			ID int `json:"id"`
		}
		if err := json.Unmarshal([]byte(line), &msg); err == nil && msg.ID == targetID {
			return line
		}
	}
}

func extractLastJSON(s string) string {
	var starts []int
	for i := 0; i < len(s); i++ {
		if s[i] == '{' {
			starts = append(starts, i)
		}
	}

	for i := len(starts) - 1; i >= 0; i-- {
		start := starts[i]
		count := 0
		inString := false
		escape := false
		for j := start; j < len(s); j++ {
			if escape {
				escape = false
				continue
			}
			if s[j] == '\\' {
				escape = true
				continue
			}
			if s[j] == '"' {
				inString = !inString
				continue
			}
			if !inString {
				if s[j] == '{' {
					count++
				} else if s[j] == '}' {
					count--
					if count == 0 {
						candidate := s[start : j+1]
						var d LLMDecision
						if err := json.Unmarshal([]byte(candidate), &d); err == nil && d.Action != "" {
							return candidate
						}
						break
					}
				}
			}
		}
	}
	return extractJSON(s)
}

func extractJSON(s string) string {
	firstBrace := strings.Index(s, "{")
	if firstBrace == -1 {
		return s
	}

	depth := 0
	inString := false
	escape := false

	for i := firstBrace; i < len(s); i++ {
		char := s[i]

		if escape {
			escape = false
			continue
		}

		if char == '\\' {
			escape = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if !inString {
			if char == '{' {
				depth++
			} else if char == '}' {
				depth--
				if depth == 0 {
					return s[firstBrace : i+1]
				}
			}
		}
	}

	// Fallback
	lastBrace := strings.LastIndex(s, "}")
	if lastBrace != -1 && firstBrace < lastBrace {
		return s[firstBrace : lastBrace+1]
	}
	return s
}

func cleanString(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func compactSnapshot(snapshot string, transcript string) string {
	lines := strings.Split(snapshot, "\n")
	var result []string

	cleanTranscript := cleanString(transcript)

	var currentElement string
	var currentSelector string
	var isKeepCurrent bool

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "• ") {
			if currentElement != "" && isKeepCurrent {
				if currentSelector != "" {
					result = append(result, fmt.Sprintf("%s (CSS: %s)", currentElement, currentSelector))
				} else {
					result = append(result, currentElement)
				}
			}
			currentElement = trimmed
			currentSelector = ""

			lowerLine := strings.ToLower(trimmed)

			// Always keep form, input, main, and textarea elements
			isMainOrForm := strings.Contains(lowerLine, "part of main") || strings.Contains(lowerLine, "in form")
			isInput := strings.Contains(lowerLine, "textarea") || strings.Contains(lowerLine, "input") || strings.Contains(lowerLine, "switch") || strings.Contains(lowerLine, "dropdown")

			// For navigation elements/links, only keep if they match a transcript keyword
			matchesTranscript := false
			if firstQuote := strings.Index(trimmed, "'"); firstQuote != -1 {
				if secondQuote := strings.Index(trimmed[firstQuote+1:], "'"); secondQuote != -1 {
					name := trimmed[firstQuote+1 : firstQuote+1+secondQuote]
					cleanName := cleanString(name)
					if len(cleanName) > 2 && (strings.Contains(cleanTranscript, cleanName) || strings.Contains(cleanName, cleanTranscript)) {
						matchesTranscript = true
					}
				}
			}

			isKeepCurrent = isMainOrForm || isInput || matchesTranscript

			if idx := strings.Index(currentElement, " ("); idx != -1 {
				currentElement = currentElement[:idx]
			}
		} else if strings.HasPrefix(trimmed, "- Selector: ") {
			currentSelector = strings.TrimPrefix(trimmed, "- Selector: ")
		}
	}

	if currentElement != "" && isKeepCurrent {
		if currentSelector != "" {
			result = append(result, fmt.Sprintf("%s (CSS: %s)", currentElement, currentSelector))
		} else {
			result = append(result, currentElement)
		}
	}

	return strings.Join(result, "\n")
}

func writeSegmentAudio(stdin io.Writer, samples []float32, start, end float64) {
	sampleRate := 16000
	startIdx := int(start * float64(sampleRate))
	endIdx := int(end * float64(sampleRate))
	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > len(samples) {
		endIdx = len(samples)
	}
	if startIdx >= endIdx {
		return
	}
	segmentSamples := samples[startIdx:endIdx]

	// Stream in 50ms chunks (800 samples)
	chunkSize := 800
	for i := 0; i < len(segmentSamples); i += chunkSize {
		limit := i + chunkSize
		if limit > len(segmentSamples) {
			limit = len(segmentSamples)
		}
		chunk := segmentSamples[i:limit]

		// Convert float32 to int16 bytes (little-endian)
		byteBuf := make([]byte, len(chunk)*2)
		for idx, s := range chunk {
			if s > 1.0 {
				s = 1.0
			} else if s < -1.0 {
				s = -1.0
			}
			val := int16(s * 32767.0)
			byteBuf[idx*2] = byte(val & 0xff)
			byteBuf[idx*2+1] = byte((val >> 8) & 0xff)
		}
		stdin.Write(byteBuf)
		time.Sleep(50 * time.Millisecond) // Simulate real-time pace
	}
}

func discoverScreenIndex() string {
	ffmpegPath := "ffmpeg"
	if _, err := os.Stat("/Users/prahaladd/.voicebrowser/ffmpeg"); err == nil {
		ffmpegPath = "/Users/prahaladd/.voicebrowser/ffmpeg"
	}
	cmd := exec.Command(ffmpegPath, "-f", "avfoundation", "-list_devices", "true", "-i", "")
	output, _ := cmd.CombinedOutput()
	lines := strings.Split(string(output), "\n")
	
	for _, line := range lines {
		if strings.Contains(line, "Capture screen") {
			idx := strings.Index(line, "Capture screen")
			if idx != -1 {
				sub := line[:idx]
				lastBrac := strings.LastIndex(sub, "]")
				firstBrac := strings.LastIndex(sub, "[")
				if lastBrac != -1 && firstBrac != -1 && lastBrac > firstBrac+1 {
					return sub[firstBrac+1 : lastBrac]
				}
			}
		}
	}
	return "2" // Fallback to index 2
}

func captureARIASnapshot(stdin io.Writer, reader *bufio.Reader, reqID *int) string {
	*reqID++
	snapshotReq := MCPRequest{
		JSONRPC: "2.0",
		Method:  "tools/call",
		Params: CallToolParams{
			Name: "aria_snapshot",
			Arguments: map[string]interface{}{
				"format": "llm-text",
				"focus":  "interactive",
			},
		},
		ID: *reqID,
	}
	sendRequest(stdin, snapshotReq)
	snapshotRespStr := readResponseByID(reader, *reqID)

	var snapshotResp ToolCallResponse
	if err := json.Unmarshal([]byte(snapshotRespStr), &snapshotResp); err != nil {
		log.Printf("[WARNING] Failed to parse ARIA snapshot response: %v", err)
		return ""
	}

	if len(snapshotResp.Result.Content) > 0 {
		return snapshotResp.Result.Content[0].Text
	}
	return ""
}

func getActiveTabWSURL(startURL string) (string, error) {
	resp, err := http.Get("http://127.0.0.1:9222/json")
	if err != nil {
		log.Printf("[SYSTEM] Chrome debugging port 9222 is not active. Automatically launching Google Chrome...")
		profileDir := "/Users/prahaladd/Projects/litelmrt/LiteRT-LM/go_test/staging/chrome_dev_profile"
		if chromeUserDataDir != "" {
			profileDir = chromeUserDataDir
		}
		_ = os.MkdirAll(profileDir, 0755)

		targetURL := "about:blank"
		if startURL != "" {
			targetURL = startURL
		}

		args := []string{
			targetURL,
			"--remote-debugging-port=9222",
			"--user-data-dir=" + profileDir,
			"--no-first-run",
			"--no-default-browser-check",
		}
		if chromeProfileDir != "" {
			args = append(args, "--profile-directory="+chromeProfileDir)
		}

		chromeCmd := exec.Command("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", args...)
		if startErr := chromeCmd.Start(); startErr != nil {
			log.Printf("[SYSTEM] Direct binary launch failed: %v. Trying Mac 'open' command...", startErr)
			openArgs := append([]string{"-a", "Google Chrome", "--args"}, args...)
			_ = exec.Command("open", openArgs...).Run()
		}
		// Wait for Chrome to boot and open the port
		time.Sleep(2 * time.Second)

		// Retry connection
		resp, err = http.Get("http://127.0.0.1:9222/json")
		if err != nil {
			return "", fmt.Errorf("Google Chrome failed to respond on debugging port 9222 after auto-launch: %w", err)
		}
	}
	defer resp.Body.Close()

	var targets []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&targets); err != nil {
		return "", err
	}

	for _, target := range targets {
		if target["type"] == "page" {
			url, _ := target["url"].(string)
			title, _ := target["title"].(string)
			if strings.Contains(strings.ToLower(url), "canva") || strings.Contains(strings.ToLower(title), "canva") {
				if wsURL, ok := target["webSocketDebuggerUrl"].(string); ok {
					log.Printf("[TARGET MATCH] Found Canva tab: %q (URL: %q)", title, url)
					return wsURL, nil
				}
			}
		}
	}

	for _, target := range targets {
		if target["type"] == "page" {
			url, _ := target["url"].(string)
			title, _ := target["title"].(string)
			if wsURL, ok := target["webSocketDebuggerUrl"].(string); ok {
				log.Printf("[TARGET FALLBACK] Using tab: %q (URL: %q)", title, url)
				return wsURL, nil
			}
		}
	}

	req, err := http.NewRequest(http.MethodPut, "http://127.0.0.1:9222/json/new", nil)
	if err == nil {
		respNew, err := http.DefaultClient.Do(req)
		if err == nil {
			defer respNew.Body.Close()
			var targetNew map[string]interface{}
			if err := json.NewDecoder(respNew.Body).Decode(&targetNew); err == nil {
				if wsURL, ok := targetNew["webSocketDebuggerUrl"].(string); ok {
					log.Printf("[TARGET NEW] Opened new page tab.")
					return wsURL, nil
				}
			}
		}
	}

	return "", fmt.Errorf("no active page tab found in Chrome")
}

func runCalibration(audioPath, onnxModelPath, whisperModelPath, modelPath, libDir string) {
	ctx := context.Background()

	log.Printf("Loading audio file for calibration: %s...", audioPath)
	audioFile, err := os.Open(audioPath)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer audioFile.Close()

	dec := wav.NewDecoder(audioFile)
	if !dec.IsValidFile() {
		log.Fatalf("Invalid WAV file format")
	}
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("Failed to read PCM data: %v", err)
	}
	pcmBuf := buf.AsFloat32Buffer()
	pcmData := pcmBuf.Data

	log.Printf("Initializing Silero VAD (Model: %s)...", onnxModelPath)
	sd, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:            onnxModelPath,
		SampleRate:           16000,
		Threshold:            0.5,
		MinSilenceDurationMs: 1500,
		SpeechPadMs:          100,
		LogLevel:             speech.LogLevelWarn,
	})
	if err != nil {
		log.Fatalf("Failed to create Silero VAD detector: %v", err)
	}

	segments, err := sd.Detect(pcmData)
	if err != nil {
		log.Fatalf("VAD detection failed: %v", err)
	}
	log.Printf("Speech segments identified: %d", len(segments))
	sd.Destroy() // Clean up VAD immediately to prevent fork deadlocks!

	log.Printf("Initializing Whisper.cpp (Model: %s)...", whisperModelPath)
	wModel, err := whisper.New(whisperModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize Whisper model: %v", err)
	}
	defer wModel.Close()

	log.Printf("Initializing LiteRT-LM Client for action classification (loading local Gemma-4 weights - this may take a few seconds)...")
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithAudioBackend("cpu"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize Gemma client: %v", err)
	}
	log.Printf("LiteRT-LM Client initialized and model weights loaded successfully!")

	type TempStep struct {
		SegmentID      int
		Transcript     string
		SpeechStartS   float64
		SpeechEndS     float64
		ActionType     string
		TimeOffsetMs   int
		SegmentTextLen int
	}
	var steps []TempStep

	type DecomposedAction struct {
		Action        string `json:"action"`
		SubTranscript string `json:"sub_transcript"`
	}

	for idx, s := range segments {
		startSample := int(s.SpeechStartAt * 16000)
		endSample := int(s.SpeechEndAt * 16000)
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(pcmData) || endSample <= startSample {
			endSample = len(pcmData)
		}
		segmentSamples := pcmData[startSample:endSample]
		if len(segmentSamples) == 0 {
			continue
		}

		wContext, err := wModel.NewContext()
		if err != nil {
			log.Printf("[WHISPER ERROR] Failed to create context: %v", err)
			continue
		}
		if err := wContext.Process(segmentSamples, nil, nil, nil); err != nil {
			log.Printf("[WHISPER ERROR] Processing failed: %v", err)
			continue
		}

		transcriptParts := []string{}
		for {
			seg, err := wContext.NextSegment()
			if err != nil {
				break
			}
			transcriptParts = append(transcriptParts, seg.Text)
		}
		transcript := strings.TrimSpace(strings.Join(transcriptParts, " "))
		if transcript != "" {
			// Query Gemma-4 model to decompose the transcript into atomic actions
			prompt := fmt.Sprintf(
				"You are an AI assistant analyzing a speech transcript segment from a website tutorial video.\n"+
					"Your task is to identify and extract explicit commands instructing the user to perform a browser action (click, type, or navigate) on the website right now.\n\n"+
					"CRITICAL RULES:\n"+
					"1. Active actions MUST use imperative verbs (e.g., 'Navigate', 'Click', 'Type', 'Go ahead and click').\n"+
					"2. Do NOT classify future-tense statements, agenda overviews, or descriptions of what will be done as active. (e.g., 'First we'll navigate to Canvas AI tools', 'I'll show you how to', 'We'll discuss how to write effective prompts'). Classify these as \"action\": \"none\".\n"+
					"3. Do NOT classify instructions related to logging in, signing in, or signing up (e.g., 'Just click the Signup button and follow the prompts', 'please sign into your account'). These are handled manually during the initial navigation step. Classify these as \"action\": \"none\".\n"+
					"4. Do NOT classify general system/browser setup talk (e.g., 'please open your web browser now') as active. Classify these as \"action\": \"none\".\n\n"+
					"Respond ONLY with a valid JSON array of objects, where each object has:\n"+
					"- \"action\": \"click\", \"type_text\", \"navigate\", or \"none\"\n"+
					"- \"sub_transcript\": the exact text portion describing this action\n\n"+
					"Example response for active single-action:\n"+
					"[\n"+
					"  {\"action\": \"click\", \"sub_transcript\": \"Go ahead and click on canva-AI.\"}\n"+
					"]\n\n"+
					"Example response for active compound-actions:\n"+
					"[\n"+
					"  {\"action\": \"click\", \"sub_transcript\": \"Click on the style drop-down.\"},\n"+
					"  {\"action\": \"click\", \"sub_transcript\": \"Click on the 16 by 9 aspect ratio drop-down.\"}\n"+
					"]\n\n"+
					"Example response for passive narration/intro/login:\n"+
					"[\n"+
					"  {\"action\": \"none\", \"sub_transcript\": \"\"}\n"+
					"]\n\n"+
					"TRANSCRIPT: %q\n\n"+
					"JSON RESPONSE:",
				transcript,
			)

			resp, err := client.Generate(ctx, prompt)
			if err != nil {
				log.Printf("[WARNING] Gemma decomposition failed: %v", err)
				// fallback: treat as single click
				steps = append(steps, TempStep{
					SegmentID:    idx + 1,
					Transcript:   transcript,
					SpeechStartS: s.SpeechStartAt,
					SpeechEndS:   s.SpeechEndAt,
					ActionType:   "click",
				})
				continue
			}

			cleanResp := cleanJSONResponse(resp)
			var subActions []DecomposedAction
			if err := json.Unmarshal([]byte(cleanResp), &subActions); err != nil {
				log.Printf("[WARNING] Failed to parse Gemma JSON decomposition response: %v (raw response: %q)", err, resp)
				// fallback
				steps = append(steps, TempStep{
					SegmentID:    idx + 1,
					Transcript:   transcript,
					SpeechStartS: s.SpeechStartAt,
					SpeechEndS:   s.SpeechEndAt,
					ActionType:   "click",
				})
				continue
			}

			log.Printf("[GEMMA DECOMPOSED] Segment %d split into %d sub-actions:", idx+1, len(subActions))
			searchStart := 0
			for _, sa := range subActions {
				sa.Action = strings.TrimSpace(strings.ToLower(sa.Action))
				if sa.Action == "none" || sa.Action == "" {
					continue
				}
				sa.SubTranscript = strings.TrimSpace(sa.SubTranscript)
				if sa.SubTranscript == "" {
					continue
				}

				// Find starting index of sa.SubTranscript in original segment transcript
				tClean := cleanString(transcript)
				mClean := cleanString(sa.SubTranscript)
				subIdx := strings.Index(tClean[searchStart:], mClean)
				startIndex := searchStart
				if subIdx != -1 {
					startIndex = searchStart + subIdx
					searchStart += subIdx + len(mClean)
				}

				// Calculate time offset inside segment
				segmentDurationS := s.SpeechEndAt - s.SpeechStartAt
				offsetMs := 0
				if len(tClean) > 0 {
					offsetMs = int((float64(startIndex) / float64(len(tClean))) * segmentDurationS * 1000.0)
				}

				log.Printf("  - Action: %s, Offset: %d ms, Text: %q", sa.Action, offsetMs, sa.SubTranscript)
				steps = append(steps, TempStep{
					SegmentID:      idx + 1,
					Transcript:     sa.SubTranscript,
					SpeechStartS:   s.SpeechStartAt,
					SpeechEndS:     s.SpeechEndAt,
					ActionType:     sa.Action,
					TimeOffsetMs:   offsetMs,
					SegmentTextLen: len(tClean),
				})
			}
		}
	}
	client.Close() // Release model resources before starting Chrome session!
	log.Printf("Extracted %d valid speech steps for calibration.", len(steps))
	log.Println("\n=======================================================================")
	log.Println("=== AUDIO ANALYSIS & TRANSCRIPTION COMPLETED SUCCESSFULY!           ===")
	log.Println("=== CONNECTING TO GOOGLE CHROME TO START INTERACTIVE CALIBRATION... ===")
	log.Println("=======================================================================")

	startURL := ""
	for _, step := range steps {
		if step.ActionType == "navigate" {
			words := strings.Fields(step.Transcript)
			for _, w := range words {
				wClean := strings.Trim(w, ".,!?:;\"'")
				if strings.HasPrefix(wClean, "www.") || strings.HasPrefix(wClean, "http://") || strings.HasPrefix(wClean, "https://") {
					startURL = wClean
					if !strings.HasPrefix(startURL, "http") {
						startURL = "https://" + startURL
					}
					break
				}
			}
			break
		}
	}

	wsURL, err := getActiveTabWSURL(startURL)
	if err != nil {
		log.Fatalf("Failed to retrieve WebSocket URL from Chrome debugging endpoint: %v", err)
	}
	log.Printf("Connecting chromedp session directly to Chrome: %s", wsURL)

	allocCtx, cancelAlloc := chromedp.NewRemoteAllocator(ctx, wsURL)
	defer cancelAlloc()

	chromeCtx, cancelCtx := chromedp.NewContext(allocCtx)
	defer cancelCtx()

	var title string
	if err := chromedp.Run(chromeCtx, chromedp.Title(&title)); err != nil {
		log.Fatalf("Failed to establish connection to Chrome tab: %v", err)
	}
	log.Printf("Connected successfully! Tab Title: %q", title)

	// Bring the connected tab to the front of the screen
	_ = chromedp.Run(chromeCtx, page.BringToFront())

	clickChan := make(chan string, 10)
	typeChan := make(chan string, 10)

	chromedp.ListenTarget(chromeCtx, func(ev interface{}) {
		switch ev := ev.(type) {
		case *cdpRuntime.EventBindingCalled:
			var val string
			if err := json.Unmarshal([]byte(ev.Payload), &val); err != nil {
				val = ev.Payload
			}
			log.Printf("[BINDING CALL] %s: %s", ev.Name, val)
			if ev.Name == "callGoCalibrateClick" {
				clickChan <- val
			} else if ev.Name == "callGoCalibrateType" {
				typeChan <- val
			}
		case *cdpRuntime.EventConsoleAPICalled:
			if len(ev.Args) > 0 {
				var val string
				if err := json.Unmarshal(ev.Args[0].Value, &val); err == nil {
					log.Printf("[BROWSER CONSOLE] %s", val)
					if strings.HasPrefix(val, "CALIBRATE_CLICK:") {
						selector := strings.TrimPrefix(val, "CALIBRATE_CLICK:")
						clickChan <- selector
					} else if strings.HasPrefix(val, "CALIBRATE_TYPE:") {
						selector := strings.TrimPrefix(val, "CALIBRATE_TYPE:")
						typeChan <- selector
					}
				}
			}
		}
	})

	jsRecorder := `
	(function() {
		if (window.__calibrate_injected) return;
		window.__calibrate_injected = true;
		console.log("CALIBRATE_INFO: Click-Recorder Active.");

		function getUniqueSelector(el) {
			if (!el) return "";
			if (el.id) {
				if (el.id.includes('--')) {
					let suffix = el.id.substring(el.id.indexOf('--'));
					return '[id$="' + suffix + '"]';
				}
				return "#" + el.id;
			}
			let attr = el.getAttribute("aria-label");
			if (attr) return '[aria-label="' + attr + '"]';

			let tag = el.nodeName.toLowerCase();
			let classes = Array.from(el.classList).filter(c => !c.startsWith('x1')).join('.');
			if (classes) {
				tag += "." + classes;
			}

			let path = [tag];
			let parent = el.parentNode;
			while (parent && parent.nodeType === Node.ELEMENT_NODE) {
				let pTag = parent.nodeName.toLowerCase();
				if (parent.id) {
					if (parent.id.includes('--')) {
						let suffix = parent.id.substring(parent.id.indexOf('--'));
						path.unshift('[id$="' + suffix + '"]');
					} else {
						path.unshift("#" + parent.id);
					}
					break;
				}
				let pAttr = parent.getAttribute("aria-label");
				if (pAttr) {
					path.unshift('[aria-label="' + pAttr + '"]');
					break;
				}
				let pClasses = Array.from(parent.classList).filter(c => !c.startsWith('x1')).join('.');
				if (pClasses) {
					pTag += "." + pClasses;
				}
				path.unshift(pTag);
				parent = parent.parentNode;
			}
			return path.join(" > ");
		}

		document.addEventListener('mousedown', function(e) {
			let target = e.target;
			while (target && target !== document.body) {
				let role = target.getAttribute("role");
				let tagName = target.tagName;
				if (tagName === 'BUTTON' || tagName === 'A' || tagName === 'INPUT' || tagName === 'TEXTAREA' || 
					role === 'button' || role === 'link' || role === 'option' || role === 'tab' || role === 'combobox') {
					break;
				}
				target = target.parentNode;
			}
			if (!target || target === document.body) {
				target = e.target;
			}
			let selector = getUniqueSelector(target);
			if (window.callGoCalibrateClick) {
				window.callGoCalibrateClick(selector);
			} else {
				console.log("CALIBRATE_CLICK:" + selector);
			}
		}, true);

		document.addEventListener('change', function(e) {
			if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
				let selector = getUniqueSelector(e.target);
				if (window.callGoCalibrateType) {
					window.callGoCalibrateType(selector);
				} else {
					console.log("CALIBRATE_TYPE:" + selector);
				}
			}
		}, true);
	})();
	`

	err = chromedp.Run(chromeCtx, chromedp.ActionFunc(func(ctx context.Context) error {
		if err := cdpRuntime.AddBinding("callGoCalibrateClick").Do(ctx); err != nil {
			return err
		}
		if err := cdpRuntime.AddBinding("callGoCalibrateType").Do(ctx); err != nil {
			return err
		}
		_, err = page.AddScriptToEvaluateOnNewDocument(jsRecorder).Do(ctx)
		return err
	}))
	if err != nil {
		log.Fatalf("Failed to register persistent page listener: %v", err)
	}

	inReader := bufio.NewReader(os.Stdin)
	var finalMappings []CalibrationMapping

	for i := 0; i < len(steps); i++ {
		step := steps[i]
		fmt.Printf("\n======================================================================\n")
		fmt.Printf("STEP %d/%d (Audio Segment %d | Time: %0.2fs - %0.2fs)\n", i+1, len(steps), step.SegmentID, step.SpeechStartS, step.SpeechEndS)
		fmt.Printf("TRANSCRIPT: %q\n", step.Transcript)
		fmt.Printf("======================================================================\n")

		action := step.ActionType
		if action == "" {
			isNavigate := strings.Contains(strings.ToLower(step.Transcript), "navigate to") || strings.Contains(strings.ToLower(step.Transcript), "www.")
			if isNavigate {
				action = "navigate"
			} else {
				action = "click"
			}
		}

		if action == "navigate" {
			url := "https://www.canva.com"
			if strings.Contains(step.Transcript, "canva.com") {
				url = "https://www.canva.com"
			}
			log.Printf("Detected navigation action. Target: %s", url)

			finalMappings = append(finalMappings, CalibrationMapping{
				Transcript: step.Transcript,
				Action:     "navigate",
				Selector:   "",
				Text:       url,
			})
			log.Printf("Navigating browser directly to: %s", url)
			if err := chromedp.Run(chromeCtx, chromedp.Navigate(url)); err != nil {
				log.Fatalf("Navigation failed: %v", err)
			}
			// Bring the tab to the front of the screen
			_ = chromedp.Run(chromeCtx, page.BringToFront())

			fmt.Println("Navigation completed. Please log in and arrange window if needed.")
			fmt.Print("Press ENTER to proceed to the next step...")
			inReader.ReadString('\n')
			continue
		}

		var res interface{}
		chromedp.Run(chromeCtx, chromedp.Evaluate(jsRecorder, &res))

		for len(clickChan) > 0 {
			<-clickChan
		}
		for len(typeChan) > 0 {
			<-typeChan
		}

		fmt.Println("\n>>> ACTION REQUIRED: Go to Chrome and perform the action (click/type) now...")

		var recordedSelector string
		var actualAction string
		select {
		case sel := <-clickChan:
			recordedSelector = sel
			if step.ActionType == "type_text" {
				actualAction = "type_text"
				fmt.Printf(">>> RECORDED INPUT FIELD CLICK. Selector: %s\n", recordedSelector)
			} else {
				actualAction = "click"
				fmt.Printf(">>> RECORDED CLICK. Selector: %s\n", recordedSelector)
			}
		case sel := <-typeChan:
			recordedSelector = sel
			actualAction = "type_text"
			fmt.Printf(">>> RECORDED TYPE. Selector: %s\n", recordedSelector)
		}

		fmt.Print("Did this action execute successfully in Chrome? (y/n): ")
		confirm, _ := inReader.ReadString('\n')
		confirm = strings.ToLower(strings.TrimSpace(confirm))

		if confirm == "y" || confirm == "yes" {
			var textPayload string
			if actualAction == "type_text" {
				fmt.Print("Enter the text you typed: ")
				textPayload, _ = inReader.ReadString('\n')
				textPayload = strings.TrimSpace(textPayload)
			}

			finalMappings = append(finalMappings, CalibrationMapping{
				Transcript:   step.Transcript,
				Action:       actualAction,
				Selector:     recordedSelector,
				Text:         textPayload,
				TimeOffsetMs: step.TimeOffsetMs,
			})
			fmt.Println("Step successfully recorded!")

			hasDropdownWord := strings.Contains(strings.ToLower(step.Transcript), "drop-down") ||
				strings.Contains(strings.ToLower(step.Transcript), "dropdown") ||
				strings.Contains(strings.ToLower(step.Transcript), "ratio") ||
				strings.Contains(strings.ToLower(step.Transcript), "style") ||
				strings.Contains(strings.ToLower(step.Transcript), "select")

			fmt.Println("\n-----------------------------------------------------------------------")
			if hasDropdownWord {
				fmt.Println(">>> [RECOMMENDED HINT: y] The transcript mentions a dropdown or selection.")
				fmt.Println(">>> You likely need to record a second click to select the specific menu option.")
			} else {
				fmt.Println(">>> Did this step require an additional action in the same voice segment?")
				fmt.Println(">>> (e.g., if you clicked a menu button, you must click the menu item next).")
			}
			fmt.Print("Record another action for this SAME segment? (y/n): ")
			more, _ := inReader.ReadString('\n')
			more = strings.ToLower(strings.TrimSpace(more))
			if more == "y" || more == "yes" {
				i--
			}
			fmt.Println("-----------------------------------------------------------------------")
		} else {
			fmt.Println("Action failed or canceled. Retrying step...")
			i--
		}
	}



	outFile, err := os.Create("canva_mappings.json")
	if err != nil {
		log.Fatalf("Failed to create mappings file: %v", err)
	}
	defer outFile.Close()

	encoder := json.NewEncoder(outFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(finalMappings); err != nil {
		log.Fatalf("Failed to write mappings: %v", err)
	}

	fmt.Println("\n=======================================================================")
	fmt.Println("Interactive calibration completed! Results saved to canva_mappings.json")
	fmt.Println("=======================================================================")
}

func cleanJSONResponse(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "```") {
		idx := strings.Index(raw, "\n")
		if idx != -1 {
			raw = raw[idx+1:]
		}
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	}
	return raw
}

func runFineTuning(audioPath, onnxModelPath, whisperModelPath, modelPath, libDir string) {
	log.Printf("=======================================================================")
	log.Printf("=== STARTING OFFLINE CALIBRATION FINE-TUNING METHOD                 ===")
	log.Printf("=======================================================================")

	// 1. Read existing mappings
	mapData, err := os.ReadFile("canva_mappings.json")
	if err != nil {
		log.Fatalf("Failed to read canva_mappings.json: %v", err)
	}
	var mappings []CalibrationMapping
	if err := json.Unmarshal(mapData, &mappings); err != nil {
		log.Fatalf("Failed to parse canva_mappings.json: %v", err)
	}
	log.Printf("Loaded %d calibrated mappings for fine-tuning.", len(mappings))

	// 2. Load audio and run speech detection/transcription/decomposition
	audioFile, err := os.Open(audioPath)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer audioFile.Close()

	dec := wav.NewDecoder(audioFile)
	if !dec.IsValidFile() {
		log.Fatalf("Invalid WAV file format")
	}
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("Failed to read PCM data: %v", err)
	}
	pcmData := buf.AsFloat32Buffer().Data

	sd, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:            onnxModelPath,
		SampleRate:           16000,
		Threshold:            0.5,
		MinSilenceDurationMs: 1500,
		SpeechPadMs:          100,
		LogLevel:             speech.LogLevelWarn,
	})
	if err != nil {
		log.Fatalf("Failed to create VAD detector: %v", err)
	}
	segments, err := sd.Detect(pcmData)
	if err != nil {
		log.Fatalf("VAD detection failed: %v", err)
	}
	sd.Destroy()

	wModel, err := whisper.New(whisperModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize Whisper model: %v", err)
	}
	defer wModel.Close()

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithAudioBackend("cpu"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize Gemma client: %v", err)
	}
	defer client.Close()

	type TempStep struct {
		SegmentID      int
		Transcript     string
		SpeechStartS   float64
		SpeechEndS     float64
		ActionType     string
		TimeOffsetMs   int
		SegmentTextLen int
	}
	var steps []TempStep

	type DecomposedAction struct {
		Action        string `json:"action"`
		SubTranscript string `json:"sub_transcript"`
	}

	for idx, s := range segments {
		startSample := int(s.SpeechStartAt * 16000)
		endSample := int(s.SpeechEndAt * 16000)
		if startSample < 0 { startSample = 0 }
		if endSample > len(pcmData) || endSample <= startSample { endSample = len(pcmData) }
		segmentSamples := pcmData[startSample:endSample]
		if len(segmentSamples) == 0 { continue }

		wContext, err := wModel.NewContext()
		if err != nil { continue }
		if err := wContext.Process(segmentSamples, nil, nil, nil); err != nil { continue }

		transcriptParts := []string{}
		for {
			seg, err := wContext.NextSegment()
			if err != nil { break }
			transcriptParts = append(transcriptParts, seg.Text)
		}
		transcript := strings.TrimSpace(strings.Join(transcriptParts, " "))
		if transcript != "" {
			prompt := fmt.Sprintf(
				"You are an AI assistant analyzing a speech transcript segment from a website tutorial video.\n"+
					"Your task is to identify and extract explicit commands instructing the user to perform a browser action (click, type, or navigate) on the website right now.\n\n"+
					"CRITICAL RULES:\n"+
					"1. Active actions MUST use imperative verbs (e.g., 'Navigate', 'Click', 'Type', 'Go ahead and click').\n"+
					"2. Do NOT classify future-tense statements, agenda overviews, or descriptions of what will be done as active. (e.g., 'First we'll navigate to Canvas AI tools', 'I'll show you how to', 'We'll discuss how to write effective prompts'). Classify these as \"action\": \"none\".\n"+
					"3. Do NOT classify instructions related to logging in, signing in, or signing up (e.g., 'Just click the Signup button and follow the prompts', 'please sign into your account'). These are handled manually during the initial navigation step. Classify these as \"action\": \"none\".\n"+
					"4. Do NOT classify general system/browser setup talk (e.g., 'please open your web browser now') as active. Classify these as \"action\": \"none\".\n\n"+
					"Respond ONLY with a valid JSON array of objects, where each object has:\n"+
					"- \"action\": \"click\", \"type_text\", \"navigate\", or \"none\"\n"+
					"- \"sub_transcript\": the exact text portion describing this action\n\n"+
					"TRANSCRIPT: %q\n\n"+
					"JSON RESPONSE:",
				transcript,
			)

			resp, err := client.Generate(ctx, prompt)
			if err != nil { continue }

			cleanResp := cleanJSONResponse(resp)
			var subActions []DecomposedAction
			_ = json.Unmarshal([]byte(cleanResp), &subActions)

			searchStart := 0
			for _, sa := range subActions {
				sa.Action = strings.TrimSpace(strings.ToLower(sa.Action))
				if sa.Action == "none" || sa.Action == "" { continue }
				sa.SubTranscript = strings.TrimSpace(sa.SubTranscript)
				if sa.SubTranscript == "" { continue }

				tClean := cleanString(transcript)
				mClean := cleanString(sa.SubTranscript)
				subIdx := strings.Index(tClean[searchStart:], mClean)
				startIndex := searchStart
				if subIdx != -1 {
					startIndex = searchStart + subIdx
					searchStart += subIdx + len(mClean)
				}

				segmentDurationS := s.SpeechEndAt - s.SpeechStartAt
				offsetMs := 0
				if len(tClean) > 0 {
					offsetMs = int((float64(startIndex) / float64(len(tClean))) * segmentDurationS * 1000.0)
				}

				steps = append(steps, TempStep{
					SegmentID:      idx + 1,
					Transcript:     sa.SubTranscript,
					SpeechStartS:   s.SpeechStartAt,
					SpeechEndS:     s.SpeechEndAt,
					ActionType:     sa.Action,
					TimeOffsetMs:   offsetMs,
					SegmentTextLen: len(tClean),
				})
			}
		}
	}

	// 3. Align extracted steps with loaded mappings and rewrite offsets
	mappingSearchStart := 0
	for stepIdx, step := range steps {
		var matchedIndices []int
		for j := mappingSearchStart; j < len(mappings); j++ {
			if cleanString(mappings[j].Transcript) == cleanString(step.Transcript) {
				matchedIndices = append(matchedIndices, j)
			} else if len(matchedIndices) > 0 {
				break
			}
		}

		if len(matchedIndices) > 0 {
			mappingSearchStart = matchedIndices[len(matchedIndices)-1] + 1
			N := len(matchedIndices)
			if N == 1 {
				mappings[matchedIndices[0]].TimeOffsetMs = step.TimeOffsetMs
				log.Printf("[FINE-TUNE] Step: %q -> Set offset to %d ms", step.Transcript, step.TimeOffsetMs)
			} else {
				segmentDurationMs := (step.SpeechEndS - step.SpeechStartS) * 1000.0
				endTimeMs := segmentDurationMs
				nextIdx := stepIdx + 1
				if nextIdx < len(steps) && steps[nextIdx].SegmentID == step.SegmentID {
					endTimeMs = float64(steps[nextIdx].TimeOffsetMs)
				}
				availableDuration := endTimeMs - float64(step.TimeOffsetMs)

				for k, mappingIdx := range matchedIndices {
					ratio := float64(k) / float64(N-1)
					cooldown := 1500.0
					if availableDuration < cooldown {
						cooldown = 0
					}
					offset := float64(step.TimeOffsetMs) + (ratio * (availableDuration - cooldown))
					mappings[mappingIdx].TimeOffsetMs = int(offset)
					log.Printf("[FINE-TUNE] Step (Multi-Click %d/%d): %q -> Set offset to %d ms", k+1, N, step.Transcript, mappings[mappingIdx].TimeOffsetMs)
				}
			}
		}
	}

	// 4. Save updated mappings file
	outFile, err := os.Create("canva_mappings.json")
	if err != nil {
		log.Fatalf("Failed to create mappings file: %v", err)
	}
	defer outFile.Close()

	encoder := json.NewEncoder(outFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(mappings); err != nil {
		log.Fatalf("Failed to write mappings: %v", err)
	}

	log.Printf("=======================================================================")
	log.Printf("=== OFFLINE CALIBRATION FINE-TUNING COMPLETED SUCCESSFULY!          ===")
	log.Printf("=======================================================================")
}

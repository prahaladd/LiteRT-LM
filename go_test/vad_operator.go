package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/go-audio/wav"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/streamer45/silero-vad-go/speech"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
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
	Transcript string `json:"transcript"`
	Action     string `json:"action"`
	Selector   string `json:"selector"`
	Text       string `json:"text,omitempty"`
}

func main() {
	ctx := context.Background()

	// Silence loader and engine initialization logs
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	libDir := "/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries"
	modelPath := "/Users/prahaladd/Projects/litelmrt/LiteRT-LM/models/gemma-4-E2B-it.litertlm"
	rulesPath := "prompts/canva_rules.txt"
	onnxModelPath := "staging/silero_vad.onnx"
	whisperModelPath := "staging/ggml-tiny.bin"
	audioPath := "explainer_audio_16k.wav"
	if len(os.Args) > 1 {
		audioPath = os.Args[1]
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
	defer func() {
		stdin.Close()
		cmd.Process.Kill()
	}()

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

	// 8. Processing Loop: Segment -> Transcribe -> Get ARIA -> Gemma Dry Run
	for idx, s := range segments {
		log.Printf("\n======================================================================")
		log.Printf("PROCESSING VOICE SEGMENT %d/%d (Start: %0.2fs, End: %0.2fs)", idx+1, len(segments), s.SpeechStartAt, s.SpeechEndAt)
		log.Printf("======================================================================")

		// Play voice segment out loud in sync
		duration := s.SpeechEndAt - s.SpeechStartAt
		if duration > 0.05 {
			log.Printf("[AUDIO] Playing segment out loud...")
			playCmd := exec.Command("afplay", "-s", fmt.Sprintf("%0.3f", s.SpeechStartAt), "-t", fmt.Sprintf("%0.3f", duration), audioPath)
			playCmd.Run() // Blocks until audio finishes playing
		}

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
					if existing.Transcript == m.Transcript {
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

			for sIdx, mm := range matchedMappings {
				if sIdx > 0 {
					log.Printf("[SYSTEM] Sleeping 1.5s to let DOM settle/animate before next sequential action...")
					time.Sleep(1500 * time.Millisecond)
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

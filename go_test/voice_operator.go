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

func main() {
	ctx := context.Background()

	// Silence loader and engine initialization logs
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	libDir := "/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries"
	modelPath := "/Users/prahaladd/Projects/litelmrt/LiteRT-LM/models/gemma-4-E2B-it.litertlm"
	rulesPath := "prompts/canva_rules.txt"

	// 1. Read System Prompt Rules
	rulesBytes, err := os.ReadFile(rulesPath)
	if err != nil {
		log.Fatalf("Failed to read system prompt rules: %v", err)
	}
	systemInstructions := string(rulesBytes)

	log.Printf("Starting step-by-step voice browser automation loop...")

	// 2. Initialize local Gemma-4 Client
	log.Printf("Initializing LiteRT-LM Client...")
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithAudioBackend("cpu"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize client: %v", err)
	}
	defer client.Close()

	// 3. Spawn the cdp-runner MCP server
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

	// Initialize MCP session
	initReq := MCPRequest{
		JSONRPC: "2.0",
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]string{
				"name":    "voice-operator-client",
				"version": "1.0.0",
			},
		},
		ID: 1,
	}
	sendRequest(stdin, initReq)
	readResponseByID(reader, 1)

	// Dynamically find and sort chunk files in correct sequence
	var chunks []string
	files, err := os.ReadDir("chunks")
	if err != nil {
		log.Fatalf("Failed to read chunks directory: %v", err)
	}
	
	chunkCount := 0
	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "chunk") && strings.HasSuffix(file.Name(), ".mp3") {
			chunkCount++
		}
	}
	for i := 1; i <= chunkCount; i++ {
		chunks = append(chunks, fmt.Sprintf("chunks/chunk%d.mp3", i))
	}

	mcpRequestID := 2

	for i, chunkPath := range chunks {
		log.Printf("\n==========================================")
		log.Printf("PROCESSING AUDIO STREAM SEGMENT %d/%d: %s", i+1, len(chunks), chunkPath)
		log.Printf("==========================================")

		// A. Fetch current page ARIA snapshot from Chrome
		log.Printf("[SYSTEM] Capturing ARIA accessibility snapshot...")
		mcpRequestID++
		snapshotReq := MCPRequest{
			JSONRPC: "2.0",
			Method:  "tools/call",
			Params: CallToolParams{
				Name: "aria_snapshot",
				Arguments: map[string]interface{}{
					"format": "llm-text",
					"focus":  "all",
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
		
		// Truncate to stay safely under 4,096 token limit of local Gemma
		if len(snapshotText) > 4000 {
			snapshotText = snapshotText[:4000]
		}
		log.Printf("[SYSTEM] Captured snapshot (%d characters).", len(snapshotText))

		// B. Load Audio Chunk
		part, err := litertlm.AudioFromFile(chunkPath)
		if err != nil {
			log.Fatalf("Failed to load audio chunk %s: %v", chunkPath, err)
		}

		// C. Formulate Prompt containing rules and context
		prompt := fmt.Sprintf(
			"%s\n\n"+
				"CURRENT PAGE ACCESSIBILITY SNAPSHOT:\n"+
				"---\n"+
				"%s\n"+
				"---\n\n"+
				"AUDIO NARRATOR STATEMENT:\n"+
				"Based on the accompanying audio segment and the ARIA snapshot above, choose the next browser tool action.\n\n"+
				"CRITICAL ENFORCED STEPS & RULES:\n"+
				"1. STEP 1 (Intent & Actionability): If the narrator is only describing a page element, placeholder, or location (e.g. 'the placeholder says \"Describe the image in your mind\"'), but does not explicitly request typing or clicking right now, you MUST output: {\"action\": \"none\", \"args\": {}}.\n"+
				"2. STEP 2 (Target Visibility): If the narrator requests an action (click/type), but the target element is NOT present in the snapshot, you MUST output: {\"action\": \"none\", \"args\": {}}.\n"+
				"3. STEP 3 (Dropdown/Combo Boxes): For custom/React combo box dropdown elements in Canva, do NOT use \"select_dropdown\". Instead, use \"click\" to expand the menu, and then " +
				"\"click\" to select the option when it appears.\n" +
				"4. General: Always specify \"clear\": true when using the \"type_text\" action.\n"+
				"5. General: Analyze the actual ARIA snapshot structure; do not blindly copy selectors from the examples.\n"+
				"6. Format: Output decision ONLY as a valid JSON object matching: {\"action\": \"...\", \"args\": {...}}",
			systemInstructions,
			snapshotText,
		)

		// D. Run local Gemma model inference
		log.Printf("Sending segment to local Gemma-4 for action selection...")
		startGen := time.Now()
		modelResponse, err := client.GenerateMulti(ctx, []litertlm.Part{part, litertlm.Text(prompt)})
		if err != nil {
			log.Fatalf("Gemma Generation failed: %v", err)
		}
		log.Printf("Model response generated in %v", time.Since(startGen))
		log.Printf("Raw Model Response: %s", strings.TrimSpace(modelResponse))

		// E. Parse JSON Tool Action from Model
		cleanJSON := extractJSON(modelResponse)
		var decision LLMDecision
		if err := json.Unmarshal([]byte(cleanJSON), &decision); err != nil {
			log.Printf("[WARNING] Failed to parse JSON tool call: %v. Raw text was: %s", err, modelResponse)
			log.Printf("Skipping this segment action.")
			continue
		}

		log.Printf("[DECISION] Action: %s, Args: %v", decision.Action, decision.Args)

		// F. Execute the action
		if decision.Action == "none" || decision.Action == "" {
			log.Printf("[SYSTEM] No browser action required. Proceeding to next segment.")
			continue
		}

		if decision.Action == "wait_for_user" {
			message, _ := decision.Args["message"].(string)
			fmt.Printf("\n==========================================\n")
			fmt.Printf("!!! USER INTERACTION REQUIRED !!!\n")
			fmt.Printf("Instruction: %s\n", message)
			fmt.Printf("Press ENTER to resume the voice browser loop...\n")
			fmt.Printf("==========================================\n")
			
			var temp string
			fmt.Scanln(&temp)
			log.Printf("[SYSTEM] Manual step confirmed. Resuming.")
			continue
		}

		// Map actions and execute
		mcpRequestID++
		actionReq := MCPRequest{
			JSONRPC: "2.0",
			Method:  "tools/call",
			Params: CallToolParams{
				Name:      decision.Action,
				Arguments: decision.Args,
			},
			ID: mcpRequestID,
		}
		
		log.Printf("[SYSTEM] Executing tool call: %s...", decision.Action)
		sendRequest(stdin, actionReq)
		readResponseByID(reader, mcpRequestID)
		log.Printf("[SYSTEM] Tool execution complete.")

		// Wait for page updates
		time.Sleep(4 * time.Second)
	}

	log.Printf("\nVoice browser automation loop complete.")
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

// extractJSON extracts the clean JSON block from potential markdown code fences in model response
func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		// Remove opening code block markdown
		lines := strings.Split(s, "\n")
		if len(lines) > 1 {
			// Check if first line contains language (e.g. ```json)
			if strings.HasPrefix(lines[0], "```") {
				lines = lines[1:]
			}
		}
		// Remove closing backticks
		for i := len(lines) - 1; i >= 0; i-- {
			if strings.HasPrefix(strings.TrimSpace(lines[i]), "```") {
				lines = lines[:i]
				break
			}
		}
		s = strings.Join(lines, "\n")
	}
	return strings.TrimSpace(s)
}

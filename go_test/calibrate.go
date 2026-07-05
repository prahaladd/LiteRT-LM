package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type Step struct {
	ID         int    `json:"id"`
	Transcript string `json:"transcript"`
	Action     string `json:"action"` // navigate, click, type_text
	Text       string `json:"text,omitempty"`
}

type CalibrationMapping struct {
	Transcript string `json:"transcript"`
	Action     string `json:"action"`
	Selector   string `json:"selector"`
	Text       string `json:"text,omitempty"`
}

type MCPRequest struct {
	JSONRPC string                 `json:"jsonrpc"`
	Method  string                 `json:"method"`
	Params  map[string]interface{} `json:"params"`
	ID      int                    `json:"id"`
}

type MCPResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Result  struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"result"`
	Error interface{} `json:"error"`
}

func main() {
	steps := []Step{
		{7, "Navigate to www.canva.com. If you're not already logged in, please sign into your account.", "navigate", "https://www.canva.com"},
		{8, "Excellent. Now look at the left side of your screen. You'll see a vertical navigation menu. About halfway down you should see an option labeled \"canva-AI\" with a small \"AI\" icon next to it. Can everyone see that? Go ahead and click on \"canva-AI\"", "click", ""},
		{9, "Perfect. You should now see a page that asks, \"What will you design today?\" This is Canvas AI Hub. Notice at the top there are three tabs, your designs, templates, and Canvas AI. Make sure Canvas AI is selected. It should be highlighted.", "click", ""},
		{10, "Now, canva a I can help with several types of content. Look just below the tabs, you'll see five buttons, image, design, doc, code, and video clip. Today we're focusing on image generation, so click on the image button.", "click", ""},
		{13, "Click on the style drop-down. You'll see various artistic styles like photographic, digital art, or illustration.", "click", ""},
		{13, "For now, let's leave it on none, so the AI interprets our prompt naturally.", "click", ""},
		{13, "Now let's look at aspect ratio. Click on the 16 by 9 drop-down.", "click", ""},
		{14, "For our demonstration, let's keep it at 16 by 9.", "click", ""},
		{18, "Let's try an example. I'm going to type MoonLitSundset.", "type_text", "MoonLitSundset"},
		{21, "Alright, we're ready. Look at the right side of the text input field for the purple circular button with an arrow. Notice it only becomes active once you've entered text. I'm going to click it now.", "click", ""},
	}

	log.Println("Starting calibration tool...")
	log.Println("Launching cdp-runner MCP server...")
	cmd := exec.Command("/Users/prahaladd/Projects/realtime-voice-browser/bin/cdp-runner")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatalf("Failed to open stdin pipe: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to open stdout pipe: %v", err)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start cdp-runner: %v", err)
	}
	defer cmd.Process.Kill()

	reader := bufio.NewReader(stdout)
	mcpRequestID := 1

	// Initialize MCP Protocol
	initReq := MCPRequest{
		JSONRPC: "2.0",
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "calibration-client",
				"version": "1.0",
			},
		},
		ID: mcpRequestID,
	}
	sendRequest(stdin, initReq)
	readResponseByID(reader, mcpRequestID)

	initializedNotification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
		"params":  map[string]interface{}{},
	}
	reqBytes, _ := json.Marshal(initializedNotification)
	stdin.Write(append(reqBytes, '\n'))

	log.Println("CDP-Runner initialized successfully.")

	mappings := []CalibrationMapping{}
	inReader := bufio.NewReader(os.Stdin)

	for i := 0; i < len(steps); i++ {
		step := steps[i]
		fmt.Printf("\n======================================================================\n")
		fmt.Printf("STEP %d/%d (Audio Segment %d)\n", i+1, len(steps), step.ID)
		fmt.Printf("TRANSCRIPT: %q\n", step.Transcript)
		fmt.Printf("======================================================================\n")

		if step.Action == "navigate" {
			log.Printf("Executing auto-navigation to: %s", step.Text)
			mcpRequestID++
			navReq := MCPRequest{
				JSONRPC: "2.0",
				Method:  "tools/call",
				Params: map[string]interface{}{
					"name": "navigate",
					"arguments": map[string]interface{}{
						"url": step.Text,
					},
				},
				ID: mcpRequestID,
			}
			sendRequest(stdin, navReq)
			readResponseByID(reader, mcpRequestID)

			mappings = append(mappings, CalibrationMapping{
				Transcript: step.Transcript,
				Action:     "navigate",
				Selector:   "",
				Text:       step.Text,
			})
			fmt.Println("Navigation successful! Proceeding...")
			continue
		}

		// Click or Type actions: capture snapshot, list elements, ask user
		for {
			mcpRequestID++
			snapReq := MCPRequest{
				JSONRPC: "2.0",
				Method:  "tools/call",
				Params: map[string]interface{}{
					"name": "aria_snapshot",
					"arguments": map[string]interface{}{
						"format": "llm-text",
						"focus":  "interactive",
					},
				},
				ID: mcpRequestID,
			}
			sendRequest(stdin, snapReq)
			snapRespStr := readResponseByID(reader, mcpRequestID)

			var snapResp MCPResponse
			json.Unmarshal([]byte(snapRespStr), &snapResp)
			snapshotText := ""
			if len(snapResp.Result.Content) > 0 {
				snapshotText = snapResp.Result.Content[0].Text
			}

			// Extract interactive elements from snapshot
			elements := parseElements(snapshotText)

			fmt.Println("\nInteractive elements on page:")
			for idx, el := range elements {
				fmt.Printf("[%d] %s (CSS: %s)\n", idx+1, el.Name, el.Selector)
			}

			// Guess best selector based on transcription text
			guessIdx := guessSelector(elements, step.Transcript)
			if guessIdx != -1 {
				fmt.Printf("\n--> Suggested Element: [%d] %s (CSS: %s)\n", guessIdx+1, elements[guessIdx].Name, elements[guessIdx].Selector)
			}

			fmt.Printf("\nEnter element number (1-%d), custom selector, 'r' to reload, or 'f' to refresh page: ", len(elements))
			choice, _ := inReader.ReadString('\n')
			choice = strings.TrimSpace(choice)

			if choice == "r" || choice == "" {
				continue
			}

			if choice == "f" || choice == "refresh" {
				log.Println("Refreshing browser page...")
				mcpRequestID++
				refreshReq := MCPRequest{
					JSONRPC: "2.0",
					Method:  "tools/call",
					Params: map[string]interface{}{
						"name":      "refresh_page",
						"arguments": map[string]interface{}{},
					},
					ID: mcpRequestID,
				}
				sendRequest(stdin, refreshReq)
				readResponseByID(reader, mcpRequestID)
				continue
			}

			var finalSelector string
			if num, err := strconv.Atoi(choice); err == nil && num >= 1 && num <= len(elements) {
				finalSelector = elements[num-1].Selector
			} else {
				finalSelector = choice
			}

			log.Printf("Executing action %s using selector: %s", step.Action, finalSelector)
			mcpRequestID++

			var actionReq MCPRequest
			if step.Action == "click" {
				actionReq = MCPRequest{
					JSONRPC: "2.0",
					Method:  "tools/call",
					Params: map[string]interface{}{
						"name": "click",
						"arguments": map[string]interface{}{
							"selector": finalSelector,
						},
					},
					ID: mcpRequestID,
				}
			} else { // type_text
				actionReq = MCPRequest{
					JSONRPC: "2.0",
					Method:  "tools/call",
					Params: map[string]interface{}{
						"name": "type_text",
						"arguments": map[string]interface{}{
							"selector": finalSelector,
							"text":     step.Text,
							"clear":    true,
						},
					},
					ID: mcpRequestID,
				}
			}

			sendRequest(stdin, actionReq)
			readResponseByID(reader, mcpRequestID)

			fmt.Print("Did this action execute successfully in Chrome? (y/n): ")
			confirm, _ := inReader.ReadString('\n')
			confirm = strings.ToLower(strings.TrimSpace(confirm))

			if confirm == "y" || confirm == "yes" {
				mappings = append(mappings, CalibrationMapping{
					Transcript: step.Transcript,
					Action:     step.Action,
					Selector:   finalSelector,
					Text:       step.Text,
				})
				break
			} else {
				fmt.Println("Action failed or did not result in expected state. Retrying step...")
			}
		}
	}

	// Write mappings to file
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

	fmt.Println("\n=======================================================")
	fmt.Println("Calibration complete! Mappings saved to canva_mappings.json")
	fmt.Println("=======================================================")
}

type ParsedElement struct {
	Name     string
	Selector string
}

func parseElements(snapshot string) []ParsedElement {
	lines := strings.Split(snapshot, "\n")
	var result []ParsedElement

	var currentElement string
	var currentSelector string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "• ") {
			if currentElement != "" {
				if currentSelector != "" {
					result = append(result, ParsedElement{Name: currentElement, Selector: currentSelector})
				}
			}
			currentElement = trimmed
			if idx := strings.Index(currentElement, " ("); idx != -1 {
				currentElement = currentElement[:idx]
			}
			currentElement = strings.TrimPrefix(currentElement, "• ")
			currentSelector = ""
		} else if strings.HasPrefix(trimmed, "- Selector: ") {
			currentSelector = strings.TrimPrefix(trimmed, "- Selector: ")
		}
	}

	if currentElement != "" && currentSelector != "" {
		result = append(result, ParsedElement{Name: currentElement, Selector: currentSelector})
	}

	return result
}

func guessSelector(elements []ParsedElement, transcript string) int {
	cleanTranscript := strings.ToLower(transcript)
	for i, el := range elements {
		name := strings.ToLower(el.Name)
		if strings.Contains(cleanTranscript, name) || strings.Contains(name, cleanTranscript) {
			return i
		}
		// Try clean string parts
		parts := strings.Fields(name)
		for _, part := range parts {
			if len(part) > 3 && strings.Contains(cleanTranscript, part) {
				return i
			}
		}
	}
	return -1
}

func sendRequest(stdin interface{}, req interface{}) {
	reqBytes, err := json.Marshal(req)
	if err != nil {
		log.Fatalf("Failed to marshal request: %v", err)
	}
	writer := stdin.(*os.File)
	_, err = writer.Write(append(reqBytes, '\n'))
	if err != nil {
		log.Fatalf("Failed to write to stdin: %v", err)
	}
}

func readResponseByID(reader *bufio.Reader, targetID int) string {
	for {
		lineBytes, err := reader.ReadBytes('\n')
		if err != nil {
			log.Fatalf("Failed to read response: %v", err)
		}
		line := string(lineBytes)

		// Filter out log lines
		if !strings.HasPrefix(line, "{") {
			continue
		}

		var resp struct {
			ID int `json:"id"`
		}
		if err := json.Unmarshal(lineBytes, &resp); err == nil && resp.ID == targetID {
			return line
		}
	}
}

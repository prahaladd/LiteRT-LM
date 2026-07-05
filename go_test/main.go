package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx := context.Background()

	// Silence loader and engine initialization logs
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	libDir := "/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries"
	modelPath := "/Users/prahaladd/Projects/litelmrt/LiteRT-LM/models/gemma-4-E2B-it.litertlm"

	var audioPath string
	var audioBytes []byte
	var textPrompt string
	var hasAudio bool

	// Check command line arguments
	if len(os.Args) > 1 {
		arg := os.Args[1]
		if isAudioFile(arg) {
			audioPath = arg
			hasAudio = true
			log.Printf("Detected audio file argument: %s", arg)
			// If there's a second argument, use it as text prompt
			if len(os.Args) > 2 {
				textPrompt = os.Args[2]
			}
		} else {
			textPrompt = arg
		}
	}

	// Check stdin (only if we don't have arguments already)
	stat, _ := os.Stdin.Stat()
	if !hasAudio && textPrompt == "" && (stat.Mode() & os.ModeCharDevice) == 0 {
		// Stdin has data piped into it
		bytes, err := io.ReadAll(os.Stdin)
		if err == nil && len(bytes) > 0 {
			// Check if it is a WAV file (starts with "RIFF")
			if len(bytes) > 4 && string(bytes[:4]) == "RIFF" {
				audioBytes = bytes
				hasAudio = true
				log.Printf("Detected audio stream (WAV) from stdin (%d bytes)", len(bytes))
			} else {
				// Treat stdin as text prompt if we don't have one yet
				if textPrompt == "" {
					textPrompt = string(bytes)
				}
			}
		}
	}

	if textPrompt == "" {
		if hasAudio {
			textPrompt = "Transcribe the audio:"
		} else {
			textPrompt = "Explain quantum computing in one sentence."
		}
	}

	log.Printf("Initializing LiteRT-LM Client...")
	log.Printf("Library Directory: %s", libDir)
	log.Printf("Model Path: %s", modelPath)

	// Create new client
	startInit := time.Now()
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
	log.Printf("Client initialized in %v", time.Since(startInit))

	// Construct audio parts AFTER client initialization
	var audioPart litertlm.Part
	if hasAudio {
		if audioPath != "" {
			part, err := litertlm.AudioFromFile(audioPath)
			if err != nil {
				log.Fatalf("Failed to load audio from file: %v", err)
			}
			audioPart = part
			log.Printf("Loaded audio from file successfully.")
		} else if len(audioBytes) > 0 {
			audioPart = litertlm.Audio(audioBytes)
			log.Printf("Loaded audio from stdin stream successfully.")
		}
	}

	log.Printf("Generating response for prompt: %q (has audio: %v)", textPrompt, hasAudio)

	fmt.Println("\nModel Response (Streaming):")
	startGen := time.Now()
	var firstTokenTime time.Duration
	var hasFirstToken bool

	if hasAudio {
		parts := []litertlm.Part{audioPart, litertlm.Text(textPrompt)}
		for chunk, err := range client.GenerateMultiStream(ctx, parts) {
			if err != nil {
				log.Fatalf("\nGeneration failed: %v", err)
			}
			if !hasFirstToken {
				firstTokenTime = time.Since(startGen)
				hasFirstToken = true
			}
			fmt.Print(chunk.Text)
			_ = os.Stdout.Sync()
		}
	} else {
		for chunk, err := range client.GenerateStream(ctx, textPrompt) {
			if err != nil {
				log.Fatalf("\nGeneration failed: %v", err)
			}
			if !hasFirstToken {
				firstTokenTime = time.Since(startGen)
				hasFirstToken = true
			}
			fmt.Print(chunk.Text)
			_ = os.Stdout.Sync()
		}
	}
	fmt.Println()
	log.Printf("\nTotal response generated in %v (Time to first token: %v)", time.Since(startGen), firstTokenTime)
}

func isAudioFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus":
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}
	return false
}

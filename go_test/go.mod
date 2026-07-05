module go_test

go 1.26.2

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20260626130357-0ae02cdb2c73
	github.com/go-audio/wav v1.1.0
	github.com/streamer45/silero-vad-go v0.2.1
	github.com/vladimirvivien/litertlm-go v0.5.0
)

require (
	github.com/chromedp/cdproto v0.0.0-20260321001828-e3e3800016bc // indirect
	github.com/chromedp/chromedp v0.15.1 // indirect
	github.com/chromedp/sysutil v1.1.0 // indirect
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/go-audio/audio v1.0.0 // indirect
	github.com/go-audio/riff v1.0.0 // indirect
	github.com/go-json-experiment/json v0.0.0-20260214004413-d219187c3433 // indirect
	github.com/gobwas/httphead v0.1.0 // indirect
	github.com/gobwas/pool v0.2.1 // indirect
	github.com/gobwas/ws v1.4.0 // indirect
	github.com/jupiterrider/ffi v0.6.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
)

replace (
	github.com/ggerganov/whisper.cpp/bindings/go => ./staging/whisper.cpp/bindings/go
	github.com/streamer45/silero-vad-go => ./staging/silero-vad-go
)

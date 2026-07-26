package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
)

type Step struct {
	ID         int    `json:"id"`
	Transcript string `json:"transcript"`
	Action     string `json:"action"` // navigate, click, type_text
	Text       string `json:"text,omitempty"`
}

type CalibrationMapping struct {
	Transcript   string `json:"transcript"`
	Action       string `json:"action"`
	Selector     string `json:"selector"`
	Text         string `json:"text,omitempty"`
	TimeOffsetMs int    `json:"time_offset_ms"`
}

func getActiveTabWSURL() (string, error) {
	resp, err := http.Get("http://127.0.0.1:9222/json")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var targets []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&targets); err != nil {
		return "", err
	}

	// 1. Try to find a tab with "canva" in the URL or title
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

	// 2. Fallback to any other page tab
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

	return "", fmt.Errorf("no active page tab found in Chrome and failed to open a new one")
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

	ctx := context.Background()

	log.Println("Starting interactive calibration tool (stable, silent mode)...")

	// 1. Connect to Chrome tab via WebSocket
	wsURL, err := getActiveTabWSURL()
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

	clickChan := make(chan string, 10)
	typeChan := make(chan string, 10)

	chromedp.ListenTarget(chromeCtx, func(ev interface{}) {
		switch ev := ev.(type) {
		case *runtime.EventBindingCalled:
			var val string
			if err := json.Unmarshal([]byte(ev.Payload), &val); err == nil {
				log.Printf("[BINDING CALL] %s: %s", ev.Name, val)
				if ev.Name == "callGoCalibrateClick" {
					clickChan <- val
				} else if ev.Name == "callGoCalibrateType" {
					typeChan <- val
				}
			}
		case *runtime.EventConsoleAPICalled:
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
		if err := runtime.AddBinding("callGoCalibrateClick").Do(ctx); err != nil {
			return err
		}
		if err := runtime.AddBinding("callGoCalibrateType").Do(ctx); err != nil {
			return err
		}
		_, err = page.AddScriptToEvaluateOnNewDocument(jsRecorder).Do(ctx)
		return err
	}))
	if (err != nil) {
		log.Fatalf("Failed to register persistent page listener: %v", err)
	}

	mappings := []CalibrationMapping{}
	inReader := bufio.NewReader(os.Stdin)

	for i := 0; i < len(steps); i++ {
		step := steps[i]
		fmt.Printf("\n======================================================================\n")
		fmt.Printf("STEP %d/%d (Audio Segment %d)\n", i+1, len(steps), step.ID)
		fmt.Printf("TRANSCRIPT: %q\n", step.Transcript)
		fmt.Printf("======================================================================\n")

		if step.Action == "navigate" {
			log.Printf("Navigating browser directly to: %s", step.Text)
			if err := chromedp.Run(chromeCtx, chromedp.Navigate(step.Text)); err != nil {
				log.Fatalf("Navigation failed: %v", err)
			}

			// jsRecorder is persistently injected on reloads

			mappings = append(mappings, CalibrationMapping{
				Transcript:   step.Transcript,
				Action:       "navigate",
				Selector:     "",
				Text:         step.Text,
				TimeOffsetMs: 0,
			})
			fmt.Println("Navigation completed. Please log in and arrange window if needed.")
			fmt.Print("Press ENTER to proceed to the next step...")
			inReader.ReadString('\n')
			continue
		}

		// Ensure listener is evaluated on active frame context
		var res interface{}
		chromedp.Run(chromeCtx, chromedp.Evaluate(jsRecorder, &res))

		// Drain stray click/type events
		for len(clickChan) > 0 {
			<-clickChan
		}
		for len(typeChan) > 0 {
			<-typeChan
		}

		fmt.Println("\n>>> ACTION REQUIRED: Go to Chrome and perform the click or text input now...")

		var recordedSelector string
		select {
		case sel := <-clickChan:
			recordedSelector = sel
			fmt.Printf(">>> RECORDED CLICK. Selector: %s\n", recordedSelector)
		case sel := <-typeChan:
			recordedSelector = sel
			fmt.Printf(">>> RECORDED TYPE. Selector: %s\n", recordedSelector)
		}

		fmt.Print("Did this action execute successfully in Chrome? (y/n): ")
		confirm, _ := inReader.ReadString('\n')
		confirm = strings.ToLower(strings.TrimSpace(confirm))

		if confirm == "y" || confirm == "yes" {
			mappings = append(mappings, CalibrationMapping{
				Transcript:   step.Transcript,
				Action:       step.Action,
				Selector:     recordedSelector,
				Text:         step.Text,
				TimeOffsetMs: 0,
			})
			fmt.Println("Step successfully recorded!")
		} else {
			fmt.Println("Action failed. Retrying step...")
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
	if err := encoder.Encode(mappings); err != nil {
		log.Fatalf("Failed to write mappings: %v", err)
	}

	fmt.Println("\n=======================================================================")
	fmt.Println("Interactive calibration completed! Results saved to canva_mappings.json")
	fmt.Println("=======================================================================")
}

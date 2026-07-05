import subprocess
import json
import sys
import time

def run_loop():
    # 1. Spawn cdp-runner (keeps running for the entire lifecycle of this script)
    proc = subprocess.Popen(
        ['/Users/prahaladd/Projects/realtime-voice-browser/bin/cdp-runner'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    
    def send_req(method, params, req_id):
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id
        }
        proc.stdin.write(json.dumps(req) + '\n')
        proc.stdin.flush()

    def read_resp(target_id):
        while True:
            line = proc.stdout.readline()
            if not line:
                return None
            try:
                resp = json.loads(line)
                if resp.get('id') == target_id:
                    return resp
            except Exception as e:
                pass

    # Initialize MCP session
    send_req("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "python-client", "version": "1.0.0"}
    }, 1)
    read_resp(1)

    print("[SYSTEM] Persistent browser session initialized. Ready for commands.")
    print("[READY]")
    sys.stdout.flush()

    req_id = 2

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        line = line.strip()
        if not line:
            continue
            
        try:
            cmd_data = json.loads(line)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON command: {e}")
            print("[READY]")
            sys.stdout.flush()
            continue
            
        action_name = cmd_data.get('action')
        action_args = cmd_data.get('args', {})
        
        if action_name == 'exit':
            print("[SYSTEM] Exiting session.")
            break
            
        # 2. Execute requested action
        if action_name != 'none':
            print(f"[SYSTEM] Executing action: {action_name}({json.dumps(action_args)})...")
            req_id += 1
            send_req("tools/call", {
                "name": action_name,
                "arguments": action_args
            }, req_id)
            action_res = read_resp(req_id)
            
            # Save screenshot if it is a screenshot tool call
            if action_name == 'screenshot' and action_res and 'result' in action_res:
                content = action_res['result'].get('content', [])
                for item in content:
                    if item.get('type') == 'image' and item.get('data'):
                        import base64
                        try:
                            img_data = base64.b64decode(item['data'])
                            screenshot_path = "/Users/prahaladd/.gemini/antigravity-cli/brain/77ac81ee-93ee-443a-9d5a-d788d3915f43/screenshot.png"
                            with open(screenshot_path, "wb") as f_img:
                                f_img.write(img_data)
                            print(f"[SYSTEM] Screenshot successfully saved to {screenshot_path}")
                        except Exception as e_img:
                            print(f"[ERROR] Failed to save screenshot: {e_img}")
            
            print(f"[SYSTEM] Action complete.")
            time.sleep(4)
        else:
            print("[SYSTEM] Skipping action, taking snapshot directly...")

        # 3. Capture ARIA snapshot (default focus is all, format is llm-text)
        print("[SYSTEM] Capturing ARIA accessibility snapshot...")
        req_id += 1
        send_req("tools/call", {
            "name": "aria_snapshot",
            "arguments": {"format": "llm-text", "focus": "all"}
        }, req_id)
        snapshot_res = read_resp(req_id)
        
        if not snapshot_res or 'result' not in snapshot_res:
            print("[ERROR] Failed to capture snapshot.")
            print("[READY]")
            sys.stdout.flush()
            continue

        content = snapshot_res['result'].get('content', [])
        snapshot_text = ""
        for item in content:
            if item.get('type') == 'text':
                snapshot_text = item.get('text')
                break

        if not snapshot_text:
            print("[ERROR] Snapshot returned empty text.")
            print("[READY]")
            sys.stdout.flush()
            continue

        # Print the ARIA snapshot directly to standard output for the agent context to consume
        print("\n=== CURRENT ARIA SNAPSHOT ===")
        print(snapshot_text)
        print("=============================\n")

        print("[READY]")
        sys.stdout.flush()

    # Clean up
    proc.stdin.close()
    proc.terminate()
    proc.wait()

if __name__ == '__main__':
    run_loop()

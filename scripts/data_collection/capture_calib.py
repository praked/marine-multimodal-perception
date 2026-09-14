#!/usr/bin/env python3
"""Live fisheye preview + checkerboard capture for recalibration.

Headless-friendly: serves an MJPEG live feed + a Capture button over HTTP so you
can watch the feed in a browser on your laptop and press ENTER (or click) to
grab a shot. Camera config mirrors capture.py EXACTLY (RGB888, 864x648 =
2592/3 x 1944/3). Saves JPGs to ~/fisheye_calib/ (resumes the count if rerun).
Open  http://<pi-ip>:8000  (plain HTTP, not https) in your laptop browser.
"""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from picamera2 import Picamera2

OUT = os.path.expanduser("~/fisheye_calib")
os.makedirs(OUT, exist_ok=True)
PORT = 8000

_lock = threading.Lock()
_state = {"frame": None, "count": len([f for f in os.listdir(OUT) if f.endswith(".jpg")])}


def _grabber():
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (int(2592 / 3), int(1944 / 3))}))
    cam.start()
    time.sleep(1.0)
    while True:
        arr = cam.capture_array()
        with _lock:
            _state["frame"] = arr


# NOTE: plain string + .replace (NOT .format): the JS below contains { } braces.
PAGE = """<!doctype html><html><body style="margin:0;background:#1c1c1c;color:#eee;
font-family:sans-serif;text-align:center">
<h2 id="c">captured: __N__</h2>
<img src="/stream" style="max-width:98vw;max-height:78vh;border:2px solid #444">
<div style="margin:12px"><button onclick="cap()"
 style="font-size:22px;padding:10px 34px">Capture &nbsp;(or press Enter)</button></div>
<p style="color:#aaa">Cover the frame EDGES + BOTTOM CORNERS across shots; TILT the board.</p>
<script>
function cap(){fetch('/capture').then(r=>r.text()).then(t=>{document.getElementById('c').innerText='captured: '+t;});}
document.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();cap();}});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/plain"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(PAGE.replace("__N__", str(_state["count"])).encode(), "text/html")
        elif self.path == "/capture":
            with _lock:
                arr = None if _state["frame"] is None else _state["frame"].copy()
            if arr is not None:
                p = os.path.join(OUT, f"calib_{_state['count']:03d}.jpg")
                cv2.imwrite(p, arr)
                _state["count"] += 1
                print(f"saved {p}  ({arr.shape[1]}x{arr.shape[0]})  total={_state['count']}", flush=True)
            self._send(str(_state["count"]).encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        arr = None if _state["frame"] is None else _state["frame"]
                    if arr is None:
                        time.sleep(0.05)
                        continue
                    ok, jpg = cv2.imencode(".jpg", arr)
                    if not ok:
                        continue
                    b = jpg.tobytes()
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


if __name__ == "__main__":
    threading.Thread(target=_grabber, daemon=True).start()
    print(f"Saving to {OUT} (already {_state['count']}).", flush=True)
    print(f"Open http://<pi-ip>:{PORT}  (plain HTTP). ENTER/click=capture. Ctrl-C to stop.", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

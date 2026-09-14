#!/usr/bin/env python3
"""Standalone per-sensor health check. Verifies each sensor produces LIVE data
that VARIES spatially and over time (catches dead/black/frozen/disconnected),
independent of the capture pipeline. Reuses the repo's radar TLV decoder."""
import os, sys, time, struct
import numpy as np
REPORT=[]
def V(name,status,detail):
    REPORT.append((name,status,detail)); print(f"[{status}] {name}: {detail}",flush=True)

def test_fisheye():
    try:
        from picamera2 import Picamera2
        c=Picamera2(); c.configure(c.create_video_configuration(main={"size":(864,648),"format":"RGB888"}))
        c.start(); time.sleep(1.0)
        fr=[c.capture_array().astype(np.float32) for _ in range(12) if not time.sleep(0.15)]
        c.stop(); a=np.stack(fr)
        m=float(a.mean()); ss=float(a[0].std()); td=float(np.abs(np.diff(a,axis=0)).mean())
        d=f"{fr[0].shape} mean={m:.0f} spatial_std={ss:.1f} temporal_diff={td:.2f}"
        if m<5: V("fisheye","FAIL","near-BLACK: "+d)
        elif m>250: V("fisheye","SUSPECT","saturated/white: "+d)
        elif ss<2: V("fisheye","SUSPECT","flat/uniform image: "+d)
        elif td<0.05: V("fisheye","SUSPECT","FROZEN (identical frames): "+d)
        else: V("fisheye","PASS",d)
        return fr[6]
    except Exception as e: V("fisheye","FAIL",f"exception {e!r}"); return None

def test_thermal():
    try:
        import cv2, glob
        # Discovery mirrors continuous_capture._open_thermal: the by-id symlink
        # first (the Lepton's /dev/video index moves once the CSI fisheye claims
        # the low ones), then an index walk as a fallback. And the by-id node
        # gets several read attempts, because the PureThermal often withholds a
        # frame on the first read after opening: probing once reported a healthy
        # camera as absent on ~half of all runs (2026-08-17).
        by_id=[os.path.realpath(p)
               for p in sorted(glob.glob("/dev/v4l/by-id/*PureThermal*index0"))]
        cap=None; node=None
        for dev in by_id+[1,0,2,3]:
            c=cv2.VideoCapture(dev)
            tries=4 if dev in by_id else 1
            ok,f=False,None
            for a in range(tries):
                ok,f=c.read()
                if ok and f is not None: break
                if a+1<tries: time.sleep(0.3)
            if ok and f is not None and f.shape[0]<=240 and f.shape[1]<=320: cap,node=c,dev; break
            c.release()
        if cap is None: V("thermal","FAIL","no thermal node answered (by-id + video0-3; disconnected?)"); return None
        fr=[]
        for _ in range(12):
            ok,f=cap.read()
            if ok and f is not None: fr.append(f.astype(np.float32))
            time.sleep(0.12)
        cap.release()
        if len(fr)<3: V("thermal","FAIL",f"only {len(fr)} frames from {node}"); return None
        a=np.stack(fr); m=float(a.mean()); ss=float(a[0].std()); td=float(np.abs(np.diff(a,axis=0)).mean())
        d=f"{node} {fr[0].shape} mean={m:.0f} spatial_std={ss:.1f} temporal_diff={td:.2f} ({len(fr)}f)"
        if m<5: V("thermal","FAIL","near-BLACK: "+d)
        elif ss<1.5: V("thermal","SUSPECT","flat/uniform (no gradient): "+d)
        elif td<0.03: V("thermal","SUSPECT","FROZEN: "+d)
        else: V("thermal","PASS",d)
        return fr[6]
    except Exception as e: V("thermal","FAIL",f"exception {e!r}"); return None

MAGIC=b'\x02\x01\x04\x03\x06\x05\x08\x07'
def u32(d): return d[0]+d[1]*256+d[2]*65536+d[3]*16777216
CLI,DATA,CFG="/dev/ttyACM0","/dev/ttyACM1","/home/vesselauser/test_config.cfg"
def read_frame(ser):
    sync=b""; dl=time.monotonic()+1.0
    while MAGIC not in sync:
        b=ser.read(1)
        if not b:
            if time.monotonic()>dl: return None
            continue
        sync=(sync+b)[-8:]
    h=ser.read(32)
    if len(h)<32: return None
    plen=u32(h[4:8]); nobj=u32(h[20:24]); ntlv=u32(h[24:28])
    p=ser.read(plen-40)
    return (p,ntlv,nobj) if len(p)>=plen-40 else None
def parse(data,ntlv,nobj):
    off=0; pts=[]
    for _ in range(ntlv):
        if off+8>len(data): break
        t=u32(data[off:off+4]); l=u32(data[off+4:off+8]); off+=8
        if t==1:
            q=off
            for _ in range(nobj):
                if q+16>len(data): break
                pts.append((struct.unpack("<f",data[q:q+4])[0],struct.unpack("<f",data[q+4:q+8])[0])); q+=16
        off+=l
    return pts
def test_radar():
    import serial
    if not (os.path.exists(CLI) and os.path.exists(DATA)):
        V("radar","FAIL",f"ports missing ACM0={os.path.exists(CLI)} ACM1={os.path.exists(DATA)} (radar USB disconnected?)"); return
    try:
        with serial.Serial(CLI,115200,timeout=1) as cli:
            cli.write(b"sensorStop\n"); time.sleep(0.1); cli.reset_input_buffer()
            for ln in open(CFG):
                ln=ln.strip()
                if ln and not ln.startswith("%"): cli.write((ln+"\n").encode()); time.sleep(0.05)
    except Exception as e: V("radar","FAIL",f"CLI config failed on {CLI}: {e!r}"); return
    frames=nonempty=tot=0; rng=[]
    try:
        with serial.Serial(DATA,921600,timeout=1) as ser:
            t0=time.monotonic()
            while time.monotonic()-t0<6.0:
                fr=read_frame(ser)
                if fr is None: continue
                p,ntlv,nobj=fr; frames+=1
                pts=parse(p,ntlv,nobj)
                if nobj>0: nonempty+=1
                tot+=len(pts); rng+=[y for x,y in pts if 0<y<10]
        with serial.Serial(CLI,115200,timeout=1) as cli: cli.write(b"sensorStop\n")
    except Exception as e: V("radar","FAIL",f"data read failed on {DATA}: {e!r}"); return
    rtxt=f" range {min(rng):.1f}-{max(rng):.1f}m" if rng else ""
    d=f"{frames} frames/6s, {nonempty} w/detections, {tot} pts{rtxt}"
    if frames==0: V("radar","FAIL","config OK but NO data frames: "+d+" (check data-port USB / cable)")
    elif tot==0: V("radar","SUSPECT","frames stream but 0 detections: "+d+" (empty scene, or RF/antenna)")
    else: V("radar","PASS",d)

def main():
    f=test_fisheye(); t=test_thermal(); test_radar()
    try:
        import cv2
        if f is not None: cv2.imwrite("/home/vesselauser/ewasr/health_fisheye.png",cv2.cvtColor(f.astype(np.uint8),cv2.COLOR_RGB2BGR))
        if t is not None: cv2.imwrite("/home/vesselauser/ewasr/health_thermal.png",t.astype(np.uint8))
    except Exception as e: print("save:",e)
    print("\n===== SENSOR HEALTH SUMMARY =====")
    for n,s,d in REPORT: print(f"{s:8} {n:8} {d}")
    bad=[n for n,s,d in REPORT if s in ("FAIL","SUSPECT")]
    print("\nVERDICT:", "ALL SENSORS HEALTHY" if not bad else f"CHECK: {', '.join(bad)}")

if __name__=="__main__":
    main()

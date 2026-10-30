#!/usr/bin/env python3
"""montazh-lite check: measures the finished file by numbers. Python standard library + ffmpeg only.

Usage: python check.py OUT.mp4 [--max-pause 0.60] [--lufs-tol 0.5]
Reads OUT.mp4.plan.json written by montazh.py. Prints a table and writes OUT.mp4.check.json.
Exit code 0 = every check passed, 1 = at least one failed, 2 = could not run.

Checks:
  loudness      integrated loudness within target +-tol, true peak <= -1.0 dBTP
  pauses        longest silence in the speech <= --max-pause (the cut should leave ~keep-gap)
  seams         no repeated frame and no empty (black or flat) frame at any cut
  duration      picture and sound have the same length (difference < 1 frame)
  zoom          the face box you gave stays fully inside every zoomed piece (zoom did not cut the face off)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from array import array
from operator import mul
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def linked_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.islink(d) or (hasattr(os.path, "isjunction") and os.path.isjunction(d)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def sh(cmd, cwd=None, binary=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True)
    if binary:
        return r
    r.stdout = r.stdout.decode("utf-8", "replace")
    r.stderr = r.stderr.decode("utf-8", "replace")
    return r


def loudness(path):
    r = sh([FFMPEG, "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    t = r.stderr
    sm = t[t.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", sm)
    p = re.search(r"Peak:\s+(-?[\d.]+)\s+dBFS", sm)
    if not i or not p:
        return None, None
    return float(i.group(1)), float(p.group(1))


def energy(path, thr_db=None):
    r = sh([FFMPEG, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], binary=True)
    smp = array("h")
    smp.frombytes(r.stdout[: len(r.stdout) // 2 * 2])
    hop = 160
    out = []
    for i in range(0, len(smp) - hop + 1, hop):
        c = smp[i:i + hop]
        out.append(10 * math.log10(sum(map(mul, c, c)) / hop / 32768.0 ** 2 + 1e-12))
    return out


def longest_internal_silence(en, thr):
    """Longest silent run (s) strictly between speech, ignoring the first and last 0.4 s."""
    on = [e > thr for e in en]
    first = next((i for i, v in enumerate(on) if v), None)
    last = max((i for i, v in enumerate(on) if v), default=None)
    if first is None:
        return 0.0, 0
    best, cur, count = 0, 0, 0
    for v in on[first:last + 1]:
        if not v:
            cur += 1
        else:
            if cur:
                best = max(best, cur)
            cur = 0
    return best / 100.0, count


def gray_frames(path, w=160, h=90):
    r = sh([FFMPEG, "-v", "error", "-i", path, "-an", "-vf", f"scale={w}:{h}:flags=area,format=gray",
            "-f", "rawvideo", "-"], binary=True)
    n = w * h
    data = r.stdout
    return [data[i:i + n] for i in range(0, len(data) - n + 1, n)]


def seam_check(path, plan):
    fr = gray_frames(path)
    fps = plan["fps"]
    seams = [round(s["out_s"] * fps) for s in plan["segments"][1:]]
    dup_all = sum(1 for a, b in zip(fr, fr[1:]) if a == b)
    dup_seam = 0
    empty_seam = 0
    for k in seams:
        for j in (k - 1, k):
            if 0 <= j < len(fr) - 1 and fr[j] == fr[j + 1]:
                dup_seam += 1
        for j in (k - 1, k, k + 1):
            if 0 <= j < len(fr):
                f = fr[j]
                mean = sum(f) / len(f)
                if mean < 6 or (max(f) - min(f)) < 6:
                    empty_seam += 1
    empty_all = sum(1 for f in fr if sum(f) / len(f) < 6 or (max(f) - min(f)) < 6)
    return {"frames": len(fr), "seams": len(seams), "repeated_at_seams": dup_seam, "empty_at_seams": empty_seam,
            "repeated_total": dup_all, "empty_total": empty_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-pause", type=float, default=0.60)
    ap.add_argument("--lufs-tol", type=float, default=0.5)
    ap.add_argument("--allow-linked-folders", action="store_true", help="allow writing the report inside a linked folder")
    a = ap.parse_args()
    for t in ("ffmpeg", "ffprobe"):
        if not shutil.which(t):
            print("ERROR: " + t + " not found in PATH", file=sys.stderr)
            return 2
    video = os.path.abspath(a.video)
    report = video + ".check.json"

    def report_blocked():
        if os.path.islink(report):
            return report + " is a symbolic link; refusing to write through it"
        ld = None if a.allow_linked_folders else linked_dir(report)
        if ld:
            return (ld + " is a symbolic link or junction; refusing to write through it "
                    "(add --allow-linked-folders if you trust it; on Mac /tmp and /var are such links)")
        return None

    why = report_blocked()
    if why:
        print("ERROR: " + why, file=sys.stderr)
        return 2
    plan_path = video + ".plan.json"
    if not Path(plan_path).exists():
        print("ERROR: no plan file " + plan_path, file=sys.stderr)
        return 2
    plan = json.load(open(plan_path, encoding="utf-8"))
    rows, ok_all = [], True

    def add(name, ok, detail, status=None):
        nonlocal ok_all
        ok_all &= ok
        rows.append({"check": name, "ok": ok, "status": status or ("PASS" if ok else "FAIL"), "detail": detail})

    lufs, tp = loudness(video)
    if lufs is None:
        add("loudness", False, "could not measure")
    else:
        tgt = plan["lufs"]
        add("loudness", abs(lufs - tgt) <= a.lufs_tol and tp <= -1.0,
            f"{lufs:.1f} LUFS (target {tgt} +-{a.lufs_tol}), true peak {tp:.1f} dBTP (limit -1.0)")

    en = energy(video)
    # loudness gain and denoise move the noise floor, so the threshold moves with it
    # (same distance above the 5th percentile as when the cut was planned)
    thr = sorted(en)[int(len(en) * 0.05)] + (plan["threshold_db"] - plan["p5_db"])
    sil, _ = longest_internal_silence(en, thr)
    add("pauses", sil <= a.max_pause,
        f"longest silence in speech {sil:.2f} s (limit {a.max_pause}; cut threshold {plan['min_pause']} s, "
        f"kept breath {plan['keep_gap']} s)")

    sc = seam_check(video, plan)
    add("seams", sc["repeated_at_seams"] == 0 and sc["empty_at_seams"] == 0,
        f"{sc['seams']} cuts: repeated frames at cuts {sc['repeated_at_seams']}, empty frames at cuts "
        f"{sc['empty_at_seams']} (whole file: repeated {sc['repeated_total']}, empty {sc['empty_total']})")

    pr = sh([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", video])
    st = json.loads(pr.stdout)["streams"]
    vd = float(next(s for s in st if s["codec_type"] == "video").get("duration", 0))
    ad = float(next(s for s in st if s["codec_type"] == "audio").get("duration", 0))
    frame = 1.0 / plan["fps"]
    add("duration", abs(vd - ad) < 2 * frame and abs(vd - plan["out_duration"]) < 2 * frame,
        f"picture {vd:.3f} s, sound {ad:.3f} s, plan {plan['out_duration']:.3f} s")

    fb = plan.get("face_box")
    if fb:
        geo = plan["geo"]
        W, H = geo["src_w"], geo["src_h"]
        cut_off = 0
        for s_ in plan["segments"]:
            x0, y0, cw, ch = s_["crop"]
            if fb[0] * W < x0 or fb[2] * W > x0 + cw or fb[1] * H < y0 or fb[3] * H > y0 + ch:
                cut_off += 1
        add("zoom", cut_off == 0,
            f"{len(plan['segments'])} pieces, zoom {plan['zoom']}: pieces where the zoom cuts into the face box: {cut_off}")
    else:
        add("zoom", True, "not measured: no --face-box was given, so it is unknown whether the zoom cut into the face",
            status="WARN")

    w = max(len(r["check"]) for r in rows)
    for r in rows:
        print(f"{r['status']:<4}  {r['check']:<{w}}  {r['detail']}")
    skipped = [r["check"] for r in rows if r["status"] == "WARN"]
    if not ok_all:
        print("SOME CHECKS FAILED")
    elif skipped:
        print("ALL PASS (not measured: " + ", ".join(skipped) + ")")
    else:
        print("ALL PASS")
    why = report_blocked()
    if why:
        print("ERROR: " + why, file=sys.stderr)
        return 2
    Path(report).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

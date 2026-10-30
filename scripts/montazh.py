#!/usr/bin/env python3
"""montazh-lite: cleans a talking-head video with ffmpeg + Python standard library only.

Three things, all local and free:
  1. pauses: finds pauses in the speech by sound energy and cuts them down;
  2. sound: rumble filter, light denoise, loudness (default -16 LUFS);
  3. zoom on the cuts: every second piece is punched in a little, so jump cuts look intentional.
The result is a clean video, ready for effects, subtitles and animation (a different skill does those).

Usage:
  python montazh.py doctor
  python montazh.py frame INPUT.mp4 --out frame.jpg [--at 0.5]
  python montazh.py run INPUT.mp4 --out OUT.mp4 [--face-box X0,Y0,X1,Y1] [--start S --end E] [--zoom 1.12]
                    [--min-pause 0.45] [--keep-gap 0.20] [--lufs -16] [--tp -2]
                    [--denoise off|light|strong] [--force] [--allow-linked-folders]
Writes OUT.mp4 and OUT.plan.json (read by check.py).

--face-box is the face rectangle in the SOURCE frame as fractions 0..1 (left,top,right,bottom).
Look at one frame and estimate it by eye; it is used to aim the zoom at the face.
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
import tempfile
from array import array
from operator import mul
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# full paths: the same program is used for every call (and a fake one can stand in during tests)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

INSTALL_HELP = ("Install the full ffmpeg build (Windows: winget install Gyan.FFmpeg; Mac: brew install ffmpeg), "
                "then restart the terminal.")


def die(msg: str, code: int = 2) -> None:
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str], cwd: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=capture, text=capture,
                          encoding="utf-8" if capture else None, errors="replace")


def probe(path: str) -> dict:
    r = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    if r.returncode != 0:
        die("ffprobe cannot read " + path + ": " + r.stderr[-300:])
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if not v or not a:
        die("input needs a video stream and an audio stream")
    try:
        n, d = str(v.get("r_frame_rate", "0/0")).split("/")
        fps = float(n) / float(d)
        duration = float(info["format"]["duration"])
        if not (fps > 0 and duration > 0):
            raise ValueError
    except (ValueError, ZeroDivisionError, KeyError):
        die("ffprobe gives no frame rate or duration for this file, so it cannot be cut. "
            "Convert it first to a normal mp4 or mov.")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {"w": w, "h": h, "fps": fps, "duration": duration,
            "rotation": rot, "has_rotation": rot != 0}


# ---------------------------------------------------------------- sound energy

def read_energy(path: str, start: float, dur: float) -> list[float]:
    """10 ms frames of RMS in dB (16 kHz mono)."""
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
           "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        die("cannot read audio: " + r.stderr.decode("utf-8", "replace")[-300:])
    smp = array("h")
    smp.frombytes(r.stdout[: len(r.stdout) // 2 * 2])
    hop = 160
    out = []
    for i in range(0, len(smp) - hop + 1, hop):
        ch = smp[i:i + hop]
        ms = sum(map(mul, ch, ch)) / hop
        out.append(10.0 * math.log10(ms / (32768.0 ** 2) + 1e-12))
    return out


def p5_db(energy: list[float]) -> float:
    """Noise floor estimate: 5th percentile of the 10 ms energies."""
    return sorted(energy)[int(len(energy) * 0.05)]


def threshold_db(energy: list[float]) -> float:
    return max(-52.0, min(-36.0, math.ceil(p5_db(energy)) + 20.0))


def voiced_blocks(energy: list[float], thr: float) -> list[tuple[int, int]]:
    """Blocks of speech in 10 ms units: [a, b). Small holes are filled, tiny islands dropped."""
    on = [e > thr for e in energy]
    blocks: list[list[int]] = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            blocks.append([i, j])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    for b in blocks:
        if merged and b[0] - merged[-1][1] < 6:  # holes under 60 ms (plosives)
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, b) for a, b in merged if b - a >= 4]  # islands under 40 ms are noise


def silence_runs(energy: list[float], thr: float) -> list[tuple[float, float]]:
    """Silent runs (seconds) between speech blocks, including the edges."""
    n = len(energy)
    bl = voiced_blocks(energy, thr)
    runs = []
    prev = 0
    for a, b in bl:
        if a > prev:
            runs.append((prev / 100.0, a / 100.0))
        prev = b
    if prev < n:
        runs.append((prev / 100.0, n / 100.0))
    return runs


# ---------------------------------------------------------------- cut plan

def plan_cuts(energy: list[float], thr: float, dur: float, fps: float,
              min_pause: float, keep_gap: float) -> list[dict]:
    """Keep-segments in frames of the (trimmed) input. Returns list of {a, b} frame ranges, b exclusive."""
    nframes = int(dur * fps)
    pad = keep_gap / 2.0
    bl = voiced_blocks(energy, thr)
    if not bl:
        die("no speech found in the chosen range (sound too quiet?)")
    removes: list[tuple[float, float]] = []
    lead_keep, tail_keep = 0.12, 0.25
    first_s = bl[0][0] / 100.0
    last_e = bl[-1][1] / 100.0
    if first_s > lead_keep + 0.10:
        removes.append((0.0, first_s - lead_keep))
    for (a0, b0), (a1, b1) in zip(bl, bl[1:]):
        ps, pe = b0 / 100.0, a1 / 100.0
        if pe - ps >= min_pause and (pe - ps) - 2 * pad >= 0.10:
            removes.append((ps + pad, pe - pad))
    if dur - last_e > tail_keep + 0.10:
        removes.append((last_e + tail_keep, dur))
    cuts = []
    for rs, re_ in removes:
        fa, fb = math.ceil(rs * fps), math.floor(re_ * fps)
        if fb - fa >= 2:
            cuts.append((fa, fb))
    segs = []
    pos = 0
    for fa, fb in cuts:
        if fa > pos:
            segs.append([pos, fa])
        pos = fb
    if pos < nframes:
        segs.append([pos, nframes])
    # glue segments shorter than 6 frames to the previous one (no flicker)
    out: list[list[int]] = []
    for s in segs:
        if out and s[1] - s[0] < 6:
            out[-1][1] = s[1]
        else:
            out.append(s)
    return [{"a": a, "b": b} for a, b in out]


# ---------------------------------------------------------------- geometry

def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def build_geometry(info: dict, zoom: float, face: tuple[float, float, float, float] | None,
                   segs: list[dict], fps: float) -> dict:
    W, H = info["w"], info["h"]
    outw, outh = (W, H) if W <= 1920 else (even(1920), even(H * 1920 / W))
    outw, outh = even(outw), even(outh)
    fx = (face[0] + face[2]) / 2 * W if face else W / 2
    fy = (face[1] + face[3]) / 2 * H if face else H * 0.42
    prev_scale = 1.0
    for i, s in enumerate(segs):
        dur = (s["b"] - s["a"]) / fps
        if i == 0:
            scale = 1.0
        elif dur < 0.7:
            scale = prev_scale
        else:
            scale = zoom if prev_scale == 1.0 else 1.0
        s["scale"] = scale
        prev_scale = scale
        cw, ch = even(W / scale), even(H / scale)
        x0 = min(max(fx - cw / 2, 0), W - cw)
        y0 = min(max(fy - ch / 2, 0), H - ch)
        s["crop"] = [even(x0), even(y0), cw, ch]
    return {"src_w": W, "src_h": H, "out_w": outw, "out_h": outh}


# ---------------------------------------------------------------- ffmpeg stages

def audio_filter_graph(segs: list[dict], fps: float, denoise: str) -> str:
    n = len(segs)
    parts = [f"[0:a:0]asplit={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, s in enumerate(segs):
        a, b = s["a"] / fps, s["b"] / fps
        d = b - a
        fd = min(0.006, d / 4)
        parts.append(f"[s{i}]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={fd:.4f},afade=t=out:st={d - fd:.6f}:d={fd:.4f}[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[cat]")
    chain = "highpass=f=70"
    if denoise == "light":
        chain += ",afftdn=nr=8:nf=-50"
    elif denoise == "strong":
        chain += ",afftdn=nr=16:nf=-45"
    parts.append(f"[cat]{chain}[aout]")
    return ";".join(parts)


def video_filter_graph(segs: list[dict], geo: dict) -> str:
    n = len(segs)
    ow, oh = geo["out_w"], geo["out_h"]
    parts = [f"[0:v:0]split={n}" + "".join(f"[v{i}]" for i in range(n))]
    for i, s in enumerate(segs):
        x0, y0, cw, ch = s["crop"]
        parts.append(f"[v{i}]trim=start_frame={s['a']}:end_frame={s['b']},setpts=PTS-STARTPTS,"
                     f"crop={cw}:{ch}:{x0}:{y0},scale={ow}:{oh}:flags=lanczos,setsar=1[c{i}]")
    tail = "".join(f"[c{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]"
    parts.append(tail)
    parts.append("[cat]format=yuv420p[vout]")
    return ";".join(parts)


def loudnorm_two_pass(wav_in: str, wav_out: str, lufs: float, tp: float, cwd: str) -> dict:
    base = f"loudnorm=I={lufs}:TP={tp}:LRA=11"
    r = run([FFMPEG, "-hide_banner", "-nostats", "-i", wav_in, "-af", base + ":print_format=json",
             "-f", "null", "-"], cwd=cwd)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        die("loudnorm pass 1 failed: " + r.stderr[-400:])
    st = json.loads(m.group(0))
    af = (base + f":measured_I={st['input_i']}:measured_TP={st['input_tp']}:measured_LRA={st['input_lra']}"
          f":measured_thresh={st['input_thresh']}:offset={st['target_offset']}:linear=true")
    r = run([FFMPEG, "-y", "-hide_banner", "-nostats", "-i", wav_in, "-af", af + ",aresample=48000",
             "-c:a", "pcm_s16le", wav_out], cwd=cwd)
    if r.returncode != 0:
        die("loudnorm pass 2 failed: " + r.stderr[-400:])
    return st


NEEDED_ENCODERS = ("libx264", "aac", "pcm_s16le", "mjpeg")
# used outside the two big graphs: loudnorm and aresample (sound pass), ebur128 and scale/format (check.py, frame)
EXTRA_FILTERS = ("loudnorm", "aresample", "ebur128", "scale", "format")


def graph_filter_names(graph: str) -> set[str]:
    """Filter names used in a filter_complex string (labels in [] dropped, chains split by , and ;)."""
    flat = re.sub(r"\[[^\]]*\]", "", graph)
    names = set()
    for part in re.split(r"[;,]", flat):
        name = part.split("=", 1)[0].strip()
        if name:
            names.add(name)
    return names


def needed_filters() -> list[str]:
    """Every filter the render builds: read from the real graph builders, so a new filter cannot be forgotten."""
    segs = [{"a": 0, "b": 50, "crop": [0, 0, 100, 100]}, {"a": 50, "b": 100, "crop": [0, 0, 100, 100]}]
    geo = {"out_w": 100, "out_h": 100}
    names = set(EXTRA_FILTERS)
    for denoise in ("light", "strong"):
        names |= graph_filter_names(audio_filter_graph(segs, 50.0, denoise))
    names |= graph_filter_names(video_filter_graph(segs, geo))
    return sorted(names)


def listed_names(text: str) -> set[str]:
    """Second column of `ffmpeg -filters` / `-encoders` lines: the real name, not text from the descriptions."""
    out = set()
    for line in text.splitlines():
        m = re.match(r"^\s*[A-Z.]{2,8}\s+(\S+)\s", line)
        if m:
            out.add(m.group(1))
    return out


def ffmpeg_problems() -> list[str]:
    """Names of missing filters and encoders in this ffmpeg build (empty list = ready)."""
    have_f = listed_names(run([FFMPEG, "-hide_banner", "-filters"]).stdout)
    have_e = listed_names(run([FFMPEG, "-hide_banner", "-encoders"]).stdout)
    problems = ["filter " + f for f in needed_filters() if f not in have_f]
    problems += ["encoder " + e for e in NEEDED_ENCODERS if e not in have_e]
    return problems


def do_doctor() -> int:
    ok = True
    for tool, path in (("ffmpeg", shutil.which("ffmpeg")), ("ffprobe", shutil.which("ffprobe"))):
        print(f"{tool}: {path or 'NOT FOUND'}")
        ok &= bool(path)
    print("python:", sys.version.split()[0], "(need 3.10+)")
    ok &= sys.version_info >= (3, 10)
    if shutil.which("ffmpeg"):
        missing = ffmpeg_problems()
        for name in [f"filter {f}" for f in needed_filters()] + [f"encoder {e}" for e in NEEDED_ENCODERS]:
            print(f"{name}: {'MISSING' if name in missing else 'ok'}")
        if missing:
            print("This ffmpeg build cannot make the video (H.264 video + AAC sound and the sound filters are "
                  "needed). " + INSTALL_HELP)
        ok &= not missing
    print("ALL OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def parse_face(s: str | None):
    if not s:
        return None
    try:
        v = [float(x) for x in s.split(",")]
        assert len(v) == 4 and 0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1
        return tuple(v)
    except Exception:
        die("--face-box must be four fractions: left,top,right,bottom (for example 0.35,0.12,0.65,0.55)")


def linked_dir(path: str) -> str | None:
    """First parent folder of path that is a symbolic link (or a Windows junction), else None."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.islink(d) or (hasattr(os.path, "isjunction") and os.path.isjunction(d)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def refuse_link(path: str, allow_dirs: bool = False) -> None:
    """Never write through a symbolic link (it could point at someone else's file or folder)."""
    if os.path.islink(path):
        die(f"{path} is a symbolic link; refusing to write through it. Choose another --out name "
            "(this is not overridden by --force).")
    ld = None if allow_dirs else linked_dir(path)
    if ld:
        die(f"the folder {ld} is a symbolic link or junction; refusing to write through it. Choose a normal "
            "folder, or add --allow-linked-folders if you trust it (on Mac /tmp and /var are such links).")


def guard_output(src: str, out: str, force: bool, allow_dirs: bool = False, sidecar: bool = True) -> None:
    """Never overwrite the input; never write through a link; never overwrite a result without --force."""
    mine = [out, out + ".plan.json"] if sidecar else [out]
    for target in mine:
        refuse_link(target, allow_dirs)
        try:
            clash = os.path.samefile(target, src) if os.path.exists(target) else False
        except OSError:
            clash = False
        if clash or os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(src)):
            die(f"--out would overwrite your original video ({src}). Choose another file name.")
    if not force:
        busy = [t for t in mine if os.path.exists(t)]
        if busy:
            die("result already exists: " + ", ".join(busy) + ". Choose another --out name or add --force.")


def do_run(a: argparse.Namespace) -> int:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(tool + " not found in PATH. " + INSTALL_HELP)
    missing = ffmpeg_problems()
    if missing:
        die("this ffmpeg build is missing: " + ", ".join(missing) + ". " + INSTALL_HELP)
    src = str(Path(a.input).resolve())
    if not Path(src).exists():
        die("input not found: " + src)
    out = os.path.abspath(a.out)  # not resolve(): a link must stay visible to guard_output
    guard_output(src, out, a.force, a.allow_linked_folders)
    info = probe(src)
    fps = info["fps"]
    t0 = a.start or 0.0
    t1 = min(a.end if a.end else info["duration"], info["duration"])
    dur = t1 - t0
    if dur <= 1.0:
        die("chosen range is too short")
    face = parse_face(a.face_box)
    print(f"[1/6] reading sound energy ({dur:.1f} s)")
    energy = read_energy(src, t0, dur)
    thr = threshold_db(energy)
    segs = plan_cuts(energy, thr, dur, fps, a.min_pause, a.keep_gap)
    geo = build_geometry(info, a.zoom, face, segs, fps)
    pos = 0.0
    for s in segs:
        s["out_s"] = pos
        pos += (s["b"] - s["a"]) / fps
    out_dur = pos
    print(f"      threshold {thr:.0f} dB, {len(segs)} pieces, {dur:.1f} s -> {out_dur:.1f} s")

    work = tempfile.mkdtemp(prefix="montazh-")
    try:
        # ---- sound
        print("[2/6] cutting and cleaning the sound")
        gaf = audio_filter_graph(segs, fps, a.denoise)
        (Path(work) / "a.txt").write_text(gaf, encoding="utf-8")
        r = run([FFMPEG, "-y", "-hide_banner", "-nostats", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
                 "-i", src, "-filter_complex_script", "a.txt", "-map", "[aout]", "-c:a", "pcm_s16le",
                 "-ar", "48000", "cut.wav"], cwd=work)
        if r.returncode != 0:
            die("sound cut failed: " + r.stderr[-500:])
        print("[3/6] loudness")
        st = loudnorm_two_pass("cut.wav", "norm.wav", a.lufs, a.tp, work)
        # ---- picture
        print("[4/6] picture: cuts and zoom")
        vgraph = video_filter_graph(segs, geo)
        (Path(work) / "v.txt").write_text(vgraph, encoding="utf-8")
        tmp_out = os.path.join(work, "out.mp4")
        cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
               "-i", src, "-i", "norm.wav", "-filter_complex_script", "v.txt", "-map", "[vout]", "-map", "1:a:0",
               "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", tmp_out]
        r = run(cmd, cwd=work)
        if r.returncode != 0:
            die("picture render failed: " + r.stderr[-800:])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        for target in (out, out + ".plan.json"):
            refuse_link(target, a.allow_linked_folders)
        shutil.move(tmp_out, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    plan = {"input": Path(src).name, "start": t0, "end": t1, "fps": fps, "threshold_db": thr,
            "p5_db": p5_db(energy), "min_pause": a.min_pause, "keep_gap": a.keep_gap, "geo": geo, "zoom": a.zoom,
            "face_box": face, "lufs": a.lufs, "tp": a.tp, "denoise": a.denoise,
            "loudnorm_pass1": st, "segments": segs, "out_duration": out_dur}
    refuse_link(out + ".plan.json", a.allow_linked_folders)
    Path(out + ".plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[5/6] done: {out}")
    # No ready-made shell line on purpose: file names can hold characters a shell would run.
    print("[6/6] Check: run check.py with these arguments, one per line (pass them as a list, "
          "do not join them into a shell command line):")
    print("    " + sys.executable)
    print("    " + os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.py"))
    print("    " + out)
    return 0


def do_frame(a: argparse.Namespace) -> int:
    """Save one frame at a fraction of the clip length (works for any clip length)."""
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(tool + " not found in PATH. " + INSTALL_HELP)
    src = str(Path(a.input).resolve())
    if not Path(src).exists():
        die("input not found: " + src)
    out = os.path.abspath(a.out)
    guard_output(src, out, a.force, a.allow_linked_folders, sidecar=False)
    if not 0.0 <= a.at <= 1.0:
        die("--at must be between 0 and 1 (0.5 = the middle of the clip)")
    info = probe(src)
    t = max(0.0, min(info["duration"] * a.at, info["duration"] - 0.1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    r = run([FFMPEG, "-y", "-hide_banner", "-nostats", "-v", "error", "-ss", f"{t:.3f}", "-i", src,
             "-frames:v", "1", "-vf", "scale=960:-1", out])
    if r.returncode != 0 or not os.path.exists(out):
        die("could not save a frame: " + r.stderr[-300:])
    print(f"frame at {t:.1f} s of {info['duration']:.1f} s saved: {out}")
    return 0


def apply_taste(a: argparse.Namespace, sub: argparse.ArgumentParser) -> None:
    """taste.json holds your habits (min_pause, keep_gap, zoom, lufs, tp, denoise).
    A flag typed on the command line wins over the file; the file wins over built-in defaults."""
    path = Path(a.taste) if a.taste else Path("taste.json")
    if not path.exists():
        return
    try:
        taste = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"{path} is not valid JSON: {e}")
    used = []
    for key, val in taste.items():
        known = ("min_pause", "keep_gap", "zoom", "lufs", "tp", "denoise")
        if key in known and getattr(a, key) == sub.get_default(key):
            setattr(a, key, val)
            used.append(key)
    if used:
        print(f"      taste from {path}: " + ", ".join(f"{k}={getattr(a, k)}" for k in used))


def main() -> int:
    ap = argparse.ArgumentParser(description="montazh-lite: simple talking-head montage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check ffmpeg, ffprobe, Python")
    fr = sub.add_parser("frame", help="save one frame (to find the face for the zoom)")
    fr.add_argument("input")
    fr.add_argument("--out", required=True, help="a .jpg or .png file name")
    fr.add_argument("--at", type=float, default=0.5, help="position in the clip, 0..1 (default 0.5 = middle)")
    fr.add_argument("--force", action="store_true")
    fr.add_argument("--allow-linked-folders", action="store_true", help="allow writing inside a linked folder")
    r = sub.add_parser("run", help="make the montage")
    r.add_argument("input")
    r.add_argument("--out", required=True)
    r.add_argument("--face-box", help="left,top,right,bottom as fractions of the source frame")
    r.add_argument("--start", type=float, default=0.0)
    r.add_argument("--end", type=float, default=0.0)
    r.add_argument("--zoom", type=float, default=1.12)
    r.add_argument("--min-pause", type=float, default=0.45, help="pauses at least this long (s) are cut")
    r.add_argument("--keep-gap", type=float, default=0.20, help="breath left in a cut pause (s)")
    r.add_argument("--lufs", type=float, default=-16.0)
    r.add_argument("--tp", type=float, default=-2.0, help="true-peak ceiling before AAC (AAC adds up to ~0.3 dB)")
    r.add_argument("--denoise", choices=["off", "light", "strong"], default="light")
    r.add_argument("--force", action="store_true", help="overwrite an existing result file")
    r.add_argument("--allow-linked-folders", action="store_true", help="allow writing inside a linked folder")
    r.add_argument("--taste", help="JSON with your preferences (default: taste.json in the current folder, if any)")
    a = ap.parse_args()
    if a.cmd == "run":
        apply_taste(a, r)
    if a.cmd == "doctor":
        return do_doctor()
    if a.cmd == "frame":
        return do_frame(a)
    return do_run(a)


if __name__ == "__main__":
    sys.exit(main())

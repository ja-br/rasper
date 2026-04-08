#!/usr/bin/env python3
"""
rasper — a terminal 16-step sequencer with sine / square / sawtooth
oscillators and per-step velocity.

Audio is generated with numpy and streamed to `aplay` (alsa-utils) via a
subprocess pipe. The UI is curses-based.

Run:  python3 sequencer.py
Quit: q
"""

import curses
import datetime
import fcntl
import json
import math
import os
import shutil
import sqlite3
import subprocess
import threading
import time

import numpy as np

SAMPLE_RATE = 44100
NUM_STEPS = 16
MIN_NOTE = 24    # C1  (~32.7 Hz)
MAX_NOTE = 108   # C8  (~4186 Hz)
MIN_BPM = 30
MAX_BPM = 300
DEFAULT_BPM = 120
DEFAULT_NOTE = 69  # A4 = 440 Hz
DEFAULT_VEL = 1.0
MIN_VEL = 0.0
MAX_VEL = 1.0
VEL_STEP = 0.05
DEFAULT_ATTACK = 0.05  # fraction of step duration
DEFAULT_DECAY = 0.30
MIN_ENV = 0.0
MAX_ENV = 1.0
ENV_STEP = 0.05
DEFAULT_OFFSET = 0.0
OFFSET_MAX = 0.40   # ±40 % of step duration
OFFSET_STEP = 0.02  # 2 % per keypress

MIN_CUTOFF = 30.0
MAX_CUTOFF = 20000.0
DEFAULT_CUTOFF = MAX_CUTOFF  # default = wide open (filter is bypassed)
CUTOFF_FACTOR = 1.15         # multiplier per keypress (log scale)
MIN_Q = 0.5
MAX_Q = 12.0
DEFAULT_Q = 0.707            # Butterworth response
Q_STEP = 0.25

AMPLITUDE = 0.3  # 0..1, global master gain

CHUNK_SIZE = 2048           # samples per mixer chunk (~46 ms at 44.1 kHz)
PIPE_BUFFER_BYTES = 16384   # OS pipe size — ~186 ms of jitter absorption
ALSA_BUFFER_US = 120000     # 120 ms ALSA ring buffer
ALSA_PERIOD_US = 25000      # 25 ms ALSA period
TRACK_KINDS = ("osc", "noise")

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def biquad_lowpass(samples, cutoff, q, sample_rate=SAMPLE_RATE):
    """Robert Bristow-Johnson lowpass biquad applied sample-by-sample.

    `cutoff` is in Hz, `q` is the resonance (Butterworth at ~0.707, sharper
    peaking as q grows). Returns a new float64 buffer the same length as
    `samples`. Bypasses (returns input unchanged) when cutoff is at or
    above the configured maximum so the default state is free.
    """
    n = len(samples)
    if n == 0:
        return samples
    nyquist = 0.5 * sample_rate
    if cutoff >= MAX_CUTOFF or cutoff >= nyquist * 0.99:
        return samples  # bypass — wide-open default
    if cutoff < 1.0:
        cutoff = 1.0

    omega = 2.0 * math.pi * cutoff / sample_rate
    cos_w = math.cos(omega)
    sin_w = math.sin(omega)
    alpha = sin_w / (2.0 * max(q, 0.001))

    a0 = 1.0 + alpha
    b0 = (1.0 - cos_w) * 0.5 / a0
    b1 = (1.0 - cos_w) / a0
    b2 = b0
    a1 = -2.0 * cos_w / a0
    a2 = (1.0 - alpha) / a0

    out = np.empty(n, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    # Tight per-sample loop. Biquad is inherently recursive, so this
    # cannot be vectorized through numpy.
    for i in range(n):
        x = float(samples[i])
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y
        x2 = x1
        x1 = x
        y2 = y1
        y1 = y
    return out


def build_envelope(n, attack, decay):
    """Trapezoidal AD envelope across `n` samples.

    `attack` and `decay` are fractions of the step duration in [0, 1].
    A small click-guard floor (~1.5 ms) is enforced so that even with
    attack=decay=0 the wave starts and ends at zero.
    """
    if n <= 0:
        return np.zeros(0)
    guard = min(64, max(1, n // 16))  # ~1.5 ms at 44.1 kHz
    a = max(guard, int(round(n * attack)))
    d = max(guard, int(round(n * decay)))
    if a + d > n:
        scale = n / float(a + d)
        a = max(1, int(a * scale))
        d = max(1, n - a)
    sustain = n - a - d
    env = np.empty(n, dtype=np.float64)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if sustain > 0:
        env[a:a + sustain] = 1.0
    if d > 0:
        env[a + sustain:] = np.linspace(1.0, 0.0, d)
    return env

WAVES = ("sine", "square", "saw", "white", "pink")
# Per-waveform gain to keep perceived loudness roughly matched to sine.
WAVE_GAIN = {
    "sine": 1.0, "square": 0.55, "saw": 0.8,
    "white": 0.5, "pink": 0.7,
}
WAVE_LABEL = {
    "sine": "sin ", "square": "sqr ", "saw": "saw ",
    "white": "wht ", "pink": "pnk ",
}
TRACK_WAVES = {
    "osc":   ("sine", "square", "saw"),
    "noise": ("white", "pink"),
}

# Color pair indices.
PAIR_TITLE = 1
PAIR_LABEL = 2
PAIR_BPM = 3
PAIR_PLAY = 4
PAIR_STOP = 5
PAIR_STEP_NUM = 6
PAIR_ON = 7
PAIR_OFF = 8
PAIR_SINE = 9
PAIR_SQUARE = 10
PAIR_SAW = 11
PAIR_NOTE = 12
PAIR_VEL_LO = 13
PAIR_VEL_MID = 14
PAIR_VEL_HI = 15
PAIR_PLAYHEAD = 16
PAIR_HELP = 17
PAIR_HELP_KEY = 18
PAIR_ATTACK = 19
PAIR_DECAY = 20
PAIR_OFFSET = 21
PAIR_NOISE_WHITE = 22
PAIR_NOISE_PINK = 23
PAIR_TRACK_ACTIVE = 24
PAIR_TRACK_DIM = 25
PAIR_CUTOFF = 26
PAIR_RESONANCE = 27

WAVE_PAIR = {
    "sine":   PAIR_SINE,
    "square": PAIR_SQUARE,
    "saw":    PAIR_SAW,
    "white":  PAIR_NOISE_WHITE,
    "pink":   PAIR_NOISE_PINK,
}


def init_colors():
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(PAIR_TITLE, curses.COLOR_MAGENTA, bg)
    curses.init_pair(PAIR_LABEL, curses.COLOR_BLUE, bg)
    curses.init_pair(PAIR_BPM, curses.COLOR_YELLOW, bg)
    curses.init_pair(PAIR_PLAY, curses.COLOR_GREEN, bg)
    curses.init_pair(PAIR_STOP, curses.COLOR_RED, bg)
    curses.init_pair(PAIR_STEP_NUM, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_ON, curses.COLOR_GREEN, bg)
    curses.init_pair(PAIR_OFF, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_SINE, curses.COLOR_CYAN, bg)
    curses.init_pair(PAIR_SQUARE, curses.COLOR_YELLOW, bg)
    curses.init_pair(PAIR_SAW, curses.COLOR_MAGENTA, bg)
    curses.init_pair(PAIR_NOTE, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_VEL_LO, curses.COLOR_GREEN, bg)
    curses.init_pair(PAIR_VEL_MID, curses.COLOR_YELLOW, bg)
    curses.init_pair(PAIR_VEL_HI, curses.COLOR_RED, bg)
    curses.init_pair(PAIR_PLAYHEAD, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_HELP, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_HELP_KEY, curses.COLOR_CYAN, bg)
    curses.init_pair(PAIR_ATTACK, curses.COLOR_CYAN, bg)
    curses.init_pair(PAIR_DECAY, curses.COLOR_MAGENTA, bg)
    curses.init_pair(PAIR_OFFSET, curses.COLOR_YELLOW, bg)
    curses.init_pair(PAIR_NOISE_WHITE, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_NOISE_PINK, curses.COLOR_MAGENTA, bg)
    curses.init_pair(PAIR_TRACK_ACTIVE, curses.COLOR_GREEN, bg)
    curses.init_pair(PAIR_TRACK_DIM, curses.COLOR_WHITE, bg)
    curses.init_pair(PAIR_CUTOFF, curses.COLOR_CYAN, bg)
    curses.init_pair(PAIR_RESONANCE, curses.COLOR_RED, bg)


def cp(n):
    return curses.color_pair(n)


def vel_pair(v):
    if v < 0.34:
        return PAIR_VEL_LO
    if v < 0.67:
        return PAIR_VEL_MID
    return PAIR_VEL_HI


PREVIEW_HEIGHT = 7
PREVIEW_CYCLES = 2
PREVIEW_TRACE_CHAR = "█"
PREVIEW_AXIS_CHAR = "─"


def white_noise(n):
    """Uniform white noise in [-1, 1]."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.random.uniform(-1.0, 1.0, n).astype(np.float64)


def pink_noise(n):
    """1/f-shaped noise via rfft bin scaling. Numpy-only, no scipy."""
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    white = np.random.uniform(-1.0, 1.0, n).astype(np.float64)
    spec = np.fft.rfft(white)
    k = np.arange(len(spec), dtype=np.float64)
    k[0] = 1.0  # avoid div by zero at DC
    spec /= np.sqrt(k)
    pink = np.fft.irfft(spec, n=n)
    peak = float(np.abs(pink).max())
    return pink / peak if peak > 0.0 else pink


def _wave_samples(wave_type, n, cycles=PREVIEW_CYCLES):
    if wave_type == "white":
        return white_noise(n)
    if wave_type == "pink":
        return pink_noise(n)
    phase = np.linspace(0.0, float(cycles), n, endpoint=False)
    frac = phase - np.floor(phase)
    if wave_type == "sine":
        return np.sin(2.0 * np.pi * phase)
    if wave_type == "square":
        return np.where(frac < 0.5, 1.0, -1.0)
    if wave_type == "saw":
        return 2.0 * frac - 1.0
    return np.zeros(n)


def draw_preview(stdscr, y, x, width, step):
    """Render a small ASCII oscilloscope of `step`'s waveform."""
    if width < 8 or PREVIEW_HEIGHT < 3:
        return
    wave_type = step["wave"]
    color = WAVE_PAIR.get(wave_type, PAIR_SINE)

    if "note" in step:
        note_name = midi_to_name(step["note"])
        freq_hz = midi_to_freq(step["note"])
        pitch_bit = f"note {note_name} ({freq_hz:.1f} Hz)  "
    else:
        pitch_bit = ""
    cutoff = step.get("cutoff", DEFAULT_CUTOFF)
    if cutoff >= MAX_CUTOFF * 0.999:
        cutoff_bit = "cutoff open"
    elif cutoff >= 1000.0:
        cutoff_bit = f"cutoff {cutoff/1000:.1f} kHz"
    else:
        cutoff_bit = f"cutoff {int(round(cutoff))} Hz"
    header = (
        f"Preview: {wave_type}  "
        f"{pitch_bit}"
        f"vel {int(round(step['vel'] * 100))}%  "
        f"atk {int(round(step['attack'] * 100))}%  "
        f"dec {int(round(step['decay'] * 100))}%  "
        f"off {int(round(step['offset'] * 100)):+d}%  "
        f"{cutoff_bit}  "
        f"Q {step.get('resonance', DEFAULT_Q):.2f}"
    )
    try:
        stdscr.addstr(y, x, header[:width], cp(PAIR_LABEL) | curses.A_BOLD)
    except curses.error:
        pass

    box_top = y + 1
    h = PREVIEW_HEIGHT
    center = h // 2

    # Zero-crossing axis.
    axis_attr = cp(PAIR_LABEL) | curses.A_DIM
    for col in range(width):
        try:
            stdscr.addstr(box_top + center, x + col, PREVIEW_AXIS_CHAR, axis_attr)
        except curses.error:
            pass

    # Compute samples and scale by velocity so the preview reflects loudness.
    samples = _wave_samples(wave_type, width) * step["vel"]
    rows = ((1.0 - samples) / 2.0 * (h - 1)).round().astype(int)
    rows = np.clip(rows, 0, h - 1)

    trace_attr = cp(color) | curses.A_BOLD
    prev = None
    for col in range(width):
        row = int(rows[col])
        if prev is not None and prev != row:
            lo, hi = (prev, row) if prev < row else (row, prev)
            for r in range(lo, hi + 1):
                try:
                    stdscr.addstr(box_top + r, x + col, PREVIEW_TRACE_CHAR, trace_attr)
                except curses.error:
                    pass
        else:
            try:
                stdscr.addstr(box_top + row, x + col, PREVIEW_TRACE_CHAR, trace_attr)
            except curses.error:
                pass
        prev = row


def _new_step(kind):
    """Default step dict for a given track kind."""
    step = {
        "on": False,
        "wave": TRACK_WAVES[kind][0],
        "vel": DEFAULT_VEL,
        "attack": DEFAULT_ATTACK,
        "decay": DEFAULT_DECAY,
        "offset": DEFAULT_OFFSET,
        "cutoff": DEFAULT_CUTOFF,
        "resonance": DEFAULT_Q,
    }
    if kind == "osc":
        step["note"] = DEFAULT_NOTE
    return step


class Track:
    """One voice in the polyphonic mixer.

    Owns its own 16 steps and an internal sample-accurate playhead so the
    `Sequencer` mixer can pull arbitrary chunk sizes without coordinating
    step boundaries across tracks. This is what makes per-track microtiming
    work — track A and track B can have completely different step lengths
    yet stay aligned at the bar boundary because their per-bar totals match.
    """

    def __init__(self, kind):
        if kind not in TRACK_KINDS:
            raise ValueError(f"unknown track kind: {kind}")
        self.kind = kind
        self.steps = [_new_step(kind) for _ in range(NUM_STEPS)]
        # Mixer state. Lives under Sequencer.lock.
        self._cur_idx = 0
        self._cur_audio = None  # cached float buffer for the active step
        self._cur_pos = 0       # samples consumed from _cur_audio

    def reset_position(self):
        self._cur_idx = 0
        self._cur_audio = None
        self._cur_pos = 0

    def step_samples(self, idx, base):
        """Microtiming-aware step length.

        Same conservation property as before: an early step steals time
        from its predecessor and a late step donates to its successor, so
        the sum across the bar always equals 16 * base (± rounding).
        """
        this_off = self.steps[idx]["offset"]
        next_off = self.steps[(idx + 1) % NUM_STEPS]["offset"]
        return max(1, int(round(base * (1.0 + next_off - this_off))))

    def render_step_audio(self, idx, n):
        """Render this step into a float64 buffer in [-1, 1].

        Pipeline: oscillator -> biquad lowpass -> gain*velocity -> AD
        envelope. The filter sits between waveform generation and the
        amplitude shaping so it carves the raw oscillator output before
        the level/envelope decisions are made.

        Returns silence if the step is off or velocity is zero. Mixing
        and master gain happen upstream in `Sequencer.render_chunk`.
        """
        step = self.steps[idx]
        if not step["on"] or step["vel"] <= 0.0 or n <= 0:
            return np.zeros(max(0, n), dtype=np.float64)
        wave_type = step["wave"]
        if self.kind == "osc":
            freq = midi_to_freq(step.get("note", DEFAULT_NOTE))
            t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
            phase = freq * t
            if wave_type == "sine":
                wave = np.sin(2.0 * np.pi * phase)
            elif wave_type == "square":
                wave = np.where((phase - np.floor(phase)) < 0.5, 1.0, -1.0)
            elif wave_type == "saw":
                wave = 2.0 * (phase - np.floor(phase)) - 1.0
            else:
                wave = np.sin(2.0 * np.pi * phase)
        else:  # noise
            if wave_type == "white":
                wave = white_noise(n)
            elif wave_type == "pink":
                wave = pink_noise(n)
            else:
                wave = np.zeros(n, dtype=np.float64)
        # Per-step lowpass biquad. Bypassed automatically when cutoff is at
        # the configured maximum so the default state has no CPU cost.
        cutoff = step.get("cutoff", DEFAULT_CUTOFF)
        resonance = step.get("resonance", DEFAULT_Q)
        wave = biquad_lowpass(wave, cutoff, resonance)
        wave *= WAVE_GAIN.get(wave_type, 1.0) * step["vel"]
        wave *= build_envelope(n, step["attack"], step["decay"])
        return wave

    def emit(self, n_samples, bpm):
        """Pull `n_samples` of audio from the track's playhead."""
        base = SAMPLE_RATE * 60.0 / bpm / 4.0
        out = np.zeros(n_samples, dtype=np.float64)
        written = 0
        # Cap iterations defensively in case of degenerate (zero-length) steps.
        guard = NUM_STEPS * 4
        while written < n_samples and guard > 0:
            if self._cur_audio is None:
                length = self.step_samples(self._cur_idx, base)
                self._cur_audio = self.render_step_audio(self._cur_idx, length)
                self._cur_pos = 0
            avail = len(self._cur_audio) - self._cur_pos
            if avail <= 0:
                self._cur_audio = None
                self._cur_idx = (self._cur_idx + 1) % NUM_STEPS
                guard -= 1
                continue
            take = min(avail, n_samples - written)
            out[written:written + take] = (
                self._cur_audio[self._cur_pos:self._cur_pos + take]
            )
            self._cur_pos += take
            written += take
            if self._cur_pos >= len(self._cur_audio):
                self._cur_audio = None
                self._cur_idx = (self._cur_idx + 1) % NUM_STEPS
        return out


class Sequencer:
    """Polyphonic mixer hosting one or more tracks."""

    def __init__(self):
        self.tracks = [Track("osc"), Track("noise")]
        self.bpm = DEFAULT_BPM
        self.playing = False
        self.play_position = 0.0  # global wall-clock in samples for visual playhead
        self.play_step = 0
        # One-shot preview voice — a float64 buffer queued by `preview_step`
        # and consumed by `render_chunk`. Both fields live under self.lock.
        self.preview_buf = None
        self.preview_pos = 0
        self.lock = threading.Lock()
        self._proc = None
        self._thread = None

    def render_chunk(self, n_samples):
        """Mix all tracks (when playing) plus the preview voice. Returns int16."""
        with self.lock:
            bpm = self.bpm
            mix = np.zeros(n_samples, dtype=np.float64)
            if self.playing:
                for track in self.tracks:
                    mix += track.emit(n_samples, bpm)
                base = SAMPLE_RATE * 60.0 / bpm / 4.0
                bar = base * NUM_STEPS
                self.play_position = (self.play_position + n_samples) % bar
                self.play_step = int(self.play_position / base) % NUM_STEPS
            # Mix in the one-shot preview voice if one is queued. This works
            # in both playing and stopped modes — the preview overlays the
            # running sequence or plays through the silence.
            if self.preview_buf is not None:
                avail = len(self.preview_buf) - self.preview_pos
                take = min(avail, n_samples)
                if take > 0:
                    mix[:take] += self.preview_buf[
                        self.preview_pos:self.preview_pos + take
                    ]
                    self.preview_pos += take
                if self.preview_pos >= len(self.preview_buf):
                    self.preview_buf = None
                    self.preview_pos = 0
        # int16 scaling/clipping touches no shared state — do it without the lock
        # so the UI thread can grab seq.lock for step edits while we finish.
        scaled = mix * AMPLITUDE * 32767.0
        np.clip(scaled, -32767.0, 32767.0, out=scaled)
        return scaled.astype(np.int16)

    def _ensure_engine(self):
        """Make sure the audio thread + aplay subprocess are running.

        Idempotent and race-free with the dying audio thread: both this
        method and the audio thread's exit transition take `self.lock`,
        so we never see a half-shutdown state where `_thread.is_alive()`
        is True but the thread is about to drop our newly-queued preview.
        """
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Either no thread has ever run or the previous one already
            # marked itself gone under the lock. Spin up a fresh aplay
            # subprocess and a new audio thread.
            self._proc = subprocess.Popen(
                [
                    "aplay", "-q",
                    "-r", str(SAMPLE_RATE),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-B", str(ALSA_BUFFER_US),
                    "-F", str(ALSA_PERIOD_US),
                    "-",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                fcntl.fcntl(
                    self._proc.stdin, fcntl.F_SETPIPE_SZ, PIPE_BUFFER_BYTES
                )
            except (OSError, AttributeError):
                pass
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def start(self):
        if self.playing:
            return
        with self.lock:
            for track in self.tracks:
                track.reset_position()
            self.play_position = 0.0
            self.play_step = 0
        self.playing = True
        self._ensure_engine()

    def preview_step(self, track_idx, step_idx):
        """Queue a one-shot of the given step. Auto-starts the engine."""
        bpm = self.bpm
        n = max(1, int(round(SAMPLE_RATE * 60.0 / bpm / 4.0)))
        with self.lock:
            track = self.tracks[track_idx]
            audio = track.render_step_audio(step_idx, n)
            self.preview_buf = np.array(audio, dtype=np.float64, copy=True)
            self.preview_pos = 0
        self._ensure_engine()

    def _loop(self):
        # Local handle on the aplay process so the cleanup in `finally`
        # never accidentally closes a NEW subprocess that a racing
        # `_ensure_engine` may have spun up between our exit decision
        # and the finally block.
        local_proc = None
        try:
            while True:
                with self.lock:
                    if not self.playing and self.preview_buf is None:
                        # Mark this thread as gone *under the lock* so any
                        # racing preview_step → _ensure_engine sees the
                        # transition and starts a fresh thread instead of
                        # bailing on a stale `is_alive() == True` check.
                        local_proc = self._proc
                        self._proc = None
                        self._thread = None
                        return
                    proc_for_chunk = self._proc
                buf = self.render_chunk(CHUNK_SIZE)
                proc_for_chunk.stdin.write(buf.tobytes())
                proc_for_chunk.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            with self.lock:
                # Same hand-off on the error path.
                if local_proc is None:
                    local_proc = self._proc
                    self._proc = None
                self._thread = None
        finally:
            if local_proc is not None:
                try:
                    local_proc.stdin.close()
                except Exception:
                    pass
                try:
                    local_proc.terminate()
                    local_proc.wait(timeout=0.5)
                except Exception:
                    pass

    def stop(self):
        self.playing = False
        with self.lock:
            self.preview_buf = None
            self.preview_pos = 0
            self.play_step = 0
            self.play_position = 0.0
        # The audio thread sees both flags fall and exits via its lock-
        # protected exit transition, which hands off `_proc` to its
        # `finally` for clean teardown.


HELP = [
    "Controls:",
    "  left/right  Select step          space      Toggle step on/off",
    "  up/down     Note +/- semitone    PgUp/PgDn  Note +/- octave",
    "  - / =       Velocity -/+ 5%      [ / ]      BPM -/+ 5",
    "  a / s       Attack -/+ 5%        d / f      Decay -/+ 5%",
    "  , / .       Offset early/late    g / h      Cutoff -/+",
    "  j / k       Resonance -/+        w          Cycle wave",
    "  c / v       Copy / Paste step    Tab        Switch track",
    "  Enter       Audition step        p          Play / Stop",
    "  S           Save to library      L          Browse / load",
    "  q           Quit",
]


def draw(stdscr, seq, track_idx, cursor, clipboards=None, message=""):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    label_attr = cp(PAIR_LABEL) | curses.A_BOLD

    # Title row with track indicator on the right.
    title = "rasper"
    stdscr.addstr(0, 0, title, cp(PAIR_TITLE) | curses.A_BOLD)
    with seq.lock:
        track_count = len(seq.tracks)
        active_track = seq.tracks[track_idx]
        active_kind = active_track.kind
        steps = [dict(s) for s in active_track.steps]
        play_step = seq.play_step
        bpm = seq.bpm
    playing = seq.playing
    indicator = f"Track {track_idx + 1}/{track_count} [{active_kind}]"
    ind_x = 14
    if ind_x + len(indicator) < w:
        stdscr.addstr(0, ind_x, "Track ", cp(PAIR_LABEL))
        stdscr.addstr(
            0, ind_x + 6, f"{track_idx + 1}",
            cp(PAIR_TRACK_ACTIVE) | curses.A_BOLD,
        )
        stdscr.addstr(0, ind_x + 6 + len(str(track_idx + 1)),
                      f"/{track_count} ", cp(PAIR_LABEL))
        kind_color = (cp(PAIR_NOTE) if active_kind == "osc"
                      else cp(PAIR_NOISE_PINK))
        stdscr.addstr(
            0, ind_x + 6 + len(str(track_idx + 1)) + len(f"/{track_count} "),
            f"[{active_kind}]", kind_color | curses.A_BOLD,
        )

    # Status row.
    stdscr.addstr(1, 0, "BPM: ", cp(PAIR_LABEL))
    stdscr.addstr(1, 5, f"{bpm:>3}", cp(PAIR_BPM) | curses.A_BOLD)
    stdscr.addstr(1, 12, "Status: ", cp(PAIR_LABEL))
    if playing:
        stdscr.addstr(1, 20, "PLAYING", cp(PAIR_PLAY) | curses.A_BOLD)
    else:
        stdscr.addstr(1, 20, "STOPPED", cp(PAIR_STOP) | curses.A_BOLD)

    # Per-kind clipboard summary, scoped to the active track's kind.
    clip = (clipboards or {}).get(active_kind)
    stdscr.addstr(1, 30, "Clip: ", cp(PAIR_LABEL))
    if clip is None:
        stdscr.addstr(1, 36, "empty", cp(PAIR_LABEL) | curses.A_DIM)
    else:
        if active_kind == "osc":
            clip_summary = (
                f"{WAVE_LABEL[clip['wave']].strip()} "
                f"{midi_to_name(clip['note'])} "
                f"v{int(round(clip['vel'] * 100))}"
            )
        else:
            clip_summary = (
                f"{WAVE_LABEL[clip['wave']].strip()} "
                f"v{int(round(clip['vel'] * 100))}"
            )
        stdscr.addstr(
            1, 36, clip_summary[: max(0, w - 37)],
            cp(PAIR_HELP_KEY) | curses.A_BOLD,
        )

    if message:
        msg_x = 56
        if msg_x < w:
            stdscr.addstr(
                1, msg_x, message[: max(0, w - msg_x - 1)],
                cp(PAIR_HELP_KEY) | curses.A_BOLD,
            )

    base_y = 3
    label_x = 0
    grid_x = 8
    cell_w = 5

    stdscr.addstr(base_y + 0, label_x, "Step:", label_attr)
    stdscr.addstr(base_y + 1, label_x, "On:",   label_attr)
    stdscr.addstr(base_y + 2, label_x, "Wave:", label_attr)
    stdscr.addstr(base_y + 3, label_x, "Note:", label_attr)
    stdscr.addstr(base_y + 4, label_x, "Vel:",  label_attr)
    stdscr.addstr(base_y + 5, label_x, "Env:",  label_attr)
    stdscr.addstr(base_y + 6, label_x, "Off:",  label_attr)
    stdscr.addstr(base_y + 7, label_x, "Flt:",  label_attr)
    stdscr.addstr(base_y + 8, label_x, "Pos:",  label_attr)

    for i in range(NUM_STEPS):
        x = grid_x + i * cell_w
        if x + cell_w >= w:
            break
        sel = (i == cursor)
        # On rows other than the playhead, the cursor cell is highlighted
        # by combining A_REVERSE on top of the cell's color.
        sel_mod = curses.A_REVERSE if sel else 0
        is_playing_here = playing and i == play_step

        # Step number — bold beat numbers (1, 5, 9, 13).
        beat_attr = cp(PAIR_STEP_NUM) | sel_mod
        if i % 4 == 0:
            beat_attr |= curses.A_BOLD
        stdscr.addstr(base_y + 0, x, f"{i + 1:>3} ", beat_attr)

        # On / off cell.
        if steps[i]["on"]:
            on_attr = cp(PAIR_ON) | curses.A_BOLD | sel_mod
            stdscr.addstr(base_y + 1, x, "[X] ", on_attr)
        else:
            off_attr = cp(PAIR_OFF) | curses.A_DIM | sel_mod
            stdscr.addstr(base_y + 1, x, "[ ] ", off_attr)

        # Waveform — colored per type.
        wave_attr = cp(WAVE_PAIR[steps[i]["wave"]]) | curses.A_BOLD | sel_mod
        stdscr.addstr(base_y + 2, x, WAVE_LABEL[steps[i]["wave"]], wave_attr)

        # Note row — only meaningful for osc tracks; show "—" for noise.
        if active_kind == "osc":
            note_attr = cp(PAIR_NOTE) | sel_mod
            if not steps[i]["on"]:
                note_attr |= curses.A_DIM
            stdscr.addstr(
                base_y + 3, x,
                f"{midi_to_name(steps[i]['note']):<4} ", note_attr,
            )
        else:
            stdscr.addstr(
                base_y + 3, x, " --  ",
                cp(PAIR_LABEL) | curses.A_DIM | sel_mod,
            )

        # Velocity — green/yellow/red gradient.
        vel = steps[i]["vel"]
        vattr = cp(vel_pair(vel)) | curses.A_BOLD | sel_mod
        if not steps[i]["on"]:
            vattr |= curses.A_DIM
        stdscr.addstr(
            base_y + 4, x, f"{int(round(vel * 100)):>3} ", vattr
        )

        # Envelope — "AA/DD" with attack in cyan and decay in magenta.
        atk_pct = int(round(steps[i]["attack"] * 99))
        dec_pct = int(round(steps[i]["decay"] * 99))
        atk_attr = cp(PAIR_ATTACK) | curses.A_BOLD | sel_mod
        dec_attr = cp(PAIR_DECAY)  | curses.A_BOLD | sel_mod
        sep_attr = cp(PAIR_LABEL)  | curses.A_DIM  | sel_mod
        if not steps[i]["on"]:
            atk_attr |= curses.A_DIM
            dec_attr |= curses.A_DIM
        stdscr.addstr(base_y + 5, x + 0, f"{atk_pct:02d}", atk_attr)
        stdscr.addstr(base_y + 5, x + 2, "/",              sep_attr)
        stdscr.addstr(base_y + 5, x + 3, f"{dec_pct:02d}", dec_attr)

        # Microtiming offset — yellow when set, dim when zero.
        off_pct = int(round(steps[i]["offset"] * 100))
        if off_pct == 0:
            off_attr = cp(PAIR_LABEL) | curses.A_DIM | sel_mod
        else:
            off_attr = cp(PAIR_OFFSET) | curses.A_BOLD | sel_mod
        if not steps[i]["on"]:
            off_attr |= curses.A_DIM
        stdscr.addstr(base_y + 6, x, f"{off_pct:+3d}  ", off_attr)

        # Filter — "CC/QQ" with cutoff in cyan and resonance in red.
        # Dim when filter is wide open (default state).
        cut_pct = cutoff_percent(steps[i]["cutoff"])
        res_pct = resonance_percent(steps[i]["resonance"])
        wide_open = steps[i]["cutoff"] >= MAX_CUTOFF * 0.999
        cut_attr = cp(PAIR_CUTOFF) | curses.A_BOLD | sel_mod
        res_attr = cp(PAIR_RESONANCE) | curses.A_BOLD | sel_mod
        sep_attr2 = cp(PAIR_LABEL) | curses.A_DIM | sel_mod
        if wide_open or not steps[i]["on"]:
            cut_attr |= curses.A_DIM
            res_attr |= curses.A_DIM
        stdscr.addstr(base_y + 7, x + 0, f"{cut_pct:02d}", cut_attr)
        stdscr.addstr(base_y + 7, x + 2, "/",              sep_attr2)
        stdscr.addstr(base_y + 7, x + 3, f"{res_pct:02d}", res_attr)

        # Playhead — bright marker on the row below.
        if is_playing_here:
            stdscr.addstr(
                base_y + 8, x, " ^  ", cp(PAIR_PLAYHEAD) | curses.A_BOLD
            )

    # Waveform preview for the cursor's step.
    preview_y = base_y + 10
    preview_x = grid_x
    preview_width = min(NUM_STEPS * cell_w, max(0, w - preview_x - 1))
    preview_total_h = PREVIEW_HEIGHT + 1  # +1 for header line
    if preview_y + preview_total_h < h:
        draw_preview(stdscr, preview_y, preview_x, preview_width, steps[cursor])
        hy = preview_y + preview_total_h + 1
    else:
        hy = base_y + 7
    help_attr = cp(PAIR_HELP) | curses.A_DIM
    key_attr = cp(PAIR_HELP_KEY) | curses.A_BOLD
    for i, line in enumerate(HELP):
        if hy + i >= h:
            break
        line = line[: max(0, w - 1)]
        if i == 0:
            stdscr.addstr(hy + i, 0, line, label_attr)
            continue
        # Render the leading "  KEY  description" segments with the
        # key highlighted in cyan and the description dimmed.
        stdscr.move(hy + i, 0)
        col = 0
        # Split into "  KEY  TEXT     KEY  TEXT" — keys come at fixed columns.
        # Simpler: highlight the first run of non-space chars after the
        # leading indent, then again after the gap.
        x_pos = 0
        in_key = False
        # Find the two key columns: 2 and ~36 (matches HELP layout).
        for col_idx, ch in enumerate(line):
            if col_idx == 2 or col_idx == 36:
                in_key = True
            if in_key and ch == " ":
                in_key = False
            attr = key_attr if in_key else help_attr
            try:
                stdscr.addstr(hy + i, col_idx, ch, attr)
            except curses.error:
                pass

    stdscr.refresh()


def adjust_note(seq, track, idx, semitones):
    if track.kind != "osc":
        return
    with seq.lock:
        n = track.steps[idx]["note"] + semitones
        track.steps[idx]["note"] = max(MIN_NOTE, min(MAX_NOTE, n))


def adjust_env(seq, track, idx, field, delta):
    with seq.lock:
        v = track.steps[idx][field] + delta
        track.steps[idx][field] = max(MIN_ENV, min(MAX_ENV, v))


def adjust_offset(seq, track, idx, delta):
    with seq.lock:
        v = track.steps[idx]["offset"] + delta
        track.steps[idx]["offset"] = max(-OFFSET_MAX, min(OFFSET_MAX, v))


def adjust_cutoff(seq, track, idx, factor):
    """Multiplicative cutoff adjust so each press moves a fixed log interval."""
    with seq.lock:
        v = track.steps[idx]["cutoff"] * factor
        track.steps[idx]["cutoff"] = max(MIN_CUTOFF, min(MAX_CUTOFF, v))


def adjust_resonance(seq, track, idx, delta):
    with seq.lock:
        v = track.steps[idx]["resonance"] + delta
        track.steps[idx]["resonance"] = max(MIN_Q, min(MAX_Q, v))


def cutoff_percent(c):
    """Cutoff as a perceptually-linear (log-scale) percent 0..99."""
    if c <= MIN_CUTOFF:
        return 0
    if c >= MAX_CUTOFF:
        return 99
    span = math.log2(MAX_CUTOFF / MIN_CUTOFF)
    return int(round(math.log2(c / MIN_CUTOFF) / span * 99))


def resonance_percent(q):
    if q <= MIN_Q:
        return 0
    if q >= MAX_Q:
        return 99
    return int(round((q - MIN_Q) / (MAX_Q - MIN_Q) * 99))


def adjust_vel(seq, track, idx, delta):
    with seq.lock:
        v = track.steps[idx]["vel"] + delta
        track.steps[idx]["vel"] = max(MIN_VEL, min(MAX_VEL, v))


def toggle_step(seq, track, idx):
    with seq.lock:
        track.steps[idx]["on"] = not track.steps[idx]["on"]


def cycle_wave(seq, track, idx):
    """Cycle through the wave types valid for this track's kind."""
    waves = TRACK_WAVES[track.kind]
    with seq.lock:
        cur = track.steps[idx]["wave"]
        if cur not in waves:
            track.steps[idx]["wave"] = waves[0]
        else:
            track.steps[idx]["wave"] = waves[(waves.index(cur) + 1) % len(waves)]


PATTERN_VERSION = 2
DB_FILENAME = "library.db"
LEGACY_DIR = "patterns"


def rasper_data_dir():
    """Base data dir, created on demand. Honors XDG_DATA_HOME."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    path = os.path.join(base, "rasper")
    os.makedirs(path, exist_ok=True)
    return path


def db_path():
    return os.path.join(rasper_data_dir(), DB_FILENAME)


def sanitize_name(name):
    """Trim and bound a user-supplied pattern name."""
    name = name.strip()
    if name.lower().endswith(".json"):
        name = name[:-5]
    return name[:64]


def _parse_step(raw, kind):
    """Validate and clamp one step for a given track kind."""
    if not isinstance(raw, dict):
        raise ValueError("step entry must be an object")
    valid_waves = TRACK_WAVES[kind]
    wave = raw.get("wave", valid_waves[0])
    if wave not in valid_waves:
        wave = valid_waves[0]
    step = {
        "on": bool(raw.get("on", False)),
        "wave": wave,
        "vel": max(MIN_VEL, min(MAX_VEL,
            float(raw.get("vel", DEFAULT_VEL)))),
        "attack": max(MIN_ENV, min(MAX_ENV,
            float(raw.get("attack", DEFAULT_ATTACK)))),
        "decay": max(MIN_ENV, min(MAX_ENV,
            float(raw.get("decay", DEFAULT_DECAY)))),
        "offset": max(-OFFSET_MAX, min(OFFSET_MAX,
            float(raw.get("offset", DEFAULT_OFFSET)))),
        "cutoff": max(MIN_CUTOFF, min(MAX_CUTOFF,
            float(raw.get("cutoff", DEFAULT_CUTOFF)))),
        "resonance": max(MIN_Q, min(MAX_Q,
            float(raw.get("resonance", DEFAULT_Q)))),
    }
    if kind == "osc":
        step["note"] = max(MIN_NOTE, min(MAX_NOTE,
            int(raw.get("note", DEFAULT_NOTE))))
    return step


def _empty_track_steps(kind):
    return [_new_step(kind) for _ in range(NUM_STEPS)]


def _parse_pattern_doc(data):
    """Validate and clamp a pattern dict; returns (bpm, [(kind, steps), ...]).

    Accepts both v1 (flat `steps`) and v2 (`tracks`) schemas. v1 patterns
    are wrapped into a single osc track, with an empty noise track appended
    so the resulting Sequencer always has both voices.
    """
    if not isinstance(data, dict):
        raise ValueError("not a rasper pattern")
    bpm = max(MIN_BPM, min(MAX_BPM, int(data.get("bpm", DEFAULT_BPM))))
    version = int(data.get("version", 1))

    if version <= 1:
        # Legacy: flat steps array, all osc.
        if "steps" not in data:
            raise ValueError("not a rasper pattern")
        raw_steps = data["steps"]
        if not isinstance(raw_steps, list) or len(raw_steps) != NUM_STEPS:
            raise ValueError(f"expected {NUM_STEPS} steps, got {len(raw_steps)}")
        osc_steps = [_parse_step(s, "osc") for s in raw_steps]
        return bpm, [
            ("osc",   osc_steps),
            ("noise", _empty_track_steps("noise")),
        ]

    # v2+: explicit tracks
    raw_tracks = data.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("v2 pattern is missing tracks")
    parsed_tracks = []
    for raw_track in raw_tracks:
        if not isinstance(raw_track, dict):
            raise ValueError("track entry must be an object")
        kind = raw_track.get("kind", "osc")
        if kind not in TRACK_KINDS:
            kind = "osc"
        raw_steps = raw_track.get("steps", [])
        if not isinstance(raw_steps, list) or len(raw_steps) != NUM_STEPS:
            raise ValueError(
                f"track expected {NUM_STEPS} steps, got {len(raw_steps)}"
            )
        parsed_tracks.append((kind, [_parse_step(s, kind) for s in raw_steps]))
    return bpm, parsed_tracks


class PatternStore:
    """SQLite-backed pattern library with JSON import/export."""

    def __init__(self, path=None):
        self.path = path or db_path()
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                name        TEXT PRIMARY KEY,
                bpm         INTEGER NOT NULL,
                steps       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        self.conn.commit()
        self._migrate_legacy()

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def _migrate_legacy(self):
        """Auto-import any leftover JSON files from the old patterns/ dir."""
        legacy = os.path.join(os.path.dirname(self.path), LEGACY_DIR)
        if not os.path.isdir(legacy):
            return
        try:
            entries = os.listdir(legacy)
        except OSError:
            return
        for entry in entries:
            if not entry.lower().endswith(".json") or entry.startswith("."):
                continue
            name = entry[:-5]
            if self.exists(name):
                continue
            try:
                with open(os.path.join(legacy, entry), "r", encoding="utf-8") as f:
                    raw = json.load(f)
                bpm, tracks = _parse_pattern_doc(raw)
                self._write(name, bpm, tracks)
            except (OSError, ValueError, json.JSONDecodeError):
                continue  # skip malformed legacy files silently

    def _write(self, name, bpm, tracks):
        """Persist a pattern. `tracks` is the parser's list of (kind, steps)."""
        payload = {
            "tracks": [
                {"kind": kind, "steps": steps} for kind, steps in tracks
            ]
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO patterns (name, bpm, steps, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (name, bpm, json.dumps(payload),
             datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def exists(self, name):
        row = self.conn.execute(
            "SELECT 1 FROM patterns WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def list_names(self):
        rows = self.conn.execute(
            "SELECT name FROM patterns ORDER BY LOWER(name)"
        ).fetchall()
        return [r[0] for r in rows]

    def save_seq(self, name, seq):
        with seq.lock:
            tracks = [
                (track.kind, [dict(s) for s in track.steps])
                for track in seq.tracks
            ]
            bpm = seq.bpm
        self._write(name, bpm, tracks)

    def load_into_seq(self, name, seq):
        row = self.conn.execute(
            "SELECT bpm, steps FROM patterns WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"pattern {name!r} not found")
        bpm_raw, payload_json = row
        payload = json.loads(payload_json)
        doc = {
            "version": PATTERN_VERSION,
            "bpm": bpm_raw,
        }
        # The DB column may hold either v1 (flat "steps") or v2 (with "tracks").
        if isinstance(payload, dict) and "tracks" in payload:
            doc["tracks"] = payload["tracks"]
        elif isinstance(payload, dict) and "steps" in payload:
            doc["version"] = 1
            doc["steps"] = payload["steps"]
        elif isinstance(payload, list):
            # Legacy raw list of steps
            doc["version"] = 1
            doc["steps"] = payload
        else:
            raise ValueError("unrecognized pattern payload")
        bpm, parsed_tracks = _parse_pattern_doc(doc)
        with seq.lock:
            seq.bpm = bpm
            new_tracks = []
            for kind, steps in parsed_tracks:
                track = Track(kind)
                track.steps = steps
                new_tracks.append(track)
            seq.tracks = new_tracks

    def delete(self, name):
        self.conn.execute("DELETE FROM patterns WHERE name = ?", (name,))
        self.conn.commit()

    def export(self, name, path):
        row = self.conn.execute(
            "SELECT bpm, steps, updated_at FROM patterns WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"pattern {name!r} not found")
        bpm, payload_json, updated_at = row
        payload = json.loads(payload_json)
        if isinstance(payload, dict) and "tracks" in payload:
            tracks_field = payload["tracks"]
        else:
            # Legacy in-DB row: re-shape to v2 on export.
            doc_in = {"version": 1, "bpm": bpm,
                      "steps": payload.get("steps") if isinstance(payload, dict) else payload}
            _, parsed = _parse_pattern_doc(doc_in)
            tracks_field = [{"kind": k, "steps": s} for k, s in parsed]
        doc = {
            "version": PATTERN_VERSION,
            "name": name,
            "bpm": bpm,
            "updated_at": updated_at,
            "tracks": tracks_field,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)


def select_menu(stdscr, title, items):
    """Modal list selector. Returns (action, item) or None on cancel.

    action is one of: 'load', 'delete', 'export'.
    """
    if not items:
        return None
    selection = 0
    stdscr.nodelay(False)
    try:
        while True:
            h, w = stdscr.getmaxyx()
            stdscr.erase()
            stdscr.addstr(
                0, 0, title[: max(0, w - 1)],
                cp(PAIR_TITLE) | curses.A_BOLD,
            )
            stdscr.addstr(
                1, 0,
                "up/down select   enter load   e export   d delete   esc cancel",
                cp(PAIR_HELP) | curses.A_DIM,
            )

            list_top = 3
            max_visible = max(1, h - list_top - 1)
            top = max(0, selection - max_visible // 2)
            top = min(top, max(0, len(items) - max_visible))
            visible = items[top:top + max_visible]
            for row, item in enumerate(visible):
                idx = top + row
                if idx == selection:
                    attr = cp(PAIR_HELP_KEY) | curses.A_BOLD | curses.A_REVERSE
                    marker = "> "
                else:
                    attr = cp(PAIR_HELP)
                    marker = "  "
                line = f"{marker}{item}"
                stdscr.addstr(list_top + row, 2, line[: max(0, w - 3)], attr)

            if len(items) > max_visible:
                stdscr.addstr(
                    h - 1, 0,
                    f"{selection + 1}/{len(items)}",
                    cp(PAIR_HELP) | curses.A_DIM,
                )
            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP:
                selection = (selection - 1) % len(items)
            elif key == curses.KEY_DOWN:
                selection = (selection + 1) % len(items)
            elif key == curses.KEY_HOME:
                selection = 0
            elif key == curses.KEY_END:
                selection = len(items) - 1
            elif key in (10, 13, curses.KEY_ENTER):
                return ("load", items[selection])
            elif key == ord("e"):
                return ("export", items[selection])
            elif key == ord("d"):
                return ("delete", items[selection])
            elif key in (27, ord("q")):
                return None
    finally:
        stdscr.nodelay(True)


def prompt(stdscr, prompt_text):
    """Read a line at the bottom of the screen via curses echo mode."""
    h, w = stdscr.getmaxyx()
    y = h - 1
    stdscr.move(y, 0)
    stdscr.clrtoeol()
    stdscr.addstr(y, 0, prompt_text, cp(PAIR_HELP_KEY) | curses.A_BOLD)
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    stdscr.nodelay(False)
    try:
        raw = stdscr.getstr(y, len(prompt_text), max(1, w - len(prompt_text) - 1))
    except curses.error:
        raw = b""
    finally:
        curses.noecho()
        curses.curs_set(0)
        stdscr.nodelay(True)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.strip()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    init_colors()

    if not shutil.which("aplay"):
        stdscr.nodelay(False)
        stdscr.addstr(0, 0, "Error: `aplay` not found. Install alsa-utils.")
        stdscr.addstr(2, 0, "Press any key to exit.")
        stdscr.refresh()
        stdscr.getch()
        return

    seq = Sequencer()
    track_idx = 0
    cursors = [0] * len(seq.tracks)
    clipboards = {kind: None for kind in TRACK_KINDS}
    message = ""
    message_until = 0.0
    store = PatternStore()

    def cur_track():
        return seq.tracks[track_idx]

    try:
        while True:
            if message and time.monotonic() > message_until:
                message = ""
            # Cursors list may be stale after a Load swapped tracks; resync.
            if len(cursors) != len(seq.tracks):
                cursors = [0] * len(seq.tracks)
                track_idx = min(track_idx, len(seq.tracks) - 1)
            draw(stdscr, seq, track_idx, cursors[track_idx], clipboards, message)
            key = stdscr.getch()

            if key == -1:
                curses.napms(40)
                continue

            track = cur_track()
            cursor = cursors[track_idx]

            if key in (ord("q"), ord("Q")):
                break
            elif key == curses.KEY_LEFT:
                cursors[track_idx] = (cursor - 1) % NUM_STEPS
            elif key == curses.KEY_RIGHT:
                cursors[track_idx] = (cursor + 1) % NUM_STEPS
            elif key in (9, ord("\t")):  # Tab
                track_idx = (track_idx + 1) % len(seq.tracks)
            elif key == curses.KEY_BTAB:  # Shift-Tab
                track_idx = (track_idx - 1) % len(seq.tracks)
            elif key in (10, 13, curses.KEY_ENTER):
                seq.preview_step(track_idx, cursors[track_idx])
            elif key == ord(" "):
                toggle_step(seq, track, cursor)
            elif key == curses.KEY_UP:
                adjust_note(seq, track, cursor, 1)
            elif key == curses.KEY_DOWN:
                adjust_note(seq, track, cursor, -1)
            elif key == curses.KEY_PPAGE:
                adjust_note(seq, track, cursor, 12)
            elif key == curses.KEY_NPAGE:
                adjust_note(seq, track, cursor, -12)
            elif key == ord("["):
                with seq.lock:
                    seq.bpm = max(MIN_BPM, seq.bpm - 5)
            elif key == ord("]"):
                with seq.lock:
                    seq.bpm = min(MAX_BPM, seq.bpm + 5)
            elif key in (ord("w"), ord("W")):
                cycle_wave(seq, track, cursor)
            elif key in (ord("="), ord("+")):
                adjust_vel(seq, track, cursor, VEL_STEP)
            elif key in (ord("-"), ord("_")):
                adjust_vel(seq, track, cursor, -VEL_STEP)
            elif key == ord("a"):
                adjust_env(seq, track, cursor, "attack", -ENV_STEP)
            elif key == ord("s"):
                adjust_env(seq, track, cursor, "attack", ENV_STEP)
            elif key == ord("d"):
                adjust_env(seq, track, cursor, "decay", -ENV_STEP)
            elif key == ord("f"):
                adjust_env(seq, track, cursor, "decay", ENV_STEP)
            elif key in (ord(","), ord("<")):
                adjust_offset(seq, track, cursor, -OFFSET_STEP)
            elif key in (ord("."), ord(">")):
                adjust_offset(seq, track, cursor, OFFSET_STEP)
            elif key == ord("g"):
                adjust_cutoff(seq, track, cursor, 1.0 / CUTOFF_FACTOR)
            elif key == ord("h"):
                adjust_cutoff(seq, track, cursor, CUTOFF_FACTOR)
            elif key == ord("j"):
                adjust_resonance(seq, track, cursor, -Q_STEP)
            elif key == ord("k"):
                adjust_resonance(seq, track, cursor, Q_STEP)
            elif key in (ord("c"), ord("C")):
                with seq.lock:
                    clipboards[track.kind] = dict(track.steps[cursor])
            elif key in (ord("v"), ord("V")):
                clip = clipboards.get(track.kind)
                if clip is not None:
                    with seq.lock:
                        track.steps[cursor] = dict(clip)
            elif key == ord("S"):
                raw = prompt(stdscr, "Save name: ")
                name = sanitize_name(raw) if raw else ""
                if name:
                    try:
                        store.save_seq(name, seq)
                        message = f"Saved {name}"
                    except sqlite3.Error as e:
                        message = f"Save failed: {e}"
                    message_until = time.monotonic() + 3.0
                elif raw:
                    message = "Invalid name"
                    message_until = time.monotonic() + 3.0
            elif key == ord("L"):
                names = store.list_names()
                if not names:
                    message = f"Library empty ({store.path})"
                    message_until = time.monotonic() + 3.0
                else:
                    title = f"Library  —  {store.path}"
                    result = select_menu(stdscr, title, names)
                    if result is not None:
                        action, name = result
                        if action == "load":
                            try:
                                store.load_into_seq(name, seq)
                                message = f"Loaded {name}"
                            except (sqlite3.Error, ValueError) as e:
                                message = f"Load failed: {e}"
                        elif action == "export":
                            export_path = os.path.abspath(f"{name}.json")
                            try:
                                store.export(name, export_path)
                                message = f"Exported to {export_path}"
                            except (OSError, sqlite3.Error, ValueError) as e:
                                message = f"Export failed: {e}"
                        elif action == "delete":
                            confirm = prompt(stdscr, f"Delete '{name}'? (y/N): ")
                            if confirm.strip().lower().startswith("y"):
                                try:
                                    store.delete(name)
                                    message = f"Deleted {name}"
                                except sqlite3.Error as e:
                                    message = f"Delete failed: {e}"
                            else:
                                message = "Cancelled"
                        message_until = time.monotonic() + 4.0
            elif key in (ord("p"), ord("P")):
                if seq.playing:
                    seq.stop()
                else:
                    seq.start()
    finally:
        seq.stop()
        store.close()


if __name__ == "__main__":
    curses.wrapper(main)

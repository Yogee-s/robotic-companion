"""
Audio Input/Output.

Microphone capture via PyAudio (with arecord fallback for Jetson).
Speaker playback by piping PCM directly to aplay (no temp files).
"""

import logging
import os
import queue
import subprocess
import threading
import time

import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

logger = logging.getLogger(__name__)


class AudioInput:
    """
    Captures audio from the microphone.

    Tries PyAudio first; falls back to arecord (always works on Jetson).
    Audio is delivered as float32 chunks at 16 kHz via a thread-safe queue.
    """

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_size = config.get("chunk_size", 512)
        self.device_name = config.get("input_device_name", "ReSpeaker")

        self._pa = None
        self._stream = None
        self._running = False
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._thread = None
        self._device_index = None
        self._use_arecord = False
        self._arecord_proc = None

        if pyaudio is not None:
            try:
                # Suppress noisy ALSA warnings during device enumeration
                import ctypes, os as _os
                _libc = ctypes.cdll.LoadLibrary("libc.so.6")
                _devnull = _os.open(_os.devnull, _os.O_WRONLY)
                _old_stderr = _os.dup(2)
                _os.dup2(_devnull, 2)
                try:
                    self._pa = pyaudio.PyAudio()
                finally:
                    _os.dup2(_old_stderr, 2)
                    _os.close(_devnull)
                    _os.close(_old_stderr)
                self._device_index = self._find_device()
            except Exception as e:
                logger.warning(f"PyAudio init failed: {e}. Will use arecord.")
                self._pa = None

    def _find_device(self) -> int | None:
        """Find audio input device by name substring."""
        if self._pa is None:
            return None

        info = self._pa.get_host_api_info_by_index(0)
        for i in range(info.get("deviceCount", 0)):
            dev = self._pa.get_device_info_by_index(i)
            name = dev.get("name", "")
            if dev.get("maxInputChannels", 0) > 0:
                if self.device_name and self.device_name.lower() in name.lower():
                    logger.info(f"Found input device: {name} (index {i})")
                    return i

        logger.warning(f"Device '{self.device_name}' not found. Using default.")
        return None

    def start(self):
        """Start capturing audio."""
        if self._running:
            return
        self._running = True

        # Try PyAudio
        if self._pa is not None:
            try:
                kwargs = {
                    "format": pyaudio.paInt16,
                    "channels": self.channels,
                    "rate": self.sample_rate,
                    "input": True,
                    "frames_per_buffer": self.chunk_size,
                }
                if self._device_index is not None:
                    kwargs["input_device_index"] = self._device_index

                self._stream = self._pa.open(**kwargs)
                self._thread = threading.Thread(
                    target=self._capture_pyaudio, daemon=True
                )
                self._thread.start()
                logger.info("Audio input started (PyAudio).")
                return
            except Exception as e:
                logger.warning(f"PyAudio open failed: {e}. Falling back to arecord.")
                self._stream = None

        # Fallback: arecord
        self._use_arecord = True
        self._thread = threading.Thread(target=self._capture_arecord, daemon=True)
        self._thread.start()
        logger.info("Audio input started (arecord).")

    def _capture_pyaudio(self):
        """Continuous capture via PyAudio."""
        while self._running:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._enqueue(audio)
            except Exception as e:
                if self._running:
                    logger.error(f"Audio capture error: {e}")
                break

    def _capture_arecord(self):
        """Continuous capture via arecord subprocess."""
        try:
            alsa_dev = self._detect_alsa_input()
            cmd = [
                "arecord", "-q", "-f", "S16_LE",
                "-r", str(self.sample_rate),
                "-c", str(self.channels), "-t", "raw",
            ]
            if alsa_dev:
                cmd.extend(["-D", alsa_dev])
                logger.info(f"arecord using device: {alsa_dev}")

            self._arecord_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            bytes_per_chunk = self.chunk_size * 2 * self.channels

            while self._running and self._arecord_proc.poll() is None:
                data = self._arecord_proc.stdout.read(bytes_per_chunk)
                if not data:
                    break
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._enqueue(audio)
        except Exception as e:
            if self._running:
                logger.error(f"arecord error: {e}")

    @staticmethod
    def _detect_alsa_input() -> str | None:
        """Detect ReSpeaker ALSA card name."""
        try:
            with open("/proc/asound/cards") as f:
                text = f.read()
            if "ArrayUAC10" in text or "ReSpeaker" in text:
                return "plughw:ArrayUAC10,0"
        except Exception:
            pass
        return None

    def _enqueue(self, audio: np.ndarray):
        """Put chunk in queue, dropping oldest if full."""
        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.put_nowait(audio)

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read one float32 audio chunk. Returns None on timeout."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_level(self) -> float:
        """Get current audio RMS level (0.0 - 1.0)."""
        try:
            chunk = self._audio_queue.queue[-1] if not self._audio_queue.empty() else None
            if chunk is not None:
                rms = np.sqrt(np.mean(chunk ** 2))
                return min(1.0, rms * 5.0)
        except (IndexError, Exception):
            pass
        return 0.0

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._arecord_proc:
            try:
                self._arecord_proc.terminate()
                self._arecord_proc.wait(timeout=2.0)
            except Exception:
                pass
            self._arecord_proc = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Audio input stopped.")

    def __del__(self):
        self.stop()
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass


class AudioOutput:
    """
    Plays audio through the speaker.

    Pipes raw PCM directly to aplay via stdin -- no temp files, no disk I/O.
    Tries USB speaker first, falls back to system default.
    """

    _alsa_device = None

    def __init__(self, config: dict):
        self.sample_rate = config.get("output_sample_rate", 22050)
        self._volume = config.get("volume", 0.8)
        self._playing = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._aplay_proc = None
        self._last_play_end: float = 0.0  # timestamp when playback last ended

        if AudioOutput._alsa_device is None:
            AudioOutput._alsa_device = self._detect_speaker()
        logger.info(f"Audio output: {AudioOutput._alsa_device or 'system default'}")

    @staticmethod
    def _detect_speaker() -> str | None:
        """Find USB speaker by ALSA card name."""
        try:
            with open("/proc/asound/cards") as f:
                cards = f.read()
            if "Device" in cards:
                return "plughw:Device,0"
        except Exception:
            pass
        return None

    def play_pcm(self, pcm_data: bytes, sample_rate: int | None = None):
        """Play raw PCM int16 bytes by piping to aplay (no temp files)."""
        self.stop()
        self._stop_event.clear()
        # Set _playing BEFORE starting the thread so callers that
        # immediately check is_playing don't see a False gap.
        self._playing = True
        sr = sample_rate or self.sample_rate
        thread = threading.Thread(
            target=self._play_thread, args=(pcm_data, sr), daemon=True
        )
        thread.start()

    def play_wav_bytes(self, pcm_data: bytes, sample_rate: int | None = None):
        """Alias for play_pcm (backward compatibility)."""
        self.play_pcm(pcm_data, sample_rate)

    def _play_thread(self, pcm_data: bytes, sample_rate: int):
        """Pipe PCM directly to aplay stdin."""
        with self._lock:
            try:
                # Apply volume
                audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
                audio *= self._volume
                raw = np.clip(audio, -32768, 32767).astype(np.int16).tobytes()

                # Try USB speaker first, then system default
                for dev in [AudioOutput._alsa_device, None]:
                    if self._stop_event.is_set():
                        break

                    cmd = [
                        "aplay", "-q",
                        "-f", "S16_LE",
                        "-r", str(sample_rate),
                        "-c", "1",
                        "-t", "raw",
                    ]
                    if dev:
                        cmd.extend(["-D", dev])

                    try:
                        self._aplay_proc = subprocess.Popen(
                            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        self._aplay_proc.stdin.write(raw)
                        self._aplay_proc.stdin.close()
                        self._aplay_proc.wait(timeout=60)
                        if self._aplay_proc.returncode == 0:
                            break
                    except subprocess.TimeoutExpired:
                        self._aplay_proc.kill()
                    except Exception as e:
                        logger.debug(f"aplay {dev or 'default'}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                self._aplay_proc = None
                self._playing = False
                self._last_play_end = time.time()

    # ------------------------------------------------------------------
    # Streaming playback: one aplay process, feed PCM chunks seamlessly
    # ------------------------------------------------------------------

    def start_stream(self, sample_rate: int):
        """Open a persistent aplay process for seamless streaming."""
        self.stop()
        self._stop_event.clear()
        self._playing = True

        sr = sample_rate
        for dev in [AudioOutput._alsa_device, None]:
            cmd = [
                "aplay", "-q",
                "-f", "S16_LE",
                "-r", str(sr),
                "-c", "1",
                "-t", "raw",
            ]
            if dev:
                cmd.extend(["-D", dev])
            try:
                self._aplay_proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return True
            except Exception as e:
                logger.debug(f"aplay stream open {dev or 'default'}: {e}")
                continue
        self._playing = False
        return False

    def write_stream(self, pcm_data: bytes):
        """Write a PCM chunk to the running aplay stream."""
        if self._aplay_proc is None or self._stop_event.is_set():
            return
        try:
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            audio *= self._volume
            raw = np.clip(audio, -32768, 32767).astype(np.int16).tobytes()
            self._aplay_proc.stdin.write(raw)
            self._aplay_proc.stdin.flush()
        except Exception as e:
            logger.debug(f"Stream write error: {e}")

    def finish_stream(self):
        """Close stdin and wait for aplay to finish playing buffered audio."""
        if self._aplay_proc is None:
            self._playing = False
            return
        try:
            self._aplay_proc.stdin.close()
            self._aplay_proc.wait(timeout=60)
        except Exception:
            pass
        finally:
            self._aplay_proc = None
            self._playing = False
            self._last_play_end = time.time()

    def stop(self):
        """Stop any active playback."""
        self._stop_event.set()
        if self._aplay_proc:
            try:
                self._aplay_proc.stdin.close()
            except Exception:
                pass
            try:
                self._aplay_proc.terminate()
            except Exception:
                pass
        start = time.time()
        while self._playing and (time.time() - start) < 0.5:
            time.sleep(0.05)
        self._aplay_proc = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def recently_played(self) -> bool:
        """True if playback ended less than 0.5s ago (echo cooldown)."""
        return (time.time() - self._last_play_end) < 0.5

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(1.0, value))

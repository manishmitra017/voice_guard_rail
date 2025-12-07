"""
Audio Recorder Module
Handles microphone input with start/stop functionality for Mac.
"""

import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
from typing import Optional
import threading


class AudioRecorder:
    """Records audio from Mac microphone with start/stop control."""

    SAMPLE_RATE = 16000  # Required by SpeechBrain model
    CHANNELS = 1  # Mono audio

    def __init__(self):
        self.recording = False
        self.audio_data = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - stores incoming audio chunks."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            with self._lock:
                self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start recording from microphone."""
        with self._lock:
            self.audio_data = []
        self.recording = True

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save to temporary file.

        Returns:
            Path to temporary WAV file, or None if no audio recorded.
        """
        self.recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self.audio_data:
                return None

            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        sf.write(
            temp_file.name,
            audio_array,
            self.SAMPLE_RATE
        )

        return temp_file.name

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording

    @staticmethod
    def list_devices():
        """List available audio input devices."""
        return sd.query_devices()

    @staticmethod
    def get_default_input_device():
        """Get default input device info."""
        return sd.query_devices(kind='input')

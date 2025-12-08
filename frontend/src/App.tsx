import { useState, useRef, useCallback } from 'react'
import './App.css'

interface EmotionResult {
  emotion: string
  emoji: string
  color: string
  confidence: number
  raw_label: string
  all_probabilities: Record<string, number>
}

interface TranscriptionResult {
  text: string
  language: string
}

interface AnalyzeResponse {
  transcription: TranscriptionResult
  emotion: EmotionResult
}

type RecordingState = 'idle' | 'recording' | 'processing'

const API_BASE = '/api'

function App() {
  const [state, setState] = useState<RecordingState>('idle')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isServerReady, setIsServerReady] = useState<boolean | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  // Check server health on mount
  useState(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setIsServerReady(data.models_loaded))
      .catch(() => setIsServerReady(false))
  })

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      setResult(null)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Use webm for better browser compatibility
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())

        // Create blob from chunks
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })

        // Send to API
        await analyzeAudio(audioBlob)
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setState('recording')

    } catch (err) {
      setError('Failed to access microphone. Please allow microphone access.')
      console.error('Microphone error:', err)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && state === 'recording') {
      mediaRecorderRef.current.stop()
      setState('processing')
    }
  }, [state])

  const analyzeAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }

      const data: AnalyzeResponse = await response.json()
      setResult(data)
      setState('idle')

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setState('idle')
    }
  }

  const getButtonText = () => {
    switch (state) {
      case 'recording': return 'Stop Recording'
      case 'processing': return 'Analyzing...'
      default: return 'Start Recording'
    }
  }

  const handleButtonClick = () => {
    if (state === 'idle') {
      startRecording()
    } else if (state === 'recording') {
      stopRecording()
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Voice Emotion Detector</h1>
        <p className="subtitle">Real-time speech emotion recognition using AI</p>
      </header>

      {isServerReady === false && (
        <div className="status-banner error">
          Server is loading models... Please wait.
        </div>
      )}

      <div className="recording-section">
        <button
          className={`record-button ${state}`}
          onClick={handleButtonClick}
          disabled={state === 'processing' || isServerReady === false}
        >
          <span className="button-icon">
            {state === 'recording' ? '⬛' : state === 'processing' ? '⏳' : '🎤'}
          </span>
          <span>{getButtonText()}</span>
        </button>

        {state === 'recording' && (
          <div className="recording-indicator">
            <span className="pulse"></span>
            Recording...
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <div className="results">
          <div className="result-card transcription">
            <h3>What You Said</h3>
            <p className="transcription-text">
              {result.transcription.text || '(No speech detected)'}
            </p>
            <span className="language-badge">{result.transcription.language}</span>
          </div>

          <div className="result-card emotion" style={{ borderColor: result.emotion.color }}>
            <h3>Detected Emotion</h3>
            <div className="emotion-display">
              <span className="emotion-emoji">{result.emotion.emoji}</span>
              <span className="emotion-label">{result.emotion.emotion}</span>
            </div>
            <div className="confidence">
              <div
                className="confidence-bar"
                style={{
                  width: `${result.emotion.confidence * 100}%`,
                  backgroundColor: result.emotion.color
                }}
              />
              <span className="confidence-text">
                {(result.emotion.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          </div>

          <div className="result-card probabilities">
            <h3>All Emotions</h3>
            <div className="probability-list">
              {Object.entries(result.emotion.all_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, prob]) => (
                  <div key={emotion} className="probability-item">
                    <span className="probability-label">{emotion}</span>
                    <div className="probability-bar-container">
                      <div
                        className="probability-bar"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      <footer>
        <p>Powered by OpenAI Whisper & HuggingFace Transformers</p>
      </footer>
    </div>
  )
}

export default App

import { useState, useRef, useCallback, useEffect } from 'react'
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
  language_name?: string
}

interface TranslationItem {
  language_code: string
  language_name: string
  flag: string
  text: string
}

interface AnalyzeResponse {
  transcription: TranscriptionResult
  emotion: EmotionResult
  audio_events?: string[]
  translations?: TranslationItem[]
}

interface LanguageInfo {
  code: string
  name: string
  flag: string
}

interface HealthInfo {
  status: string
  models_loaded: boolean
  sensevoice_enabled: boolean
  translation_enabled: boolean
}

type RecordingState = 'idle' | 'recording' | 'processing'

const API_BASE = '/api'

// Popular languages for quick selection
const POPULAR_LANGUAGES = ['es', 'fr', 'de', 'zh', 'ja', 'ko', 'pt', 'it', 'ru', 'ar', 'hi']

// Convert audio blob to WAV format
async function convertToWav(audioBlob: Blob, sampleRate: number = 16000): Promise<Blob> {
  const audioContext = new AudioContext({ sampleRate })
  const arrayBuffer = await audioBlob.arrayBuffer()
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

  const channelData = audioBuffer.getChannelData(0)

  let samples: Float32Array
  if (audioBuffer.sampleRate !== sampleRate) {
    const ratio = audioBuffer.sampleRate / sampleRate
    const newLength = Math.round(channelData.length / ratio)
    samples = new Float32Array(newLength)
    for (let i = 0; i < newLength; i++) {
      samples[i] = channelData[Math.round(i * ratio)]
    }
  } else {
    samples = channelData
  }

  const pcmData = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
  }

  const wavBuffer = new ArrayBuffer(44 + pcmData.length * 2)
  const view = new DataView(wavBuffer)

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + pcmData.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, pcmData.length * 2, true)

  const pcmOffset = 44
  for (let i = 0; i < pcmData.length; i++) {
    view.setInt16(pcmOffset + i * 2, pcmData[i], true)
  }

  await audioContext.close()
  return new Blob([wavBuffer], { type: 'audio/wav' })
}

function App() {
  const [state, setState] = useState<RecordingState>('idle')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null)

  // Translation state
  const [availableLanguages, setAvailableLanguages] = useState<LanguageInfo[]>([])
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(['es', 'fr'])
  const [showLanguageSelector, setShowLanguageSelector] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  // Fetch health and languages on mount
  useEffect(() => {
    // Check health
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then((data: HealthInfo) => setHealthInfo(data))
      .catch(() => setHealthInfo({ status: 'error', models_loaded: false, sensevoice_enabled: false, translation_enabled: false }))

    // Fetch available languages
    fetch(`${API_BASE}/languages`)
      .then(res => res.json())
      .then((data: LanguageInfo[]) => setAvailableLanguages(data))
      .catch(() => {})
  }, [])

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      setResult(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      })

      let mimeType = 'audio/webm;codecs=opus'
      if (MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/wav'
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop())
        const audioBlob = new Blob(chunksRef.current, { type: mimeType })

        setState('processing')
        try {
          const wavBlob = await convertToWav(audioBlob)
          await analyzeAudio(wavBlob)
        } catch (convErr) {
          console.error('Conversion error:', convErr)
          setError('Failed to process audio. Please try again.')
          setState('idle')
        }
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
    }
  }, [state])

  const analyzeAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.wav')

      // Build URL with translation params
      let url = `${API_BASE}/analyze?language=auto`
      if (selectedLanguages.length > 0 && healthInfo?.translation_enabled) {
        url += `&translate_to=${selectedLanguages.join(',')}`
      }

      const response = await fetch(url, {
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

  const toggleLanguage = (code: string) => {
    setSelectedLanguages(prev => {
      if (prev.includes(code)) {
        return prev.filter(l => l !== code)
      }
      if (prev.length >= 5) {
        return prev // Max 5 languages
      }
      return [...prev, code]
    })
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

  const getLanguageFlag = (code: string): string => {
    const lang = availableLanguages.find(l => l.code === code)
    return lang?.flag || '🏳️'
  }

  const getLanguageName = (code: string): string => {
    const lang = availableLanguages.find(l => l.code === code)
    return lang?.name || code
  }

  return (
    <div className="container">
      <header>
        <h1>Voice Emotion Detector</h1>
        <p className="subtitle">
          Real-time speech emotion recognition with multilingual translation
        </p>
        {healthInfo?.sensevoice_enabled && (
          <span className="feature-badge sensevoice">SenseVoice</span>
        )}
        {healthInfo?.translation_enabled && (
          <span className="feature-badge translation">Translation</span>
        )}
      </header>

      {healthInfo && !healthInfo.models_loaded && (
        <div className="status-banner loading">
          Server is loading models... Please wait.
        </div>
      )}

      {/* Language Selection */}
      {healthInfo?.translation_enabled && (
        <div className="language-section">
          <div className="language-header">
            <span>Translate to:</span>
            <button
              className="language-toggle"
              onClick={() => setShowLanguageSelector(!showLanguageSelector)}
            >
              {showLanguageSelector ? 'Done' : 'Edit'}
            </button>
          </div>

          <div className="selected-languages">
            {selectedLanguages.map(code => (
              <span key={code} className="language-chip" onClick={() => toggleLanguage(code)}>
                {getLanguageFlag(code)} {getLanguageName(code)}
                <span className="remove">×</span>
              </span>
            ))}
            {selectedLanguages.length === 0 && (
              <span className="no-languages">No languages selected</span>
            )}
          </div>

          {showLanguageSelector && (
            <div className="language-selector">
              <div className="language-grid">
                {availableLanguages
                  .filter(lang => POPULAR_LANGUAGES.includes(lang.code) || selectedLanguages.includes(lang.code))
                  .map(lang => (
                    <button
                      key={lang.code}
                      className={`language-option ${selectedLanguages.includes(lang.code) ? 'selected' : ''}`}
                      onClick={() => toggleLanguage(lang.code)}
                    >
                      <span className="flag">{lang.flag}</span>
                      <span className="name">{lang.name}</span>
                    </button>
                  ))}
              </div>
              {selectedLanguages.length >= 5 && (
                <p className="language-limit">Maximum 5 languages</p>
              )}
            </div>
          )}
        </div>
      )}

      <div className="recording-section">
        <button
          className={`record-button ${state}`}
          onClick={handleButtonClick}
          disabled={state === 'processing' || !healthInfo?.models_loaded}
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
          {/* Original Transcription */}
          <div className="result-card transcription">
            <h3>What You Said</h3>
            <p className="transcription-text">
              {result.transcription.text || '(No speech detected)'}
            </p>
            <div className="transcription-meta">
              <span className="language-badge">
                {getLanguageFlag(result.transcription.language)} {result.transcription.language_name || result.transcription.language}
              </span>
              {result.audio_events && result.audio_events.length > 0 && (
                <span className="audio-events">
                  Detected: {result.audio_events.join(', ')}
                </span>
              )}
            </div>
          </div>

          {/* Translations */}
          {result.translations && result.translations.length > 0 && (
            <div className="result-card translations">
              <h3>Translations</h3>
              <div className="translations-list">
                {result.translations.map(translation => (
                  <div key={translation.language_code} className="translation-item">
                    <div className="translation-header">
                      <span className="translation-flag">{translation.flag}</span>
                      <span className="translation-lang">{translation.language_name}</span>
                    </div>
                    <p className="translation-text">{translation.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Emotion Result */}
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

          {/* All Emotions */}
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
        <p>
          Powered by {healthInfo?.sensevoice_enabled ? 'SenseVoice' : 'OpenAI Whisper'} & HuggingFace
          {healthInfo?.translation_enabled && ' + NLLB Translation'}
        </p>
      </footer>
    </div>
  )
}

export default App

import { useState } from 'react'
import './App.css'

const messageTypes = [
  { value: 'sms', label: 'SMS' },
  { value: 'email', label: 'Email' },
]

function App() {
  const [message, setMessage] = useState('')
  const [source, setSource] = useState('sms')
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')
  const [isChecking, setIsChecking] = useState(false)

  function clearFeedback() {
    setPrediction(null)
    setError('')
  }

  async function checkMessage(event) {
    event.preventDefault()
    if (isChecking || !message.trim()) return

    clearFeedback()
    setIsChecking(true)

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message, source }),
      })

      if (!response.ok) throw new Error('Prediction request failed')

      const { label } = await response.json()
      if (label !== 'spam' && label !== 'ham') throw new Error('Invalid prediction')
      setPrediction(label)
    } catch {
      setError('Couldn’t check this message. Please try again.')
    } finally {
      setIsChecking(false)
    }
  }

  return (
    <div className="page">
      <header className="site-header">
        <span className="brand-mark" aria-hidden="true"><i /></span>
        <span>Spam or Ham?</span>
      </header>

      <main className="workspace">
        <div className="intro">
          <h1>Spam or <span>Ham?</span></h1>
          <p>Paste an SMS or email to see if it’s likely spam.</p>
        </div>

        <form className="checker" onSubmit={checkMessage} aria-label="Message checker">
          <fieldset className="source-field" disabled={isChecking}>
            <legend>Message type</legend>
            <div className="source-options">
              {messageTypes.map(({ value, label }) => (
                <label className="source-option" key={value}>
                  <input
                    type="radio"
                    name="source"
                    value={value}
                    checked={source === value}
                    onChange={() => {
                      setSource(value)
                      clearFeedback()
                    }}
                  />
                  <span className={`source-icon source-icon--${value}`} aria-hidden="true" />
                  {label}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="message-field">
            <label htmlFor="message">Message text</label>
            <div className="text-box">
              <textarea
                id="message"
                placeholder="Type or paste your message here…"
                value={message}
                readOnly={isChecking}
                required
                onChange={(event) => {
                  setMessage(event.target.value)
                  clearFeedback()
                }}
              />
              <div className="text-tools">
                <span>{message.length.toLocaleString()} characters</span>
                <button
                  className="clear-button"
                  type="button"
                  disabled={!message || isChecking}
                  onClick={() => {
                    setMessage('')
                    clearFeedback()
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
          </div>

          <button className="submit-button" disabled={!message.trim() || isChecking}>
            {isChecking ? 'Checking…' : 'Check for spam'}
          </button>

          <div role="status">
            {prediction && (
              <div className={`feedback feedback--${prediction}`}>
                <strong>Prediction</strong>
                <span>{prediction === 'spam' ? 'Likely spam' : 'Likely ham'}</span>
              </div>
            )}
          </div>
          {error && <p className="feedback feedback--error" role="alert">{error}</p>}
        </form>

        <footer className="disclaimer">
          <span className="info-icon" aria-hidden="true">i</span>
          <p>This is only a prediction. It can be wrong, so don’t rely on it alone. Verify suspicious messages independently.</p>
        </footer>
      </main>
    </div>
  )
}

export default App

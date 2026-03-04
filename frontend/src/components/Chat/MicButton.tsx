import type { SpeechState } from '../../hooks/useSpeech';

interface MicButtonProps {
  state: SpeechState;
  onClick: () => void;
  disabled?: boolean;
}

export function MicButton({ state, onClick, disabled }: MicButtonProps) {
  const title =
    state === 'recording'
      ? 'Stop recording'
      : state === 'transcribing'
        ? 'Transcribing...'
        : 'Voice input';

  return (
    <button
      className={`mic-btn ${state !== 'idle' ? `mic-${state}` : ''}`}
      onClick={onClick}
      disabled={disabled || state === 'transcribing'}
      title={title}
      style={{
        background: state === 'recording' ? '#e74c3c' : 'transparent',
        border: '1px solid var(--border, #555)',
        borderRadius: '8px',
        padding: '8px',
        cursor: disabled || state === 'transcribing' ? 'default' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: '36px',
        height: '36px',
        color: state === 'recording' ? '#fff' : 'var(--text, #cdd6f4)',
        opacity: disabled || state === 'transcribing' ? 0.5 : 1,
        animation: state === 'recording' ? 'pulse 1.5s ease-in-out infinite' : 'none',
      }}
    >
      {state === 'transcribing' ? (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="28" strokeDashoffset="10">
            <animateTransform attributeName="transform" type="rotate" from="0 8 8" to="360 8 8" dur="1s" repeatCount="indefinite" />
          </circle>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M5 3a3 3 0 0 1 6 0v5a3 3 0 0 1-6 0V3z" />
          <path d="M3.5 6.5A.5.5 0 0 1 4 7v1a4 4 0 0 0 8 0V7a.5.5 0 0 1 1 0v1a5 5 0 0 1-4.5 4.975V15h3a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1h3v-2.025A5 5 0 0 1 3 8V7a.5.5 0 0 1 .5-.5z" />
        </svg>
      )}
    </button>
  );
}

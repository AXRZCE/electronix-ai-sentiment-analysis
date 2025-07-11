import './style.css'

interface PredictionResponse {
  label: string;
  score: number;
}

class SentimentAnalyzer {
  private apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  async predictSentiment(text: string): Promise<PredictionResponse> {
    const response = await fetch(`${this.apiUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.apiUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

const analyzer = new SentimentAnalyzer();

function createApp(): void {
  const app = document.querySelector<HTMLDivElement>('#app')!;

  app.innerHTML = `
    <div class="container">
      <header>
        <h1>🎭 Sentiment Analysis</h1>
        <p>Enter text below to analyze its sentiment</p>
        <div id="status" class="status"></div>
      </header>

      <main>
        <div class="input-section">
          <textarea
            id="textInput"
            placeholder="Enter your text here..."
            rows="6"
          ></textarea>

          <button id="predictBtn" type="button">
            Analyze Sentiment
          </button>
        </div>

        <div id="results" class="results hidden">
          <h3>Results:</h3>
          <div class="result-card">
            <div class="label">
              <span>Sentiment:</span>
              <span id="sentimentLabel" class="sentiment-label"></span>
            </div>
            <div class="score">
              <span>Confidence:</span>
              <span id="confidenceScore" class="confidence-score"></span>
            </div>
            <div class="progress-bar">
              <div id="progressFill" class="progress-fill"></div>
            </div>
          </div>
        </div>

        <div id="loading" class="loading hidden">
          <div class="spinner"></div>
          <p>Analyzing sentiment...</p>
        </div>

        <div id="error" class="error hidden">
          <p id="errorMessage"></p>
        </div>
      </main>
    </div>
  `;

  setupEventListeners();
  checkBackendStatus();
}

function setupEventListeners(): void {
  const textInput = document.getElementById('textInput') as HTMLTextAreaElement;
  const predictBtn = document.getElementById('predictBtn') as HTMLButtonElement;

  predictBtn.addEventListener('click', handlePredict);

  textInput.addEventListener('input', () => {
    const hasText = textInput.value.trim().length > 0;
    predictBtn.disabled = !hasText;
  });

  // Allow Enter key to trigger prediction (with Ctrl/Cmd)
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handlePredict();
    }
  });
}

async function handlePredict(): Promise<void> {
  const textInput = document.getElementById('textInput') as HTMLTextAreaElement;
  const text = textInput.value.trim();

  if (!text) return;

  showLoading();
  hideError();
  hideResults();

  try {
    const result = await analyzer.predictSentiment(text);
    showResults(result);
  } catch (error) {
    showError(error instanceof Error ? error.message : 'An error occurred');
  } finally {
    hideLoading();
  }
}

function showResults(result: PredictionResponse): void {
  const resultsDiv = document.getElementById('results')!;
  const sentimentLabel = document.getElementById('sentimentLabel')!;
  const confidenceScore = document.getElementById('confidenceScore')!;
  const progressFill = document.getElementById('progressFill')!;

  sentimentLabel.textContent = result.label.toUpperCase();
  sentimentLabel.className = `sentiment-label ${result.label}`;

  confidenceScore.textContent = `${(result.score * 100).toFixed(1)}%`;

  progressFill.style.width = `${result.score * 100}%`;
  progressFill.className = `progress-fill ${result.label}`;

  resultsDiv.classList.remove('hidden');
}

function showLoading(): void {
  document.getElementById('loading')!.classList.remove('hidden');
}

function hideLoading(): void {
  document.getElementById('loading')!.classList.add('hidden');
}

function showError(message: string): void {
  const errorDiv = document.getElementById('error')!;
  const errorMessage = document.getElementById('errorMessage')!;

  errorMessage.textContent = message;
  errorDiv.classList.remove('hidden');
}

function hideError(): void {
  document.getElementById('error')!.classList.add('hidden');
}

function hideResults(): void {
  document.getElementById('results')!.classList.add('hidden');
}

async function checkBackendStatus(): Promise<void> {
  const statusDiv = document.getElementById('status')!;

  try {
    const isHealthy = await analyzer.checkHealth();
    if (isHealthy) {
      statusDiv.innerHTML = '<span class="status-online">🟢 Backend Online</span>';
    } else {
      statusDiv.innerHTML = '<span class="status-offline">🔴 Backend Offline</span>';
    }
  } catch {
    statusDiv.innerHTML = '<span class="status-offline">🔴 Backend Offline</span>';
  }
}

// Initialize the app
createApp();

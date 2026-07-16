const form = document.querySelector('#generator-form');
const promptInput = document.querySelector('#prompt');
const seedInput = document.querySelector('#seed');
const modeInput = document.querySelector('#mode');
const sizeInput = document.querySelector('#size');
const generateButton = document.querySelector('#generate');
const statusText = document.querySelector('#status');
const placeholder = document.querySelector('#placeholder');
const loader = document.querySelector('#loader');
const loaderLabel = document.querySelector('#loader-label');
const progressTrack = document.querySelector('#progress-track');
const progressFill = document.querySelector('#progress-fill');
const image = document.querySelector('#result-image');
const processedButton = document.querySelector('#processed-view');
const rawButton = document.querySelector('#raw-view');
const download = document.querySelector('#download');
const device = document.querySelector('#device');
const exportSize = document.querySelector('#export-size');
const historySection = document.querySelector('#history-section');
const historyGrid = document.querySelector('#history-grid');
let activeCategory = null;

let processedUrl = '';
let spriteUrl = '';
let rawUrl = '';
let progressTimer = null;

function setView(view) {
  const raw = view === 'raw';
  image.src = raw ? rawUrl : processedUrl;
  processedButton.classList.toggle('active', !raw);
  rawButton.classList.toggle('active', raw);
}

function renderProgress(state) {
  if (state.phase === 'loading') {
    loaderLabel.textContent = 'LOADING SDXL';
    progressTrack.hidden = true;
    statusText.textContent = 'Loading model weights. The first run can take a few minutes.';
  } else if (state.phase === 'generating' && state.total > 0) {
    loaderLabel.textContent = 'RENDERING';
    progressTrack.hidden = false;
    progressFill.style.width = `${Math.round((state.step / state.total) * 100)}%`;
    statusText.textContent = `Denoising step ${state.step} of ${state.total}.`;
  } else if (state.phase === 'exporting') {
    loaderLabel.textContent = 'EXPORTING';
    progressFill.style.width = '100%';
    statusText.textContent = 'Cropping and exporting the transparent sprite.';
  }
}

function startProgressPolling() {
  stopProgressPolling();
  progressTimer = setInterval(async () => {
    try {
      const response = await fetch('/api/progress');
      if (response.ok) renderProgress(await response.json());
    } catch (_) {
      // transient poll failure, keep last state
    }
  }, 700);
}

function stopProgressPolling() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  progressFill.style.width = '0%';
  progressTrack.hidden = true;
  loaderLabel.textContent = 'RENDERING';
}

function setBusy(busy) {
  generateButton.disabled = busy;
  loader.hidden = !busy;
  if (busy) {
    placeholder.hidden = true;
    image.hidden = true;
    statusText.classList.remove('error');
    statusText.textContent = 'Starting generation.';
    startProgressPolling();
  } else {
    stopProgressPolling();
  }
}

function showResult(entry, message) {
  processedUrl = entry.image_url;
  spriteUrl = entry.sprite_url;
  rawUrl = entry.raw_url;
  download.href = spriteUrl;
  download.download = `spritelab_${entry.size}px.png`;
  setView('processed');
  placeholder.hidden = true;
  image.hidden = false;
  download.classList.remove('disabled');
  download.removeAttribute('aria-disabled');
  device.textContent = (entry.device || 'gpu').toUpperCase();
  exportSize.textContent = `${entry.size} × ${entry.size}`;
  statusText.classList.remove('error');
  statusText.textContent = message;
}

function renderHistory(entries) {
  historyGrid.replaceChildren();
  historySection.hidden = entries.length === 0;
  entries.forEach((entry) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'history-item';
    button.title = `${entry.prompt} (seed ${entry.seed})`;
    const thumb = document.createElement('img');
    thumb.src = entry.sprite_url;
    thumb.alt = entry.prompt;
    thumb.loading = 'lazy';
    button.appendChild(thumb);
    button.addEventListener('click', () => {
      promptInput.value = entry.prompt;
      seedInput.value = entry.seed;
      if (entry.mode) modeInput.value = entry.mode;
      if (entry.size) sizeInput.value = String(entry.size);
      activeCategory = entry.category || null;
      showResult(entry, `Loaded from history. ${entry.mode} mode, seed ${entry.seed}.`);
    });
    historyGrid.appendChild(button);
  });
}

async function refreshHistory() {
  try {
    const response = await fetch('/api/history');
    if (!response.ok) return;
    renderHistory((await response.json()).entries);
  } catch (_) {
    // history is best-effort
  }
}

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    promptInput.value = button.dataset.prompt;
    activeCategory = button.dataset.category || null;
    promptInput.focus();
  });
});

document.querySelector('#random-seed').addEventListener('click', () => {
  seedInput.value = Math.floor(Math.random() * 4_294_967_296);
});

processedButton.addEventListener('click', () => processedUrl && setView('processed'));
rawButton.addEventListener('click', () => rawUrl && setView('raw'));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setBusy(true);
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: promptInput.value.trim(),
        seed: Number(seedInput.value),
        mode: modeInput.value,
        size: Number(sizeInput.value),
        category: activeCategory,
      }),
    });
    if (!response.ok) {
      let detail = `Generation failed with status ${response.status}`;
      try {
        const payload = await response.json();
        if (payload.detail) {
          detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
        }
      } catch (_) {
        // keep status fallback
      }
      throw new Error(detail);
    }
    const result = await response.json();
    showResult(result, `Complete. ${result.mode} mode, seed ${result.seed}.`);
    refreshHistory();
  } catch (error) {
    placeholder.hidden = false;
    statusText.classList.add('error');
    statusText.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

(async () => {
  refreshHistory();
  try {
    const response = await fetch('/api/progress');
    if (!response.ok) return;
    const state = await response.json();
    if (state.phase === 'loading') {
      statusText.textContent = 'SDXL model is warming up in the background.';
    } else if (state.model_loaded) {
      statusText.textContent = 'Model loaded. Ready to generate.';
    }
  } catch (_) {
    // keep default status
  }
})();

const form = document.querySelector('#generator-form');
const promptInput = document.querySelector('#prompt');
const seedInput = document.querySelector('#seed');
const modeInput = document.querySelector('#mode');
const sizeInput = document.querySelector('#size');
const generateButton = document.querySelector('#generate');
const statusText = document.querySelector('#status');
const placeholder = document.querySelector('#placeholder');
const loader = document.querySelector('#loader');
const image = document.querySelector('#result-image');
const processedButton = document.querySelector('#processed-view');
const rawButton = document.querySelector('#raw-view');
const download = document.querySelector('#download');
const device = document.querySelector('#device');
const exportSize = document.querySelector('#export-size');

let processedUrl = '';
let spriteUrl = '';
let rawUrl = '';

function setView(view) {
  const raw = view === 'raw';
  image.src = raw ? rawUrl : processedUrl;
  processedButton.classList.toggle('active', !raw);
  rawButton.classList.toggle('active', raw);
}

function setBusy(busy) {
  generateButton.disabled = busy;
  loader.hidden = !busy;
  if (busy) {
    placeholder.hidden = true;
    image.hidden = true;
    statusText.classList.remove('error');
    statusText.textContent = 'Rendering on the GPU. First generation also loads SDXL.';
  }
}

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    promptInput.value = button.dataset.prompt;
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
      }),
    });
    if (!response.ok) {
      throw new Error(`Generation failed with status ${response.status}`);
    }
    const result = await response.json();
    processedUrl = result.image_url;
    spriteUrl = result.sprite_url;
    rawUrl = result.raw_url;
    download.href = spriteUrl;
    download.download = `spritelab_${result.size}px.png`;
    setView('processed');
    image.hidden = false;
    download.classList.remove('disabled');
    download.removeAttribute('aria-disabled');
    device.textContent = result.device.toUpperCase();
    exportSize.textContent = `${result.size} × ${result.size}`;
    statusText.textContent = `Complete. ${result.mode} mode, seed ${result.seed}.`;
  } catch (error) {
    placeholder.hidden = false;
    statusText.classList.add('error');
    statusText.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

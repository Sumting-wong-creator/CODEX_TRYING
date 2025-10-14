const summaryLine = document.getElementById('summary-line');
const bulletsEl = document.getElementById('bullets');
const closeBtn = document.getElementById('close');

window.addEventListener('message', event => {
  if (event.data?.type === 'awa-topbar-update') {
    renderSummary(event.data.summary || {});
  }
});

closeBtn.addEventListener('click', () => {
  parent.postMessage({ type: 'awa-topbar-close' }, '*');
});

function renderSummary(summary) {
  summaryLine.textContent = summary.line || 'HAWA overview unavailable';
  bulletsEl.innerHTML = '';
  (summary.bullets || []).slice(0, 3).forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    li.dir = 'auto';
    bulletsEl.appendChild(li);
  });
}

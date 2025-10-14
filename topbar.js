(() => {
  const host = document.currentScript?.getRootNode()?.host;
  if (!host) return;
  const summaryEl = host.querySelector('.summary');
  if (summaryEl && host.textContent) {
    summaryEl.textContent = host.textContent;
  }
})();

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function restoreCodeBlocks(html, blocks) {
  return blocks.reduce((acc, block, index) => {
    const token = `__CODE_BLOCK_${index}__`;
    const escaped = escapeHtml(block.code);
    const languageClass = block.lang ? ` class="language-${block.lang}"` : '';
    const replacement = `<pre><code${languageClass}>${escaped}</code></pre>`;
    return acc.replace(token, replacement);
  }, html);
}

function buildLists(paragraph) {
  const lines = paragraph.split(/\n/);
  if (!lines.every(line => /^[-*]\s+/.test(line))) {
    return `<p>${paragraph.replace(/\n/g, '<br>')}</p>`;
  }
  const items = lines.map(line => `<li>${line.replace(/^[-*]\s+/, '')}</li>`).join('');
  return `<ul>${items}</ul>`;
}

function convertMarkdown(text) {
  if (!text) return '';
  const codeBlocks = [];
  let replaced = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang = '', code = '') => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push({ lang: lang.trim(), code });
    return token;
  });

  replaced = escapeHtml(replaced);
  replaced = replaced.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  replaced = replaced.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  replaced = replaced.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  replaced = replaced.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  replaced = replaced.replace(/\*(.+?)\*/g, '<em>$1</em>');
  replaced = replaced.replace(/`([^`]+)`/g, '<code>$1</code>');
  replaced = replaced.replace(/\[(.+?)\]\((https?:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  const paragraphs = replaced.trim().split(/\n{2,}/);
  const html = paragraphs.map(paragraph => {
    if (/^([-*]\s+)/.test(paragraph.trim())) {
      return buildLists(paragraph.trim());
    }
    return `<p>${paragraph.replace(/\n/g, '<br>')}</p>`;
  }).join('');

  return restoreCodeBlocks(html, codeBlocks);
}

export function renderMarkdown(container, text) {
  const html = convertMarkdown(text);
  container.innerHTML = html;
  if (window.hljs) {
    container.querySelectorAll('pre code').forEach(block => {
      window.hljs.highlightElement(block);
    });
  }
}

export function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[(.+?)\]\((.+?)\)/g, '$1')
    .replace(/[*_#>-]/g, '')
    .trim();
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[char]);
}

export function markdownToHtml(markdown) {
  if (!markdown) return '';
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inCode = false;
  let codeLang = '';
  const listStack = [];

  const closeLists = level => {
    while (listStack.length > level) {
      html.push('</ul>');
      listStack.pop();
    }
  };

  lines.forEach(line => {
    const fenceMatch = line.match(/^```(.*)/);
    if (fenceMatch) {
      if (inCode) {
        html.push(`</code></pre>`);
        inCode = false;
        codeLang = '';
      } else {
        inCode = true;
        codeLang = fenceMatch[1].trim();
        const className = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : '';
        html.push(`<pre dir="auto"><code${className}>`);
      }
      return;
    }

    if (inCode) {
      html.push(`${escapeHtml(line)}\n`);
      return;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeLists(0);
      const level = heading[1].length;
      html.push(`<h${level} dir="auto">${inlineFormat(heading[2])}</h${level}>`);
      return;
    }

    const list = line.match(/^([*+-]|\d+\.)\s+(.*)$/);
    if (list) {
      const level = listStack.length;
      if (!listStack.length) {
        html.push('<ul>');
        listStack.push('ul');
      }
      html.push(`<li dir="auto">${inlineFormat(list[2])}</li>`);
      return;
    }

    closeLists(0);
    if (line.trim() === '') {
      html.push('');
      return;
    }
    html.push(`<p dir="auto">${inlineFormat(line)}</p>`);
  });

  if (inCode) {
    html.push('</code></pre>');
  }
  closeLists(0);
  return html.join('\n');
}

function inlineFormat(text) {
  if (!text) return '';
  let result = escapeHtml(text);
  result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  result = result.replace(/\*(.+?)\*/g, '<em>$1</em>');
  result = result.replace(/`([^`]+)`/g, '<code>$1</code>');
  result = result.replace(/\[(.+?)\]\((https?:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return result;
}

export function sanitizeHtml(html) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const disallowed = ['script', 'style', 'iframe', 'object'];
  disallowed.forEach(tag => {
    doc.querySelectorAll(tag).forEach(node => node.remove());
  });
  doc.querySelectorAll('*').forEach(node => {
    Array.from(node.attributes).forEach(attr => {
      if (/^on/i.test(attr.name) || attr.name === 'style') {
        node.removeAttribute(attr.name);
      }
      if (attr.name === 'href' && node.getAttribute('href') && !/^https?:/i.test(node.getAttribute('href'))) {
        node.removeAttribute('href');
      }
    });
  });
  return doc.body.innerHTML;
}

export function renderMarkdown(target, markdown) {
  const html = sanitizeHtml(markdownToHtml(markdown));
  target.innerHTML = html;
  if (window.hljs && typeof window.hljs.highlightAll === 'function') {
    window.hljs.highlightAll();
  }
}

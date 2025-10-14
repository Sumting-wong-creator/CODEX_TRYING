# HAWA (Hyper Agentic Web Assistant)

HAWA is a Manifest V3 Chrome extension that embeds a cautious Gemini 2.5 Flash agent directly in Chrome's side panel. It runs entirely on-device except for API calls to Gemini.

## Installation

1. Go to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked** and choose this folder.
4. Pin the action icon so HAWA is always one click away.

## Getting Started

1. Open the extension **Options** page (right-click the toolbar icon → *Options*).
2. Paste your Gemini 2.5 Flash API key. HAWA validates it against the Gemini models endpoint before saving.
3. (Optional) Add a passphrase to encrypt the key locally with AES-GCM—the verified key is also stored as the `GEMINI_API_KEY` environment value so the agent can connect immediately.
4. Add any trusted domains to the allow-list. HAWA blocks risky actions elsewhere.

## Using HAWA

- **Ask mode** (default) chats about the current page. When you send a message, HAWA automatically captures the page context.
- **Agent mode** opens a dedicated tab with an animated overlay and an emergency stop button. HAWA performs actions there within your guardrails.
- Use the **Summarize** quick action to get a structured digest of the page instantly.
- Press **New chat** to reset the conversation with a gentle transition animation.

## Safety Guardrails

- Transactions with totals other than zero are blocked automatically.
- Navigation, typing, clicking, and scrolling are restricted to domains on your allow-list.
- Before form submissions or cart interactions, the content script prompts for confirmation.
- Page-level instructions are ignored unless you enable them in the chat bar.

## Debugging

Open the side panel and press `Ctrl+Shift+J` (or `Cmd+Option+J` on macOS) to inspect the sidebar logs. The background service worker also emits detailed streaming diagnostics visible at `chrome://extensions` → *Service Worker*.

## Privacy

All chat history, preferences, and encrypted credentials stay in `chrome.storage.local`. Nothing is sent anywhere except the text needed for Gemini responses.

Enjoy calm, minimal, system-aware theming that adapts to your light or dark mode automatically.

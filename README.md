# Agentic Web Assistant

Agentic Web Assistant is a Manifest V3 Chrome extension that provides an offline-first Gemini 2.5 Flash copilot for browsing. All processing happens locally except streaming calls to the Gemini API.

## Installation

1. Open `chrome://extensions/` in Chrome or Edge.
2. Enable **Developer mode** in the top-right corner.
3. Click **Load unpacked** and select this folder.
4. Pin the extension to keep quick actions handy.

## Permissions

The extension requests the following permissions:

- `activeTab`, `tabs` – allow the assistant to understand and interact with the current page when you approve it.
- `scripting` – injects the sidebar UI, topbar, and content script helpers.
- `storage` – stores your settings, encrypted API key, and recent chats locally.
- `sidePanel` – opens the persistent chat panel.
- `contextMenus` – exposes “Summarize” and “Claim Epic” actions in the page menu.
- `notifications` – used for safety and unlock reminders.
- Host permissions (`https://*/*`, `http://*/*`) – required so the assistant can read and act on any page you allow.

## Safety & Guardrails

- Actions such as navigation, typing, or clicking run only on domains in your allow-list (configured in **Options**).
- Requests with non-zero totals are blocked automatically to prevent unapproved purchases.
- A confirmation modal appears before form submits or cart interactions.
- Page-level instructions are filtered unless you toggle **Allow page instructions** in the sidebar or options.

## API Key Storage

Add your Gemini API key in the options page. You can encrypt it with a custom passphrase; the key can be unlocked for 30 minutes using AES-GCM (PBKDF2 key derivation). No key data leaves your machine.

## Hebrew & RTL Tips

- All chat bubbles use `dir="auto"`, and the sidebar fully supports RTL scripts like Hebrew.
- Currency detection surfaces ₪, $, and “חינם” pricing cues inside the page context summary.

## Troubleshooting

- If streaming stalls, click **Stop** and resend your message.
- Missing API key? Open the options page and ensure it is saved or unlocked.
- To clear history, start a new chat from the sidebar header.

Enjoy safer, more transparent browsing with Agentic Web Assistant!

# LingBot demo web client

TypeScript source of the browser page served by `apps/lingbot/demo.py`. The server serves
`../static/index.html`, `../static/styles.css` and `../static/app.js`. `app.js` is compiled from
`src/app.ts` and committed, so the demo itself needs no Node.

```bash
cd apps/lingbot/web
npm install          # TypeScript only
npm run build        # src/app.ts -> ../static/app.js (strict type-check)
npm run watch        # rebuild on save
npm run verify       # fails if the committed ../static/app.js is out of date
npm run e2e          # drive a running demo in headless Chrome (start the server first)
```

`src/app.ts` types every message of the websocket protocol (`demo.py`, `demo_waiting_room.py`)
and looks up every element of `index.html` once at startup with its type, so a renamed element
fails on page load rather than mid-session. The page loads it as an ES module.

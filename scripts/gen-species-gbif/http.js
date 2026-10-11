// http.js: polite JSON/text fetching for the species generator.
//
// Every request goes through one queue (at most `concurrency` in flight), carries
// a User-Agent naming the project, and retries network errors, 429 and 5xx with
// exponential backoff (Retry-After honoured). With a cache directory, responses
// are kept on disk keyed by method + URL + body, so a re-run (or a tweak to the
// thresholds) does not hit the APIs again.

const fs = require('fs');

const USER_AGENT = 'brovisionml-gen-species-gbif/1.0 (+https://github.com/wlejon/brovisionml; species lists for zero-shot classification)';

function fnv1a(str, seed) {
    let h = seed >>> 0;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, '0');
}

function cacheKey(method, url, body) {
    const s = method + ' ' + url + '\n' + (body || '');
    return fnv1a(s, 0x811c9dc5) + fnv1a(s, 0x050c5d1f) + fnv1a(s, 0x1b873593);
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

class Http {
    constructor({ concurrency = 4, retries = 6, cacheDir = null, log = () => {} } = {}) {
        this.concurrency = Math.max(1, Math.min(4, concurrency | 0));
        this.retries = retries;
        this.cacheDir = cacheDir;
        this.log = log;
        this.active = 0;
        this.waiting = [];
        this.stats = { requests: 0, cached: 0, retried: 0 };
        if (cacheDir) fs.mkdirSync(cacheDir, { recursive: true });
    }

    async slot() {
        if (this.active < this.concurrency) { this.active++; return; }
        await new Promise((r) => this.waiting.push(r));
        this.active++;
    }

    release() {
        this.active--;
        const next = this.waiting.shift();
        if (next) next();
    }

    // Returns the response body as text. `opts`: { method, body, headers }.
    async text(url, opts = {}) {
        const method = opts.method || 'GET';
        const key = this.cacheDir ? cacheKey(method, url, opts.body) : null;
        const file = key ? `${this.cacheDir}/${key}` : null;
        if (file && fs.existsSync(file)) {
            this.stats.cached++;
            return fs.readFileSync(file, 'utf8');
        }
        await this.slot();
        try {
            for (let attempt = 0; ; attempt++) {
                let wait = 1000 * Math.pow(2, attempt);
                let why;
                try {
                    this.stats.requests++;
                    const res = await fetch(url, {
                        method,
                        body: opts.body,
                        headers: Object.assign({ 'User-Agent': USER_AGENT }, opts.headers || {}),
                    });
                    if (res.ok) {
                        const body = await res.text();
                        // opts.retryIf(body): a reason to retry a 200 (an API's
                        // "busy, come back later" answer), never cached.
                        const busy = opts.retryIf ? opts.retryIf(body) : null;
                        if (!busy) {
                            if (file) fs.writeFileSync(file, body, 'utf8');
                            return body;
                        }
                        wait = Math.max(wait, 5000);
                        throw new Error(busy);
                    }
                    if (res.status !== 429 && res.status < 500) {
                        const err = new Error(`HTTP ${res.status} for ${url}`);
                        err.status = res.status;
                        throw err;
                    }
                    const ra = Number(res.headers.get && res.headers.get('retry-after'));
                    if (ra > 0) wait = Math.max(wait, ra * 1000);
                    why = `HTTP ${res.status}`;
                } catch (e) {
                    if (e.status) throw e;
                    why = String(e && e.message || e);
                }
                if (attempt >= this.retries) throw new Error(`giving up on ${url}: ${why}`);
                this.stats.retried++;
                this.log(`  retry ${attempt + 1}/${this.retries} in ${wait} ms (${why}): ${url.slice(0, 120)}`);
                await delay(wait);
            }
        } finally {
            this.release();
        }
    }

    async json(url, opts = {}) {
        return JSON.parse(await this.text(url, opts));
    }
}

// Runs fn over items with the queue's concurrency (the queue enforces the cap;
// this only keeps the number of pending promises bounded).
async function mapLimit(items, limit, fn) {
    const out = new Array(items.length);
    let next = 0;
    const worker = async () => {
        while (next < items.length) {
            const i = next++;
            out[i] = await fn(items[i], i);
        }
    };
    await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
    return out;
}

module.exports = { Http, mapLimit, USER_AGENT };

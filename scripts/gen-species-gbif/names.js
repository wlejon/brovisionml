// names.js: English common names from Wikidata (CC0), through the Wikidata
// Action API (search + wbgetentities) rather than the SPARQL endpoint, which
// rate-limits hard.
//
// Items are found by GBIF taxon ID (P846, `haswbstatement`, many keys OR'd per
// search); species no item claims by GBIF ID are looked up by taxon name
// (P225). When several items claim one key, the one whose P225 is the
// backbone's scientific name wins. The name, in order:
//   1. a taxon common name (P1843, English, best rank) equal to the item's
//      English Wikipedia title (any case; a "(bird)"-style suffix dropped),
//   2. a P1843 equal to the English label,
//   3. the English Wikipedia title, 4. the English label (neither when it is
//      just the scientific name),
//   5. the first capitalised P1843.
// The Wikipedia title comes first because English Wikipedia titles taxa by
// their most used English name (for birds, the IOC name), where P1843 often
// lists every regional name at the same rank.

const API = 'https://www.wikidata.org/w/api.php';

const norm = (s) => s.trim().replace(/[’‘]/g, "'").replace(/\s+/g, ' ');
const lc = (s) => s.toLowerCase();

// Capitalise each space-separated word ("black-capped chickadee" ->
// "Black-capped Chickadee"), the convention bird lists use.
// "and", "of", "the" stay lower case inside a name ("Black and Gold Bumblebee").
const SMALL = new Set(['and', 'of', 'the', 'or', 'in', 'on', 'with', 'a', 'an']);
function titleCase(s) {
    return s.split(' ').map((w, i) => (i > 0 && SMALL.has(w.toLowerCase()) ? w.toLowerCase()
        : w.charAt(0).toUpperCase() + w.slice(1))).join(' ');
}

// A Latin name passed off as a common one: the binomial in another gender or
// spelling ("Melissodes Bimaculata" for Melissodes bimaculatus), or a name
// with an author ("Diplolepis Polita (Ashmead"). Not a name that only starts
// with the genus ("Strangalepta Flower Longhorn").
function latinName(name, scientific) {
    if (/[()0-9]/.test(name)) return true;
    const w = name.trim().split(/\s+/), s = String(scientific).toLowerCase().split(' ');
    // "Eurema Lisa" for Pyrisitia lisa: the epithet under an older genus.
    return w.length <= 2 && (w[0].toLowerCase() === s[0] ||
                             (w.length === 2 && s.length > 1 && w[1].toLowerCase() === s[s.length - 1]));
}

// Of several spellings of one name, the one capitalising the most words, then
// the fewest letters ("Red-throated Loon" over "Red-throated loon" and over
// "Red-Throated Loon").
function bestCase(list) {
    const words = (s) => s.split(' ').filter((w) => /^[A-Z]/.test(w)).length;
    const caps = (s) => (s.match(/[A-Z]/g) || []).length;
    return list.slice().sort((a, b) => (words(b) - words(a)) || (caps(a) - caps(b)))[0];
}

// A label or title that lists names ("American Mink, Mink"): the first.
const firstName = (s) => s.split(/, |; /)[0];

// Reads only, a few dozen requests a run: no maxlag (it tracks the query
// service's lag too, which can stall a read for minutes), but an answer that
// says the API is busy is retried rather than taken.
function url(params) {
    const p = Object.assign({ format: 'json', formatversion: '2' }, params);
    return API + '?' + Object.entries(p).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

const busy = (body) => (/"code":\s*"(maxlag|ratelimited|readonly)"/.test(body) ? 'wikidata busy' : null);

async function search(http, expr) {
    const res = await http.json(url({ action: 'query', list: 'search', srsearch: expr,
                                      srnamespace: '0', srlimit: '500', srprop: '' }), { retryIf: busy });
    if (res.error) throw new Error(`wikidata search: ${res.error.info || res.error.code}`);
    return (res.query && res.query.search || []).map((h) => h.title);
}

async function entities(http, ids) {
    const out = [];
    for (let i = 0; i < ids.length; i += 50) {
        const res = await http.json(url({ action: 'wbgetentities', ids: ids.slice(i, i + 50).join('|'),
                                          props: 'labels|claims|sitelinks', languages: 'en',
                                          sitefilter: 'enwiki' }), { retryIf: busy });
        if (res.error) throw new Error(`wikidata wbgetentities: ${res.error.info || res.error.code}`);
        out.push(...Object.values(res.entities || {}));
    }
    return out;
}

// Truthy values of a property (preferred rank if any, else normal), as the
// SPARQL wdt: prefix would give them.
function truthy(entity, prop, map) {
    const st = (entity.claims && entity.claims[prop]) || [];
    const live = st.filter((s) => s.rank !== 'deprecated' && s.mainsnak && s.mainsnak.datavalue);
    const best = live.some((s) => s.rank === 'preferred') ? live.filter((s) => s.rank === 'preferred') : live;
    return best.map((s) => map(s.mainsnak.datavalue.value)).filter((v) => v !== null && v !== undefined);
}

function describe(e) {
    return {
        qid: e.id,
        gbif: truthy(e, 'P846', (v) => String(v)),
        taxonName: truthy(e, 'P225', (v) => v),
        common: truthy(e, 'P1843', (v) => (v.language === 'en' ? norm(v.text) : null)),
        label: e.labels && e.labels.en ? norm(e.labels.en.value) : '',
        title: e.sitelinks && e.sitelinks.enwiki ? norm(e.sitelinks.enwiki.title).replace(/ \([^)]*\)$/, '') : '',
    };
}

// An item -> { qid, name, alt[], via } or null when it has no usable name.
function pick(d, scientific) {
    const sci = lc(scientific);
    const ok = (c) => c && lc(c) !== sci && !latinName(c, scientific);
    const usable = [...new Set(d.common)].filter(ok);
    const same = (x) => usable.filter((c) => lc(c) === lc(x));
    let name = '', via = '';
    if (d.title && same(d.title).length) { name = bestCase(same(d.title)); via = 'P1843=enwiki'; }
    else if (d.label && same(d.label).length) { name = bestCase(same(d.label)); via = 'P1843=label'; }
    else if (ok(d.title)) { name = titleCase(d.title); via = 'enwiki'; }
    else if (ok(d.label)) { name = titleCase(d.label); via = 'label'; }
    else if (usable.length) { name = titleCase(usable.find((c) => /^[A-Z]/.test(c)) || usable[0]); via = 'P1843'; }
    if (!name) return null;
    name = titleCase(firstName(name));
    const seen = new Set([lc(name)]);
    const alt = [];
    for (const c of usable) if (!seen.has(lc(c))) { seen.add(lc(c)); alt.push(c); }
    return { qid: d.qid, name, alt, via };
}

// Among items for one species: the one whose P225 is the scientific name, then
// one with a usable name.
function choose(items, scientific) {
    const ranked = items.slice().sort((a, b) =>
        (b.taxonName.includes(scientific) - a.taxonName.includes(scientific)) ||
        (b.common.length > 0) - (a.common.length > 0));
    for (const d of ranked) {
        const p = pick(d, scientific);
        if (p) return p;
    }
    return null;
}

// species: [{ key, scientific }]. Returns Map key -> { qid, name, alt, via, match }.
async function wikidataNames(http, species, { batch = 40, log = () => {} } = {}) {
    const result = new Map();
    const keys = species.map((s) => String(s.key));
    const want = new Set(keys);
    const qids = new Set();
    for (let i = 0; i < keys.length; i += batch) {
        const expr = 'haswbstatement:' + keys.slice(i, i + batch).map((k) => `P846=${k}`).join('|');
        for (const q of await search(http, expr)) qids.add(q);
    }
    const byKey = new Map();
    for (const d of (await entities(http, [...qids])).map(describe)) {
        for (const k of d.gbif) if (want.has(k)) (byKey.get(k) || byKey.set(k, []).get(k)).push(d);
    }
    for (const s of species) {
        const items = byKey.get(String(s.key));
        const p = items && choose(items, s.scientific);
        if (p) result.set(String(s.key), Object.assign(p, { match: 'P846' }));
    }
    const missing = species.filter((s) => !result.has(String(s.key)));
    log(`  wikidata: ${result.size} named by GBIF ID (P846), ${missing.length} to try by taxon name (P225)`);
    for (const s of missing) {
        const hits = await search(http, `haswbstatement:"P225=${s.scientific}"`);
        if (!hits.length) continue;
        const items = (await entities(http, hits)).map(describe).filter((d) => d.taxonName.includes(s.scientific));
        const p = choose(items, s.scientific);
        if (p) result.set(String(s.key), Object.assign(p, { match: 'P225' }));
    }
    return result;
}

module.exports = { API, wikidataNames, titleCase, latinName };

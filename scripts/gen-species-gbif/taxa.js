// taxa.js: the `taxa` map of a species list, the English names of its higher
// taxa, so a classifier can fall back from a species to its genus, family,
// order or class ("Darners (Aeshnidae)") when no species is certain.
//
//   "taxa": { "Aeshna":    { "rank": "genus",  "common": "Mosaic darners", "commonSource": "wikidata" },
//             "Aeshnidae": { "rank": "family", "common": "Darners", ... }, ... }
//
// One entry per genus (the first word of `scientific`), family, order and
// class the rows use; a row without a rank (the backbone's Squamata and
// Testudines are classes with no order) adds nothing for it. `rank` is the row
// field the name is in. `common` is a short English group name, from:
//   1. the list's editorial table (config "taxaNames", commonSource "override"),
//   2. Wikidata (CC0): the item by GBIF ID (P846), else by taxon name (P225)
//      and taxon rank (P105); its English taxon common names (P1843), label,
//      aliases and enwiki title, the first that is a vernacular name rather
//      than a Latin one (see pickWikidata),
//   3. GBIF's English vernacular names for the backbone key, most attested,
//   4. none: the entry keeps only its rank.
// The backbone key of each taxon comes from the rows' species records (a
// generated list) or from species/match with the rank and the row's higher
// taxa (a list made elsewhere, such as the eBird-named upper-midwest.json).

const gbif = require(bro.appDir + '/gbif.js');
const { mapLimit } = require(bro.appDir + '/http.js');

const WIKIDATA_API = 'https://www.wikidata.org/w/api.php';
const RANKS = ['class', 'order', 'family', 'genus'];
// Wikidata items of the taxon ranks (P105).
const RANK_QID = { class: 'Q37517', order: 'Q36602', family: 'Q35409', genus: 'Q34740' };

const norm = (s) => String(s).trim().replace(/[’‘]/g, "'").replace(/\s+/g, ' ');
const lc = (s) => s.toLowerCase();

// The taxa the rows use: Map name -> { name, rank, ctx } where ctx holds the
// higher taxa of the first row naming it (for species/match) and keys the
// backbone keys when the rows carry gbifKey (filled by keysFromSpecies).
function collect(rows) {
    const taxa = new Map();
    for (const r of rows) {
        const genus = String(r.scientific || '').trim().split(/\s+/)[0];
        const names = { class: r.class, order: r.order, family: r.family, genus };
        for (const rank of RANKS) {
            const name = names[rank];
            if (!name) continue;
            if (!taxa.has(name))
                taxa.set(name, { name, rank, commons: [], ctx: { kingdom: r.kingdom, phylum: r.phylum, class: r.class,
                                                                 order: r.order, family: r.family } });
            if (r.common && r.common !== r.scientific) taxa.get(name).commons.push(r.common);
        }
    }
    return taxa;
}

// Backbone keys from the species records of a generated list (`info`: the
// /species/{key} records, any order): genusKey, familyKey, orderKey, classKey,
// taken only where the record's name at that rank is the taxon's.
function keysFromSpecies(taxa, info) {
    const field = { class: ['class', 'classKey'], order: ['order', 'orderKey'],
                    family: ['family', 'familyKey'], genus: ['genus', 'genusKey'] };
    for (const b of info) {
        for (const t of taxa.values()) {
            if (t.key) continue;
            const [nf, kf] = field[t.rank];
            if (b[nf] === t.name && b[kf]) t.key = b[kf];
        }
    }
}

// Backbone keys by species/match (strict, with the rank and the higher taxa
// as hints) for the taxa keysFromSpecies did not cover.
async function keysByMatch(http, taxa) {
    const todo = [...taxa.values()].filter((t) => !t.key);
    await mapLimit(todo, 4, async (t) => {
        const p = [['name', t.name], ['rank', t.rank.toUpperCase()], ['strict', 'true']];
        for (const k of ['kingdom', 'phylum', 'class', 'order'])
            if (t.ctx[k] && k !== t.rank) p.push([k, t.ctx[k]]);
        if (t.rank === 'genus' && t.ctx.family) p.push(['family', t.ctx.family]);
        const m = await http.json(`${gbif.API}/species/match?${gbif.query(p)}`);
        if (m.usageKey && m.matchType === 'EXACT' && lc(m.canonicalName || '') === lc(t.name) &&
            (!m.kingdom || !t.ctx.kingdom || m.kingdom === t.ctx.kingdom)) {
            t.key = m.status === 'ACCEPTED' || !m.acceptedUsageKey ? m.usageKey : m.acceptedUsageKey;
            t.keyVia = 'match';
        }
    });
}

// --- Wikidata -------------------------------------------------------------

function wdUrl(params) {
    const p = Object.assign({ format: 'json', formatversion: '2' }, params);
    return WIKIDATA_API + '?' + Object.entries(p).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}
const busy = (body) => (/"code":\s*"(maxlag|ratelimited|readonly)"/.test(body) ? 'wikidata busy' : null);

async function wdSearch(http, expr) {
    const res = await http.json(wdUrl({ action: 'query', list: 'search', srsearch: expr,
                                        srnamespace: '0', srlimit: '500', srprop: '' }), { retryIf: busy });
    if (res.error) throw new Error(`wikidata search: ${res.error.info || res.error.code}`);
    return (res.query && res.query.search || []).map((h) => h.title);
}

async function wdEntities(http, ids) {
    const out = [];
    for (let i = 0; i < ids.length; i += 50) {
        const res = await http.json(wdUrl({ action: 'wbgetentities', ids: ids.slice(i, i + 50).join('|'),
                                            props: 'labels|aliases|claims|sitelinks', languages: 'en',
                                            sitefilter: 'enwiki' }), { retryIf: busy });
        if (res.error) throw new Error(`wikidata wbgetentities: ${res.error.info || res.error.code}`);
        out.push(...Object.values(res.entities || {}));
    }
    return out;
}

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
        rank: truthy(e, 'P105', (v) => v.id),
        common: truthy(e, 'P1843', (v) => (v.language === 'en' ? norm(v.text) : null)),
        label: e.labels && e.labels.en ? norm(e.labels.en.value) : '',
        aliases: (e.aliases && e.aliases.en || []).map((a) => norm(a.value)),
        title: e.sitelinks && e.sitelinks.enwiki ? norm(e.sitelinks.enwiki.title).replace(/ \([^)]*\)$/, '') : '',
    };
}

// A Latin name, not a vernacular one: the taxon's own name or any of the
// item's taxon names, a name with an author or a year, or one capitalised
// word with a Latin ending (Aeshna, Odonata, Aeshnidae, Anisoptera).
function latin(s, names) {
    const l = lc(s);
    if (names.some((n) => lc(n) === l)) return true;
    if (/[0-9(),]/.test(s)) return true;
    const w = s.split(' ');
    if (names.some((n) => lc(w[0]) === lc(n))) return true;
    if (w.length === 1 && /^[A-Z][a-z]+$/.test(s) && /(formes|idae|inae|oidea|ptera)$/.test(s)) return true;
    return w.length === 1 && /^[A-Z][a-z]+$/.test(s) && /(a|ae|us|um|ix|ops|is|o|ini)$/.test(s);
}

// Words a group name keeps capitalised inside a sentence-case name.
const PROPER = new Set(['America', 'American', 'Americas', 'North', 'South', 'Central', 'New', 'Old', 'World',
    'European', 'Eurasian', 'Asian', 'Asiatic', 'African', 'Australian', 'Nearctic', 'Holarctic', 'Palearctic',
    'Palaearctic', 'Neotropical', 'Mexican', 'Canadian', 'Japanese', 'Chinese', 'Indian', 'English', 'Arctic',
    'Ameroasian', 'Atlantic', 'Pacific', 'Carolina', 'Virginia', 'Baltimore', 'Io', 'Cecropia', 'Polyphemus',
    'Luna', 'Promethea', 'Isabella', 'Colorado', 'Japan', 'Africa', 'Asia', 'Europe', 'Australia', 'June', 'May',
    'Christmas', 'Hercules', 'Goliath', 'Atlas', 'Mediterranean', 'Oriental', 'Western', 'Eastern']);
// "Western"/"Eastern" stay only as the first word, where every word does; a
// possessive (Fowler's, Cooper's, McKay's) keeps its capital.
const KEEP_INNER = (w) => /^[A-Z].*'s$/.test(w) ||
    (PROPER.has(w.replace(/[^A-Za-z]/g, '')) && !/^(Western|Eastern)$/.test(w));

// Sentence case: the first letter capitalised, later words (and the parts of
// a hyphenated word after the first) lower case unless they are proper ("New
// World warblers", "Fowler's toads", "White-throated swifts").
function sentence(s) {
    return s.split(' ').map((x, i) => x.split('-').map((p, j) =>
        (i === 0 && j === 0 ? p.charAt(0).toUpperCase() + p.slice(1) : KEEP_INNER(p) ? p : p.toLowerCase()))
        .join('-')).join(' ');
}

// A group name, not one animal: the last word plural ("darners", "geese",
// "allies"), or several names joined with "and".
const IRREGULAR = new Set(['mice', 'geese', 'lice', 'fish', 'deer', 'sheep', 'moose', 'bison', 'cattle', 'teeth',
                           'people', 'swine', 'grouse', 'quail', 'kin']);
function plural(s) {
    const w = lc(s).replace(/[^a-z' -]/g, '').trim().split(/[ -]/).pop();
    if (/ and |, /.test(s)) return true;
    if (IRREGULAR.has(w)) return true;
    return /s$/.test(w) && !/(ss|us|is)$/.test(w);
}

// An English word made from the Latin name ("accipitrids", "strigiforms",
// "phasianids", "passerines"): a name, but not the one a field guide heads a
// group with, so a plain name ("hawks", "owls") from either source goes first.
const anglicized = (s) => !/ /.test(s) && /(ids|iforms|ines|ans|morphs|ates|ids)$/i.test(s);

// The plural of a group name made from one animal's name: "Box turtle" ->
// "Box turtles", "Mudpuppy" -> "Mudpuppies", "Horsefly" -> "Horseflies".
function pluralize(s) {
    const m = /^(.*?)(\S+)$/.exec(s);
    const [, head, last] = m;
    const l = lc(last);
    const irregular = { mouse: 'mice', goose: 'geese', louse: 'lice', fish: 'fish', deer: 'deer', moose: 'moose',
                        sheep: 'sheep', grouse: 'grouse', quail: 'quail', bison: 'bison' };
    if (irregular[l]) return head + last.slice(0, 1) + irregular[l].slice(1);
    if (/[^aeiou]y$/.test(l)) return head + last.slice(0, -1) + 'ies';
    if (/(s|x|z|ch|sh)$/.test(l)) return head + last + 'es';
    return head + last + 's';
}

// A candidate name cleaned of what makes it not a group name on its own: a
// leading article, a trailing rank word ("Cricket frog genus", "the skink
// family"), a disambiguation suffix.
function clean(s) {
    s = norm(s);
    if (/\bindet\b|\bunidentified\b|\bsp\.$|\?/i.test(s)) return '';
    if (s === s.toUpperCase()) s = s.toLowerCase();   // "DOGS", "WEASELS AND RELATIVES"
    return s.replace(/ & /g, ' and ').replace(/^(the|a|an) /i, '')
        .replace(/ (genus|family|subfamily|superfamily|order|suborder|infraorder|class|subclass)$/i, '')
        .replace(/ \([^)]*\)$/, '').trim();
}

// The names one Wikidata item offers, in order: its English taxon common names
// (P1843; one equal to the label, the enwiki title or an alias first), then
// the label, the aliases and the enwiki title; Latin names left out. null when
// none is left.
function pickWikidata(d, name) {
    const names = [name].concat(d.taxonName);
    const ok = (s) => s && !latin(s, names);
    const common = [...new Set(d.common.map(clean))].filter(ok);
    const others = [d.label, ...d.aliases, d.title].map(clean).filter(ok);
    const otherSet = new Set(others.map(lc));
    const all = common.filter((c) => otherSet.has(lc(c))).concat(common, others);
    const cands = all.filter((c, i) => all.findIndex((x) => lc(x) === lc(c)) === i);
    if (!cands.length) return null;
    return { qid: d.qid, name: cands[0], cands, alt: cands.slice(1) };
}

// Wikidata names for the taxa (Map name -> taxon with .key). Items by GBIF ID
// first, then by P225 + P105 (the rank) for taxa no item claims by key; a name
// two items of that rank carry (a plant and an animal genus) is skipped.
async function wikidataTaxa(http, taxa, log) {
    const result = new Map();
    const byKey = new Map();
    for (const t of taxa.values()) if (t.key) byKey.set(String(t.key), t);
    const keys = [...byKey.keys()];
    const qids = new Set();
    for (let i = 0; i < keys.length; i += 40) {
        const expr = 'haswbstatement:' + keys.slice(i, i + 40).map((k) => `P846=${k}`).join('|');
        for (const q of await wdSearch(http, expr)) qids.add(q);
    }
    const items = new Map();   // taxon name -> [describe]
    for (const d of (await wdEntities(http, [...qids])).map(describe)) {
        for (const k of d.gbif) {
            const t = byKey.get(k);
            if (t) (items.get(t.name) || items.set(t.name, []).get(t.name)).push(d);
        }
    }
    for (const t of taxa.values()) {
        const its = (items.get(t.name) || []).sort((a, b) =>
            b.taxonName.includes(t.name) - a.taxonName.includes(t.name));
        for (const d of its) {
            const p = pickWikidata(d, t.name);
            if (p) { result.set(t.name, Object.assign(p, { match: 'P846' })); break; }
        }
        if (its.length) t.wikidata = its[0].qid;
    }
    const missing = [...taxa.values()].filter((t) => !t.wikidata);
    log(`  taxa wikidata: ${taxa.size - missing.length} items by GBIF ID (P846), ${missing.length} to try by taxon name (P225)`);
    await mapLimit(missing, 4, async (t) => {
        const hits = await wdSearch(http, `haswbstatement:"P225=${t.name}"`);
        if (!hits.length) return;
        const its = (await wdEntities(http, hits)).map(describe)
            .filter((d) => d.taxonName.includes(t.name) && d.rank.includes(RANK_QID[t.rank]));
        if (its.length !== 1) return;
        t.wikidata = its[0].qid;
        const p = pickWikidata(its[0], t.name);
        if (p) result.set(t.name, Object.assign(p, { match: 'P225' }));
    });
    return result;
}

// GBIF's English vernaculars for the key, most attested first, without
// lists or groups with a slash ("Snow/Ross's Goose") and Latin names.
async function gbifNames(http, t) {
    if (!t.key) return [];
    const out = [];
    for (const x of await gbif.englishVernaculars(http, t.key)) {
        if (/[/()]/.test(x.name)) continue;
        const n = clean(x.name.split(/;/)[0]);
        if (n && !latin(n, [t.name]) && !out.some((o) => lc(o) === lc(n))) out.push(n);
    }
    return out;
}

// Names too general to name a family or genus: GBIF attaches "Animals" and
// "Arthropods" to many keys.
const GENERIC = new Set(['animals', 'animal', 'arthropods', 'arthropod', 'insects', 'insect', 'invertebrates',
                         'organisms', 'plants', 'plant']);

// The words of a name, as stems ("flies" -> "fly", "bees" -> "bee"), without
// the words every group name has.
const FILLER = new Set(['and', 'the', 'with', 'allies', 'ally', 'common', 'typical', 'true', 'other', 'relative',
                        'family', 'group', 'of', 'or', 'new', 'old', 'world', 'american', 'north', 'eurasian']);
function stems(s) {
    return lc(s).split(/[^a-z]+/).filter((w) => w.length >= 3)
        .map((w) => (/ies$/.test(w) ? w.slice(0, -3) + 'y' : /[^s]s$/.test(w) ? w.slice(0, -1) : w))
        .filter((w) => !FILLER.has(w));
}

// A name sourced by a lookup that can land on a homonym (GBIF's vernaculars
// are matched by name; a Wikidata item found by P225) is kept for a family or
// genus only when it shares a word with what the list already says of that
// taxon: its rows' common names, or its parent's name. "False indigos" (a
// plant) for the sphinx moth genus Amorpha shares none with "Walnut Sphinx".
function fits(name, context) {
    const a = stems(name);
    if (!a.length) return false;
    return a.some((w) => context.has(w) || [...context].some((c) => c.length >= 4 && w.length >= 4 &&
                                                              (c.startsWith(w) || w.startsWith(c))));
}

// The group name of a taxon: a plural (or "X and Y") name from Wikidata, else
// one from GBIF, else Wikidata's first name made plural, else GBIF's.
function choose(wdCands, gbCands) {
    const plain = (s) => plural(s) && !anglicized(s);
    const w = wdCands.find(plain);
    if (w) return { common: w, source: 'wikidata' };
    const g = gbCands.find(plain);
    if (g) return { common: g, source: 'gbif' };
    const wa = wdCands.find(plural);
    if (wa) return { common: wa, source: 'wikidata' };
    const ga = gbCands.find(plural);
    if (ga) return { common: ga, source: 'gbif' };
    if (wdCands.length) return { common: pluralize(wdCands[0]), source: 'wikidata' };
    if (gbCands.length) return { common: pluralize(gbCands[0]), source: 'gbif' };
    return null;
}

// rows -> { taxa: {name: {rank, common?, commonSource?, wikidata?}}, summary }.
// opts: { http, info (species records of a generated list), overrides
// ({name: {common, why}}), log }.
async function buildTaxa(rows, { http, info = null, overrides = {}, log = () => {} }) {
    const taxa = collect(rows);
    if (info) keysFromSpecies(taxa, info);
    await keysByMatch(http, taxa);
    const wd = await wikidataTaxa(http, taxa, log);
    const out = {};
    const sources = {};
    const applied = [], unused = new Set(Object.keys(overrides));
    const list = [...taxa.values()].sort((a, b) =>
        RANKS.indexOf(a.rank) - RANKS.indexOf(b.rank) || a.name.localeCompare(b.name));
    // GBIF's name of every taxon (only used without a Wikidata one, but kept
    // for the report).
    const gnames = await mapLimit(list, 4, (t) => gbifNames(http, t));
    const report = [];
    const named = new Map();   // taxon -> chosen common, for its children's context
    list.forEach((t, i) => {
        const e = { rank: t.rank };
        const o = overrides[t.name];
        const w = wd.get(t.name);
        // A family's parent is its order (else, as for the backbone's
        // Squamata, its class); a genus's its family.
        const parent = t.rank === 'genus' ? t.ctx.family : t.rank === 'family' ? (t.ctx.order || t.ctx.class) : null;
        const context = new Set(t.commons.flatMap(stems).concat(parent && named.get(parent) ? stems(named.get(parent)) : []));
        const low = t.rank === 'family' || t.rank === 'genus';
        const keep = (n, byName) => !low || (!GENERIC.has(lc(n)) && (!byName || fits(n, context)));
        const wc = w ? w.cands.filter((n) => keep(n, w.match === 'P225')) : [];
        const gc = gnames[i].filter((n) => keep(n, true));
        const picked = choose(wc, gc);
        let common = null, source = 'none';
        if (o) {
            unused.delete(t.name);
            applied.push({ taxon: t.name, rank: t.rank, common: o.common || null,
                           was: picked ? sentence(picked.common) : null, why: o.why || '' });
            if (o.common) { common = o.common; source = 'override'; }
        } else if (picked) { common = sentence(picked.common); source = picked.source; }
        if (common) { e.common = common; e.commonSource = source; named.set(t.name, common); }
        if (t.wikidata) e.wikidata = t.wikidata;
        if (t.key) e.gbifKey = Number(t.key);
        sources[source] = (sources[source] || 0) + 1;
        out[t.name] = e;
        report.push({ taxon: t.name, rank: t.rank, common: e.common, source,
                      wikidata: w ? w.cands.join(' | ') : '', gbif: gnames[i].join(' | ') });
    });
    const byRank = {};
    for (const t of list) byRank[t.rank] = (byRank[t.rank] || 0) + 1;
    return { taxa: out, summary: { taxa: list.length, byRank, commonSource: sources },
             overrides: { applied, unused: [...unused] }, report };
}

// The `taxa` block as it is written into a list: one entry a line, between
// `meta` (or the list's header) and `species`.
function taxaJson(taxa, indent = '  ') {
    const lines = Object.entries(taxa).map(([k, v]) => `${indent}  ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
    return `${indent}"taxa": {\n${lines.join(',\n')}\n${indent}}`;
}

// Puts a taxa block into a list file's text, replacing one already there and
// leaving every other byte as it was: the block goes in as a line before the
// top-level "species" key.
function insertTaxa(text, taxa) {
    const block = taxaJson(taxa);
    const old = /\n  "taxa": \{\n[\s\S]*?\n  \},?\n/.exec(text);
    if (old) text = text.slice(0, old.index) + '\n' + text.slice(old.index + old[0].length);
    const at = text.indexOf('\n  "species": [');
    if (at < 0) throw new Error('no top-level "species" array (expected a line starting with  "species": [)');
    return text.slice(0, at) + '\n' + block + ',' + text.slice(at);
}

module.exports = { buildTaxa, taxaJson, insertTaxa, collect, latin, RANKS };

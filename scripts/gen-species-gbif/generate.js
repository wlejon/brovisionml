// generate.js: build an openly licensed regional species list from GBIF.
//
//   bro-headless --no-gpu scripts/gen-species-gbif scripts/gen-species-gbif/generate.js \
//       --config scripts/gen-species-gbif/regions/upper-midwest.json --out <dir> [options]
//
// Writes <dir>/species.json (the zero-shot label set: top-level name,
// description, taxonomy, regions, species[] with common, scientific, kingdom ..
// family, regions, accidental, plus records per region, gbifKey, commonSource
// and a meta block), <dir>/datasets.tsv (every contributing GBIF dataset with
// its record count and DOI: the attribution), <dir>/ATTRIBUTION.md
// and <dir>/LICENSE (CC BY 4.0). See brovisionml/docs/species-lists.md.
//
// Options (each overrides the config file):
//   --config <file>            region/taxon config (regions/*.json)
//   --out <dir>                output folder (required)
//   --name <name>              list name
//   --region CODE=CC:Province  a region (repeatable; replaces the config's).
//                              Province may list spellings with '|';
//                              CODE=gadm:USA.24_1 filters by GADM level-1 id instead
//   --group label=taxa         a taxon group (repeatable; replaces the config's).
//                              taxa: comma list of GBIF keys or Name:RANK
//   --min-records N            records a species needs in a region to count there
//   --accidental-share F       below this share of a region's records (in every
//                              region it counts in) a species is marked accidental
//   --year RANGE               GBIF year filter, e.g. 1990,* ('' for none)
//   --name-sources LIST        common-name sources in order (default wikidata,gbif)
//   --exclude-basis LIST       basisOfRecord values not counted
//                              (default LIVING_SPECIMEN,FOSSIL_SPECIMEN; '' for none)
//
// Editorial corrections (dataset exclusions, merges, drops, name overrides)
// live in the config only; see editorial.js.
//   --cache <dir>              keep API responses on disk (re-runs are offline)
//   --concurrency N            requests in flight (1..4, default 4)
//   --taxa-report <file>       also write a TSV of every taxa entry with the
//                              Wikidata and GBIF names it was chosen from
//
// Every list carries a `taxa` map (taxa.js): the English names of the genera,
// families, orders and classes of its rows. Names set by hand come from
// taxa-names.json beside this script, then the config's "taxaNames".
//
//   bro-headless --no-gpu scripts/gen-species-gbif scripts/gen-species-gbif/generate.js \
//       --add-taxa <list.json> [--out <file>] [--config <region.json>] [--cache <dir>]
//
// adds (or replaces) the `taxa` map of a list made elsewhere (the eBird-named
// upper-midwest.json), leaving every other byte of the file as it was; it is
// written back in place unless --out names another file.

const GENERATOR = 'brovisionml/scripts/gen-species-gbif';
const VERSION = '1.1.0';

const fs = require('fs');
const { Http, mapLimit } = require(bro.appDir + '/http.js');
const gbif = require(bro.appDir + '/gbif.js');
const { wikidataNames, titleCase, latinName, API: WIKIDATA_API } = require(bro.appDir + '/names.js');
const out = require(bro.appDir + '/output.js');
const editorial = require(bro.appDir + '/editorial.js');
const taxaMod = require(bro.appDir + '/taxa.js');

const log = (...a) => console.log(...a);

function parseArgs(argv) {
    const o = { region: [], group: [] };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (!a.startsWith('--')) throw new Error(`unexpected argument "${a}"`);
        const k = a.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        const v = argv[++i];
        if (v === undefined) throw new Error(`${a} needs a value`);
        if (k === 'region' || k === 'group') o[k].push(v); else o[k] = v;
    }
    return o;
}

function parseRegion(s) {
    const m = /^([^=]+)=(.+)$/.exec(s);
    if (!m) throw new Error(`--region "${s}": expected CODE=CC:Province or CODE=gadm:GID`);
    const [, code, rest] = m;
    const i = rest.indexOf(':');
    if (i < 0) throw new Error(`--region "${s}": expected CODE=CC:Province`);
    const a = rest.slice(0, i), b = rest.slice(i + 1);
    if (a.toLowerCase() === 'gadm') return { code, gadmGid: b };
    return { code, country: a.toUpperCase(), stateProvince: b.split('|') };
}

function parseGroup(s) {
    const m = /^([^=]+)=(.+)$/.exec(s);
    if (!m) throw new Error(`--group "${s}": expected label=key,key or label=Name:RANK,...`);
    const taxa = m[2].split(',').map((t) => {
        t = t.trim();
        if (/^\d+$/.test(t)) return Number(t);
        const [name, rank] = t.split(':');
        return rank ? { name, rank: rank.toUpperCase() } : { name };
    });
    return { label: m[1], taxa };
}

// brokit's fs resolves a relative path against the app folder, not the shell's
// working directory, so paths from the command line are made absolute here.
const abs = (p) => (!p || /^([A-Za-z]:)?[\\/]/.test(p) ? p : `${process.cwd()}/${p}`);

// The hand-set taxa names: taxa-names.json beside this script (shared by every
// list), then the config's own "taxaNames" over it. Keys starting with "_" are
// comments.
function taxaNames(cfg) {
    const shared = JSON.parse(fs.readFileSync(bro.appDir + '/taxa-names.json', 'utf8'));
    const all = Object.assign({}, shared, cfg.taxaNames || {});
    for (const k of Object.keys(all)) if (k.startsWith('_')) delete all[k];
    return all;
}

// The taxa report: one line per entry, with what each source offered.
function taxaReport(file, built) {
    const lines = ['taxon\trank\tcommon\tsource\twikidata\tgbif'];
    for (const r of built.report)
        lines.push([r.taxon, r.rank, r.common || '', r.source, r.wikidata || '', r.gbif || '']
            .map((v) => String(v).replace(/[\t\n]/g, ' ')).join('\t'));
    fs.writeFileSync(file, lines.join('\n') + '\n', 'utf8');
}

// --add-taxa: the taxa map of an existing list, put into its text.
async function addTaxa(args) {
    for (const k of ['config', 'out', 'cache', 'addTaxa', 'taxaReport']) args[k] = abs(args[k]);
    const cfg = args.config ? JSON.parse(fs.readFileSync(args.config, 'utf8')) : {};
    const http = new Http({ concurrency: Number(args.concurrency || 4), cacheDir: args.cache || null, log });
    const text = fs.readFileSync(args.addTaxa, 'utf8');
    const doc = JSON.parse(text);
    const built = await taxaMod.buildTaxa(doc.species, { http, overrides: taxaNames(cfg), log });
    const next = taxaMod.insertTaxa(text, built.taxa);
    const check = JSON.parse(next);
    if (JSON.stringify(check.species) !== JSON.stringify(doc.species)) throw new Error('rows changed; not written');
    const target = args.out || args.addTaxa;
    fs.writeFileSync(target, next, 'utf8');
    if (args.taxaReport) taxaReport(args.taxaReport, built);
    log(`wrote ${target}: taxa ${JSON.stringify(built.summary)}`);
    if (built.overrides.unused.length) log(`  taxa overrides naming no taxon here: ${built.overrides.unused.length}`);
}

function loadConfig(args) {
    for (const k of ['config', 'out', 'cache', 'taxaReport']) args[k] = abs(args[k]);
    const cfg = args.config ? JSON.parse(fs.readFileSync(args.config, 'utf8')) : {};
    if (args.region.length) cfg.regions = args.region.map(parseRegion);
    if (args.group.length) cfg.groups = args.group.map(parseGroup);
    if (args.name) cfg.name = args.name;
    cfg.filters = Object.assign({ licenses: ['CC0_1_0', 'CC_BY_4_0'], occurrenceStatus: 'PRESENT' },
                                cfg.filters || {});
    if (args.year !== undefined) cfg.filters.year = args.year || undefined;
    // GBIF's search has no negated basisOfRecord, so an exclusion is the
    // allowed values listed. An explicit filters.basisOfRecord wins.
    if (args.excludeBasis !== undefined)
        cfg.filters.excludeBasisOfRecord = args.excludeBasis ? args.excludeBasis.split(',') : [];
    // An explicit basisOfRecord list excludes every other value.
    if (cfg.filters.basisOfRecord && !cfg.filters.excludeBasisOfRecord)
        cfg.filters.excludeBasisOfRecord = gbif.BASIS_OF_RECORD.filter((b) => !cfg.filters.basisOfRecord.includes(b));
    if (!cfg.filters.excludeBasisOfRecord) cfg.filters.excludeBasisOfRecord = ['LIVING_SPECIMEN', 'FOSSIL_SPECIMEN'];
    for (const b of cfg.filters.excludeBasisOfRecord)
        if (!gbif.BASIS_OF_RECORD.includes(b)) throw new Error(`basisOfRecord "${b}" is not one of ${gbif.BASIS_OF_RECORD.join(' ')}`);
    if (!cfg.filters.basisOfRecord && cfg.filters.excludeBasisOfRecord.length)
        cfg.filters.basisOfRecord = gbif.BASIS_OF_RECORD.filter((b) => !cfg.filters.excludeBasisOfRecord.includes(b));
    cfg.minRecords = Number(args.minRecords ?? cfg.minRecords ?? 10);
    cfg.accidentalShare = Number(args.accidentalShare ?? cfg.accidentalShare ?? 2e-5);
    cfg.nameSources = args.nameSources ? args.nameSources.split(',') : (cfg.nameSources || ['wikidata', 'gbif']);
    for (const s of cfg.nameSources)
        if (s !== 'wikidata' && s !== 'gbif') throw new Error(`name source "${s}": expected wikidata or gbif`);
    cfg.out = args.out || cfg.out;
    cfg.cache = args.cache || cfg.cache || null;
    cfg.taxaReport = args.taxaReport || null;
    cfg.concurrency = Number(args.concurrency || cfg.concurrency || 4);
    if (!cfg.out) throw new Error('--out <dir> is required');
    if (!cfg.name) cfg.name = cfg.out.replace(/\/+$/, '').split('/').pop();
    if (!cfg.regions || !cfg.regions.length) throw new Error('no regions (config "regions" or --region)');
    if (!cfg.groups || !cfg.groups.length) throw new Error('no taxon groups (config "groups" or --group)');
    for (const r of cfg.regions) {
        if (!r.code || !(r.gadmGid || (r.country && r.stateProvince)))
            throw new Error(`region ${JSON.stringify(r)} needs code and country + stateProvince (or gadmGid)`);
    }
    return cfg;
}

// Pick a common name from the sources in `order` (default wikidata, then gbif):
// Wikidata's (names.js), or GBIF's (the backbone's own vernacularName, then the
// most-attested English vernacular); with neither, the scientific name.
async function commonName(http, sp, wd, order) {
    // GBIF vernaculars are sometimes lists ("American Mink, Mink": keep the
    // first) or name a group of species ("Snow/Ross's Goose", "Canada Goose
    // (canadensis Group)": skip).
    const first = (s) => s.split(/[,;]/)[0].trim().replace(/’/g, "'");
    // A group noun is not a species' name ("Moth", "Weevil", "Hover Fly").
    const generic = (s) => /^(moth|butterfly|weevil|beetle|fly|bug|wasp|bee|spider|ant|midge|cranefly|crane fly|hover ?fly|stink bug|leaf beetle|ground beetle|sawfly|aphid|leafhopper)s?$/i.test(s);
    const clean = (s) => s && !/[/()]/.test(first(s)) && !latinName(first(s), sp.scientific) && !generic(first(s));
    for (const src of order) {
        if (src === 'wikidata' && wd) return { common: wd.name, source: 'wikidata', wikidata: wd.qid, alt: wd.alt };
        if (src !== 'gbif') continue;
        if (clean(sp.vernacularName)) return { common: titleCase(first(sp.vernacularName)), source: 'gbif' };
        const v = (await gbif.englishVernaculars(http, sp.key)).find((x) => clean(x.name));
        if (v) return { common: titleCase(first(v.name)), source: 'gbif' };
    }
    return { common: sp.scientific, source: 'none' };
}

async function main() {
    const t0 = Date.now();
    const args = parseArgs(bro.app.argv || []);
    if (args.addTaxa) return addTaxa(args);
    const cfg = loadConfig(args);
    const http = new Http({ concurrency: cfg.concurrency, cacheDir: cfg.cache, log });
    const codes = cfg.regions.map((r) => r.code);
    log(`gen-species-gbif ${VERSION}: ${cfg.name}, regions ${codes.join(' ')}, ` +
        `groups ${cfg.groups.map((g) => g.label).join(' ')}, minRecords ${cfg.minRecords}`);

    // 1. Taxon groups -> backbone keys.
    for (const g of cfg.groups) {
        g.resolved = await mapLimit(g.taxa, 4, (t) => gbif.resolveTaxon(http, t));
        g.keys = g.resolved.map((t) => t.key);
        log(`  ${g.label}: ${g.resolved.map((t) => `${t.name} (${t.rank} ${t.key})`).join(', ')}`);
    }

    // 2. Per region x group: species facet + dataset facet.
    const jobs = [];
    for (const r of cfg.regions) for (const g of cfg.groups) jobs.push({ r, g });
    const queries = [];
    const perSpecies = new Map();   // key -> { group, records: {code: n} }
    const totals = {};              // code -> label -> records
    const datasets = new Map();     // datasetKey -> records
    const excluded = [];            // records of excludeDatasets taken out
    await mapLimit(jobs, 4, async ({ r, g }) => {
        const pairs = gbif.occurrenceParams(cfg.filters, r, g);
        const sp = await gbif.facetAll(http, pairs, 'speciesKey');
        const ds = await gbif.facetAll(http, pairs, 'datasetKey');
        const removed = await editorial.subtractDatasets(http, cfg, pairs, r, sp, ds);
        excluded.push(...removed.map((x) => Object.assign(x, { group: g.label })));
        queries.push({ region: r.code, group: g.label, url: gbif.searchUrl(pairs, [['limit', '0']]),
                       records: sp.total, species: sp.counts.size, datasets: ds.counts.size,
                       excludedRecords: removed.reduce((n, x) => n + x.records, 0) });
        (totals[r.code] = totals[r.code] || {})[g.label] = sp.total;
        for (const [k, n] of sp.counts) {
            const e = perSpecies.get(k) || { group: g.label, records: {} };
            e.records[r.code] = (e.records[r.code] || 0) + n;
            perSpecies.set(k, e);
        }
        for (const [k, n] of ds.counts) datasets.set(k, (datasets.get(k) || 0) + n);
        log(`  ${r.code} ${g.label}: ${sp.total} records, ${sp.counts.size} species, ${ds.counts.size} datasets`);
    });
    queries.sort((a, b) => codes.indexOf(a.region) - codes.indexOf(b.region) || a.group.localeCompare(b.group));
    const seen = perSpecies.size;

    // 2b. Editorial merges (backbone duplicates, misfilings) and drops.
    const { merges, drops } = await editorial.mergeAndDrop(http, cfg, perSpecies, log);

    // 3. Threshold: a region counts for a species at minRecords records there
    // (the group's own minRecords when it has one).
    const minOf = (label) => {
        const g = cfg.groups.find((x) => x.label === label);
        return Number(g && g.minRecords != null ? g.minRecords : cfg.minRecords);
    };
    // How many species each group would keep at other cuts, to choose one by.
    for (const g of cfg.groups) {
        const best = [...perSpecies.values()].filter((e) => e.group === g.label)
            .map((e) => Math.max(0, ...Object.values(e.records)));
        const at = [3, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000].map((n) => `${n}:${best.filter((b) => b >= n).length}`);
        log(`  ${g.label}: species with N+ records in some region ${at.join(' ')}`);
    }
    const kept = [];
    let dropped = 0;
    for (const [key, e] of perSpecies) {
        const min = minOf(e.group);
        const regions = codes.filter((c) => (e.records[c] || 0) >= min);
        if (!regions.length) { dropped++; continue; }
        const share = Math.max(...regions.map((c) => e.records[c] / totals[c][e.group]));
        kept.push({ key, group: e.group, regions, records: e.records, share });
    }
    log(`  ${seen} species seen, ${perSpecies.size} after merges/drops, ${kept.length} kept at >= ` +
        `${cfg.groups.map((g) => `${minOf(g.label)} (${g.label})`).join(', ')} records in a region, ${dropped} dropped`);

    // 4. Backbone names and higher taxonomy.
    const info = await mapLimit(kept, 4, (s) => gbif.species(http, s.key));
    let backboneDataset = '';
    kept.forEach((s, i) => {
        const b = info[i];
        backboneDataset = backboneDataset || b.datasetKey || '';
        Object.assign(s, { scientific: b.canonicalName || b.species || b.scientificName,
                           kingdom: b.kingdom, phylum: b.phylum, class: b.class, order: b.order,
                           family: b.family, vernacularName: b.vernacularName || '',
                           status: b.taxonomicStatus, authorship: b.authorship || '' });
        if (b.taxonomicStatus && b.taxonomicStatus !== 'ACCEPTED')
            log(`  note: ${s.scientific} (${s.key}) is ${b.taxonomicStatus} in the backbone`);
    });
    // Hybrids ("Anas rubripes x platyrhynchos") are not labels a classifier should offer.
    const hybrid = (s) => /\s[x×]\s|×/.test(s.scientific);
    const hybrids = kept.filter(hybrid).map((s) => s.scientific);
    if (hybrids.length) {
        log(`  dropped ${hybrids.length} hybrid(s): ${hybrids.join(', ')}`);
        for (let i = kept.length - 1; i >= 0; i--) if (hybrid(kept[i])) kept.splice(i, 1);
    }

    // 5. Common names.
    const wd = await wikidataNames(http, kept, { log });
    const names = await mapLimit(kept, 4, (s) => commonName(http, s, wd.get(String(s.key)), cfg.nameSources));
    kept.forEach((s, i) => Object.assign(s, names[i]));
    const overrides = editorial.overrideNames(cfg, kept);
    log(`  name overrides: ${overrides.applied.length} applied` +
        (overrides.unused.length ? `, ${overrides.unused.length} matched no species: ${overrides.unused.join(', ')}` : ''));

    // Two species under one name are a backbone duplicate (a merge) or a
    // wrong name (an override): say so, for the region's review.
    const byName = new Map();
    for (const s of kept) {
        const k = s.common.toLowerCase();
        byName.set(k, (byName.get(k) || []).concat(`${s.scientific} (${s.source})`));
    }
    for (const [k, v] of byName) if (v.length > 1) log(`  note: "${k}" names ${v.join(', ')}`);

    // 6. Rows, grouped as the config orders the groups, then taxonomically.
    const groupOrder = cfg.groups.map((g) => g.label);
    kept.sort((a, b) => groupOrder.indexOf(a.group) - groupOrder.indexOf(b.group)
        || (a.order || '').localeCompare(b.order || '') || (a.family || '').localeCompare(b.family || '')
        || a.scientific.localeCompare(b.scientific));
    const rows = kept.map((s) => {
        const row = { common: s.common, scientific: s.scientific, kingdom: s.kingdom, phylum: s.phylum,
                      class: s.class, order: s.order, family: s.family, regions: s.regions,
                      accidental: s.share < cfg.accidentalShare };
        row.records = Object.fromEntries(codes.filter((c) => s.records[c]).map((c) => [c, s.records[c]]));
        row.gbifKey = Number(s.key);
        row.commonSource = s.source;
        if (s.wikidata) row.wikidata = s.wikidata;
        if (s.alt && s.alt.length) row.commonAlt = s.alt;
        return row;
    });

    // 6b. The higher taxa of the rows and their names (taxa.js).
    const taxa = await taxaMod.buildTaxa(rows, { http, info, overrides: taxaNames(cfg), log });
    if (cfg.taxaReport) taxaReport(cfg.taxaReport, taxa);
    log(`  taxa: ${JSON.stringify(taxa.summary)}` + (taxa.overrides.applied.length
        ? `, ${taxa.overrides.applied.length} names set by hand` : ''));

    // 7. Datasets (titles, DOIs) for the citation.
    const dsKeys = [...datasets.keys()];
    log(`  fetching ${dsKeys.length} dataset records`);
    const dsInfo = (await mapLimit(dsKeys, 4, (k) => gbif.dataset(http, k)))
        .map((d) => Object.assign(d, { records: datasets.get(d.key) }))
        .sort((a, b) => b.records - a.records || a.key.localeCompare(b.key));

    // 8. Summary + files.
    const count = (f) => rows.reduce((m, r) => (m[f(r)] = (m[f(r)] || 0) + 1, m), {});
    const byRegion = Object.fromEntries(codes.map((c) => [c, rows.filter((r) => r.regions.includes(c)).length]));
    const summary = {
        species: rows.length, speciesSeen: seen, droppedBelowThreshold: dropped,
        droppedHybrids: hybrids, merged: merges.length, dropped: drops.length,
        nameOverrides: overrides.applied.length,
        excludedDatasetRecords: excluded.reduce((n, x) => n + x.records, 0),
        accidental: rows.filter((r) => r.accidental).length,
        byClass: count((r) => r.class), byRegion,
        commonNameSource: count((r) => r.commonSource),
        taxa: taxa.summary,
        records: totals, datasets: dsInfo.length,
        datasetRecords: dsInfo.reduce((n, d) => n + d.records, 0),
    };
    const meta = {
        generator: GENERATOR, version: VERSION, generated: new Date().toISOString(),
        gbifApi: gbif.API, taxonomyDataset: backboneDataset,
        licenseFilter: cfg.filters.licenses, filters: cfg.filters,
        regions: cfg.regions, groups: cfg.groups.map((g) => ({ label: g.label, taxa: g.resolved })),
        minRecords: cfg.minRecords,
        groupMinRecords: Object.fromEntries(cfg.groups.map((g) => [g.label, minOf(g.label)])),
        accidentalShare: cfg.accidentalShare,
        commonNames: { order: cfg.nameSources.concat(['none']),
                       wikidata: 'item by P846 (GBIF ID), else P225 (taxon name); name = P1843 (taxon common name, en) ' +
                                 'matching the enwiki title, else P1843 matching the en label, else the enwiki title, else the en label',
                       gbif: 'backbone vernacularName, else the most attested /species/{key}/vernacularNames (eng)',
                       none: 'the scientific name',
                       wikidataApi: WIKIDATA_API },
        taxaNames: { order: ['override', 'wikidata', 'gbif', 'none'],
                     override: 'taxa-names.json of the generator, then the config\'s taxaNames',
                     wikidata: 'item by P846 (GBIF ID), else P225 (taxon name) with P105 (rank); the first English ' +
                               'P1843 (taxon common name), label, alias or enwiki title that is not a Latin name',
                     gbif: 'the most attested English /species/{key}/vernacularNames of the backbone key',
                     none: 'rank only, no common' },
        editorial: {
            excludeDatasets: (cfg.excludeDatasets || []).map((x) => Object.assign({}, x,
                { removed: excluded.filter((e) => e.dataset === x.key) })),
            merges, drops,
            nameOverrides: overrides.applied, unusedNameOverrides: overrides.unused,
            taxaNameOverrides: taxa.overrides.applied,
        },
        queries, totals: summary,
    };
    const regionNames = cfg.regions.map((r) => r.code).join(', ');
    const doc = {
        name: cfg.name,
        description: ((new Set(cfg.groups.map((g) => minOf(g.label)))).size === 1
            ? `${cfg.groups.map((g) => g.label).join(' and ')} with at least ${minOf(cfg.groups[0].label)} `
            : `${cfg.groups.map((g) => `${g.label} (at least ${minOf(g.label)})`).join(' and ')} with that many `) +
            `CC0 / CC BY 4.0 GBIF occurrence records (${cfg.filters.year ? 'years ' + cfg.filters.year + ', ' : ''}` +
            `occurrenceStatus ${cfg.filters.occurrenceStatus || 'any'}) in at least one of ${regionNames}; ` +
            `accidental:true below ${cfg.accidentalShare} of a region's records everywhere it counts. ` +
            `GBIF Backbone Taxonomy names, English common names from Wikidata (CC0) else GBIF, ` +
            `with hand-made merges and name overrides listed in meta.editorial. See ATTRIBUTION.md.`,
        taxonomy: 'GBIF Backbone Taxonomy',
        regions: codes,
        meta,
        taxa: taxa.taxa,
        species: rows,
    };
    fs.mkdirSync(cfg.out, { recursive: true });
    fs.writeFileSync(`${cfg.out}/species.json`, out.speciesJson(doc), 'utf8');
    fs.writeFileSync(`${cfg.out}/datasets.tsv`, out.datasetsTsv(dsInfo), 'utf8');
    fs.writeFileSync(`${cfg.out}/ATTRIBUTION.md`, out.attribution(doc, cfg, dsInfo), 'utf8');
    fs.writeFileSync(`${cfg.out}/LICENSE`, await out.licenseText(http), 'utf8');

    log(`wrote ${cfg.out}: ${rows.length} species (${JSON.stringify(summary.byClass)}), ` +
        `${summary.accidental} accidental, ${dsInfo.length} datasets`);
    log(`  by region ${JSON.stringify(byRegion)}`);
    log(`  common names ${JSON.stringify(summary.commonNameSource)}`);
    log(`  http: ${http.stats.requests} requests, ${http.stats.cached} cached, ${http.stats.retried} retries; ` +
        `${((Date.now() - t0) / 1000).toFixed(1)} s`);
}

// Headless runs a classic script to its end and exits, so pump the event loop
// (timers, fetch completions) until main settles.
let done = false, failed = null;
main().catch((e) => { failed = e; }).finally(() => { done = true; });
while (!done) { advanceTime(16); wallSleep(16); }
if (failed) {
    console.error('gen-species-gbif failed: ' + (failed.stack || failed));
    process.exit(1);
}

// editorial.js: the hand-made corrections a region config may carry on top of
// the data, each with a `why` that is copied into species.json's meta.
//
//   "excludeDatasets": [{"key": "<datasetKey>", "regions": ["WI"], "why": ".."}]
//       records of that dataset (in those regions, or all) are not counted:
//       for captive collections and the like. Subtracted per region x group with
//       a datasetKey-filtered facet, so totals and datasets.tsv agree.
//   "merge": [{"from": <taxon>, "into": <taxon>, "why": ".."}]
//       backbone duplicates and misfilings: from's records are added to into's,
//       region by region, before the threshold; from's row disappears.
//   "drop":  [{"taxon": <taxon>, "why": ".."}]
//       a species removed outright.
//   "names": {"<speciesKey or scientific name>": {"common": "..", "why": ".."}}
//       a common name that replaces the sourced one (commonSource "override").
//
// A <taxon> is a GBIF backbone key or a scientific name (resolved with the
// backbone's species/match; an inexact match is an error).

const gbif = require(bro.appDir + '/gbif.js');

async function resolveRef(http, ref) {
    if (typeof ref === 'number' || /^\d+$/.test(String(ref))) return { key: String(ref), name: String(ref) };
    const t = await gbif.resolveTaxon(http, { name: ref });
    return { key: String(t.key), name: t.name };
}

// Records of the excluded datasets in this region x group, subtracted in place
// from the species and dataset facets. Returns what was removed.
async function subtractDatasets(http, cfg, pairs, region, sp, ds) {
    const removed = [];
    for (const x of cfg.excludeDatasets || []) {
        if (x.regions && !x.regions.includes(region.code)) continue;
        const sub = await gbif.facetAll(http, pairs.concat([['datasetKey', x.key]]), 'speciesKey');
        if (!sub.total) continue;
        for (const [k, n] of sub.counts) {
            const left = (sp.counts.get(k) || 0) - n;
            if (left > 0) sp.counts.set(k, left); else sp.counts.delete(k);
        }
        sp.total -= sub.total;
        const dleft = (ds.counts.get(x.key) || 0) - sub.total;
        if (dleft > 0) ds.counts.set(x.key, dleft); else ds.counts.delete(x.key);
        removed.push({ dataset: x.key, region: region.code, records: sub.total, species: sub.counts.size });
    }
    return removed;
}

// Applies merge and drop to perSpecies (key -> {group, records}) in place.
async function mergeAndDrop(http, cfg, perSpecies, log) {
    const merges = [], drops = [];
    for (const m of cfg.merge || []) {
        const from = await resolveRef(http, m.from), into = await resolveRef(http, m.into);
        const src = perSpecies.get(from.key);
        const entry = { from: from.name, fromKey: Number(from.key), into: into.name, intoKey: Number(into.key),
                        why: m.why || '', records: src ? Object.assign({}, src.records) : {} };
        merges.push(entry);
        if (!src) { log(`  merge: ${from.name} has no records, nothing to move`); continue; }
        const dst = perSpecies.get(into.key) || { group: src.group, records: {} };
        for (const [c, n] of Object.entries(src.records)) dst.records[c] = (dst.records[c] || 0) + n;
        perSpecies.set(into.key, dst);
        perSpecies.delete(from.key);
    }
    for (const d of cfg.drop || []) {
        const t = await resolveRef(http, d.taxon);
        const src = perSpecies.get(t.key);
        drops.push({ taxon: t.name, key: Number(t.key), why: d.why || '', records: src ? src.records : {} });
        perSpecies.delete(t.key);
    }
    if (merges.length || drops.length) log(`  editorial: ${merges.length} merge(s), ${drops.length} drop(s)`);
    return { merges, drops };
}

// Replaces common names from cfg.names; returns the overrides applied and
// the entries that matched no species.
function overrideNames(cfg, species) {
    const table = cfg.names || {};
    const applied = [];
    const used = new Set();
    for (const s of species) {
        const k = table[String(s.key)] ? String(s.key) : (table[s.scientific] ? s.scientific : null);
        if (!k) continue;
        used.add(k);
        const o = table[k];
        applied.push({ scientific: s.scientific, gbifKey: Number(s.key), common: o.common, was: s.common,
                       wasSource: s.source, why: o.why || '' });
        Object.assign(s, { common: o.common, source: 'override' });
    }
    const unused = Object.keys(table).filter((k) => !used.has(k));
    return { applied, unused };
}

module.exports = { subtractDatasets, mergeAndDrop, overrideNames };

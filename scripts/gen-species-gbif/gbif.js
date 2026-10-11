// gbif.js: the GBIF API calls the species generator makes.
//
// Occurrence counts come from the occurrence search API's facets (limit=0, so
// no records are transferred), names and higher taxonomy from the species API
// (the GBIF Backbone Taxonomy), dataset titles/DOIs from the registry.

const API = 'https://api.gbif.org/v1';

// Every basisOfRecord value the occurrence API knows.
const BASIS_OF_RECORD = ['HUMAN_OBSERVATION', 'MACHINE_OBSERVATION', 'OBSERVATION', 'OCCURRENCE',
                         'PRESERVED_SPECIMEN', 'MATERIAL_SAMPLE', 'MATERIAL_CITATION',
                         'LIVING_SPECIMEN', 'FOSSIL_SPECIMEN'];

// Query parameters shared by every occurrence search: the licence filter and
// the record filters, then the region, then the taxon group. Returned as
// [key, value] pairs, since GBIF repeats a key to OR its values.
function occurrenceParams(filters, region, group) {
    const p = [];
    for (const lic of filters.licenses) p.push(['license', lic]);
    if (filters.occurrenceStatus) p.push(['occurrenceStatus', filters.occurrenceStatus]);
    if (filters.year) p.push(['year', filters.year]);
    for (const b of filters.basisOfRecord || []) p.push(['basisOfRecord', b]);
    if (filters.hasGeospatialIssue !== undefined && filters.hasGeospatialIssue !== null)
        p.push(['hasGeospatialIssue', String(filters.hasGeospatialIssue)]);
    if (region.gadmGid) p.push(['gadmGid', region.gadmGid]);
    if (region.country) p.push(['country', region.country]);
    for (const s of [].concat(region.stateProvince || [])) p.push(['stateProvince', s]);
    for (const k of group.keys) p.push(['taxonKey', String(k)]);
    return p;
}

function query(pairs) {
    return pairs.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&');
}

function searchUrl(pairs, extra = []) {
    return `${API}/occurrence/search?${query(pairs.concat(extra))}`;
}

// Every value of facet `field` with its record count, paging through
// facetOffset until a page comes back short. Also returns the query's total.
async function facetAll(http, pairs, field, pageSize = 1000) {
    const counts = new Map();
    let total = 0;
    for (let offset = 0; ; offset += pageSize) {
        const url = searchUrl(pairs, [['limit', '0'], ['facet', field],
            ['facetLimit', String(pageSize)], ['facetOffset', String(offset)]]);
        const res = await http.json(url);
        total = res.count;
        const facet = (res.facets || [])[0];
        const page = facet ? facet.counts : [];
        for (const { name, count } of page) counts.set(name, (counts.get(name) || 0) + count);
        if (page.length < pageSize) break;
    }
    return { total, counts };
}

// A taxon given as a key (number or numeric string) or as {name, rank}; the
// name form is resolved with the backbone's species/match.
async function resolveTaxon(http, spec) {
    if (typeof spec === 'number' || /^\d+$/.test(String(spec))) {
        const s = await http.json(`${API}/species/${spec}`);
        return { key: s.key, name: s.canonicalName || s.scientificName, rank: s.rank };
    }
    const name = typeof spec === 'string' ? spec : spec.name;
    const params = [['name', name], ['strict', 'true']];
    if (spec.rank) params.push(['rank', spec.rank]);
    if (spec.kingdom) params.push(['kingdom', spec.kingdom]);
    const m = await http.json(`${API}/species/match?${query(params)}`);
    if (!m.usageKey || m.matchType === 'NONE')
        throw new Error(`taxon "${name}" does not match the GBIF backbone`);
    if (m.matchType !== 'EXACT')
        throw new Error(`taxon "${name}" only matches the backbone ${m.matchType} (${m.canonicalName}); give its key`);
    const key = m.status === 'ACCEPTED' || !m.acceptedUsageKey ? m.usageKey : m.acceptedUsageKey;
    return { key, name: m.canonicalName, rank: m.rank };
}

async function species(http, key) {
    return http.json(`${API}/species/${key}`);
}

// English vernacular names from the backbone's sources, most-attested first.
// GBIF tags English as 'eng'.
async function englishVernaculars(http, key) {
    const tally = new Map();
    for (let offset = 0; ; offset += 1000) {
        const res = await http.json(`${API}/species/${key}/vernacularNames?limit=1000&offset=${offset}`);
        for (const v of res.results || []) {
            if (v.language !== 'eng' || !v.vernacularName) continue;
            const name = v.vernacularName.trim().replace(/’/g, "'");
            const k = name.toLowerCase();
            const t = tally.get(k) || { name, n: 0, preferred: false };
            t.n++;
            if (v.preferred) t.preferred = true;
            // Keep the capitalised spelling when sources disagree only in case.
            if (/^[A-Z]/.test(name) && !/^[A-Z]/.test(t.name)) t.name = name;
            tally.set(k, t);
        }
        if (res.endOfRecords !== false) break;
    }
    return [...tally.values()].sort((a, b) => (b.preferred - a.preferred) || (b.n - a.n));
}

async function dataset(http, key) {
    try {
        const d = await http.json(`${API}/dataset/${key}`);
        return { key, title: d.title || '', doi: d.doi || '', license: d.license || '',
                 publisher: d.publishingOrganizationKey || '' };
    } catch (e) {
        return { key, title: '', doi: '', license: '', publisher: '', error: String(e.message || e) };
    }
}

module.exports = { API, BASIS_OF_RECORD, occurrenceParams, searchUrl, query, facetAll, resolveTaxon, species,
                   englishVernaculars, dataset };

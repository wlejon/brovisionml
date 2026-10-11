// output.js: the files the species generator writes besides the rows themselves.

const LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/legalcode.txt';

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December'];

function citeDate(iso) {
    const d = new Date(iso);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

// Pretty-printed, but one species per line, the way the hand-made lists read.
function speciesJson(doc) {
    const head = Object.assign({}, doc);
    delete head.species;
    let s = JSON.stringify(head, null, 2);
    s = s.slice(0, s.lastIndexOf('}')).replace(/\s+$/, '');
    s += ',\n  "species": [\n' + doc.species.map((r) => '    ' + JSON.stringify(r)).join(',\n') + '\n  ]\n}\n';
    return s;
}

const tsvCell = (v) => String(v ?? '').replace(/[\t\r\n]+/g, ' ');

function datasetsTsv(ds) {
    const lines = ['datasetKey\trecords\tdoi\tlicense\ttitle'];
    for (const d of ds) lines.push([d.key, d.records, d.doi, d.license, d.title].map(tsvCell).join('\t'));
    return lines.join('\n') + '\n';
}

function attribution(doc, cfg, ds) {
    const m = doc.meta;
    const t = m.totals;
    const date = citeDate(m.generated);
    const groups = m.groups.map((g) => `- **${g.label}:** ` +
        g.taxa.map((x) => `${x.name} (${x.rank.toLowerCase()}, key ${x.key})`).join(', ')).join('\n');
    const regions = m.regions.map((r) => `| ${r.code} | ` + (r.gadmGid ? `gadmGid=${r.gadmGid}` :
        `country=${r.country}, stateProvince=${[].concat(r.stateProvince).join(' or ')}`) + ' |').join('\n');
    const queries = m.queries.map((q) => `| ${q.region} | ${q.group} | ${q.records} | ${q.species} | ${q.datasets} |`).join('\n');
    const urls = m.queries.map((q) => `- ${q.region} ${q.group}: <${q.url}>`).join('\n');
    const top = ds.slice(0, 15).map((d) => `| ${d.title || d.key} | ${d.doi ? 'https://doi.org/' + d.doi : d.key} | ${d.records} |`).join('\n');
    const cmd = `bro-headless --no-gpu scripts/gen-species-gbif scripts/gen-species-gbif/generate.js \\\n` +
        `    --config scripts/gen-species-gbif/regions/<region>.json --out <dir> --cache <cache dir>`;
    return `# ${doc.name}: sources, licences and method

\`species.json\` is a regional species list (the label set of a zero-shot species
classifier) compiled from GBIF occurrence data by \`${m.generator}\` ${m.version}
on ${date} (${m.generated}). \`datasets.tsv\` lists every GBIF dataset whose records
were counted, with its record count and DOI.

## Citation

Please cite GBIF in its derived-dataset form (the list: *${doc.name}, species with
CC0 / CC BY 4.0 occurrence records in ${doc.regions.join(', ')}*):

\`\`\`
GBIF.org (${date}) Derived dataset GBIF.org https://doi.org/10.15468/dd.<pending>
DOI: (pending GBIF derived-dataset registration)
\`\`\`

The derived-dataset DOI is minted by registering this list at
https://www.gbif.org/derived-dataset/register with \`datasets.tsv\` (dataset key
and record count per dataset) and the search URLs below. Until then, the
constituent datasets are cited through \`datasets.tsv\` (${t.datasets} datasets,
${t.datasetRecords} records).

## Licences

- **This list** (\`species.json\`, \`datasets.tsv\`): Creative Commons Attribution
  4.0 International (CC BY 4.0), full text in \`LICENSE\`. Attribution: GBIF.org
  and the data publishers in \`datasets.tsv\`, as cited above.
- **Occurrence records:** only records licensed CC0 1.0 or CC BY 4.0 were counted
  (GBIF licence filter \`${m.licenseFilter.map((l) => 'license=' + l).join('&')}\`);
  CC BY-NC 4.0 records are excluded. The filter is per record, so a dataset whose
  default licence is CC BY-NC can still contribute its CC0 / CC BY records (the
  \`license\` column of \`datasets.tsv\` is the dataset's default). Only counts per
  species are used, no records are redistributed.
- **Scientific names and higher taxonomy:** GBIF Backbone Taxonomy
  (dataset ${m.taxonomyDataset || 'd7dddbf4-2cf0-4f39-9b2a-bb099caae36c'},
  https://doi.org/10.15468/39omei), CC BY 4.0.
- **English common names:** Wikidata (CC0 1.0) where an item carries one
  (\`commonSource: "wikidata"\`, with the item id in \`wikidata\`); otherwise the GBIF
  Backbone's English vernacular names (\`commonSource: "gbif"\`, CC BY 4.0); a species
  with neither keeps its scientific name (\`commonSource: "none"\`).

## Query

GBIF occurrence search API (${m.gbifApi}/occurrence/search), \`limit=0\`, counted with
\`facet=speciesKey\` (species list) and \`facet=datasetKey\` (datasets), paged with
\`facetOffset\`. Filters on every query: ${Object.entries(m.filters).filter(([, v]) => v !== undefined)
    .map(([k, v]) => [].concat(v).map((x) => `\`${k === 'licenses' ? 'license' : k}=${x}\``).join(' or ')).join(', ')}.

| Region | GBIF filter |
|---|---|
${regions}

Taxon groups (GBIF \`taxonKey\`, OR'd within a group):

${groups}

| Region | Group | Records | Species seen | Datasets |
|---|---|---|---|---|
${queries}

Search URLs (as run; add \`facet=speciesKey\` or \`facet=datasetKey\`):

${urls}

## Method

1. Each region x group search gives a record count per species (\`speciesKey\`
   facet; records identified to subspecies roll up to their species, records not
   identified to species are not counted).
2. A species counts in a region when it has at least **${m.minRecords}** records there
   (\`minRecords\`), which drops one-off misidentifications and mis-georeferenced
   records; it is kept when it counts in at least one region. \`regions\` lists the
   regions it counts in and \`records\` its record count in every region it has any.
   ${t.speciesSeen} species were seen, ${t.droppedBelowThreshold} fell below the threshold everywhere.
3. \`accidental\` is true when, in every region the species counts in, its records
   are below **${m.accidentalShare}** of that region's records for its group
   (\`accidentalShare\`): vagrants that birders chase pile up records, but still a
   tiny share of a well-watched region's total.
4. Names and higher taxonomy (kingdom to family) come from the GBIF Backbone
   species record of each \`speciesKey\` (\`gbifKey\`). Common names as above.
5. Rows are ordered by group (${m.groups.map((g) => g.label).join(', ')}), then order,
   family and scientific name. \`meta\` in \`species.json\` records the parameters,
   the queries and the totals.

## Contents

${t.species} species: ${Object.entries(t.byClass).map(([k, v]) => `${k} ${v}`).join(', ')};
${t.accidental} marked accidental. Per region: ${Object.entries(t.byRegion).map(([k, v]) => `${k} ${v}`).join(', ')}.
Common names: ${Object.entries(t.commonNameSource).map(([k, v]) => `${k} ${v}`).join(', ')}.

Largest contributing datasets (all ${t.datasets} in \`datasets.tsv\`; the first holds
${ds.length ? (100 * ds[0].records / t.datasetRecords).toFixed(1) : 0}% of the counted records):

| Dataset | DOI | Records |
|---|---|---|
${top}

## Regenerating

From the brovisionml checkout (see \`docs/species-lists.md\`):

\`\`\`
${cmd}
\`\`\`
`;
}

async function licenseText(http) {
    return http.text(LICENSE_URL);
}

module.exports = { speciesJson, datasetsTsv, attribution, licenseText, LICENSE_URL };

# Regional species lists (GBIF)

A zero-shot species classifier (BioCLIP, through `bro.lm.loadClip`) scores an
image against a fixed label set: one prompt per species, built from its
taxonomy and common name. The label set is a regional species list, a JSON
file in `brovisionml-data/species/`. `scripts/gen-species-gbif/` generates one
from openly licensed data, so the list can be published with the weights:

- **which species occur** where: GBIF occurrence records licensed CC0 1.0 or
  CC BY 4.0, counted per species and region;
- **scientific names and higher taxonomy:** the GBIF Backbone Taxonomy (CC BY 4.0);
- **English common names:** Wikidata (CC0), else GBIF's vernacular names.

The output is CC BY 4.0 (attribution to GBIF.org and the contributing
datasets). It replaces, for shipping, the hand-built `upper-midwest.json`, whose
taxonomy comes from the eBird API under terms that are not an open licence and
which therefore stays local.

## Running it

The generator is plain JavaScript run by `bro-headless` (it uses brokit's
`fetch` and `fs`; no Node, no Python). From a brovisionml checkout:

```bash
bro-headless --no-gpu scripts/gen-species-gbif scripts/gen-species-gbif/generate.js \
    --config scripts/gen-species-gbif/regions/upper-midwest.json \
    --out ../brovisionml-data/species/upper-midwest-gbif \
    --cache /tmp/gbif-cache
```

One command per region file. It takes about a minute (around 800 requests, at
most 4 in flight); with `--cache` a re-run with different thresholds reads
every response from disk. Requests carry a User-Agent naming the project and
retry 429 / 5xx / network errors with exponential backoff (Retry-After
honoured).

It writes, into `--out`:

| File | What |
|---|---|
| `species.json` | the list: `name`, `description`, `taxonomy`, `regions`, `meta`, `species[]` |
| `datasets.tsv` | every GBIF dataset whose records were counted: key, record count, DOI, default licence, title |
| `ATTRIBUTION.md` | the GBIF citation (derived-dataset form, DOI pending), licences, the exact search URLs, method and totals |
| `LICENSE` | the CC BY 4.0 legal code, fetched verbatim from creativecommons.org |

### Rows

```json
{"common": "Black-capped Chickadee", "scientific": "Poecile atricapillus",
 "kingdom": "Animalia", "phylum": "Chordata", "class": "Aves", "order": "Passeriformes",
 "family": "Paridae", "regions": ["MN", "WI", "IA", "ND", "SD", "MB", "ON"], "accidental": false,
 "records": {"MN": 785399, "...": 0}, "gbifKey": 2487805, "commonSource": "wikidata", "wikidata": "Q282687"}
```

The first nine fields are the shape the classifier reads (the prompt is
`a photo of <kingdom> <phylum> <class> <order> <family> <scientific> with common
name <common>.`; `accidental: false` filters on the flag, `classes` on `class`).
The rest are provenance: `records` per region (regions under the threshold
included, so the cut can be re-judged), the backbone key, which source gave
the common name (`wikidata` / `gbif` / `none`), the Wikidata item, and
`commonAlt`, the item's other English names, when it has any.

`meta` records the generator version, the generation time, the API base, the
licence filter and record filters, the regions and resolved taxon keys, the
thresholds, every search URL with its totals, and the totals of the list
(species by class and region, common-name sources, datasets).

## Parameters

A region file (`regions/*.json`) holds everything; flags override it.

| Config key | Flag | Default | Meaning |
|---|---|---|---|
| `name` | `--name` | output folder name | the list's `name` |
| `regions` | `--region CODE=CC:Province` (repeatable) | | `{code, country, stateProvince}`; `stateProvince` may be an array of spellings, or `{code, gadmGid}` filters by GADM level-1 polygon instead (`--region MN=gadm:USA.24_1`) |
| `groups` | `--group label=taxa` (repeatable) | | `{label, taxa}`: taxa are backbone keys or `{name, rank}` resolved with species/match (an inexact match is an error); the keys of one group are OR'd in one search |
| `filters` | `--year` | licences CC0_1_0 + CC_BY_4_0, `occurrenceStatus=PRESENT` | also `year` (`"1990,*"` in upper-midwest), `basisOfRecord` (array), `hasGeospatialIssue` |
| `minRecords` | `--min-records` | 10 | records a species needs in a region to count there |
| `accidentalShare` | `--accidental-share` | 0.00002 | the `accidental` cut, below |
| `nameSources` | `--name-sources` | `wikidata,gbif` | common-name sources in order |
| | `--out`, `--cache`, `--concurrency` (1..4) | | |

`upper-midwest.json` covers MN, WI, IA, ND, SD (US) and MB, ON (CA) by
`country` + `stateProvince`, birds (class Aves, key 212), and the medium and
large mammals seen in yards and on trail cameras: squirrels and chipmunks
(Sciuridae), beaver, porcupine, muskrat (*Ondatra*), rabbits and hares,
deer, canids, cats, bears, raccoon, skunks, mustelids and the opossum. Small
rodents, shrews and bats are left out, as photos rarely tell them apart.

## Method

1. **Counts.** For every region x group, an occurrence search with `limit=0`
   and `facet=speciesKey` (paged with `facetOffset`) gives the record count of
   each species. Records identified to subspecies roll up to their species;
   records identified only to genus are not counted. A second facet,
   `datasetKey`, over the same search gives the contributing datasets.
2. **Licence filter.** `license=CC0_1_0&license=CC_BY_4_0` is per record, so CC
   BY-NC records are left out even inside a dataset that also holds open ones.
   It keeps 99% of the bird records (20.29 M of 20.47 M in Minnesota), as eBird's
   EOD dataset, most of them, is CC BY 4.0. Mammals lose far more: most
   iNaturalist observations are CC BY-NC, so the mammal counts are hundreds where
   birds have millions.
3. **Modern records.** `year=1990,*` drops museum specimens of extinct species
   (Passenger Pigeon, Carolina Parakeet) and historic records of extirpated
   ones; `occurrenceStatus=PRESENT` drops absence records.
4. **Threshold, `minRecords` = 10.** A species counts in a region at 10 or more
   records there, and is kept when it counts in any region. In the
   upper-midwest run 739 species keys have a record, 731 of them not hybrids;
   with N or more records in some region: 636 at 3, 612 at 5, 597 at 10, 571 at
   20, 533 at 50 (hybrids dropped). The 5 to 9
   band is a mix of one-off vagrants (Kelp Gull, Grace's Warbler), escapes
   (Silver Teal, Monk Parakeet, Yellow-fronted Canary) and naming artefacts
   (Common Snipe for Wilson's Snipe, Audubon's Warbler as a species), while a
   vagrant that birders chase collects dozens of eBird checklists in a few
   days. 10 is the
   smallest cut that drops most of the noise without losing chased rarities;
   `records` keeps every region's count, so a consumer can apply a stricter one.
5. **`accidental`.** True when, in every region the species counts in, its
   records are under `accidentalShare` of that region's records for its group.
   A share rather than a count because the regions differ 25-fold in records
   (ON 60 M birds, SD 2.4 M). 2e-5 (20 records per million) was chosen by
   comparing with the regular/accidental status of the state and provincial
   checklists on the 505 bird species both lists hold: it agrees on 483 (1e-5:
   469, 5e-5: 471). Escaped and captive birds with records (the cranes at a
   Wisconsin crane collection, peafowl, budgerigars) fall under it too.
6. **Hybrids** (`Anas rubripes x platyrhynchos`) are dropped.
7. **Names.** Higher taxonomy and the scientific name are the backbone's
   (`/v1/species/{key}`). The common name is Wikidata's: the item with P846
   (GBIF taxon ID) = the key, else P225 (taxon name) = the scientific name,
   found with the Action API (`haswbstatement` search + `wbgetentities`; the
   SPARQL endpoint rate-limits too hard to rely on). Of the item's names, a
   P1843 (taxon common name) equal to its English Wikipedia title wins, then one
   equal to its English label, then the title, then the label. Without an item
   or name, GBIF's: the backbone's `vernacularName`, else the most attested
   English entry of `/vernacularNames`, skipping group names
   ("Snow/Ross's Goose"); else the scientific name.
8. **Order.** Rows by group, then order, family, scientific name.

## Caveats

- **Taxonomy is the GBIF backbone's, not Clements/AOS.** The backbone lags the
  checklists: it keeps *Phalacrocorax auritus*, *Bubulcus ibis*, *Grus
  canadensis*, *Regulus calendula*, *Accipiter cooperii*, and lumps some recent
  North American splits into their Old World parent (*Setophaga petechia*,
  *Numenius phaeopus*, *Accipiter gentilis*, *Larus argentatus*, *Circus
  cyaneus*, *Cistothorus platensis*, *Gallinula chloropus*, *Anas crecca*). It
  also files the American Goshawk as *Falco atricapillus* in Falconidae, which
  has no common name anywhere (`commonSource: "none"`).
- **Wikidata names follow the item's concept and Wikipedia's (IOC) usage.** A
  lumped backbone species gets the Old World daughter's name (*Setophaga
  petechia* is "Mangrove Warbler" in Wikidata, though nearly every record here
  is a Yellow Warbler), and some names are British (Sand Martin, Grey Plover,
  Rough-legged Buzzard). `--name-sources gbif,wikidata` prefers GBIF's names,
  which follow the backbone's concept ("Yellow Warbler", "Herring Gull") but
  include eBird-style names and more gaps. Either way, check the list's names
  before shipping a region.
- **Region filter.** `stateProvince` is the publisher's text, interpreted by
  GBIF; a few records geolocated just across a border are included (22 k of
  Minnesota's 20 M bird records fall in Wisconsin). `gadmGid` filters by
  coordinates instead.
- **Data dominance.** eBird's EOD dataset holds ~98% of the counted bird
  records; it is published to GBIF under CC BY 4.0, which is what makes the
  list redistributable, and it is cited through `datasets.tsv`.

## Publishing

1. Register a derived dataset at https://www.gbif.org/derived-dataset/register
   with `datasets.tsv` (dataset keys and record counts) and the search URLs in
   `ATTRIBUTION.md`; GBIF mints a `10.15468/dd.*` DOI.
2. Replace the `DOI: (pending GBIF derived-dataset registration)` line and the
   `<pending>` in the citation of `ATTRIBUTION.md` with it.
3. Ship `species.json`, `datasets.tsv`, `ATTRIBUTION.md` and `LICENSE` together.

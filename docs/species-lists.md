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
| `species.json` | the list: `name`, `description`, `taxonomy`, `regions`, `meta`, `taxa`, `species[]` |
| `datasets.tsv` | every GBIF dataset whose records were counted: key, record count, DOI, default licence, title |
| `ATTRIBUTION.md` | the citation (GBIF.org and the datasets in `datasets.tsv`), licences, the exact search URLs, method and totals |
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
the common name (`wikidata` / `gbif` / `override` / `none`), the Wikidata item, and
`commonAlt`, the item's other English names, when it has any.

### Taxa

Every list carries a top-level `taxa` map, the English names of the genera
(the first word of `scientific`), families, orders and classes its rows use, so
a classifier can fall back from a species to a higher rank ("Darners
(Aeshnidae)") when no species clears its threshold:

```json
"taxa": {
  "Odonata":   {"rank": "order",  "common": "Dragonflies and damselflies", "commonSource": "wikidata", "wikidata": "Q25375", "gbifKey": 789},
  "Aeshnidae": {"rank": "family", "common": "Darners", "commonSource": "override", "wikidata": "Q516862", "gbifKey": 4209},
  "Aeshna":    {"rank": "genus",  "common": "Mosaic darners", "commonSource": "override", "wikidata": "Q746347", "gbifKey": 1425165}
}
```

`rank` is the row field the name is in, so a row without a rank adds nothing for
it (the herps' `Squamata` and `Testudines` are classes, with no order).
`common` is a short group name in sentence case, proper words kept ("New World
warblers"); an entry with no name worth giving keeps only `rank` (most insect
genera). `taxa.js` picks it:

1. **By hand**, from `scripts/gen-species-gbif/taxa-names.json` (shared by every
   list), then the config's `taxaNames`, each with a `why`
   (`commonSource: "override"`).
2. **Wikidata (CC0):** the item by the backbone key (P846), else by taxon name
   (P225) with the rank (P105), skipped when two items match (the plant and
   animal *Morus*). Its English taxon common names (P1843), label, aliases and
   enwiki title are the candidates, cleaned of a leading article and a trailing
   rank word ("Cricket frog genus"), Latin names left out.
3. **GBIF's** English vernacular names for the key, most attested first.

Of the candidates, a plural or "X and Y" name from Wikidata wins, then one from
GBIF, then anglicised Latin ("accipitrids"), then Wikidata's singular made
plural ("Box turtle" to "Box turtles"). A GBIF name, or a Wikidata one found by
name, for a family or genus must share a word with the common names of the
rows under it or its parent's name: GBIF matches vernaculars by name, so a
homonym brings a plant's ("False indigos" for the sphinx moth genus *Amorpha*)
or nothing useful ("Animals"). The backbone keys come from the rows' species
records in a generated list, or from species/match (rank, higher taxa as hints)
for a list made elsewhere.

What the overrides fix: an order named for one of its families (Hymenoptera
"Ants", Accipitriformes "Hawks", Pelecaniformes "Herons"), British names
(Paridae "Tits", Aeshnidae "Hawker dragonflies", Tettigoniidae "Bushcrickets",
*Sympetrum* "Darters", *Riparia* "Sand martins"), slang or odd sources
(Trochilidae "Hummers", Haematopodidae "Seapies", *Grus* "Anthropoides"), and
the bird families field guides head with several names (Anatidae "Ducks, geese
and swans", Ardeidae "Herons, egrets and bitterns", Rallidae "Rails, gallinules
and coots"). `--taxa-report <file>` writes every entry with each source's
candidates, for review.

`--add-taxa <list.json>` adds the map to a list the generator did not make (the
eBird-named `upper-midwest.json`), or refreshes it in a published one without
re-counting (`upper-midwest-gbif`): the block goes in as one line per entry
before `"species"`, every other byte stays, and the rows are checked unchanged
before the file is written.

`meta` records the generator version, the generation time, the API base, the
licence filter and record filters, the regions and resolved taxon keys, the
thresholds, every search URL with its totals, the editorial corrections
(`meta.editorial`: excluded datasets with the records removed, merges with the
records moved, drops, name overrides with the name they replaced, each with
its `why`), and the totals of the list (species by class and region,
common-name sources, override, merge and drop counts, datasets).

## Parameters

A region file (`regions/*.json`) holds everything; flags override it.

| Config key | Flag | Default | Meaning |
|---|---|---|---|
| `name` | `--name` | output folder name | the list's `name` |
| `regions` | `--region CODE=CC:Province` (repeatable) | | `{code, country, stateProvince}`; `stateProvince` may be an array of spellings, or `{code, gadmGid}` filters by GADM level-1 polygon instead (`--region MN=gadm:USA.24_1`) |
| `groups` | `--group label=taxa` (repeatable) | | `{label, taxa}`: taxa are backbone keys or `{name, rank}` resolved with species/match (an inexact match is an error); the keys of one group are OR'd in one search |
| `filters` | `--year` | licences CC0_1_0 + CC_BY_4_0, `occurrenceStatus=PRESENT` | also `year` (`"1990,*"` in upper-midwest), `hasGeospatialIssue`, `basisOfRecord` (array, the values to count; overrides the next) |
| `filters.excludeBasisOfRecord` | `--exclude-basis A,B` | `LIVING_SPECIMEN, FOSSIL_SPECIMEN` | basisOfRecord values not counted (the search lists every other value: GBIF has no negated filter) |
| `excludeDatasets` | | none | `[{key, regions?, why}]`: a dataset's records are not counted (in those regions, or all), subtracted per search with a `datasetKey` facet |
| `merge` | | none | `[{from, into, why}]`: backbone duplicates and misfilings; `from`'s per-region counts are added to `into`'s before the threshold. Taxa are keys or scientific names |
| `drop` | | none | `[{taxon, why}]`: a species removed outright |
| `names` | | none | `{"<speciesKey or scientific name>": {common, why}}`: a common name replacing the sourced one (`commonSource: "override"`) |
| `minRecords` | `--min-records` | 10 | records a species needs in a region to count there; a group's own `minRecords` overrides it for that group. The log prints, per group, how many species have N+ records in some region, to choose it by |
| `taxaNames` | | none | `{"<taxon>": {common, why}}`: names of higher taxa over `taxa-names.json` (see Taxa) |
| `filtersNote`, `minRecordsNote` | | none | why the filters and the cut are what they are; copied into `ATTRIBUTION.md` |
| `accidentalShare` | `--accidental-share` | 0.00002 | the `accidental` cut, below |
| `nameSources` | `--name-sources` | `wikidata,gbif` | common-name sources in order |
| | `--out`, `--cache`, `--concurrency` (1..4), `--taxa-report`, `--add-taxa` | | |

The editorial keys are explained under "Editorial corrections" below.

`upper-midwest.json` covers MN, WI, IA, ND, SD (US) and MB, ON (CA) by
`country` + `stateProvince`, birds (class Aves, key 212), and the medium and
large mammals seen in yards and on trail cameras: squirrels and chipmunks
(Sciuridae), beaver, porcupine, muskrat (*Ondatra*), rabbits and hares,
deer, canids, cats, bears, raccoon, skunks, mustelids and the opossum. Small
rodents, shrews and bats are left out, as photos rarely tell them apart.

`upper-midwest-herps.json` covers the same seven regions for reptiles and
amphibians (78 species: 31 snakes and lizards, 14 turtles, 33 frogs, toads
and salamanders), written to `species/upper-midwest-herps/`.
`bro.vision.wildlife` merges it with the eBird-named `upper-midwest.json`
by default, so the birds keep their checklist names and the herps come from
open data. Its differences from the birds config:

- **Groups.** The GBIF backbone has no class Reptilia: turtles (Testudines)
  and snakes and lizards (Squamata) are classes of their own there, so the
  reptile group is those two classes, and a row's `class` is `Testudines` or
  `Squamata` with no `order`. (BioCLIP 2 scored the region's garter snakes a
  little higher with class Reptilia and the order filled in, 0.92 against
  0.89 on one burst, but the list keeps the backbone's taxonomy like the
  others.)
- **`minRecords` 5.** Open records are scarce for herps (most iNaturalist
  observations are CC BY-NC): every species with 5 to 9 records in a region
  is a native of it (Western Ratsnake, Six-lined Racerunner, Smooth
  Softshell, Western Ribbon Snake, Sagebrush Lizard, Northern Dusky
  Salamander), while those under 5 are mostly escapes and misplaced records
  (Pig-nosed Turtle, Brown Anole, Cottonmouth).
- **Editorial.** Dropped: "Amblystoma spec" (genus-only records under a
  doubtful name), "Ambystoma unisexual" (the unisexual salamander complex,
  not a species) and *Testudo nebrascensis* (Badlands fossil tortoises
  catalogued with a recent year). Merged: Ontario's ratsnake records on
  *Pantherophis alleghaniensis* into *P. spiloides* (Gray Ratsnake), and the
  doubtful second key of *Desmognathus fuscus*. Renamed to the names this
  region's field guides use: Fowler's Toad (Wikidata's label was "Bufo
  Fowleri"), Bullsnake (the subspecies here; Wikidata names the Pacific
  one), Northern Water Snake, Gray Ratsnake, Eastern Box Turtle, Red-eared
  Slider, Smallmouth Salamander, Northern Dusky Salamander. Garter snakes
  keep Wikidata's spaced form (Common Garter Snake; field guides also write
  Common Gartersnake).

`upper-midwest-arthropods.json` covers the same seven regions for insects
(class Insecta) and spiders and harvestmen (orders Araneae and Opiliones),
written to `species/upper-midwest-arthropods/`: 1350 species, Lepidoptera
730, Coleoptera 122, Araneae 111, Hymenoptera 102, Diptera 90, Odonata 88,
Hemiptera 62, Orthoptera 19, Trichoptera 7, Opiliones 5, and 14 in seven
smaller orders (none accidental: at these counts no species is under 2e-5 of a
region's records). Its differences:

- **Observations only** (`basisOfRecord` HUMAN_OBSERVATION and
  MACHINE_OBSERVATION). In Ontario 72% of the open insect records are
  DNA-barcoded trap material (iBOL, 763 k MATERIAL_SAMPLE; the Centre for
  Biodiversity Genomics, 377 k PRESERVED_SPECIMEN): counted, the most recorded
  species are the malaise traps' midges, gall midges and parasitoid wasps.
  Observations (iNaturalist's CC0 / CC BY ones, BugGuide, Bumble Bee Watch, the
  Lepidopterists' Society, PollardBase) are the species people see and
  photograph.
- **`minRecords` 40 for insects, 10 for spiders** (a group cut). Insects with
  N+ observations in some region: 1881 at 20, 1070 at 50, 650 at 100; 40 keeps
  1234 (after the merge below). Spiders have about a thirtieth of the records: 116 at 10, 73 at 20.
  Ontario's 286 k insect observations set most of the list (1247 species count
  there, 332 in MN).
- **Editorial.** Merged: *Eurema lisa* into *Pyrisitia lisa* (Little Yellow,
  two backbone keys). Named: butterflies the backbone files under an old name
  with no English one (*Euptychia cymela*, *Elkalyce comyntas*, *Incisalia*,
  *Mitoura*, ...), British names (Camberwell Beauty, Small White, Four-spotted
  Chaser, Black Darter, Vapourer, Seven-spot Ladybird, German Wasp), GBIF
  vernaculars that name a family or group ("Soldier Beetle", "Crab Spider",
  "Wolf Spiders", "Cutworm"), Wikidata items carrying another species' name
  (*Hesperia comma* as Silver-spotted Skipper, *Sciaphilus asperatus* as
  Strawberry Root Weevil) and Spongy Moth for *Lymantria dispar*. A species
  with no English name anywhere keeps its scientific name (66). The generator
  now leaves out, as a species name, a group noun ("Moth", "Weevil", "Hover
  Fly") and a Latin binomial dressed as a name ("Melissodes Bimaculata",
  "Eurema Lisa"), and logs every common name two species share.

`species/pets/species.json` is hand-made (CC0, facts): the domestic dog,
*Canis lupus familiaris*, and the domestic cat, *Felis catus*, with
`regions: []` and its `taxa` from `--add-taxa`. The dog has the backbone's
accepted name, a subspecies (*Canis familiaris* is a synonym there): BioCLIP 2's
TreeOfLife-200M labels were resolved against the GBIF backbone (TaxonoPy) and
cut to seven ranks, so its dog images were captioned *Canis lupus*, and the
three-word name keeps the row apart from the Gray Wolf's (lists merge by
scientific name). *Felis catus* is also a row of `upper-midwest-gbif`, with
the same name; merged, either stands.

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
3. **Modern, wild records.** `year=1990,*` drops museum specimens of extinct
   species (Passenger Pigeon, Carolina Parakeet) and historic records of
   extirpated ones; `occurrenceStatus=PRESENT` drops absence records; living and
   fossil specimens (zoo and aviary animals, fossils) are not counted. That last
   filter changes no species in upper-midwest (it removes 4 small datasets): the
   captive birds that do show up are preserved specimens, such as the eggs of
   the International Crane Foundation's flock in a Wisconsin egg collection,
   which `excludeDatasets` takes out (7 crane species).
4. **Threshold, `minRecords` = 10.** A species counts in a region at 10 or more
   records there, and is kept when it counts in any region. In the
   first upper-midwest run (before the editorial corrections) 739 species keys
   had a record, 731 of them not hybrids; with N or more records in some region:
   636 at 3, 612 at 5, 597 at 10, 571 at 20, 533 at 50. The 5 to 9
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
   469, 5e-5: 471). Escaped birds seen in the wild (peafowl, budgerigars,
   Mandarin Duck) fall under it too.
6. **Editorial corrections** from the config (below): dataset exclusions while
   counting, then merges and drops, before the threshold, so a merged species is
   judged on its summed counts. **Hybrids** (`Anas rubripes x platyrhynchos`) are
   dropped.
7. **Names.** Higher taxonomy and the scientific name are the backbone's
   (`/v1/species/{key}`). The common name is Wikidata's: the item with P846
   (GBIF taxon ID) = the key, else P225 (taxon name) = the scientific name,
   found with the Action API (`haswbstatement` search + `wbgetentities`; the
   SPARQL endpoint rate-limits too hard to rely on). Of the item's names, a
   P1843 (taxon common name) equal to its English Wikipedia title wins, then one
   equal to its English label, then the title, then the label. Without an item
   or name, GBIF's: the backbone's `vernacularName`, else the most attested
   English entry of `/vernacularNames`, skipping group names
   ("Snow/Ross's Goose"); else the scientific name. Last, the config's `names`
   table replaces the names it lists (`commonSource: "override"`).
8. **Order.** Rows by group, then order, family, scientific name.

## Editorial corrections: why they exist

The data sources are kept as they are, and a region config corrects them in
the open. Each entry carries a `why`, and `ATTRIBUTION.md` and `meta.editorial`
list them all. Wikidata stays the first name source, and the overrides fix the
two ways it goes wrong for a North American list:

- **Backbone lumps with an Old World relative.** The GBIF backbone lags the
  checklists and keeps some American species lumped with their Old World (or
  tropical) relative, so nearly all of a familiar bird's records sit on the
  Old World name: *Setophaga petechia* (Yellow Warbler), *Anas crecca*
  (Green-winged Teal), *Numenius phaeopus* (Whimbrel), *Accipiter gentilis*
  (American Goshawk), *Circus cyaneus* (Northern Harrier), *Cistothorus
  platensis* (Sedge Wren), *Gallinula chloropus* (Common Gallinule), and among
  mammals *Cervus elaphus* (Elk). Wikidata's item for that name follows the
  current split and is named after the Old World bird ("Mangrove Warbler",
  "Eurasian Teal", "Hen Harrier", "Red Deer"), so the override gives the name
  of the bird actually recorded. The same lumps leave a small duplicate row for
  the American name (*Setophaga aestiva*, *Anas carolinensis*, *Numenius
  hudsonicus*), and the backbone also files the American Goshawk as *Falco
  atricapillus* in Falconidae with no common name anywhere; `merge` folds
  these into the row holding the records. Records under a pre-split name of an
  American species (*Larus argentatus*, *Lanius excubitor*, *Melanitta fusca*,
  *Troglodytes troglodytes*) are merged into it the same way.
- **British names.** Wikidata and English Wikipedia name taxa by IOC / British
  usage: Sand Martin, Grey Plover, Rough-legged Buzzard, Rock Dove, Little Auk,
  Black-necked Grebe, Common Starling, Stoat. The override gives the North
  American name (Bank Swallow, Black-bellied Plover, Rough-legged Hawk, ...).

`--name-sources gbif,wikidata` prefers GBIF's names instead, which follow the
backbone's concepts ("Yellow Warbler", "Herring Gull") but carry eBird group
names and their own oddities ("Eurasian Moorhen"); the overrides are the
smaller, reviewable fix. Hyphenation (Whistling Duck / Whistling-Duck,
Screech Owl / Screech-Owl) is left as Wikidata has it. A new region needs its
own review of names before it ships.

## Caveats

- **Taxonomy is the GBIF backbone's, not Clements/AOS.** Besides the lumps
  above it keeps older genera (*Phalacrocorax auritus*, *Bubulcus ibis*, *Grus
  canadensis*, *Regulus calendula*, *Accipiter cooperii*); the classifier's
  prompts use these names.
- **Region filter.** `stateProvince` is the publisher's text, interpreted by
  GBIF; a few records geolocated just across a border are included (22 k of
  Minnesota's 20 M bird records fall in Wisconsin). `gadmGid` filters by
  coordinates instead.
- **Data dominance.** eBird's EOD dataset holds ~98% of the counted bird
  records; it is published to GBIF under CC BY 4.0, which is what makes the
  list redistributable, and it is cited through `datasets.tsv`.

## Publishing

Ship `species.json`, `datasets.tsv`, `ATTRIBUTION.md` and `LICENSE` together:
`datasets.tsv` is the attribution, every constituent dataset with its DOI. The
lists are not registered as GBIF derived datasets (that needs a GBIF account);
GBIF asks for the registration, CC BY 4.0 does not require it.

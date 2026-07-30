# Reference audit

Audit date: 24 July 2026.

This file records the reference-level verification performed for the active
manuscript in `body.tex`. Metadata were checked against the publisher page,
DOI record, official proceedings page, or the primary paper. It does not
assert that every cited result has been independently reproduced.

## Corrected during this audit

- `keshavarz2011`: corrected the final DOI digit to
  [10.1177/0018720811403736](https://doi.org/10.1177/0018720811403736).
- `ang2023`: corrected the first author to Samuel Ang and added
  [10.3389/frvir.2023.1027552](https://doi.org/10.3389/frvir.2023.1027552).
- `kaufman2012`: added Ori Stitelman, the article-number page range, and
  [10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579).
- `kundu2022truvr`, `kundu2023litevr`, and `islam2021`: corrected Ripan
  Kundu/Rifatul Islam/Kevin Desai names as applicable and added the IEEE
  conference DOIs.
- `garrido2022`, `roberts2017`, `cho2014`, and `vaswani2017`: replaced
  abbreviated author lists with the full author lists.

## Active-reference status

| Key | Verification source | Status |
|---|---|---|
| `stanney1997` | [SAGE DOI](https://doi.org/10.1177/107118139704100292) | title, authors, venue, pages, year, DOI verified |
| `rebenitsch2016` | [Springer DOI](https://doi.org/10.1007/s10055-016-0285-9) | verified |
| `duzmanska2018` | [Frontiers DOI](https://doi.org/10.3389/fpsyg.2018.02132) | verified |
| `garrido2022` | [Springer DOI](https://doi.org/10.1007/s10055-022-00636-4) | full author list and metadata verified |
| `liao2020` | [IEEE DOI](https://doi.org/10.1109/ACCESS.2020.3008165) | verified |
| `magalhaes2021` | [IEEE DOI](https://doi.org/10.1109/ACCESS.2021.3084863) | verified |
| `shimada2023` | [IEEE DOI](https://doi.org/10.1109/ACCESS.2023.3312216) | verified |
| `kundu2025relaxvr` | [IEEE DOI](https://doi.org/10.1109/ACCESS.2025.3566958) | verified |
| `ang2023` | [Frontiers DOI](https://doi.org/10.3389/frvir.2023.1027552) | verified after author correction |
| `kundu2022truvr` | [IEEE DOI](https://doi.org/10.1109/ISMAR55827.2022.00096) | authors, pages, year, DOI verified |
| `kundu2023litevr` | [IEEE DOI](https://doi.org/10.1109/VR55154.2023.00076) | authors, pages, year, DOI verified |
| `islam2021` | [IEEE DOI](https://doi.org/10.1109/ISMAR52148.2021.00017) | authors, pages, year, DOI verified |
| `lopes2020` | [ACM DOI](https://doi.org/10.1145/3424636.3426906) | verified |
| `gavgani2017` | [Elsevier DOI](https://doi.org/10.1016/j.autneu.2016.12.004) | verified |
| `uyan2024` | [Elsevier DOI](https://doi.org/10.1016/j.displa.2024.102704) | verified |
| `setu2024` | [IEEE DOI](https://doi.org/10.1109/ISMAR62088.2024.00121) | full author list, pages, year, DOI verified |
| `kaufman2012` | [ACM DOI](https://doi.org/10.1145/2382577.2382579) | journal version verified |
| `roberts2017` | [Wiley DOI](https://doi.org/10.1111/ecog.02881) | full author list and metadata verified |
| `bai2018` | [primary arXiv record](https://arxiv.org/abs/1803.01271) | verified |
| `cho2014` | [ACL Anthology](https://aclanthology.org/D14-1179/) | full author list, pages, year, DOI verified |
| `vaswani2017` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | full author list, title, venue, year verified |
| `kingma2015` | [primary arXiv record](https://arxiv.org/abs/1412.6980) | authors, title, ICLR year verified |
| `keshavarz2011` | [SAGE DOI](https://doi.org/10.1177/0018720811403736) | verified after DOI correction |
| `cohen1968` | [APA DOI](https://doi.org/10.1037/h0026256) | verified |

## Data-source provenance

Corrected 25 July 2026. An earlier version of this note stated that Maze,
Simulations and Terrain were folders within one openly shared SAVELab
collection associated with `setu2024`. That is not accurate and the manuscript
no longer says it.

The public release accompanying `setu2024` is the VRWalking dataset
repository (`vrwalking2024`,
<https://github.com/Jyotinag/VRWalking_Dataset>). Its data paper describes a
single real-walking maze study with 39 recruited participants, 36 of whom
appear in its own case study. It contains the Maze source only; Simulations
and Terrain are neither included in nor described by that release.

Simulations and Terrain were obtained from the SAVELab through the group's
data request form and are available to other researchers by the same route.
All three sources originate from the same research group, which is why the
earlier conflation was easy to make, but "same group" is not "same release"
and the two must not be stated interchangeably.

Consequences recorded in the manuscript: the Data and Code Availability
section gives the two access routes separately and notes that the single-task
endpoint depends on the request-only sources; the Ethics and Consent section
applies the `setu2024` consent statement to Maze only and does not extend it
to the 47 participants in Simulations and Terrain; approving bodies and
protocol identifiers for those two sources were not available and are not
inferred. **Corrected again 25 July 2026.** An earlier version of this note stated that
`islam2021` remains in Related Work only and is not used to support data
provenance. That is now reversed. The Simulations source *is* the dataset
published in `islam2021`: that paper describes five environments -- Beach City,
Road Side, Furniture Shop, SeaVoyage and Roller Coaster -- recorded with an HTC
Vive Pro Eye with FMS at 30 s intervals, matching the five condition folders
(`beach`, `walk`, `room`, `sea`, `roller`) and the recording parameters of the
data we received. `islam2021` reports 30 participants recruited and 27
analyzed; our inclusion rule yields 25. The manuscript therefore cites
`islam2021` as the source of Simulations in Methods and in Data Availability.

Two consequences follow. First, Simulations is not a seated source: only Roller
Coaster and SeaVoyage are seated, while Road Side involves walking, Beach City
controller locomotion with object manipulation, and Furniture Shop room-scale
teleportation with visual search. The manuscript no longer calls this subset
"seated" and groups Simulations with Terrain as the *single-task* subset, the
shared property being the absence of a concurrent memory and attention task.
Second, `islam2021` reports no ethics approval identifier or consent language
that could be cited, so the ethics statement remains conservative for this
source.

Third, on availability: `kundu2023litevr` footnotes a public download for this
dataset at `tinyurl.com/2p92p45h`. That link was checked on 25 July 2026 and
does not resolve. Simulations is therefore not publicly downloadable through any
route we could verify, and the SAVELab data request form remains the only
working one. The Data Availability section reflects this.

The public `setu2024` paper reports 39 recruited participants and a 36-person
case-study subset after excluding three incomplete records. The released
files used here yield 37 Maze participants with the 14 common channels and at
least one valid 30-second window. The paper states both counts and makes clear
that the latter follows this benchmark's released-file inclusion rule.

The source paper reports signed informed consent and the right to stop the
study. It does not provide an ethics committee name or approval identifier in
the public text checked for this audit, so the manuscript does not invent
one. Before submission, the repository record should still preserve the exact
SAVELab archive URL, archive version, reuse terms, and file manifest.

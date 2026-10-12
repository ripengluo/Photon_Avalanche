# Scope for the next large-scale kMC campaign (multi-Ln, multi-wavelength)

Date: 2026-10-11. Evidence base: (1) finished Tm photon-avalanche (PA) demo in
this repo, (2) full statistical scan of the SUNSET-1..4 kMC datasets, (3) a
Crossref/Semantic Scholar literature sweep on PA and UCNP systems at 980, 800,
1064, and 1550 nm excitation.

## 1. Bottom line

Run a five-phase campaign. Phases A and B are PA systems and get full 12-point
power sweeps (the power dimension is the whole point for PA). Phase C covers
1550 nm Er with a medium-density power scan because its PA status in NaYF4 is
genuinely unknown and worth discovering. Phases D and E are conventional UC at
980/800 nm and get sparse 4-6-point power sets. About 80% of runs stay within
routine experimental reach (<=4 layers, spherical, sizes/concentrations with
direct literature precedent); about 20% go beyond (5-7 layers,
c-axis-elongated prisms, clustered doping). Estimated total: ~40 run
configurations, ~45-115 node-days on 24-core cm1-class hardware (floor set by
Tm, ceiling by the 34-level Er systems), compressible roughly linearly with
NERSC chaining via `submit_kmc_chain_nersc.sh`.

## 2. What exists now

- `tm_npt_kmc_production.py`: self-contained Tm 12-level NPT-rate kMC pipeline
  (geometry generation, rate DB build, prepared checkpoint schedule, power
  parallelism, replay/summary). Validated against experiment in
  `kmc_validation/VALIDATION_REPORT.md`: threshold 10-12.5 kW/cm2 (exp. 7-20),
  max log-log slope 41-59 (exp. 22-32), 800 nm band dominance, three-regime
  power curve.
- Finished sweeps: `10nm-8p0Tm` (50 A core + 25 A shell, 1452 Tm sites) and
  `14nm-8p0Tm` (70 A core + 25 A shell, 2953 Tm sites), 1064 nm, 12 powers
  (3k-50k W/cm2) x 8 seeds x 70M steps. The 14 nm sweep took ~59 h wall on one
  24-core node, i.e. ~2.5 node-days per PA sweep at this size.
- Rate model is currently Tm-specific: `SK_input.json` pins 1064 nm,
  Tm-derived ESA/CR scales, and NPT's semi-empirical ET constants.

## 3. SUNSET gap analysis

SUNSET (previous UCNP kMC dataset, NanoParticleTools schema) statistics from a
full scan of all four releases (IID/OOD splits included):

| Dataset | Docs | Layers | Outer radius (A) | Dopants (n_levels) | Surface quench |
|---|---|---|---|---|---|
| SUNSET-1 | 5571 | 1-4 (80% = 4) | 10-114 | Yb(2)/Er(34)/Nd(24) | no |
| SUNSET-2 | 1909 | 3 | 40 fixed | Yb/Er (+trace Tm) | Surface6 30% |
| SUNSET-3 | 1569 | 2 | 34 fixed | Yb/Er (+trace Tm) | no |
| SUNSET-4 | 16790 | 3 | 40 fixed | Yb/Er/Tm + Surface6 | Surface6 30% |

Key observations:

- SUNSET records carry no excitation-wavelength or power field; NPT's
  SpectralKinetics defaults are 976 nm at 1e7 W/cm2, so SUNSET is a
  single-condition composition-to-spectrum dataset. The entire power dimension
  (PA thresholds, slopes, bistability) is absent. That dimension is our unique
  contribution; do not replicate SUNSET composition sweeps.
- Coverage: spherical only; host NaYF4; ions limited to Yb/Er/Nd/Tm; no Ho,
  no Pr, no Sm/Dy/Eu/Tb/Gd as active optical centers; no shape axis; doping is
  uniform random (no clustering); concentrations span 0.0001-1.0 per layer.
- SUNSET-1-OOD is 5-layer (206 docs), i.e. "beyond 4 layers" starts at 5.
- NPT species data exists for Yb(2), Tm(12), Nd(24), Er(34), Ho(23), Gd(15),
  Sm(24), Eu(22), Tb(9), Dy(22), so rate-model generation for any of these
  ions is supported by the same machinery SUNSET used.

## 4. Literature map for the four pump wavelengths

PA = established photon avalanche (dense power scan required). UC =
conventional upconversion (sparse power set suffices). Discovery = plausible
PA, unproven in NaYF4.

| Pump | System | Class | Key evidence |
|---|---|---|---|
| 1064 | NaYF4:Tm (2-32%) | PA (8% demo done) | Lee et al., Nature 589, 230 (2021) 10.1038/s41586-020-03092-9; Liang et al., Nat. Nanotechnol. 17, 524 (2022); Dudek et al., Adv. Opt. Mater. 10, 2201052 (2022) (8-100% Tm); Kwock et al., JPCC (2021) 10.1021/acs.jpcc.1c07721 (surface sensitivity); Skripka et al., Nano Lett. 23, 7100 (2023) |
| 1064 | KMgF3:Tm | PA (alt host) | Zhang et al., Nano Lett. (2023) 10.1021/acs.nanolett.3c02377: slope ~27, threshold 16.6 kW/cm2 |
| 1064 | NaYF4:Nd | Discovery | PA-like UC in NdAl3(BO3)4 NPs at 1064 nm, da Silva et al., APL (2020) 10.1063/5.0024619; >200th-order bistable PA in Nd:KPb2Cl5, Skripka et al., Nat. Photonics (2024) 10.1038/s41566-024-01577-x. Nd PA in NaYF4 untested - high-value target |
| ~1064 | NaYF4:Ho | PA (2025, host TBD) | Bednarkiewicz et al., Nat. Photonics (2025) 10.1038/s41566-025-01705-1 (parallel PA in Ho); Lu et al., Sci. China Mater. (2025) 10.1007/s40843-025-3556-6. Pump wavelength not in public abstracts - confirm from full text |
| 1550 | NaYF4:Er up to NaErF4; Er/Yb | UC / PA unknown | Przybylska et al., J. Alloys Compd. (2020) 10.1016/j.jallcom.2020.154797 (808/975/1532 nm core/shell); Tyminski et al., Part. Part. Syst. Charact. (2020) 10.1002/ppsc.202000068; Wang et al., Chem. Eng. J. (2023) 10.1016/j.cej.2023.143558. Er PA reported only in phosphors at 980 (Kumar et al., J. Alloys Compd. 2021 10.1016/j.jallcom.2020.158196); 1550 PA in NaYF4 unestablished - discovery |
| 980 | NaYF4:Yb/Er, Yb/Tm | UC | Wurth et al., Nano Res. (2022) 10.1007/s12274-022-4570-5 (Yb/Er concentration dependence); Rabouw et al., ACS Nano (2018) 10.1021/acsnano.8b01545 (quenching pathways); Wiesholler et al., Nanoscale (2019) 10.1039/c9nr03127h (980 vs 808) |
| 800 | Nd-sensitized core/shell (Nd->Yb->Er/Tm) | UC | Arboleda et al., Chem. Mater. (2019) 10.1021/acs.chemmater.8b04057 (high-Nd 800 nm); Liao et al., Opt. Mater. (2019) 10.1016/j.optmat.2019.109489 (808 nm core-shell-shell) |
| ~852 (800 band) | NaYF4:Pr,Yb | PA | Dudek et al., Nanoscale (2023) 10.1039/d3nr04409b: PA at 852 nm, best 0.5%Pr/15%Yb@NaYF4, thresholds 281-286 kW/cm2, slopes 8.6-9.0, with rate-equation modelling - direct template for our kMC |

Reviews used for classification: Liu et al., Nano Lett. (2024)
10.1021/acs.nanolett.4c04524; Passini et al., Acc. Chem. Res. (2026)
10.1021/acs.accounts.5c00913; Huang et al., ACS Appl. Opt. Mater. (2024)
10.1021/acsaom.4c00334; Bednarkiewicz et al., Nanoscale Horiz. (2019)
10.1039/c9nh00089e; Skripka et al., Mater. Horiz. (2025) 10.1039/d4mh01798f.

## 5. Proposed campaign

Reference architecture: hexagonal NaYF4, 70 A core + 25 A shell (the validated
14 nm-core geometry), 8 seeds, 70M steps per power, prepared-checkpoint
protocol as in PLAN.md. PA sweeps reuse the centered-gaussian 12-power sampler
with the center moved to the expected threshold.

### Phase A - 1064 nm PA depth study (Tm series + discovery ions), dense scans

| # | Composition / architecture | Layers | Power protocol |
|---|---|---|---|
| A1 | NaYF4:Tm 4% @ NaYF4 | 2 | 12 powers, 3k-50k, center 12k |
| A2 | NaYF4:Tm 16% @ NaYF4 | 2 | 12 powers, 3k-50k, center 12k |
| A3 | NaYF4:Tm 32% @ NaYF4 | 2 | 12 powers, 3k-50k, center 12k |
| A4 | NaYF4:Nd 10% @ NaYF4 (discovery) | 2 | 12 powers, 1k-200k wide scan |
| A5 | NaYF4:Ho 10% @ NaYF4 (discovery; pump per full-text check) | 2 | 12 powers, 1k-200k wide scan |

A1-A3 complete the Dudek-style concentration series against the finished 8%
runs (threshold and slope vs Tm% is the cleanest untested prediction of the
current model). A4/A5 test whether Nd and Ho avalanche in NaYF4 at 1064 nm as
they do in borate/chloride hosts; densify around any steep onset.

### Phase B - 852/800-class PA (Pr/Yb), dense scan

| # | Composition / architecture | Layers | Power protocol |
|---|---|---|---|
| B1 | NaYF4:0.5%Pr,15%Yb @ NaYF4 | 2 | 12 powers, 10k-500k, center 280k |
| B2 | NaYF4:1%Pr,30%Yb @ NaYF4 | 2 | 12 powers, 10k-500k |

Thresholds near 280 kW/cm2 (Dudek 2023) are ~20x higher than the Tm ANP range;
the 50k preparation schedule in PLAN.md must be re-derived for this range.

### Phase C - 1550 nm Er, medium scan, PA discovery

| # | Composition / architecture | Layers | Power protocol |
|---|---|---|---|
| C1 | NaYF4:Er 2% @ NaYF4 | 2 | 8 powers, 1k-100k |
| C2 | NaYF4:Er 20% @ NaYF4 | 2 | 8 powers, 1k-100k |
| C3 | NaErF4 @ NaYF4 (100% Er limit) | 2 | 8 powers, 1k-100k |
| C4 | NaYF4:Er 20% @ NaYF4:Yb 20% (sensitized shell) | 3 | 8 powers, 1k-100k |

Er(34 levels) is the heaviest supported species: per-pair channel count scales
roughly as levels^2 vs Tm(12), so budget ~3-8x Tm cost per step and pilot C1
small before committing.

### Phase D - 980 nm canonical UC, sparse power sets

| # | Composition / architecture | Layers | Power protocol |
|---|---|---|---|
| D1 | NaYF4:Yb20%,Er2% @ NaYF4 | 2 | 4 powers, 1e2-1e5 |
| D2 | NaYF4:Yb60%,Er2% @ NaYF4 | 2 | 4 powers |
| D3 | NaYF4:Yb20%,Tm0.5% @ NaYF4 | 2 | 4 powers |
| D4 | NaYF4:Yb20%,Er2% @ NaYF4:Yb20% (active shell) | 3 | 4 powers |

These anchor the 980 nm comparison against SUNSET's fixed-condition
compositions; the power dependence is exactly what SUNSET lacks.

### Phase E - architecture axes: beyond-4-layer, elongation, clustering

20% of total runs go here. Elongation uses NPT's existing `PrismConstraint`
(hexagonal prism = c-axis-elongated beta-NaYF4 habit) at fixed volume;
clustering needs a pair-correlated placement option (current doping is uniform
random substitution).

| # | Axis | Configuration | Layers / shape | Power protocol |
|---|---|---|---|---|
| E1 | layers>4 | 5-layer 800 nm cascade: NaYF4:Yb,Er @ NaYF4:Yb @ NaYF4:Nd @ NaYF4:Nd @ NaYF4 | 5 | 4 powers at 800, 4 at 980 |
| E2 | layers>4 | 6-layer Tm PA stack (alternating Tm/inert shells) | 6 | 12 powers (PA) |
| E3 | shape | Phase-A reference (Tm 8%) as prism, c/a = 2 | 2-layer equiv. | 12 powers (PA) |
| E4 | shape | same, c/a = 4 | 2-layer equiv. | 12 powers (PA) |
| E5 | shape | D1 as prism, c/a = 2 | 2 | 4 powers |
| E6 | clustering | Tm 8%, clustered (nearest-neighbor pair-biased) vs random baseline | 2 | 12 powers (PA) |
| E7 | clustering | D1 with Er clustered vs random | 2 | 4 powers |

### Budget check

32 of 40 configurations (80%) are <=4 layers with literature-precedent
compositions; 8 (20%) are beyond routine capability (E1-E2 >4 layers; E3-E7
shape/clustering perturbations). Dense PA sweeps: A1-A5, B1-B2, E2-E4, E6 =
12 sweeps x ~2.5 node-days = ~30 node-days for Tm-like cost; 34/24/23-level
Er/Nd/Ho systems run slower per step, so plan 60-90 node-days for the dense
set. Sparse sets (C1-C4, D1-D4, E5, E7 + E1's two pumps): ~12 sweeps x
0.5-1 node-day = ~6-12 node-days. Total roughly 40-100 node-days on cm1-class
hardware; NERSC chaining compresses wall time ~linearly with nodes.

## 6. Implementation gaps before launch

1. Multi-species rate model: `tm_npt_kmc_production.py` is Tm-only
   (`--tm-fraction`, Tm-specific ESA/CR scales, 1064 nm pinned in
   `SK_input.json`). Each new system needs its own SK parameter JSON
   (wavelength, pump channel, per-ion cross sections); NPT's species data
   (Section 3) supplies line strengths for all ions above.
2. Multi-layer/multi-species geometry: the generator currently builds
   core+shell(+surface) with one dopant. SUNSET's `dopant_specifications`
   schema (layer index, concentration, species) is the right model to adopt.
3. Elongation: swap `SphericalConstraint` for `PrismConstraint` at fixed
   volume; verify the interaction-radius bookkeeping (`GeometryFactor`) stays
   correct for anisotropic site distributions.
4. Clustering: add correlated-placement doping (pair-biased substitution) as
   an option next to uniform random.
5. Power-range generalization: the fitted 50k preparation schedule in PLAN.md
   is calibrated to 3k-50k W/cm2; Phase B (to 500k) needs a refit.
6. Confirm the Ho PA pump wavelength from the 2025 full texts before
   finalizing A5's SK parameters.

## 7. Open questions / risks

- Ho PA excitation wavelength unverified (abstracts paywalled); A5 is blocked
  on this one fact.
- Nd (A4) and Er at 1550 (Phase C) are discovery targets - negative results
  are possible and still useful as boundaries of the PA design space, but size
  the scans accordingly (wide first pass, densify only on signal).
- Er(34)/Nd(24)/Ho(23) rate DBs are larger than Tm(12); first build of each
  species should be a `--dry-run` plus a short NPMC smoke test before any
  production sweep.
- The beyond-capability runs (E1-E2, 5-6 layers) have no SUNSET counterpart
  to validate against; treat trends as predictions and cross-check one against
  experimental multishell data where available.

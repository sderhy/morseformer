# NEXT — handoff post-P0 closeout (2026-05-19)

## §1. État actuel

**Version courante : v0.6.3** (commit `69b4b62`, tag `v0.6.3`, GitHub +
PyPI + HF). Acoustic recommandé inchangé : `rnnt_phase5_5.pt`. Cwget-
parity live-validée sur ragchew2 IC-7300.

### Les deux P0 du rapport de dette sont fermés

| P0 | Sujet | Commit | Tests |
|---|---|---|---|
| **P0-A** | `ValidationConfig.matching()` propage maintenant les 18 champs audio (operator + channel + empty/post-silence) depuis `DatasetConfig` | `bd1045c` | 20/20 (8 nouveaux : propagation par champ + différentiel jittered vs clean + override + mode pseudo-Morse) |
| **P0-B** | `python -m eval.release_gate` — ship-decision unique, 10 catégories calibrées sur `rnnt_phase5_5` + 0.5 pp margin, JSON report + exit 0/1 | `92a6596` | 4/4 (orchestrator end-to-end via mini-manifest silence+latency) |

Le bug `ValidationConfig.matching()` documenté dans
`project_phase4_0b_result` est résolu. Le release gate ferme le
post-mortem de `project_phase5_9_failure` (vibes-driven ship-then-
revert). Voir `project_release_gate_v1` (memory).

### Bilan plateaux — inchangé

| Phase | Mécanisme | Verdict |
|-------|-----------|---------|
| 5.6 retrain | digit-threshold inférence | shippé inférence-only en v0.5.2 |
| 5.7 amateur-corpus | 5NN cut-numbers, run-on prosigns | shippé v0.5.3, modeste |
| 5.8 EN-literary | Moby Dick + classiques | reverti en v0.6.2 |
| 5.9 letter-groups | densification ciphers | régression 6/6 |
| 5.10 callsign bump | mix 0.10→0.18 funded from random | régression 6/7 cible incluse |
| 7.0 beam search | frame-synchronous beam | strictement pire que greedy |
| 7.1 ITU prior | callsign-shape rescorer | diversity collapse, aucun effet |

Plafond structurel maintenu. Mais on a maintenant les outils pour
mesurer une future tentative honnêtement.

---

## §2. Plan d'attaque immédiat — aucun (les deux P0 sont fermés)

Cette section est volontairement vide. Les deux P0 du
`reports/technical_debt_2026-05-18.html` sont résolus, le projet est
techniquement prêt soit pour une release plateau (Option A) soit pour
une nouvelle campagne de training disciplinée (Options C / E / F).

**Comment décider la suite** :

1. Tourner `python -m eval.release_gate` sur `rnnt_phase5_5` pour
   confirmer que la baseline passe les seuils dans la version actuelle
   du code (vérification de non-régression silencieuse depuis
   2026-05-09).
2. Si la baseline passe, **choisir une option du §3 ci-dessous**.
3. Si la baseline échoue, comprendre *pourquoi* avant tout autre
   travail — c'est un signal que quelque chose a régressé sans qu'on
   le voie.

---

## §3. Options en aval — par ordre d'ambition

### Option A — plateau pur (recommandation par défaut)

Accepter v0.6.3 comme version finale. Maintenance utilisateur seulement.
**Coût** : 0. **Risque** : 0.

### Option B — packaging / UX

~~À faire~~ **DONE en v0.6.3** (PyPI, GUI desktop, Gradio demo, README
recentré, CI 3.10-3.13, dev setup, runtime guardrail).

### Option C — collecte data + bench plus large

Préparer le terrain pour une hypothétique Phase 8 : RBN pipeline,
transcription ragchew2 callsigns, mining WebSDR + W1AW, FAV22 time-
stretch. **Coût** : 1-2 semaines, indépendant du modèle. **Gain**
immédiat : 0. Conditionnel sur Phase 8.

**Important** : tout nouveau corpus aligné mérite d'être ajouté au
manifest `release_gate_v1.json` (ou un `v2.json` si la calibration
baseline change) — pas seulement à un bench script ad hoc.

### Option D — Phase 7 alternatif : post-process greedy

Levenshtein + ITU regex sur la sortie greedy pour corriger les
callsigns *presque-justes*. Coût ~3-4 h. Risque faux positifs. Gain
potentiel sur ragchew2, invisible sur LCWO v1 propre.

**Avec le release gate** : l'ajout doit produire un net positif
sur la catégorie `callsign_lcwo_001` (baseline 1.13 %, seuil 1.63 %)
sans dégrader les 6 autres catégories prose / contest / silence /
word-gaps.

### Option E — Phase 8 scale-up

d=192/256, training plus long. Phase 3-scale avait montré que doubler
à data constante ne bouge rien — sans Option C d'abord, ça reproduit
le plateau plus lentement.

**Avec les P0 fermés** : on saura mesurer honnêtement, donc un E sans
C est moins risqué qu'avant (le run produit au moins un signal réel,
même si on s'attend à un plateau).

### Option F — switch d'architecture

wav2vec / Conformer plus moderne / whisper custom decoder. Coût 1-2
mois. Risque très élevé. À ne considérer que si Options C + E sont
faites et plafond toujours là.

---

## §4. Anti-recommandations

- **Ne pas re-tenter** : Phase 5.x bump-and-amputate sur PHASE_3_4_MIX
  (5 échecs), Phase 7 beam dans sa forme actuelle (diversity collapse
  documenté), n'importe quel curriculum letter-groups densifié.
- **Ne pas re-proposer Option B** (packaging) — c'est fait.
- **Ne pas ship un acoustic candidate sans `release_gate` PASS.** C'est
  maintenant le critère unique de décision (cf
  `project_release_gate_v1`). Pas de "live test then ship" sans pass
  du gate.
- **Ne pas étendre `release_gate_v1.json` discrètement** : tout
  changement de baseline ou de catégorie doit incrémenter la version
  (v2.json) et expliquer le mouvement de seuils dans le commit.

---

## §5. Pointers

- Release gate : `python -m eval.release_gate --acoustic rnnt_phase5_5`
- Manifest : `eval/release_gate_v1.json`
- Bench LCWO standalone : `python -m eval.bench_lcwo --models rnnt_phase5_5 --presets live`
- Bench callsign : `eval/bench_lcwo.py` clip 7 (callsign_lcwo_001)
- Bench latency : `python -m eval.bench_latency`
- Rapport de dette : `reports/technical_debt_2026-05-18.html` (mis à
  jour 2026-05-19 pour marquer les P0 résolus)
- Tags : `git tag --list 'v*' | tail -5`
- Memory clés : `project_release_gate_v1`, `project_phase7_failure`,
  `project_v0_6_2_release`, `project_bench_lcwo_v1`,
  `project_phase4_0b_result` (bug d'origine, désormais fixé)

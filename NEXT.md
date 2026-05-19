# NEXT — handoff post-v0.6.3 (2026-05-19)

## §1. État actuel

**Version courante : v0.6.3** (commit `69b4b62`, tag `v0.6.3`, GitHub + PyPI
+ HF). Acoustic recommandé inchangé : `rnnt_phase5_5.pt` (vocab 49, greedy +
`digit_threshold=0.90`, central-zone-commit streaming). Cwget-parity live-
validée sur ragchew2 IC-7300 (cf `project_live_observations_v0_5_2`).

### Ce qui a été livré depuis le précédent NEXT (post-v0.6.2)

| Commit | Sujet |
|---|---|
| `18c7b42` | Phase 5.10 curriculum — **régression 6/7 clips, abandonnée** |
| `f9b85fd` | audit fixes (CHANGELOG split, E2E test, latency bench, callsign tests) |
| `2cf4593` + `69b4b62` | **v0.6.3** packaging — `__init__` aligné, CI 3.10-3.13, ruff |
| `144c912` + `e8f7e8e` | **GUI PySide6 desktop + Gradio demo + WSL2 PulseAudio backend** |
| `c637aa1` | MODEL_CARD refresh v0.6.3 |
| `6f02cc9` → `8f18816` | dev setup, `reports/technical_debt_2026-05-18.html`, runtime guard 3.10-3.13, fix `sample_prose(max_chars=)`, test Gradio prose preset |

Tests : `pytest -q` → **362 passed, 7 skipped**. Ruff vert. Python 3.10-3.13
supportés, runtime guardrail dans `morseformer/__init__.py`.

### Bilan plateaux — Phase 5.10 ajoutée

| Phase | Mécanisme | Verdict |
|-------|-----------|---------|
| 5.6 retrain | digit-threshold inférence | shippé inférence-only en v0.5.2 |
| 5.7 amateur-corpus | 5NN cut-numbers, run-on prosigns | shippé v0.5.3, modeste |
| 5.8 EN-literary | Moby Dick + classiques | reverti en v0.6.2 |
| 5.9 letter-groups | densification ciphers | régression 6/6 |
| **5.10 callsign bump** | mix 0.10→0.18 funded from random | **régression 6/7 cible incluse** |
| 7.0 beam search | frame-synchronous beam | strictement pire que greedy |
| 7.1 ITU prior | callsign-shape rescorer | diversity collapse, aucun effet |

**Plafond structurel confirmé** avec modèle/data actuels. Le rapport de dette
2026-05-18 fixe la priorité avant tout nouveau retrain : réparer le signal
de validation et automatiser le release gate.

---

## §2. Plan d'attaque immédiat — résoudre les deux P0 du rapport de dette

Code-only, sans retrain, ~1-2 jours. **Bloquant pour toute nouvelle campagne
d'entraînement.**

### P0-A. `ValidationConfig.matching()` ne propage pas l'enveloppe complète

**Symptôme** : la val mesure de la "CW idéale" pendant que le train tourne
sur du jittered/noisy → trade-offs invisibles à l'eval interne (cf
`project_phase4_0b_result`).

**Fichier** : `morseformer/data/validation.py:73` (`ValidationConfig.matching`).
Actuellement propagé : `target_duration_s`, `sample_rate`, `freq_hz`,
`text_mix`, `frontend`, `keying`, `max_text_retries`, `pre_quiet_zone_range_s`,
`post_quiet_zone_min_s`. **Manquant** : channel (snr, rx_filter), jitter,
dash/dot ratio, gap inflation, word-gap inflation, QRM/QSB/QRN, drift,
empty samples / empty_sample_probability.

**Étapes** :

1. Auditer `DatasetConfig` (`morseformer/data/dataset.py` ou
   `morseformer/data/synth.py`) pour la liste complète des champs qui
   influencent l'audio.
2. Étendre `ValidationConfig` avec les champs manquants ; défaut = no-op
   pour rester bit-identique avec les configs 3.x existantes.
3. Étendre `ValidationConfig.matching()` pour les recopier.
4. Câbler les nouveaux champs dans le rendu (`_render_one` et helpers
   en aval), avec test que `seed` figé → bit-identique sample.

**Tests à ajouter** (`tests/test_validation.py`) :

* Pour chaque champ : `ValidationConfig.matching(ds_cfg)` lit bien
  `ds_cfg.<champ>`.
* Test différentiel : un `DatasetConfig` jittered produit un val set
  visiblement bruité (énergie moyenne, variance ≠ clean), pas le set
  "ideal CW" qu'on a aujourd'hui.

**Critère de succès** : sur le mix Phase 5.5 réutilisé, le CER de
validation interne au step N change de façon mesurable (typiquement +1
à +5 pp) une fois la propagation faite — preuve qu'on voit enfin la
même distribution que le train.

### P0-B. Release gate automatisé

**Symptôme** : ship v0.6.0 (phase5_8) puis revert v0.6.2 parce que la
décision était vibes-driven. Le rapport de dette demande une commande
unique de décision.

**Fichier nouveau** : `eval/release_gate.py`. S'appuie sur l'existant —
`eval/bench_lcwo.py` est déjà conçu pour ce rôle (cf docstring : *"the
reproducible source-of-truth that NEXT.md asks for before any further
acoustic retrain"*).

**Étapes** :

1. Manifeste versionné `eval/release_gate_v1.json` : pour chaque
   catégorie (LCWO clean, websdr, callsigns, silence, word-gaps,
   contest, prose FR/EN), seuils `max_cer` et `max_wer` calibrés sur
   `rnnt_phase5_5` + marge de non-régression (typiquement +0.5 pp
   absolu).
2. `python -m eval.release_gate --acoustic <name> [--lm <name>]` qui :
   - exécute `bench_lcwo` + `bench_latency` + un smoke silence/word-gaps
   - écrit `reports/release_gate_<acoustic>_<date>.json`
   - imprime un tableau coloré pass/fail par catégorie
   - exit code 0 si tout passe les seuils, ≠ 0 sinon
3. Section README sur "comment ship une release" qui appelle uniquement
   `release_gate`. Pas de claim live-only.
4. Test smoke `tests/test_release_gate.py` avec un mini-manifest et
   un acoustic baseline pour vérifier que le binaire tourne.

**Critère de succès** : rejouer mentalement la décision v0.6.0 →
`release_gate(rnnt_phase5_8)` doit faire échouer (la mean CER passait
de 2.60 à 3.93 %). Idem pour `rnnt_phase5_10` qui régressait sur sa
propre cible.

---

## §3. Options en aval — par ordre d'ambition

### Option A — plateau pur (recommandation par défaut après §2)

Accepter v0.6.3 comme version finale. Le projet est cwget-parity
live-validée, ce qui satisfait `project_ambition` recadré. Pas de
nouveau code, pas de retrain. Maintenance utilisateur seulement.

**Coût** : 0. **Risque** : 0.

### Option B — packaging / UX

~~À faire~~ → **DONE en v0.6.3.** PyPI shippé, GUI desktop PySide6
(`morseformer gui`), Gradio demo (`demo/app.py`), README recentré,
CI 3.10-3.13, dev setup documenté, runtime guardrail.

### Option C — collecte data + bench plus large

Inchangé depuis le précédent NEXT. Préparer le terrain pour une
hypothétique Phase 8 : RBN pipeline, transcription ragchew2 callsigns,
mining WebSDR + W1AW, FAV22 time-stretch.

**Coût** : 1-2 semaines, indépendant du modèle.

**Gain** : 0 immédiat. Conditionnel sur Phase 8.

### Option D — Phase 7 alternatif : post-process greedy

Inchangé. Levenshtein + ITU regex sur la sortie greedy pour corriger
les callsigns *presque-justes* (MW0BPL → MW0BGL). Coût ~3-4 h. Risque
faux positifs. Gain potentiel sur ragchew2 hard callsigns, invisible
sur LCWO v1 propre.

### Option E — Phase 8 scale-up

Inchangé. d=192/256, training plus long. Phase 3-scale a montré que
doubler à data constante ne bouge rien — sans Option C d'abord, ça
reproduit le plateau plus lentement.

### Option F — switch d'architecture

Inchangé. wav2vec / Conformer plus moderne / whisper custom decoder.
Coût 1-2 mois. Risque très élevé. À ne considérer que si Option C +
§2 sont faits et plafond toujours là.

---

## §4. Anti-recommandations

- **Ne pas relancer de retrain tant que §2 n'est pas fini.** Sans
  signal de validation fiable et release gate automatisé, chaque
  nouvelle phase risque de répliquer le scénario phase5_8/5_9/5_10
  (ship-then-revert ou veto post-hoc).
- **Ne pas re-tenter** : Phase 5.x bump-and-amputate sur PHASE_3_4_MIX
  (5 échecs), Phase 7 beam dans sa forme actuelle (diversity collapse
  documenté), n'importe quel curriculum letter-groups densifié.
- **Ne pas re-proposer Option B** (packaging) — c'est fait.

---

## §5. Pointers

- Bench LCWO : `python -m eval.bench_lcwo --models rnnt_phase5_5 --presets live`
- Bench callsign : inclus dans le manifest LCWO v1 (clip 7)
- Bench latency : `python -m eval.bench_latency`
- Rapport de dette : `reports/technical_debt_2026-05-18.html`
- Phase 7 scaffold off : `--beam-width 4 --callsign-prior-weight 1.0` (no-op)
- Tags : `git tag --list 'v*' | tail -5`
- Memory clés : `project_phase7_failure`, `project_v0_6_2_release`,
  `project_phase5_10_failure`, `project_bench_lcwo_v1`,
  `project_phase4_0b_result` (origine du bug P0-A)

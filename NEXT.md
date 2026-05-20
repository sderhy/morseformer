# NEXT — handoff post-Phase 8-10 + word-splitter (2026-05-21)

## §1. État actuel

**Version recommandée : v0.6.3 (acoustic `rnnt_phase5_5`).** Inchangé.
Cwget-parity live-validée. Tous les outils du gate v1 passent (+ le
nouveau post-process word splitter activé sur le preset `prose`).

### Bilan de la session 2026-05-19 → 2026-05-21

| Travail | Statut | Commit |
|---|---|---|
| Audit real-QSO `g3ses` + `g6pz` (28 clips ~30 min) | gap 8× confirmé | `edf2ae1` |
| Real-audio prep pipeline | 159 + 102 chunks alignés, score ~0.78 | `0e45f9e` |
| Phase 8 fine-tune | FAIL gate (5/10) | non shippé |
| Phase 8a recipe-conservative | FAIL gate (5/10) | non shippé |
| Phase 9 jargon enrichment + word_gap floor | FAIL gate (6/10) — closest | code shippé, weights non |
| Phase 10 real-audio word_gap augmentation | FAIL gate (2/10) — backfired | code shippé, weights non |
| **Post-process word splitter** | **shippé ON sur preset `prose`** | `6c0a5c6` |

### Verdict

**Le fine-tune real-audio naïf sur 30 min ne marche pas.** La régression
structurelle sur `word_gap_inflation_6×` (1.5 % → 9-14 %) vient du
mismatch de distribution : real audio = word gaps ~1×, synth = U(1,8).
Le modèle collapse vers la moyenne et perd la queue 6×. Tentatives de
fix (jargon, augmentation par insertion silence) n'ont pas suffi —
sans **forced alignment** du corpus real, l'augmentation injecte du
silence à des positions ~aléatoires et empire le problème.

**Phase 5.5 reste recommandé.** Le commit `6c0a5c6` ship en parallèle :

1. **Jargon templates Phase 9** (text.py) — pour toute future Phase 11+
   sans real-audio mix
2. **Real-audio augmentation infra** (real_audio.py) — OFF par défaut,
   disponible pour expérimentation
3. **Word splitter post-process** (decoding/word_splitter.py + preset
   integration) — gain -1 à -1.4 pp WER sur g3ses/g6pz held-out

---

## §2. Plan d'attaque immédiat — aucun

Le projet est dans un état stable. Le gate passe, la dette technique
est traitée, le post-process apporte un gain modeste mais réel sur
ragchew. Pas de chantier bloquant.

---

## §3. Options en aval

### Option A — plateau (recommandation)

Accepter `v0.6.3 + word splitter ON pour prose` comme version finale.

### Option B — packaging / UX

DONE en v0.6.3. RAS.

### Option C — collecte data + forced alignment

Si on veut un *vrai* Phase 11, deux conditions préalables :

1. **Plus de corpus real** : ≥ 2 h d'audio multi-opérateurs / multi-
   bandes avec transcripts time-aligned, OU
2. **Forced alignment** sur le corpus existant : CTC alignment pour
   identifier les positions exactes de chaque caractère dans l'audio
   → enable une vraie data augmentation word-gap. ~1-2 jours d'engineering
   (utiliser `torchaudio.functional.forced_align` ou similaire).

Avec l'une des deux, Phase 11 peut envisager : `DatasetConfig.phase_9`
(jargon) + real-audio mix 0.20 avec augmentation **alignée**. Cibles :
g6pz CER < 12 % sans régresser le gate.

**Coût** : 2 jours data + 1 jour training. **Risque** : modéré (on a
maintenant les bons outils pour mesurer).

### Option D — Phase 7 alternatif

Inchangé. Post-process callsign (Levenshtein vs ITU prefix). Peut
empiler par-dessus le word splitter pour les ragchews.

### Option E — Phase 8 scale-up

Inchangé. Sans Option C d'abord, va reproduire le plateau.

### Option F — switch d'architecture

Inchangé. Wav2Vec / Conformer plus moderne / whisper. 1-2 mois.

---

## §4. Anti-recommandations

- **Ne pas relancer un retrain real-audio sur le corpus actuel sans
  forced alignment.** Quatre échecs consécutifs (Phase 8/8a/9/10) le
  confirment. Le data augmentation par interpolation linéaire backfire.
- **Ne pas ship Phase 8 / 9 / 10 weights.** Tous FAIL le gate par
  régression structurelle sur `word_gap_inflation_6×`. Phase 5.5 reste.
- **Ne pas désactiver le post-process word splitter sur le preset
  `prose`** sans benchmark. Gain réel mesuré sur g3ses (-1 pp WER) et
  g6pz (-1.4 pp WER).
- **Ne pas étendre `release_gate_v1.json` pour relaxer `word_gap_inflation_6×`**
  sans une raison documentée — c'est exactement le test qui a attrapé
  les 4 retrains ratés.

---

## §5. Pointers

- Release gate : `python -m eval.release_gate --acoustic rnnt_phase5_5`
- Audit real OTA : `python -m scripts.audit_real_qso --device cuda`
  (avec `--post-segment` pour A/B le splitter)
- Decode prose avec splitter (auto) : `morseformer decode file.wav --preset prose`
- Decode live sans splitter : `morseformer decode file.wav --no-post-segment`
- Real-audio prep : `python -m scripts.prepare_real_qso --device cuda`
- Phase 9 curriculum dispo : `--curriculum phase9` dans `train_rnnt`
- Phase 10 augmentation : `--real-audio-word-gap-augment-prob 0.5`
- Word splitter dict : `morseformer/decoding/word_splitter.py` — ajouter
  un mot = ajouter à `_AMATEUR_HIGH` ou `_ENGLISH_COMMON`
- Memory clés : `project_audit_real_qso_2026_05_19`,
  `project_phase8_to_10_results`, `project_release_gate_v1`,
  `project_v0_6_2_release`

# NEXT — handoff post 2026-05-21

## §1. État actuel

**Acoustic shippé : `rnnt_phase5_5` (v0.6.3) inchangé.** Quatre retrains
real-audio (Phase 8 / 8a / 9 / 10) ont tous FAIL le gate par régression
structurelle word_gap_inflation. Détails dans
`memory/project_phase8_to_10_results.md`.

**Inférence améliorée cette session** (commits `bd1045c` → `714eec0`) :

| Domaine | Changement | Commit |
|---|---|---|
| P0-A dette | `ValidationConfig.matching()` propage 18 champs audio | `bd1045c` |
| P0-B dette | `python -m eval.release_gate` (10 catégories, exit 0/1) | `92a6596` |
| Display | `format_output` ajoute `\n` après `=` / `KN` | `edf2ae1` |
| Real data | `prepare_real_qso.py` (resample + decode + align JSONL) | `0e45f9e` |
| Jargon + infra | Phase 9 templates + Phase 10 augmentation knobs (OFF default) | `6c0a5c6` |
| **Word splitter** | `morseformer/decoding/word_splitter.py` (~330 lignes, 22 tests) | `6c0a5c6` |
| **Preset `prose`** | switch no-LM + splitter ON par défaut (LM hurts ragchew) | `714eec0` |
| Structural rules | DE both directions + EE + punctuation aération + callsign reconstruction | `714eec0` |

**Tests** : 22 word_splitter + 28 validation + 4 release_gate + 8 postprocess
+ existants ⇒ ~410 passed. Ruff clean.

---

## §2. Prochaine session — direction décidée : interfaces de décodage

Le modèle est stable, le bench est outillé, le post-process ship. La
prochaine itération porte sur **l'UX du décodeur**. Voir
`memory/handoff_post_2026_05_21.md` pour la map des threads.

### Threads candidats (à ordonner par le user)

1. **Cohérence preset across surfaces** — le preset `prose` (no-LM
   + splitter) est-il propagé dans GUI et Gradio ?
2. **Toggle `--post-segment` en GUI/Gradio** — flag CLI shippé,
   à porter côté UX
3. **Live decode UX** — séparateurs visuels pour `=` / `KN` dans
   le streaming, formatage par transmission
4. **Batch mode CLI** — `morseformer decode dir/*.wav` ou flag
5. **JSON output** — `--format json` pour pipelines downstream
6. **HTTP/REST serveur** — `morseformer serve` exposant decode en endpoint
7. **Standalone Windows package** — PyInstaller bundle
8. **Live transcription markers** — coloration / annotations dans la GUI

### Points d'entrée code
- CLI dispatch : `morseformer/cli/__init__.py`
- Presets (truth source) : `morseformer/cli/presets.py`
- GUI : `morseformer/gui/app.py` + `live_tab.py` + `file_tab.py`
- Gradio HF Space : `demo/app.py`
- Streaming decoder API : `morseformer/decoding/streaming.py`
- Audio capture : `morseformer/gui/audio_capture.py` (PulseAudio + ALSA)

---

## §3. Options modèle en aval (inchangées, en attente)

### Option A — plateau (recommandation par défaut)

`v0.6.3` + word splitter ON pour prose = version finale.

### Option B — packaging / UX

Couverte par §2 ci-dessus pour cette prochaine session.

### Option C — collecte data + forced alignment

Pour un VRAI Phase 11 retrain, deux pré-requis non remplis :
1. ≥ 2 h de corpus real multi-opérateurs, OU
2. **Forced alignment** sur le corpus existant (CTC alignment via
   `torchaudio.functional.forced_align`) → unlock vraie data
   augmentation word-gap

### Options D / E / F

Inchangées (post-process callsign, scale-up, switch d'architecture).

---

## §4. Anti-recommandations

- Ne pas relancer de retrain real-audio sans forced alignment
  (4 échecs Phase 8 / 8a / 9 / 10 le confirment)
- Ne pas ship Phase 8 / 9 / 10 weights (tous FAIL gate)
- Ne pas re-introduire LM dans `prose` default sans test live qui
  démontre un gain net (le LM hurts ragchew confirmé sur g6pz/G1
  + g3ses/C7)
- Ne pas désactiver le splitter sur preset `prose` par défaut sans
  alternative readability équivalente
- Ne pas ship d'interface (GUI/Gradio/web) qui n'expose pas un
  équivalent du flag `--post-segment` côté UX

---

## §5. Pointers

- Release gate : `python -m eval.release_gate --device cuda`
- Audit real OTA : `python -m scripts.audit_real_qso --device cuda --post-segment`
- Decode ragchew (lisible) : `morseformer decode file.wav --preset prose`
- Decode literary prose (LM opt-in) : `morseformer decode file.wav --preset prose --lm lm_phase5_2 --fusion-weight 0.7`
- Word splitter dict : `morseformer/decoding/word_splitter.py` —
  ajouter un mot = ajouter à `_AMATEUR_HIGH` ou `_ENGLISH_COMMON`
- Tags : `git tag --list 'v*' | tail -5`
- Memory clés : `handoff_post_2026_05_21`, `project_phase8_to_10_results`,
  `project_release_gate_v1`, `project_audit_real_qso_2026_05_19`

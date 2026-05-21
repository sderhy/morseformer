# NEXT — handoff post 2026-05-21 (révision décodage)

## §1. État actuel

**Acoustic : `rnnt_phase5_5` (v0.6.3) inchangé.** Quatre retrains
real-audio (Phase 8 / 8a / 9 / 10) tous FAIL le gate par régression
structurelle `word_gap_inflation_6×`. Détails dans
`memory/project_phase8_to_10_results.md`.

**Inférence shippée cette session** (commits `bd1045c` → `4bd8ae9`) :
- P0-A `ValidationConfig.matching()` + P0-B `release_gate` (10 cat)
- Word splitter post-process (~330 lignes, 22 tests) — ON par défaut
  pour preset `prose`
- `format_output` newlines après `=` / `KN`
- `prepare_real_qso.py` (resample + decode + align JSONL)
- Phase 9 jargon + Phase 10 augmentation infra (OFF par défaut)
- `prose` preset switch no-LM + structural pre-pass étendu (DE both
  directions + EE + punctuation aération + callsign reconstruction)

---

## §2. Rétrospective honnête de la session

Décision prise après live test g6pz/G1 + g3ses/C7 : on aurait pu faire
**significativement mieux** sur le décodage si on avait re-séquencé.

### Ce qu'on aurait dû faire AVANT les retrains

**Splitter le premier, retrain ensuite.** On a fait l'inverse — 4 retrains
avant le post-process. Si le word splitter avait été shippé day 1, on
aurait su immédiatement que le run-on est *post-processable* et les
retrains auraient eu un scope différent (acoustic-char accuracy seule,
pas la segmentation des mots). Le splitter coûte ~3 h et donne -1.4 pp
WER ; on a brûlé ~6 h de GPU + ~6 h dev sur les 4 retrains qui n'ont
rien shippé.

**Forced alignment du corpus.** Le bug structurel word_gap_inflation
n'est pas un bug de modèle — c'est qu'on n'a jamais su *où* sont les
word boundaries dans l'audio real. Sans cette info, le data augmentation
de Phase 10 a injecté du silence à des positions interpolées
linéairement, et le modèle a appris "en real audio, silence ≠ word gap"
— exactement l'inverse de ce qu'on voulait. La bonne approche est :
forced-align d'abord, augmentation correcte ensuite.

### Ce qu'on aurait dû explorer côté décodage

**N-best + rescoring dict-aware.** Le greedy RNN-T sort UN chemin. Avec
beam N=4 et un rescorer privilégiant les chemins dont les tokens
matchent le dict amateur, on aurait pu choisir la meilleure segmentation
*avant* la post-process. C'est différent du Phase 7 beam (qui avait
diversity-collapsed sur ITU prior) parce que le scoring serait par
couverture dictionnaire, pas par regex callsign.

**KenLM bigram amateur** au lieu du neural prose LM. Le `lm_phase5_2`
actuel (~5 M params, entraîné sur Wikipedia + amateur mix) **hurts**
le ragchew parce qu'il a appris des bigrammes literary. Un KenLM 3-gram
entraîné UNIQUEMENT sur amateur idioms (~10 k QSO synthétiques + nos
templates) ferait ~10 KB, runtime négligeable, et aiderait au lieu de
hurter.

**Confidence-aware splitting.** Aujourd'hui le splitter découpe TOUS
les tokens ≥ 6 chars avec 90 % coverage. Avec la confidence per-frame
de l'encoder, on saurait *où* le modèle hésite et on activerait le
splitter chirurgicalement — évite le faux positif sur du LCWO clean
(les -3 pp sur `word_gap_inflation_6×` du gate).

### Ce qu'on a bien fait

- **Le gate avant les retrains** (P0-B). Sans ça on shippait Phase 8 et
  on rejouait phase5_8 → 5_5. Le gate a payé son investissement
  immédiatement.
- **Abandon honnête** de Phase 8/9/10. Pas de bricolage de seuils pour
  "sauver" un mauvais candidate.
- **LM retiré du `prose` preset basé sur live evidence**, pas sur le
  bench synthétique qui disait l'inverse.

---

## §3. Plan d'attaque prochaine session — **forced alignment + KenLM amateur**

Investir là où le levier est dominant. Estimation totale : **~1.5 jour
dev + ~30-60 min retrain GPU**. Gain attendu : -3 à -5 pp WER sur g6pz
held-out, possiblement passage du gate `word_gap_inflation_6×`.

### Étape A — Forced alignment du corpus real (½ jour)

1. **Implémenter `scripts/force_align_real_qso.py`** qui prend l'audio
   resamplé + le label normalisé et produit un per-character timestamp
   via `torchaudio.functional.forced_align` (utiliser la CTC head du
   modèle Phase 5.5).
2. **Re-générer `data/real/g3ses_aligned.jsonl`** avec un champ
   supplémentaire `char_starts_s: list[float]` (timestamp de chaque
   char dans l'audio).
3. **Validation** : sanity-check 10 chunks en plottant audio +
   timestamps, vérifier que les positions des `' '` (espaces) sont
   bien au milieu des silences inter-mots.

### Étape B — Augmentation word-gap *correcte* (¼ jour)

1. **Re-écrire `_augment_word_gap`** dans `morseformer/data/real_audio.py`
   pour utiliser les `char_starts_s` au lieu de l'interpolation linéaire.
   Insertion silence aux **vraies** positions de word boundaries.
2. **Ajouter test différentiel** : un chunk augmenté à inflation 6×
   doit avoir un audio mesurablement plus long entre la fin du mot N
   et le début du mot N+1.

### Étape C — KenLM 3-gram amateur (½ jour)

1. **Générer corpus textuel synthétique** : 100 k échantillons depuis
   `DatasetConfig.phase_9` (qui a tout le jargon nouveau). Format texte
   pur.
2. **Entraîner KenLM 3-gram** via `kenlm` (déjà dépendance? sinon ajout
   minimal — ~10 KB binaire). Vocab 49.
3. **Implémenter `morseformer/decoding/lm_kenlm.py`** : wrapper qui
   expose `score_word(text)` et `score_sequence(text)`.
4. **Intégrer dans le splitter** comme alternative à la coverage
   heuristique : `score = sum(kenlm.score_word(w) for w in candidate_split)`.
   Choisir la segmentation au meilleur score LM.

### Étape D — Phase 11 retrain (½ jour incl. GPU)

1. **Lancer `train_rnnt.py --curriculum phase9
   --real-audio-jsonl data/real/g3ses_aligned.jsonl
   --real-audio-word-gap-augment-prob 0.5
   --real-audio-word-gap-augment-inflation-range "1.5,6.0"
   --pretrained-rnnt checkpoints/phase5_5/best_rnnt.pt
   --real-audio-probability 0.20 --total-steps 20000`**
   — same recipe que Phase 10 mais l'augmentation utilise maintenant
   les VRAIES positions.
2. **Audit + gate** comme d'habitude. Cibles ship :
   - g6pz hors freq-OOD CER < 15 %
   - `word_gap_inflation_6×` ≤ 5 % (relax de 2.5, justifié par real-mix)
   - Pas de FAIL sur les 7 LCWO clips

### Étape E (optionnelle, si A-D ne suffit pas) — N-best beam + dict rescore

Plus engineering-heavy, à différer si Phase 11 passe le gate seul.

---

## §4. Décodage interfaces (déprioritisé)

Repoussé. La précédente version de NEXT pointait vers les interfaces
(GUI/Gradio/live UX) comme prochaine direction — révisé après
réflexion §2. Le bottleneck est la **qualité de décodage**, pas
l'affichage. Les UX peuvent attendre — elles affichent du décodage
médiocre joliment vs. afficher du meilleur décodage.

Threads à reprendre **après** Phase 11 (si on shipp un nouvel acoustic) :
cohérence presets sur GUI/Gradio, toggle `--post-segment` UI, batch
CLI, JSON output, mini HTTP server, segment markers live, etc. La map
complète reste dans `memory/handoff_post_2026_05_21.md` §"Threads
candidats".

---

## §5. Anti-recommandations renforcées

- **Ne PAS relancer un retrain real-audio sans Étape A (forced
  alignment).** Le 5e échec aurait la même cause que les 4 premiers —
  augmentation à des positions ~aléatoires. **Forced align d'abord,
  retrain ensuite.**
- **Ne pas re-introduire LM dans `prose` default** sans test live qui
  démontre un gain net (le neural prose LM hurts confirmé deux fois).
  Pour Phase 11+, prioriser KenLM amateur (lighter, targeted).
- **Ne pas étendre le splitter sans confidence-aware gating** s'il
  cause de nouvelles régressions LCWO. La heuristique "≥6 chars
  + 90 % coverage" est dumb mais sûre ; la prochaine version doit être
  smarter, pas plus aggressive.
- **Ne pas ship Phase 8 / 9 / 10 weights.** Tous FAIL gate.
- **Anti-pattern de session** : ne plus engager du temps GPU/dev sur
  des retrains avant d'avoir le post-process activé pour mesurer ce
  que le post-process seul donne. Splitter d'abord, retrain ensuite.

---

## §6. Pointers

- Release gate : `python -m eval.release_gate --device cuda`
- Audit real OTA : `python -m scripts.audit_real_qso --device cuda --post-segment`
- Decode ragchew : `morseformer decode file.wav --preset prose`
- Decode literary (opt-in LM) : `morseformer decode file.wav --preset prose --lm lm_phase5_2 --fusion-weight 0.7`
- Real audio prep (aujourd'hui linéaire) : `python -m scripts.prepare_real_qso --device cuda`
- Forced align (à écrire Étape A) : `torchaudio.functional.forced_align`
  (déjà dispo via `torchaudio>=2.0` qu'on a en dépendance)
- KenLM (à intégrer Étape C) : `pip install https://github.com/kpu/kenlm/archive/master.zip`
  + binaire ~150 KB
- Tags : `git tag --list 'v*' | tail -5`
- Memory clés : `handoff_post_2026_05_21`, `project_phase8_to_10_results`,
  `project_release_gate_v1`, `project_audit_real_qso_2026_05_19`
- Réflexion source : chat 2026-05-21, "Pense tu qu'on pouvait faire mieux"

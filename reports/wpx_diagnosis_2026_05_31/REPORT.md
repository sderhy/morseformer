# WPX Contest Diagnosis — 2026-05-31

## TL;DR

**Cause confirmée du mauvais résultat WPX = densité multi-stations dans la passe-bande, pas le WPM ni le SNR.**

- WPM des signaux primaires : 26-32 WPM → **dans la plage de training** [18, 32]
- SNR estimé : 15-23 dB → **sain** (training va jusqu'à −10 dB sans casser)
- Stations simultanées dans la passe-bande 300-1500 Hz : **6 à 8 dans 98 % des fenêtres de 3 s** sur tous les clips analysés
- Le training a **toujours** vu mono-station (un seul carrier) → mismatch structurel

C'est la même classe d'échec que les pile-ups de QSO standard (mode d'échec déjà connu, jamais corrigé), amplifiée par la densité contest constante.

## Méthode

5 clips WPX analysés. Pour chaque clip :
- FFT globale (60 s) → top 8 carriers ≥ 10× noise floor entre 300-1500 Hz
- Sliding window 3 s pas 1.5 s → distribution du nombre de stations détectées
- Bandpass ±80 Hz autour du carrier primaire → Hilbert + Savitzky-Golay → estimation WPM (1.2/dit) et SNR (médiane on / médiane off)

## Résultats

| Fichier | Durée | Carrier primaire | WPM (60 s) | SNR dB (60 s) | Stations / fenêtre 3 s |
|---|---|---|---|---|---|
| `cwcwwS50RS.wav` | 113 s | 595 Hz | 32 | 19.5 | **7-8 dans 73/74 fenêtres (99 %)** |
| `cwcww11.wav` | 36 s | 601 Hz | 59* | 14.3 | 7-8 dans 14/21 fenêtres (67 %) |
| `cwcww5.wav` | 28 s | 575 Hz | 33 | 22.7 | **7-8 dans 17/17 fenêtres (100 %)** |
| `cwcwwDA1A.wav` | 107 s | 600 Hz | n/a | 15.4 | 7-8 dans 57/70 fenêtres (81 %) |
| `cwcwwHA8A.wav` | 300 s | 621 Hz | 26 | 19.4 | **6-8 dans 5208/5345 fenêtres (97 %)** |

*cwcww11 WPM = 59 = artefact (signal trop court ou pile-up qui pourrit la détection de dit). Les autres mesures concordent.

### Distribution carriers — exemple S50RS (le clip qui décode en bouillie)

| Carrier (Hz) | Force / noise floor |
|---|---|
| 594.5 | **7115×** ← primaire (S50RS) |
| 654.3 | 415× |
| 504.4 | 334× |
| 706.8 | 180× |
| 452.1 | 91× |
| 757.8 | 44× |
| 399.8 | 15× |
| 811.4 | 14× |

8 carriers détectables. Le primaire écrase les autres en énergie (× 17 vs le 2e), mais les harmoniques de modulation des autres stations injectent de l'énergie large bande qui entre dans la décision du modèle.

## Interprétation

Le modèle (rnnt_phase11b) :
1. **Repère bien le carrier dominant** (callsigns extraits dans tous les clips)
2. **Hallucine sur le signal résiduel** des stations adjacentes quand le primaire est en key-up (silence du primaire ≠ silence vrai)
3. **Boucle sur les répétitions courtes** (callsign × 3, exchange × 2) parce que le RNN-T amplifie les patterns vus en training

Le training pipeline (`morseformer/data/generator.py`) n'a pas de mode "pile-up simulé". Les augmentations actuelles :
- AWGN, jitter, dash:dot ratio, word_gap_inflation, real-audio mix (test.wav)
- **AUCUNE** : 2e station à fréquence proche, key-clicks d'opérateur voisin, AGC modulé par signal fort adjacent

## Artefacts

- `summary.json` — données brutes
- `cwcww*_spec.png` — spectrogrammes 200-1500 Hz + envelope RMS

Les spectrogrammes confirment visuellement les 6-8 carriers parallèles.

## Recommandations

### Option A : ne pas retrainer (plateau)
Ship-as-is. Le contest est un cas d'usage non-target dans la roadmap actuelle (focus QSO standard / prose). Documenter "contest = best effort" dans le README.

### Option B : Phase 13 = contest curriculum (risqué)
Ajouter au mix synthétique :
- 2-4 stations en parallèle à ±50/100/150 Hz, énergies ratio 1:0.2 / 1:0.5
- Sérials WPX 001-9999 (format `5NN 1234`) → on a déjà 5NN cut-numbers
- Word_gap_inflation × 1.5 (les ops contest empilent)

**Risque** : 6 retrains consécutifs ont échoué le release_gate depuis Phase 8. Ajouter une distribution radicalement nouvelle (multi-station) sur la même base (5.5 / 11b) peut casser le gate **encore**. Précaution : entraîner en **add-on layer** (LoRA-style) plutôt qu'en fine-tune complet, ou faire 2 modèles (general + contest preset).

### Option C : multi-channel inference (sans retrainer)
Pre-process audio : isoler le carrier primaire par notch filter étroit (±30 Hz) avant d'envoyer au décodeur. Coût zéro côté training, gain attendu visible si la diagnosis est correcte.

**Recommandation forte : Option C d'abord** (1-2 h de travail, vérifie l'hypothèse sans toucher au modèle). Si gain confirmé → Option B comme amélioration ; sinon → Option A.

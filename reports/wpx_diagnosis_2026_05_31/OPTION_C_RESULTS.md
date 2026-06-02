> **MAJ 2026-06-02 :** ce rapport teste surtout narrow=60. Le factoriel
> `FACTORIAL_2026_06_02.md` a depuis montré que **bw=100 est le défaut optimal**
> pour le preset `contest` (récupère DA1A, zéro régression LCWO/websdr) alors que
> bw=60 sur-rétrécit (régresse websdr-fav22 et re-hallucine cwcww5). Le preset
> `contest` est donc à **100 Hz**, pas 60.

# Option C — Pre-process narrow bandpass : résultats

## TL;DR

**Gain partiel confirmé** : narrow `--bandwidth=60 Hz` (vs default 200) **améliore 3/5 fichiers**, avec un gain majeur sur `cwcwwDA1A.wav` (la structure de QSO devient lisible). 2/5 fichiers ne bougent pas — la cause est ailleurs.

**Le carrier-fix seul (--freq=détecté vs 600 Hz) a ZÉRO impact** — output identique 4/5 fichiers. Le front-end tolère bien les ±25 Hz d'offset.

→ **La bandwidth est le levier**, pas le carrier offset. Mon hypothèse "pile-up = problème dominant" est **partiellement validée** : ça concerne 3 fichiers sur 5.

## Matrice testée (5 fichiers × 4 conditions)

- `baseline` : `--freq=600 --bandwidth=200` (default actuel)
- `carrierfix` : `--freq=<détecté> --bandwidth=200`
- `narrow100` : `--freq=<détecté> --bandwidth=100`
- `narrow60` : `--freq=<détecté> --bandwidth=60`

## Résultats détaillés

### cwcwwS50RS.wav — pas d'amélioration
| Condition | RNN-T (extrait) |
|---|---|
| baseline | `550RS0R -50RS0R 5NS50RS0R 55RS0R 455050 ...BBBBB` |
| narrow60 | `450RS0R -50R0 4S5RS0 -5RS0R -5050R ...UBKIIERRBBKELIB` |

**Verdict** : RNN-T coincé en boucle sur `S50RS` répété. Le narrowing ne change rien. **Failure mode différent** = pathologie du décodeur quand un opérateur fait du rapid-fire callsign repeats (typique pile-up DX). À traiter avec une heuristique anti-loop ou beam search, pas avec un filtre.

### cwcww11.wav — gain marginal
| Condition | RNN-T |
|---|---|
| baseline | `B2F4HYYF4HYY5 5NN T11 UCQOM7M OM7M` |
| narrow60 | `BKK2FFF4HYYF4HYY 453 5NN T11AUCQ OM7M OM7M E` |

`5NN T11` reste détecté, plus de chars autour. Marginal.

### cwcww5.wav — gain mineur (prosign CQ fixé)
| Condition | RNN-T |
|---|---|
| baseline | `7AATEST F4HYY F4HYY588 5NN TT5 XHA7A **COT** HA7A HA7A` |
| narrow60 | `7AATEST F4HYY F4HYYN5 5NN TT5 XHA7A **CQ** HA7A HA7A` |

`COT` → `CQ` : la hallucination prosign disparaît. Gain qualitatif.

### cwcwwDA1A.wav — GAIN MAJEUR
| Condition | RNN-T |
|---|---|
| baseline | `114HYY OF4?4HYY .4F4?4HYYY M4YYN1185NNN 118 RMÉA11ACQQZA1ATESTGGG4AAA3NN186TUU-1R SK+CQA1JSTBB` |
| narrow60 | `A1ASTEFF44HYY FF4?4HYY +4F4? F4HYYY F4HYY 5NN 1185NN 118 UKBKRR DA1LS4CCQZDA1A TESTGGG4AAV **5NN 1186** BTU DA1R MUCQ DA1A TESTBK` |

Avec narrow60, on **lit la structure du QSO** :
- DA1A appelle CQ TEST
- F4HYY répond
- F4HYY envoie 5NN 1185 (probablement ton serial sortant)
- DA1A confirme et envoie son `5NN 1186` (ton serial entrant)
- BK, DA1A relance CQ

C'est **utilisable**. Le baseline était inexploitable.

### cwcwwHA8A.wav — quantité augmentée (signal et bruit)
| Condition | Longueur RNN-T | HA8A mentions | callsign-like |
|---|---|---|---|
| baseline | 4613 chars | 96 | 27 |
| carrierfix | 5349 | 114 | 37 |
| narrow100 | 5358 | (n/a) | (n/a) |
| narrow60 | 6189 | 129 | 47 |

Narrow60 **récupère 33 % de mentions HA8A en plus et 74 % de callsigns en plus**, mais avec −5 marqueurs 5NN. Net positif probable (plus de signal récupéré sur les callers du pile-up multi-station), mais à juger sur transcription humaine.

## Conclusion sur l'hypothèse

| Mode d'échec hypothétique | Confirmé ? | Évidence |
|---|---|---|
| Carrier offset training/inference | **NON** | --freq fix = output identique 4/5 |
| Pile-up multi-station QRM | **OUI** (3/5) | narrow60 améliore DA1A, cwcww5, cwcww11, HA8A |
| RNN-T loop sur callsign rapid-fire | **OUI** (1 cas) | S50RS bloqué quel que soit le filtre |
| Bruit de contest générique | inconnu | non testé |

L'hypothèse **"pile-up dominant"** était partiellement juste. Le narrowing à 60 Hz est une victoire facile pour ~60 % des fichiers.

## Recommandations actionnables

### 1. Ajouter une option `--bandwidth` au CLI `morseformer decode` (5 min)
Actuellement seul `scripts/decode_audio.py` l'expose. Le wrapper `morseformer/cli/decode.py` doit forwarder le flag.

### 2. Ajouter un preset `contest_narrow` (15 min)
Default `bandwidth=60` (vs 200), `confidence_threshold=0.6`, `digit_threshold=0.9`.

### 3. **Avant de défaut** : valider sur les bancs existants
- LCWO bench (clean, mono-station) → vérifier que narrow60 ne régresse pas
- AWGN guard (faible SNR) → narrower BPF = moins de bruit OK, mais aussi moins de signal si carrier drift

### 4. Côté RNN-T loop (S50RS)
Phase 13.1 candidate : **émission cap par caractère répété** dans le décodeur (postprocess), exemple : interdire > 3 répétitions consécutives d'un même token. Coût zéro côté training.

### 5. Test live conseillé
Tu peux activer le narrow60 dès maintenant sans toucher au modèle :
```bash
python scripts/decode_audio.py mon_fichier.wav --ckpt release/rnnt_phase11b.pt \
    --bandwidth 60 --confidence-threshold 0.6 --digit-threshold 0.9
```
ou via la GUI si on ajoute le flag. À tester sur d'autres fichiers WPX pour confirmer.

"""Training-data generation for the morseformer acoustic model.

All text sources are synthetic and hardcoded in this package — no
external corpus dependency. The synthetic pipeline is:

    itu_prefixes  →  callsign generator  ─┐
    Q-codes / CW abbreviations            │
    QSO grammar templates                 ├─►  text sampler  →  audio synth  →  features
    numeric / punctuation samples         │
    English-word corpus                   ┘

Phase 2.0 trains on clean audio only. Channel impairments are added in
Phase 2.1 once the model has demonstrated that it has learned the CW
code on noise-free inputs.
"""

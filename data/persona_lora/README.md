# Persona LoRA Preference Data

`ramsay_style_preferences.json` contains a small preference dataset for the main assignment LoRA-DPO/TRl track. Each row has:

- `prompt`: user instruction
- `chosen`: preferred Ramsay-inspired but still useful response
- `rejected`: safer, more neutral baseline response

The assignment manual formats these rows into TRL conversational DPO records with `prompt`, `chosen`, and `rejected` as role-message lists.

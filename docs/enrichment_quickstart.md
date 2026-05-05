# Enrichment Quickstart (ORA/GSEA/ULM)

This short tutorial focuses on interpretation-first analysis after embedding.

## Install Optional Enrichment Extras

```bash
pip install -e ".[enrichment]"
```

## End-to-End Example

```python
import pandas as pd
import spVIPESmulti

# Assume combined AnnData is already prepared and registered.
model = spVIPESmulti.model.spVIPESmulti(combined)
model.train(max_epochs=200)
model.embed(batch_size=512)

# Minimal long-format network (source -> target).
network = pd.DataFrame(
    {
        "source": ["TF1", "TF1", "TF1", "TF1", "TF1", "TF2", "TF2", "TF2", "TF2", "TF2"],
        "target": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5", "Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
    }
)

res = model.get_enrichment_scores(
    network,
    methods=["ora", "gsea", "ulm"],
    obsm_key="X_spvm_enrichment",
    uns_key="spvm_enrichment",
)

summary = model.summarize_enrichment(res["scores_df"], groupby="groups")
report = model.interpretation_report(
    res["scores_df"],
    groupby="groups",
    label_key="cell_type",  # optional
)
```

## Visualization Helpers

```python
import spVIPESmulti

# Heatmap of enrichment programs (group-aggregated).
ax = spVIPESmulti.pl.enrichment_heatmap(
    res["scores_df"],
    group_labels=combined.obs["groups"],
    top_n=20,
)

# Combined embedding + enrichment dashboard.
fig = spVIPESmulti.pl.interpretation_dashboard(
    combined,
    res["scores_df"],
    groupby="groups",
)
```

## Notes

- If decoupler is missing, `get_enrichment_scores(...)` raises an actionable error with the install command.
- The API writes deterministic output keys to `adata.obsm` and `adata.uns` by default.
- ULM is included by default in the recommended method set.

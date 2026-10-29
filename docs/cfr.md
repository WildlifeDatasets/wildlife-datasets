# CFR and SSI: Evidence behind animal identification

In identification of individual animals, images of the same individual often share similar backgrounds, locations, or capture conditions. 
As a result, AI models may learn to associate these contextual cues with individual identities rather than relying on the animal’s visual characteristics. 
This can lead to overly optimistic evaluation results and poor generalization to new environments. 
Standard retrieval metrics, such as Recall@k, cannot reveal this problem: they measure whether the correct individual was retrieved, but not what visual information supported the match.

**Counterfactual Foreground Retrieval (CFR)** and the **Shortcut Signal Indicator (SSI)** address this limitation by evaluating retrieval performance across different views of the same image, allowing us to distinguish identity information associated with the animal from information available in its surroundings.

## Understanding CFR and SSI

CFR and SSI evaluate retrieval performance using three views of each image:

- **Full image:** The original image.
- **Foreground:** Only the animal, with the background removed.
- **Shortcut:** Image with removed animal, i.e., background only view.

By comparing retrieval performance across these views, we can check how much identity information is available from the animal itself and how much can be recovered from shortcuts.
The CFR and SSI are then capturing different aspects:

| Metric | What it measures                                                            | Interpretation |
|---|-----------------------------------------------------------------------------|---|
| **CFR** | Retrieval performance supported by full image and animal foreground. | Higher values indicate stronger foreground-supported retrieval. |
| **SSI** | Identity information recoverable from shortcut views relative to the full image. | Higher values indicate stronger shortcut signals. |

For example, a model with **high CFR and low SSI** achieves strong retrieval performance using the animal foreground, while little identity information can be recovered from shortcuts. 
Conversely, **high CFR and high SSI** indicate that both the animal and shortcut views contain useful identity information.

## Computing CFR and SSI

To compute CFR and SSI, you only need to infer your model on each image view and collect the predicted individual identities. 
The different views are available through `dataset[i].cfr_views` (see the [mask guide](masks.md) for details).

Once you have the predictions, both metrics can be computed directly using the `wildlife_datasets.metrics` module:

```python
from wildlife_datasets import metrics

predictions = {
    "full": full_predictions,
    "foreground": foreground_predictions,
    "shortcut": shortcut_predictions,
    "silhouette": silhouette_predictions,  # Optional
}

cfr_score = metrics.cfr(dataset, predictions)
ssi_score = metrics.ssi(dataset, predictions)
```

Each entry in `predictions` should contain the predicted individual IDs in dataset order, either as a single ID per image or as a ranked list of IDs for Recall@k.

All views must be evaluated against the same masked gallery of known individuals, with self-matches excluded. The functions automatically retrieve the ground-truth identities and mask coverage from the dataset and compute the required Recall@k values and random baselines.

Both metrics range from 0 to 1 when defined.

**Note:** For species where shape is an important identification cue, such as whales identified by their flukes, use `silhouette_is_shortcut=False` when computing SSI.

## Example Notebook

For an example evaluation pipeline, please refere to the [CFR evaluation notebook](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/notebooks/cfr.ipynb). 
It demonstrates how to generate predictions using MegaDescriptor-T-224 and compute Recall@k, random Recall@k, CFR, and SSI.

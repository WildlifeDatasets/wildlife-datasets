import os

from wildlife_datasets.datasets import names_all

meta_datasets = ["AnimalCLEF2025", "AnimalCLEF2026", "WildlifeReID10k"]
merge_datasets = {
    "BalearicLizard": "BalearicLizardSegmented",
    "BirdIndividualID": "BirdIndividualIDSegmented",
    "SealID": "SealIDSegmented",
    "SeaTurtleID2022": "SeaTurtleIDHeads",
}
pdf_icon = "https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/pdf_icon.png"


def dataset_section(name: str, summary: dict, docs_dir: str) -> str:
    if name not in merge_datasets:
        header = f"### [{name}]({summary['url']})"
    else:
        header = f"### [{name}, {merge_datasets[name]}]({summary['url']})"
    if summary.get("publication_url"):
        header += f' <a href="{summary["publication_url"]}" target="_blank"><img src="{pdf_icon}" alt="publication link" width="20"></a>'
    if name in merge_datasets:
        header += f' {{ #{name.lower()} data-toc-label="{name}" }}'
    lines = [header, summary.get("description", "")]
    image = f"images/grid_{name}.png"
    if os.path.exists(os.path.join(docs_dir, image)):
        lines.append(f"![]({image})")
    elif name in merge_datasets:
        image = f"images/grid_{merge_datasets[name]}.png"
        if os.path.exists(os.path.join(docs_dir, image)):
            lines.append(f"![]({image})")
    return "\n\n".join(line for line in lines if line)


def datasets_markdown(docs_dir: str) -> str:
    summaries = {c.display_name(): c.summary for c in names_all}
    names = sorted(summaries, key=str.lower)
    merge_values = list(merge_datasets.values())
    lines = ["## Overview of MetaDatasets"]
    for name in names:
        if name in meta_datasets and name not in merge_values:
            lines += [dataset_section(name, summaries[name], docs_dir)]
    lines += ["## Overview of Datasets"]
    for name in names:
        if name not in meta_datasets and name not in merge_values:
            lines += [dataset_section(name, summaries[name], docs_dir)]
    return "\n\n".join(lines) + "\n"


def on_page_markdown(markdown: str, page, config, files) -> str:
    if page.file.src_uri == "datasets.md":
        return markdown.replace("<!-- datasets -->", datasets_markdown(config["docs_dir"]))
    return markdown

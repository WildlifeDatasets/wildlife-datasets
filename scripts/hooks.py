import os

from wildlife_datasets.datasets import names_all

merge_datasets = {
    "BalearicLizard": "BalearicLizardSegmented",
    "BirdIndividualID": "BirdIndividualIDSegmented",
    "SealID": "SealIDSegmented",
    "SeaTurtleID2022": "SeaTurtleIDHeads",
}
pdf_icon = "https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/pdf_icon.png"


def get_summaries(meta: bool) -> dict[str, dict]:
    summaries = {c.display_name(): c.summary for c in names_all}
    merge_values = list(merge_datasets.values())
    names = sorted(summaries, key=str.lower)
    summaries_subset = {}
    for name in names:
        if summaries[name].get("original_dataset", True) != meta and name not in merge_values:
            summaries_subset[name] = summaries[name]
    return summaries_subset


def publication_link(summary: dict, width: int) -> str:
    if not summary.get("publication_url"):
        return ""
    return f'<a href="{summary["publication_url"]}" target="_blank"><img src="{pdf_icon}" alt="publication link" width="{width}"></a>'


def dataset_link(name: str, summary: dict) -> str:
    return f"[{name}]({summary['url']})" if summary.get("url") else name


def dataset_section(name: str, summary: dict, docs_dir: str) -> str:
    if name not in merge_datasets:
        header = f"### {dataset_link(name, summary)}"
    else:
        header = f"### {dataset_link(f'{name}, {merge_datasets[name]}', summary)}"
    if summary.get("publication_url"):
        header += " " + publication_link(summary, 20)
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


def datasets_markdown(meta: bool, docs_dir: str) -> str:
    summaries = get_summaries(meta)
    return "\n\n".join(dataset_section(name, summary, docs_dir) for name, summary in summaries.items())


def preprocessing_markdown(meta: bool) -> str:
    summaries = get_summaries(meta)
    lines = [
        f"| {'MetaDataset' if meta else 'Dataset'} | Publication | Processing requirements | Dataset modifications |",
        "|--|:--|:--|:--|",
    ]
    for name, summary in summaries.items():
        cells = [
            dataset_link(name, summary),
            publication_link(summary, 16),
            summary.get("processing") or "",
            summary.get("modifications") or "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:
    docs_dir = config["docs_dir"]
    if page.file.src_uri == "datasets.md":
        markdown = markdown.replace("<!-- datasets-meta -->", datasets_markdown(True, docs_dir))
        markdown = markdown.replace("<!-- datasets -->", datasets_markdown(False, docs_dir))
    elif page.file.src_uri == "preprocessing.md":
        markdown = markdown.replace("<!-- preprocessing-meta -->", preprocessing_markdown(True))
        markdown = markdown.replace("<!-- preprocessing -->", preprocessing_markdown(False))
    return markdown

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import eda, tracing

DEVELOPMENT_DIR = Path(__file__).resolve().parents[1] / "data" / "development"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "eda"
TRACE_PATH = Path(__file__).resolve().parents[1] / "reports" / "trace" / "pipeline.jsonl"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = eda.load_annotations(DEVELOPMENT_DIR)
    # EDA operates on the whole dataset at once (not per-image), so it gets
    # a single trace record rather than one per image.
    with tracer.step("dataset", "eda", {"num_images": len(annotations["images"])}):
        run_eda(annotations)


def run_eda(annotations: dict) -> None:
    categories = eda.category_distribution(annotations)
    categories.to_csv(OUTPUT_DIR / "category_distribution.csv", index=False)
    top = categories.head(30)
    plt.figure(figsize=(10, 8))
    plt.barh(top["name"][::-1], top["num_annotations"][::-1])
    plt.xlabel("num_annotations")
    plt.title("Top 30 categorías por número de instancias")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.png")
    plt.close()

    instances = eda.instances_per_image(annotations)
    instances.to_csv(OUTPUT_DIR / "instances_per_image.csv")
    plt.figure()
    plt.hist(instances, bins=range(0, int(instances.max()) + 2))
    plt.xlabel("instancias por imagen")
    plt.ylabel("num imágenes")
    plt.title("Distribución de instancias por imagen")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "instances_per_image.png")
    plt.close()

    sizes = eda.object_size_distribution(annotations)
    sizes.to_csv(OUTPUT_DIR / "object_sizes.csv", index=False)
    plt.figure()
    log_bins = pd.Series(sizes["area_ratio"]).clip(lower=1e-5)
    bins = [10 ** e for e in [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]]
    plt.hist(log_bins, bins=bins)
    plt.xscale("log")
    for q in (1 / 3, 2 / 3):
        plt.axvline(sizes["area_ratio"].quantile(q), color="red", linestyle="--")
    plt.xlabel("area_ratio (área objeto / área imagen, escala log)")
    plt.ylabel("num instancias")
    plt.title("Distribución de area_ratio (líneas rojas: terciles propuestos small/medium/large)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "object_sizes.png")
    plt.close()

    occlusion = eda.occlusion_stats(annotations)
    occlusion.to_csv(OUTPUT_DIR / "occlusion.csv", index=False)
    plt.figure()
    plt.hist(occlusion["max_neighbor_iou"], bins=40)
    plt.xlabel("IoU máximo con el vecino más cercano (bbox)")
    plt.ylabel("num instancias")
    plt.title("Proxy de oclusión/crowding por instancia")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occlusion.png")
    plt.close()

    neg_summary = eda.negative_and_non_exhaustive_categories(annotations)
    with open(OUTPUT_DIR / "neg_not_exhaustive_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(neg_summary, f, ensure_ascii=False, indent=2)

    print(f"EDA guardado en {OUTPUT_DIR}")
    print(f"- {len(categories)} categorías con anotaciones, top: {categories.iloc[0]['name']} ({categories.iloc[0]['num_annotations']})")
    print(f"- instancias por imagen: media={instances.mean():.1f}, mediana={instances.median():.0f}, máx={instances.max()}")
    print(f"- area_ratio terciles propuestos: {sizes['area_ratio'].quantile(1/3):.4f} / {sizes['area_ratio'].quantile(2/3):.4f}")
    print(f"- imágenes con neg_category_ids: {neg_summary['images_with_neg_category_ids']}")
    print(f"- imágenes con not_exhaustive_category_ids: {neg_summary['images_with_not_exhaustive_category_ids']}")


if __name__ == "__main__":
    main()

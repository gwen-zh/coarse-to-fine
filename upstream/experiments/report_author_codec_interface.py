"""Descriptive paired reporting for the frozen author->own-codec pilot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_author_codec_interface import ROOT, GEOMETRY_KEYS, METRIC_KEYS, verify


def main():
    p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
    result = json.loads((ROOT / "complete.json").read_text())
    if len(result["rows"]) != 4 or not result["codec_unchanged"] or result["optimizer_updates"] != 0:
        raise RuntimeError("Incomplete or non-read-only interface evaluation")
    summary = result["two_seed_development_macro"]
    available = all(summary[a]["metrics_available"] for a in ["original", "projected"])
    report = dict(status="complete_awaiting_review", scope=p["scope"],
        original=summary["original"], projected=summary["projected"], rows=result["rows"],
        native_oracle_diagnostic=result["native_oracle_diagnostic"],
        optimizer_updates=0, codec_unchanged=True, independent_test_claim=False,
        test_coordinates_opened=False, R3_opened=False,
        report_source_sha256=sha(__file__), protocol_sha256=sha(ROOT / "protocol.json"))
    if available:
        report["paired_macro_delta"] = {k: summary["projected"][k] - summary["original"][k] for k in METRIC_KEYS}
        report["paired_macro_delta"]["geometry"] = {k: summary["projected"]["geometry"][k] - summary["original"]["geometry"][k] for k in GEOMETRY_KEYS}
        report["mean_aligned_input_CA_displacement_A"] = float(np.mean([
            r["paired_input_displacement"]["mean_aligned_CA_RMSD_A"] for r in result["rows"]]))
        report["maximum_preserved_block_center_roundoff_A"] = max(r["projected"]["maximum_center_roundoff_A"] for r in result["rows"])
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
        panels = [
            ("mean_sample_nearest_reference_ca_rmsd_A", "Generated → nearest reference (Å)", False, 1),
            ("mean_reference_nearest_sample_ca_rmsd_A", "Reference → nearest generated (Å)", False, 1),
            ("cn_rate", "C–N outliers (%)", True, 100),
            ("js_pairwise_distance_divergence_nats", "Pair-distance JS divergence (nats)", False, 1),
            ("reference_ca_c_n_angle_mae_degrees", "CA–C–N angle error (degrees)", True, 1),
            ("reference_c_n_ca_angle_mae_degrees", "C–N–CA angle error (degrees)", True, 1)]
        colors = ["#737373", "#3577a7"]
        for ax, (key, label, geom, scale) in zip(axes.flat, panels):
            values = [(summary[a]["geometry"] if geom else summary[a])[key] * scale for a in ["original", "projected"]]
            ax.bar([0, 1], values, color=colors, width=.6)
            ax.set_xticks([0, 1], ["Author", "Author → own codec"], fontsize=9)
            for i, value in enumerate(values):
                ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
            ax.set_ylabel(label); ax.set_ylim(0, max(values) * 1.22 + 1e-9)
            ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax = axes[1, 2]; names = [t["protein"] for t in p["targets"]]; positions = np.arange(len(names))
        for offset, arm, color in [(-.18, "original", colors[0]), (.18, "projected", colors[1])]:
            spread = [np.mean([r[arm]["mean_pairwise_ca_rmsd_A"] for r in result["rows"] if r["protein"] == name]) for name in names]
            ax.bar(positions + offset, spread, width=.34, color=color, label="Author" if arm == "original" else "Author → own codec")
        reference = [next(r["original"]["reference_mean_pairwise_ca_rmsd_A"] for r in result["rows"] if r["protein"] == name) for name in names]
        ax.scatter(positions, reference, marker="_", s=250, color="black", label="Native reference", zorder=5)
        ax.set_xticks(positions, names); ax.set_ylabel("Within-ensemble CA spread (Å)")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax = axes[1, 3]
        displacement = [np.mean([r["paired_input_displacement"]["mean_aligned_CA_RMSD_A"] for r in result["rows"] if r["protein"] == name]) for name in names]
        native = [next(r["paired_native_reconstruction_CA_RMSD_A"] for r in result["native_oracle_diagnostic"] if r["protein"] == name) for name in names]
        ax.bar(positions - .18, displacement, width=.34, color=colors[1], label="Generated input → codec")
        ax.bar(positions + .18, native, width=.34, color="#aa7744", label="Native input → codec (oracle)")
        ax.set_xticks(positions, names); ax.set_ylabel("Paired encode/decode CA RMSD (Å)")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Can our frozen representation preserve an AA-only structural prior?\n"
            "2 reused development proteins · 2 seeds · 64 raw samples/group · no training or quantization", fontsize=12)
        directory = Path("paper/figures/author_codec_interface_v1"); directory.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            fig.savefig(directory / ("Author_Codec_Interface." + ext), dpi=170)
        plt.close(fig)
        report["figure"] = str(directory / "Author_Codec_Interface.pdf")
    atomic_json(ROOT / "report.json", report)
    print("AUTHOR_CODEC_INTERFACE_REPORT", json.dumps({k: v for k, v in report.items() if k not in ["rows", "original", "projected"]}), flush=True)


if __name__ == "__main__":
    main()

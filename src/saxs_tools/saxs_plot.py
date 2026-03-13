import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import re
from scipy.signal import find_peaks

def peak_to_axis_fraction(ax, y):
    y0, y1 = ax.get_ylim()

    if ax.get_yscale() == "log":
        return (np.log(y) - np.log(y0)) / (np.log(y1) - np.log(y0))
    else:
        return (y - y0) / (y1 - y0)

def read_saxs_curve_with_trust(curve_path):
    curve_path = Path(curve_path)

    trust = 0.0
    with curve_path.open("r", encoding="utf8") as f:
        for line in f:
            if "Trust" in line:
                trust = float(line.strip().split()[-1])

    data = np.loadtxt(curve_path, comments=["@", "#"])
    q = data[:, 0]
    intensity = data[:, 1]

    return q, intensity, trust


def find_saxs_peaks(q, intensity, min_height=10.0, max_q=3.5):
    peaks, props = find_peaks(intensity, height=min_height)

    q_peaks = q[peaks]
    intensity_peaks = intensity[peaks]

    mask = q_peaks <= max_q
    q_peaks = q_peaks[mask]
    intensity_peaks = intensity_peaks[mask]

    if len(q_peaks) > 0:
        q1 = q_peaks[0]
        q_ratios = q_peaks / q1
    else:
        q_ratios = np.array([])

    return q_peaks, q_ratios, intensity_peaks


def write_peaks_txt(output_txt, q_peaks, q_ratios, intensity_peaks):
    output_txt = Path(output_txt)
    with output_txt.open("w", encoding="utf8") as f:
        f.write("Peaks\n")
        f.write(f"{'Peak':<5} {'q (nm^-1)':<12} {'q/q1':<10} {'I (a.u.)':<12}\n")
        for i, (qp, qr, ip) in enumerate(zip(q_peaks, q_ratios, intensity_peaks), start=1):
            f.write(f"{i:<5} {qp:<12.3f} {qr:<10.2f} {ip:<12.2f}\n")


def plot_saxs_curve(
    q,
    intensity,
    trust,
    output_png,
    title="Martini SAXS Profile",
    peak_height=10.0,
    peak_max_q=3.5,
    xlim=(0.0, 3.5),
    ylim=None,
):
    q_peaks, q_ratios, intensity_peaks = find_saxs_peaks(
        q, intensity, min_height=peak_height, max_q=peak_max_q
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(q, intensity, label="Simulation", linewidth=2.5)
    ax.plot(q_peaks, intensity_peaks, "o")

    ax.set_xlabel("q (1/nm)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    ax.set_title(title, fontsize=16)
    

    #ymin, ymax = ax.get_ylim()

    ax.set_yscale("log")
    ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    for i, (qp, ip) in enumerate(zip(q_peaks, intensity_peaks)):
        ymax_frac = peak_to_axis_fraction(ax, ip)
        ymax_frac = np.clip(ymax_frac, 0.0, 1.0)
        line = mlines.Line2D(
            [qp, qp],
            [0.0, ymax_frac],
            transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
            linestyle="dashed",
            color="gray",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.add_line(line)

        ax.annotate(
            f"{qp:.2f}",
            xy=(qp, 0),
            xycoords=("data", "axes fraction"),
            xytext=(-15, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        if i < 6:
            ax.text(qp, ip * 1.05, f"{q_ratios[i]:.2f}", ha="center", fontsize=10)

    

    ax.axvspan(0.0, trust, color="gray", alpha=0.2)
    ax.tick_params(length=6, width=1.5, labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return q_peaks, q_ratios, intensity_peaks

def add_plot_parser(subparsers):
    plot_parser = subparsers.add_parser("plot", help="Plot SAXS output and detect peaks")
    plot_parser.add_argument(
        "--out-base",
        default=None,
        help="Base output path used during compute. Default: latest martini_saxs_<timestamp> in the search directory.",
    )
    plot_parser.add_argument(
        "--search-dir",
        default=".",
        help="Directory to search for latest martini_saxs_<timestamp> output if --out-base is omitted.",
    )
    plot_parser.add_argument("--peak-height", type=float, default=10.0)
    plot_parser.add_argument("--peak-max-q", type=float, default=3.5)
    plot_parser.add_argument("--title", default=None)
    plot_parser.add_argument("--x-min", type=float, default=0.0)
    plot_parser.add_argument("--x-max", type=float, default=3.5)
    plot_parser.add_argument("--y-min", type=float, default=None)
    plot_parser.add_argument("--y-max", type=float, default=None)
    return plot_parser

def plot_from_out_base(
    out_base,
    peak_height=10.0,
    peak_max_q=3.5,
    title=None,
    xlim=(0.0, 3.5),
    ylim=None,
):
    out_base = Path(out_base)
    curve_path = Path(f"{out_base}_saxs_curve.dat")
    output_png = Path(f"{out_base}_plot.png")
    output_txt = Path(f"{out_base}_peaks.txt")

    q, intensity, trust = read_saxs_curve_with_trust(curve_path)

    if title is None:
        title = f"Martini SAXS Profile ({out_base.name})"

    q_peaks, q_ratios, intensity_peaks = plot_saxs_curve(
        q=q,
        intensity=intensity,
        trust=trust,
        output_png=output_png,
        title=title,
        peak_height=peak_height,
        peak_max_q=peak_max_q,
        xlim=xlim,
        ylim=ylim,
    )

    write_peaks_txt(output_txt, q_peaks, q_ratios, intensity_peaks)

    return output_png, output_txt

def find_latest_martini_saxs_out_base(search_dir="."):
    search_dir = Path(search_dir)
    pattern = re.compile(r"^(martini_saxs_\d{8}_\d{6})_saxs_curve\.dat$")

    candidates = []
    for path in search_dir.glob("martini_saxs_*_saxs_curve.dat"):
        match = pattern.match(path.name)
        if match:
            base = match.group(1)
            candidates.append((path.stat().st_mtime, search_dir / base))

    if not candidates:
        raise FileNotFoundError(
            f"No martini_saxs_<timestamp> results found in '{search_dir}'. "
            "Either run 'martini_saxs compute' first or provide --out-base explicitly."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def run_plot_command(args):
    out_base = args.out_base
    if out_base is None:
        out_base = find_latest_martini_saxs_out_base(args.search_dir)

    ylim = None
    if args.y_min is not None or args.y_max is not None:
        if args.y_min is None or args.y_max is None:
            raise ValueError("Provide both --y-min and --y-max, or neither.")
        ylim = (args.y_min, args.y_max)

    output_png, output_txt = plot_from_out_base(
        out_base=out_base,
        peak_height=args.peak_height,
        peak_max_q=args.peak_max_q,
        title=args.title,
        xlim=(args.x_min, args.x_max),
        ylim=ylim,
    )

    print(f"Wrote plot: {output_png}")
    print(f"Wrote peaks: {output_txt}")

if __name__=="__main__":
    pass
"""Plot descriptive calibration covariance spectra; no evaluation or rank selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'gpt2_gate_subspace_run');ap.add_argument('--out',type=Path,default=ROOT/'figures11');args=ap.parse_args()
s = json.loads((args.run/'calibration_spectra.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for layer in range(4, 12):
    for ax, space in zip(axes, ['b', 'r']):
        ax.plot(range(1, 13), s[str(layer)][space]['cumulative_explained_variance'], 'o-', ms=3, label=f'L{layer}')
for ax, title in zip(axes, ['Centered allocation logits b', 'Centered logits after subtracting log(q)']):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Number of principal directions retained')
    ax.set_ylabel('Cumulative calibration variance')
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].legend(ncol=2, fontsize=8, loc='lower right')
fig.suptitle('Eight calibration passages: covariance describes variation, not causal necessity', fontsize=12)
out=args.out;out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'gate_spectrum.png', dpi=170)
fig.savefig(out/'gate_spectrum.svg')
plt.close(fig)

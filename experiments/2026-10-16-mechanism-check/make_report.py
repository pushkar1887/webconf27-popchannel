"""Render only measured diagnostics and the predeclared continue/pivot decision."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def interval_text(values):
    return '[' + ', '.join(f'{v:.6f}' for v in values) + ']'


def main():
    r = json.loads((ROOT / 'results.json').read_text())
    p = json.loads((ROOT / 'protocol.json').read_text())
    m = json.loads((ROOT / 'fit_manifest.json').read_text())
    v = json.loads((ROOT / 'verification.json').read_text())
    decision = 'CONTINUE ONLY WITH A NARROWED MECHANISM STUDY' if r['gate_pass'] else 'CLOSE THE CURRENT COMPRESSION PROPOSAL; PIVOT'
    figure_dir = ROOT / 'figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0))
    panels = [('high_minus_low', 'Does higher contextual disagreement add damage?'), ('neutral_minus_low', 'Does low-disagreement grouping beat random pairing?')]
    for ax, (key, title) in zip(axes, panels):
        for pos, result in enumerate(r['pairing_results']):
            contrast = result['contrasts'][key]
            point = contrast['estimate']
            lo, hi = contrast['block_bootstrap_95']
            ax.errorbar(point, pos, xerr=[[point - lo], [hi - point]], fmt='o', capsize=5, color='#1e6091', markersize=7)
        ax.axvline(0, color='#555555', linewidth=1)
        ax.axvline(.001, color='#aaaaaa', linestyle='--', linewidth=1)
        ax.set_yticks(range(3), [f"Seed {z['seed']} ({z['matched_users']:,} matched users)" for z in r['pairing_results']])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, loc='left', pad=14)
        ax.set_xlabel('Difference in log loss (positive supports the proposed effect)')
        ax.set_xlim(-.007, .007)
        ax.set_xticks(np.arange(-.006, .0061, .002))
        ax.grid(axis='x', alpha=.2)
        ax.margins(y=.35)
    fig.suptitle('Mechanism check: effects do not repeat across pairings', fontsize=14, fontweight='bold', x=.03, ha='left')
    fig.text(.03, .025, 'Bars: pointwise 95% matched-block bootstrap intervals; every bar spans zero. Dashed line: predefined 0.001 development threshold.\nThree seeds reuse the same source data; they are sensitivity checks, not independent confirmations. Validation only; no test rerun.', fontsize=9)
    fig.tight_layout(rect=[.01, .10, .99, .94])
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(figure_dir / ('mechanism_contrasts.' + ext), dpi=180, facecolor='white')
    plt.close(fig)
    lines = ['# Bounded mechanism check: executed decision', '', f'**Decision: {decision}.**', '',
        'The fixed primary gate fails. Contextual disagreement selected in training half A does not have established repeatability in disjoint half B; the validation sharing contrast changes sign across pairing seeds. This closes the current count-sharing/rare-context/tilted proposal under the agreed stopping rule. It does not establish a universal impossibility result about context-aware compression.', '',
        '## Question and contract', '',
        '> When users have similar overall behavior and training activity, but different response patterns across interfaces, does forcing them to share count parameters cause extra held-out prediction damage?', '',
        '- Real KuaiRand-Pure data from the already audited cache, unchanged UTC chronology, duplicate quarantine and native long_view labels.',
        '- Only training and validation splits were accessed by this diagnostic. Standard/random test outcomes were not re-evaluated. Both train and validation had prior development use; this is exploratory evidence.',
        '- Choose the interface pair with the largest number of training users having >=20 events in both, without using outcomes for pair selection. Interfaces 0 and 1 were selected, with 1,284 users before halving.',
        '- Randomly partition train records into disjoint A/B halves with seed 20261016. Each eligible user must retain >=8 records in each chosen interface in each half. A defines profiles and pairings; B fits models and checks profile repeatability. Random partitions do not establish statistical independence for correlated behavior.',
        '- Form four-user blocks with pairwise-caliper controls on A overall logit, A mean interface logit, and full-train activity in both interfaces. Maximum logit ranges .5; maximum context-specific activity ratio 2. Match greedily under three seeds, without validation labels.',
        '- Enumerate all three perfect two-user pairings in each block. Choose low/high disagreement from A only. Neutral is expected loss over the three equally probable pairing outcomes, not prediction averaging.',
        '- All shared models use two users per group, the same dense 15-context table and fixed-length mapping, identical alpha=10 and residual weight=1, and common B video/interface tables. This isolates a modest sharing intervention, not a severe compression regime.',
        '- Evaluate the same matched validation events for each strategy within a seed. Primary metric: per-event binary log loss. Also save equal-context loss, individual-user reference, matched-block bootstrap and user-bootstrap sensitivity.',
        '- A within-block weighted regression controls marginal-rate difference, overall-rate difference, activity mismatch, mean activity, mean marginal component and A item-exposure overlap. This describes adjusted association; it does not identify intrinsic preferences or remove all exposure confounding.', '',
        f"Training users eligible after the A/B support requirements: **{m['eligible_users']:,}**.", '',
        '## Measured sharing contrasts', '',
        'Positive high-minus-low supports extra damage from contextual disagreement. Positive neutral-minus-low supports a useful low-disagreement pairing rule. The rows use slightly different training-matched cohorts; comparisons are paired within a row, not between row loss levels.', '',
        '| Pairing seed | Matched users | Validation events | High minus low loss | 95% block interval | Neutral minus low loss | 95% block interval |',
        '|---:|---:|---:|---:|---|---:|---|']
    for z in r['pairing_results']:
        a, b = z['contrasts']['high_minus_low'], z['contrasts']['neutral_minus_low']
        lines.append(f"| {z['seed']} | {z['matched_users']:,} | {z['validation_events']:,} | {a['estimate']:.6f} | {interval_text(a['block_bootstrap_95'])} | {b['estimate']:.6f} | {interval_text(b['block_bootstrap_95'])} |")
    lines.extend(['', 'Every sharing-contrast interval above spans zero. The high-minus-low point estimate reverses sign in two of three seeds. The low-disagreement rule has no established gain over neutral pairing.', '',
        '## Does the selected contextual difference repeat?', '',
        'Contextual contrast is the difference between the two smoothed interface logits. Pair disagreement is its squared difference between paired users. Low/high selection uses A only; the next table measures high-minus-low disagreement using B.', '',
        '| Seed | A high-minus-low disagreement | B high-minus-low disagreement | B 95% block interval |', '|---:|---:|---:|---|'])
    for z, fit in zip(r['pairing_results'], m['pairing_seeds']):
        result = z['contrasts']['B_profile_repeatability']
        lines.append(f"| {z['seed']} | {fit['mean_A_high_minus_low']:.6f} | {result['estimate']:.6f} | {interval_text(result['block_bootstrap_95'])} |")
    lines.extend(['', 'B point differences are positive but all B intervals include zero. The large A separation was the criterion used to choose the pairings; its size cannot be treated as independent evidence. Failure to establish B repeatability does not prove that user differences are absent.', '',
        '## Adjusted association and fixed gate', '',
        '| Seed | Adjusted contextual-disagreement coefficient, per feature SD | 95% block interval |', '|---:|---:|---|'])
    for z in r['pairing_results']:
        result = z['contrasts']['adjusted_context_coefficient_per_feature_sd']
        lines.append(f"| {z['seed']} | {result['estimate']:.6f} | {interval_text(result['block_bootstrap_95'])} |")
    lines.extend(['', 'The primary seed has a positive adjusted association, but that result does not repeat across the other pairings. It does not override the failed reliability and sharing-intervention checks.', '',
        'The four effects use Bonferroni-adjusted one-sided bounds at family alpha .05. A .001 log-loss minimum effect is an analyst-chosen development threshold, not a proven practical-importance boundary.', '',
        '| Predeclared gate condition | Result |', '|---|---|'])
    for key, value in r['gate_conditions'].items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines.extend(['', f"Combined gate: **{'GO for a narrowed exploratory study' if r['gate_pass'] else 'NO-GO'}**.", '',
        '## Optimization sensitivity of the previous prototype', '',
        '- Extended the previous full-train clustering heuristic to at most 100 rounds for B=64/256, frequency/tilted lambda=.3, three common initializations: 12 fits and 24 saved round-12/final model snapshots.',
        '- All stopped because the heuristic found no accepted proposal, after 10–69 accepted rounds; none reached the 100-round limit. This is heuristic stalling, not certified optimality.',
        '- Several previous 12-round fits were unfinished. Continuing changed validation loss in both directions; improving the grouping objective does not guarantee better prediction.',
        '- The frequency/tilted validation ordering also changes across initializations. This does not establish a stable new-method improvement.',
        '- The four shared seed-11 round-12 snapshots exactly reproduce the earlier E0 assignment maps. The extended snapshots were evaluated only on validation; the old test-context sweep was not rerun. No claim is made that the original test conclusion has been exhaustively proved invariant to optimization.', '',
        '| B | Seed | Method | Accepted rounds | Round-12 validation loss | Final validation loss |', '|---:|---:|---|---:|---:|---:|'])
    for row in r['optimizer_results']:
        lines.append(f"| {row['budget']} | {row['seed']} | {'Frequency' if row['lambda'] is None else 'Tilted .3'} | {row['rounds']} | {row['round12_validation_loss']:.6f} | {row['final_validation_loss']:.6f} |")
    lines.extend(['', 'The round-12 label means the state at round 12 or an earlier heuristic stop. Several runs can improve the train objective while validation loss worsens. No validation-based checkpoint selection was used to rescue the earlier result.', '',
        '## Research decision', '',
        'Close this specific compression proposal and preserve the evidence. Do not start a three-day neural campaign, expand the tilt grid, select a favorable pairing seed, or present the positive primary regression coefficient as the paper result.', '',
        'Recommended next direction from the existing literature dossier: the agentic Web-analysis mechanism study, holding the same posts fixed while testing whether update history changes which supported viewpoints survive. This is a proposed pivot, not a validated replacement. It needs its own closest-baseline/novelty screen, semantic reference and development pilot before model-scale experiments. No new pivot experiment was executed here.', '',
        '## Verification, limits and runtime', '',
        f"Independent verification: **{v['status']}**. Replayed all {v['pair_predictors_replayed']} pair predictors and {v['optimizer_snapshots_replayed']} optimizer snapshots, checked matched calipers and equal grouped bytes, reconstructed B count tables, replayed full bootstrap contrast vectors and sampled adjusted-regression resamples, and recomputed the gate.", '',
        f"Successful CPU fitting {r['fit_seconds']:.2f}s; evaluation/bootstrap {r['evaluation_seconds']:.2f}s; independent verification {v['elapsed_seconds']:.2f}s. These exclude implementation, inspection and rendering. No GPU, Kaggle operation, NVIDIA generation, new download, package installation or external write occurred.", '',
        'Limits: native long_view/documentation inconsistency remains unresolved; observational logged responses are not intrinsic preferences; matching and linear adjustment do not remove all confounding; A/B events may remain dependent; three seeds reuse data; two-user groups do not reproduce GraphHash/CCE/BACO; exploratory validation cannot certify generalization, novelty or main-track readiness.', '',
        'Existing NumPy and Matplotlib were reused. Sonatype Guide is unavailable in this session, so no component-safety check is claimed; no packages were installed or upgraded.', '',
        'Verification commands: `python -m unittest discover -v` and `python verify_mechanism.py`. Re-running the frozen fitting/evaluation stages requires a new output directory; the scripts refuse overwrite.', '',
        'Main artifacts: protocol.json, fit_manifest.json, results.json, pair_metrics.csv, adjustment_pairs.csv, optimizer_validation_metrics.csv, training_halves.npz, blocks_s*.npz, block_aggregates.npz, bootstrap_contrasts.npz, pair_predictions_validation.npz, optimizer_predictions_validation.npz, models/, verification.json.', '',
        '![Validation mechanism contrasts](figures/mechanism_contrasts.png)', ''])
    (ROOT / 'MECHANISM_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    files = {}
    for file in sorted(ROOT.rglob('*')):
        if file.is_file() and '__pycache__' not in str(file) and file.name != 'artifact_manifest.json':
            files[str(file.relative_to(ROOT))] = {'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    (ROOT / 'artifact_manifest.json').write_text(json.dumps({'files': files, 'total_bytes': sum(row['bytes'] for row in files.values())}, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(ROOT / 'MECHANISM_REPORT.md'), 'figure': str(figure_dir / 'mechanism_contrasts.png'), 'decision': decision, 'artifact_bytes': sum(row['bytes'] for row in files.values())}))


if __name__ == '__main__':
    main()

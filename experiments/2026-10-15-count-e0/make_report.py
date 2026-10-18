"""Render the executed exploratory result; reads receipts, does not fit models."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_csv(name):
    return list(csv.DictReader((ROOT / name).open(encoding='utf-8')))


def main():
    results = json.loads((ROOT / 'results.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'fit_manifest.json').read_text())
    verification = json.loads((ROOT / 'verification.json').read_text())
    gates = json.loads((ROOT / 'gates.json').read_text())
    curves, contexts, raw = [read_csv(f) for f in ['sweep_metrics.csv', 'context_frequencies.csv', 'raw_metrics.csv']]
    by_curve = {(r['model'], r['cohort'], r['sweep'], float(r['gamma'])): r for r in curves}
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    panels = [
        ('all', 'all_flatten', 256, 'Primary: all users, all 15 interfaces, B=256'),
        ('all', 'qualified_flatten', 256, 'Secondary: all users, 5 qualified interfaces, B=256'),
        ('supported', 'qualified_flatten', 64, 'Exploratory: supported users, 5 interfaces, B=64'),
        ('supported', 'qualified_flatten', 256, 'Secondary: supported users, 5 interfaces, B=256'),
    ]
    colors = {'frequency': '#1e6091', 'tilted': '#c0392b', 'marginal': '#687787'}
    gamma = np.array([0., .25, .5, .75, 1.])
    for ax, (cohort, sweep, budget, title) in zip(axes.flat, panels):
        for prefix, label in [('frequency', 'Frequency-weighted'), ('tilted', 'Tilted, lambda=0.3'), ('marginal', 'Marginal similarity')]:
            name = f'{prefix}_B{budget}' + ('_L0.3' if prefix == 'tilted' else '')
            rows = [by_curve[name, cohort, sweep, g] for g in gamma]
            values = np.array([float(r['sharing_penalty']) for r in rows])
            ax.plot(gamma, values, marker='o', markersize=4, color=colors[prefix], linewidth=1.9, label=label)
            if prefix != 'marginal':
                ax.fill_between(gamma, [float(r['delta_ci_low']) for r in rows], [float(r['delta_ci_high']) for r in rows], color=colors[prefix], alpha=.12)
        ax.axhline(0, color='#888888', linewidth=.8, linestyle='--')
        ax.set_title(title, loc='left', fontsize=10.5)
        ax.set_xlabel('Context flattening gamma (0 = train mix; 1 = uniform)')
        ax.set_ylabel('Excess log loss vs individual users')
        ax.set_xticks(gamma)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='best', frameon=False)
    fig.suptitle('Count-level E0: the primary go/no-go gate fails', fontsize=16, fontweight='bold', x=.06, ha='left')
    fig.text(.06, .025, 'Lower excess loss is better. Shading: pointwise 95% user-bootstrap intervals. Fixed empirical context reweighting; exploratory test period.\nSame bytes at each B; one behavioral initialization; capped heuristic. Qualified contexts chosen from training only: tabs 0, 1, 2, 4, 8.', fontsize=9)
    fig.tight_layout(rect=[.02, .07, .99, .94])
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(figures / ('context_mix_sweeps.' + extension), dpi=180, facecolor='white')
    plt.close(fig)
    gate = results['primary_gate']
    lines = [
        '# Count-level E0: executed go/no-go experiment', '',
        '**Decision: NO-GO for escalating the current KuaiRand count-sharing setup into a full experiment campaign.**', '',
        'The fixed primary gate fails: frequency-weighted sharing damage decreases under interface flattening, and the tilted objective has no established advantage in the growth contrast. This is an exploratory diagnostic, not a universal rejection of context-aware compression.', '',
        '## What was executed', '',
        '- Real, previously audited KuaiRand-Pure data; existing duplicate quarantine, UTC chronology and native long_view label unchanged.',
        '- Training: 1,051,988 interactions; validation: 59,072; standard evaluation: 278,684. Random-exposure outcomes were not used in this E0 evaluation.',
        '- 27 grouped count predictors: budgets 64/256/1024 times random (3 seeds), marginal similarity, frequency-weighted behavioral grouping, and tilted grouping at lambda 0.1/0.3/1/3.',
        '- Three references: individual-user/context counts, video/context counts, and global rate. The individual reference exactly reproduces the previous probe predictions.',
        '- All-user cohort: 25,748 evaluation users, 278,684 events. Supported cohort: 4,211 observed evaluation users, 69,792 events, selected by >=10 train events in >=2 interfaces; 4,229 training users qualify overall.',
        '- Four context sweeps, five gamma values each, both cohorts; 1,200 saved model/cohort/sweep rows. Per-user/context loss sums and counts are saved so aggregation is auditable.',
        '- No neural training, GraphHash/CCE/BACO execution, Kaggle operation, NVIDIA inference, new download, package installation, or external write in this spike.', '',
        '## Frozen mathematical contract', '',
        'For each train user and interface, estimate r_us=(positive_us+10*p_context_s)/(n_us+10). Replace r_us by p_context_s if n_us<10 for the grouping profile. This equals shrinking the residual logit to zero; it is not deconfounded preference estimation.', '',
        'Let m_gs be the arithmetic mean of r_us within group g, and c_gs the mean KL(Bern(r_us)||Bern(m_gs)) across its users. The frequency objective is sum_g |g|/N * sum_s p_train(s)c_gs. The tilted objective is sum_g |g|/N * lambda*log(sum_s p_train(s)*exp(c_gs/lambda)). Users, not training events, receive equal weight in this grouping objective.', '',
        'Frequency and tilted variants start from the identical seed-11 balanced random partition, use the same nonempty-group/maximum-size constraint, and use the same capped generalized Lloyd heuristic. Every accepted step reduces its exact recorded objective. Iterations are capped at 12; these are not globally optimal solutions and several fits hit the cap.', '',
        'After grouping, aggregate the actual train positive/negative counts by group and interface and smooth with alpha=10. Predict sigmoid(logit(video/interface probability)+logit(group/interface probability)-logit(interface probability)). All grouped and individual effects have fixed weight 1. No grouping hyperparameter was selected on this E0 test.', '',
        'The group table is fitted from actual counts, including low-support cells. The support threshold affects grouping features, not which labels are discarded. Unseen users receive zero personal residual.', '',
        'All grouped methods at a given B store the same dense B*15 float32 probabilities and the same fixed-length int16 mapping; common item/context/global arrays are identical. Bytes refer to allocated numerical inference arrays, excluding archive/container overhead and training buffers. Reference byte counts include the common bundle even when a control does not use every member.', '',
        '| Groups B | Grouped predictor array bytes |', '|---|---:|',
    ]
    for b, size in verification['equal_bytes_at_each_budget'].items():
        lines.append(f'| {b} | {size:,} |')
    lines.extend([
        '', '## Primary observed result', '',
        'The primary comparison was fixed before fitting: B=256, lambda=0.3, all users, all-context flattening. Define Delta(gamma)=loss(grouped,q_gamma)-loss(individual,q_gamma), with q_gamma(s) proportional to p_train(s)^(1-gamma). The underlying test records within each interface remain fixed.', '',
        '| Gamma | Frequency-weighted sharing penalty | Tilted sharing penalty |', '|---:|---:|---:|',
    ])
    for g in gamma:
        f = by_curve['frequency_B256', 'all', 'all_flatten', g]
        t = by_curve['tilted_B256_L0.3', 'all', 'all_flatten', g]
        lines.append(f"| {g:g} | {float(f['sharing_penalty']):.6f} | {float(t['sharing_penalty']):.6f} |")
    def ci(value):
        return '[' + ', '.join(f'{v:.6f}' for v in value) + ']'
    lines.extend([
        '', f"Frequency penalty change, gamma 0 to 1: **{gate['frequency_penalty_growth']:.6f}**, pointwise user-bootstrap 95% interval {ci(gate['frequency_growth_95_ci'])}. The expected increase is absent in this primary setup; the estimated change is negative.", '',
        f"Frequency-minus-tilted penalty growth: **{gate['growth_advantage']:.6f}**, interval {ci(gate['growth_advantage_95_ci'])}. This interval includes zero, so a tilted robustness advantage is not established.", '',
        f"Tilted-minus-frequency loss at gamma=0: **{gate['tilted_minus_frequency_at_gamma0']:.6f}**, interval {ci(gate['in_mix_difference_95_ci'])}.", '',
        'The three fixed primary conditions use Bonferroni-adjusted one-sided bootstrap bounds at family alpha .05. Ordinary-mixture noninferiority uses an analyst-chosen margin of .001 log loss; failing to reject harm is not treated as proof of equivalence.', '',
        '| Primary condition | Outcome |', '|---|---|',
        '| Frequency sharing penalty grows | FAIL |',
        '| Tilted objective reduces that growth | FAIL |',
        '| Tilted ordinary-mixture loss is within the .001 margin | PASS |',
        '| Combined gate | NO-GO |', '',
        'These intervals condition on the models, users and observed period. They are not simultaneous bands over gamma or a temporal-generalization guarantee.', '',
        '## Narrower results and counterevidence', '',
        '- At B=64 on supported users and the five train-qualified interfaces, the frequency penalty increases by 0.005973; its pointwise 95% interval is [0.000414, 0.012077]. This is a narrower exploratory signal, not a confirmation selected independently of the grid.',
        '- In that same comparison, the lambda=.3 tilted penalty grows by 0.006192, slightly more. Its growth advantage is -0.000219 with interval [-0.003296, 0.002485]. The proposed remedy is not established there.',
        f"- Across {len(gates)} budget/lambda/cohort/sweep diagnostics, none passes all three conditions. This does not validate a multiplicity-free grid search or establish that all alternative methods fail.",
        '- Supported-user all-15-interface intervals retain only 545/1000 bootstrap resamples because some rare interfaces disappear when users are resampled. They are conditional-on-presence summaries and are not dependable nominal-coverage intervals. The qualified-context analyses retain all 1000 resamples.', '',
        '## Observed-mixture reference table', '',
        'This table uses the actual standard-test mix. Gamma=0 in the sweep uses the training mix; those losses differ by definition.', '',
        '| Predictor | Observed standard-test log loss |', '|---|---:|',
    ])
    for name in ['ungrouped', 'item_only', 'marginal_B256', 'frequency_B256', 'tilted_B256_L0.3']:
        r = next(x for x in raw if x['model'] == name and x['cohort'] == 'all')
        lines.append(f"| {name} | {float(r['log_loss']):.6f} |")
    lines.extend(['', '## Requested interface frequencies', '',
        'Percentages below use each cleaned split as the denominator. Tab IDs are retained; no unsupported semantic interface names are invented.', '',
        '| Tab | Train rows | Train % | Standard test % | Random test % | Users with >=10 train events in tab | Qualified |',
        '|---:|---:|---:|---:|---:|---:|---|'])
    for c in contexts:
        lines.append(f"| {c['tab']} | {int(c['train_rows']):,} | {100*float(c['train_fraction']):.4f} | {100*float(c['test_standard_fraction']):.4f} | {100*float(c['test_random_fraction']):.4f} | {c['train_users_with_at_least_10']} | {c['train_qualified_context']} |")
    lines.extend([
        '', 'The literal rarest target is tab 14: only 10 train events and 20 standard-test events, with no train user reaching the per-context threshold. It supplies no supported per-user grouping-profile variation at the chosen threshold. The secondary rarest qualified target is tab 8.', '',
        '## Corrections to the supplied interpretation', '',
        '- Differences between separately fitted count models do not form a decomposition of intrinsic user information. The proposed 25% contextual-signal calculation is not established.',
        '- Near-equality between one random-bucket predictor and item-only prediction does not identify the budgets at which behavioral grouping must work.',
        '- Shrinking low-support profiles to zero reduces noisy differences but can also erase real differences. It does not guarantee that false disagreement disappears.',
        '- The random-exposure reversal remains unexplained. Calibration and item exposure changes are plausible mechanisms; moving the result to an appendix does not establish its cause. The optional recalibration check was not run, and even a successful recalibration would not uniquely identify the cause.',
        '- The exponential tilted objective is only a candidate here. This run establishes neither algorithmic novelty nor superiority to closest published embedding methods.', '',
        '## What the decision permits', '',
        'Stop escalation to a three-day neural campaign under the current broad thesis. A future experiment needs a justified supported-context mechanism, a grouping objective aligned to that estimand, optimizer/convergence evidence, and independent confirmation data. Neither retuning on this test nor changing the headline to an attractive secondary slice makes the present result confirmatory.', '',
        'The narrow supported-user B=64 signal may motivate a bounded follow-up, but the tilted method does not currently close that gap. Switching datasets would be a new test, not a demonstrated repair. No new dataset or neural campaign was run.', '',
        '## Verification, runtime and reproducibility', '',
        f"- Independent verification: {verification['status']}; all 30 saved predictors replayed; 60 raw metric rows and 1,200 sweep rows/intervals checked; 300 bootstrap context vectors independently replayed.",
        '- The individual reference matches the previous probe predictions exactly; grouped count tables and group capacities were independently reconstructed from train records.',
        '- The original label/documentation mismatch remains unresolved. Artifact verification does not turn the labels into duration-rule-validated ground truth.',
        f"- Successful fitting: {results['fit_seconds']:.2f} CPU seconds; evaluation/bootstrap: {results['evaluation_seconds']:.2f}; independent verification: {verification['elapsed_seconds']:.2f}. These exclude coding, inspection, plotting and earlier setup.",
        '- Existing NumPy and Matplotlib used; no installs or upgrades. Sonatype Guide connector is unavailable, so its component check could not be completed; no dependency-safety claim is made.',
        '- Reproduction: run `python -m unittest -v test_e0_math`; `python verify_e0.py` verifies the existing run. Fresh preparation/fitting/evaluation requires a new output directory because scripts refuse to overwrite frozen receipts.', '',
        'Artifacts: protocol.json, fit_manifest.json, results.json, gates.json, context_frequencies.csv, raw_metrics.csv, per_context_metrics.csv, sweep_metrics.csv, predictions_standard.npz, per_user_context_loss_sums.npz, bootstrap_context_deltas.npz, verification.json, models/, figures/.', '',
        '![Executed sweep plots](figures/context_mix_sweeps.png)', '',
    ])
    (ROOT / 'E0_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'dependency_receipt.json').write_text(json.dumps({'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'installed_or_upgraded': [], 'sonatype_connector_available': False, 'sonatype_check_completed': False}, indent=2), encoding='utf-8')
    artifacts = {}
    for file in sorted(ROOT.rglob('*')):
        if file.is_file() and '__pycache__' not in str(file) and file.name != 'artifact_manifest.json':
            artifacts[str(file.relative_to(ROOT))] = {'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    (ROOT / 'artifact_manifest.json').write_text(json.dumps({'files': artifacts, 'total_bytes': sum(v['bytes'] for v in artifacts.values())}, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(ROOT / 'E0_REPORT.md'), 'figure': str(figures / 'context_mix_sweeps.png'), 'artifact_bytes': sum(v['bytes'] for v in artifacts.values()), 'gate': 'NO-GO'}))


if __name__ == '__main__':
    main()

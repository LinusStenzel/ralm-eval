import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

def compare_metrics(evaluation_control, evaluation_experiment):
    log_perplexity = {
        'control_correct': np.log(evaluation_control['perplexity_correct'].values),
        'experiment_correct': np.log(evaluation_experiment['perplexity_correct'].values),
        'control_incorrect': np.log(evaluation_control['perplexity_incorrect'].values),
        'experiment_incorrect': np.log(evaluation_experiment['perplexity_incorrect'].values),
    }

    f1_score = {
        'control_correct': evaluation_control['f1_correct'].values,
        'experiment_correct': evaluation_experiment['f1_correct'].values,
        'control_incorrect': evaluation_control['f1_incorrect'].values,
        'experiment_incorrect': evaluation_experiment['f1_incorrect'].values,
    }

    similarity = {
        'control_correct': evaluation_control['similarity_correct'].values,
        'experiment_correct': evaluation_experiment['similarity_correct'].values,
        'control_incorrect': evaluation_control['similarity_incorrect'].values,
        'experiment_incorrect': evaluation_experiment['similarity_incorrect'].values,
    }

    metrics = [log_perplexity, f1_score, similarity]
    names = ["Perplexity", "F1 Score", "Similarity"]
    fig, axss = plt.subplots(2, 3, figsize=(20, 8), sharey='row', sharex='col')
    axss = axss.T
    fig.suptitle('Comparison of Control and Experiment RALM Across Different Metrics', fontsize=16)
    
    metrics_comparison = {}
    for metric, name, axs in zip(metrics, names, axss):
        if name == "Perplexity":
            all_data = np.concatenate(list(log_perplexity.values()))
            bins = np.histogram_bin_edges(all_data, bins='auto')
        else:
            bins = np.linspace(0, 1, 21)

        comparison_correct = plot_metrics(axs[0], metric['control_correct'], metric['experiment_correct'], 'Correct Answers', name, bins)
        comparison_incorrect = plot_metrics(axs[1], metric['control_incorrect'], metric['experiment_incorrect'], 'Incorrect Answers', name, bins)
        metrics_comparison[name] =  {'correct_answers': comparison_correct, 'incorrect_answers': comparison_incorrect}

    plt.tight_layout()
    return metrics_comparison, fig


def plot_metrics(ax, control, experiment, label, name, bins):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = np.min(np.diff(bin_centers)) * 0.8

    data_pairs = [('Control', control, 'blue'), ('Experiment', experiment, 'red')]

    x_grid = np.linspace(bins[0], bins[-1], 100)

    densities = {}
    means = {}
    for group, data, color in data_pairs:
        counts, _ = np.histogram(data, bins=bins, density=True)
        counts /= np.sum(counts) * (len(x_grid) / len(counts))
        density = gaussian_kde(data)(x_grid) / np.sum(gaussian_kde(data)(x_grid))
        densities[group] = density
        means[group] = np.mean(data).item()

        ax.bar(bin_centers, counts, width=bar_width, align='center', alpha=0.1, color=color)
        ax.plot(x_grid, density, color=color, alpha=1, label=f"{group} KDE")
        ax.fill_between(x_grid, density, alpha=0.2, color=color)
        ax.axvline(np.mean(data), color=color, linestyle='dashed', linewidth=1)

    _, p_value = ks_2samp(densities['Control'], densities['Experiment'])
    control_cdf, experiment_cdf = np.cumsum(densities['Control']), np.cumsum(densities['Experiment'])
    area_between_cdfs = np.trapz(control_cdf - experiment_cdf, x_grid).item()
    distance_text = f"Area Between CDFs: {area_between_cdfs:.3f}\nK-S test p-value: {p_value:.3f}\nMean Control: {means['Control']:.3f}\nMean Experiment: {means['Experiment']:.3f}"
    ax.text(0.98, 0.7, distance_text, transform=ax.transAxes, fontsize=10,
            va='center', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'{label} - {name}')
    ax.set_xlabel('Score' if name != "Perplexity" else 'Log Perplexity')
    ax.set_ylabel('Prob. Density')
    ax.legend()
    return {'mean_control': round(means['Control'],3), 'mean_experiment': round(means['Experiment'],3), 'area_between_cdfs': round(area_between_cdfs,3), 'p_value': round(p_value,3)}
"""
Analysis module for plotting and analysis functions of training results
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def moving_average(data, window=100):
    """
    Calculates moving average of a time series
    
    Args:
        data: List/array of values
        window: Window size for moving average
    
    Returns:
        Numpy array with moving average
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_scores(scores, title="Training Progress", window=100):
    """
    Plots training results with moving average
    
    Args:
        scores: List of scores per episode
        title: Graph title
        window: Window size for moving average
    """
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score per Episode', alpha=0.6)
    
    if len(scores) >= window:
        moving_avg = moving_average(scores, window)
        plt.plot(range(window - 1, len(scores)), moving_avg, 
                label=f'{window}-Episode Moving Avg', linewidth=2, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_carracing_scores(scores, window=100):
    """Plotta i risultati del training CarRacing"""
    plot_training_scores(scores, "CarRacing Training Progress", window)


def plot_comparison_scores(scores_dict, window=100, title="Training Comparison"):
    """
    Plots comparison between different algorithms/configurations
    
    Args:
        scores_dict: Dict with algorithm_name -> scores_list
        window: Window size for moving average
        title: Graph title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Reward per episode
    for i, (name, scores) in enumerate(scores_dict.items()):
        color = colors[i % len(colors)]
        ax1.plot(scores, alpha=0.6, label=name, color=color)
        
        # Moving average
        if len(scores) >= window:
            ma = moving_average(scores, window)
            ax1.plot(range(window - 1, len(scores)), ma, 
                    label=f'MA {name}', color=color, linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Final scores histogram (last 100 episodes)
    for i, (name, scores) in enumerate(scores_dict.items()):
        final_scores = scores[-100:] if len(scores) >= 100 else scores
        color = colors[i % len(colors)]
        ax2.hist(final_scores, alpha=0.5, bins=20, label=name, color=color)
    
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Final Scores Distribution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def episodes_to_target(scores, target=180):
    """
    Finds number of episodes needed to reach a target score
    
    Args:
        scores: List of scores
        target: Target score to reach
    
    Returns:
        Number of episodes to reach target
    """
    for i, score in enumerate(scores):
        if score >= target:
            return i + 1
    return len(scores)


def analyze_training_results(scores_dict, target_score=180, window=50):
    """
    Analizza e confronta i risultati di training di diversi algoritmi
    
    Args:
        scores_dict: Dict con nome_algoritmo -> lista_scores
        target_score: Score target per analisi convergenza
        window: Finestra per calcolo stabilità
    
    Returns:
        Dict con statistiche comparative
    """
    results = {}
    
    print("=== RESULTS ANALYSIS ===")
    
    for name, scores in scores_dict.items():
        # Basic statistics
        final_score = scores[-1] if scores else 0
        final_scores = scores[-100:] if len(scores) >= 100 else scores
        mean_final = np.mean(final_scores)
        std_final = np.std(final_scores)
        
        # Convergence
        episodes_target = episodes_to_target(scores, target_score)
        
        # Stability (std of last episodes)
        stability_window = min(window, len(scores))
        stability = np.std(scores[-stability_window:]) if stability_window > 0 else 0
        
        results[name] = {
            'final_score': final_score,
            'mean_final_100': mean_final,
            'std_final_100': std_final,
            'episodes_to_target': episodes_target,
            'stability': stability,
            'total_episodes': len(scores)
        }
        
        print(f"\n{name}:")
        print(f"  Score finale: {final_score:.2f}")
        print(f"  Media ultimi 100: {mean_final:.2f} ± {std_final:.2f}")
        print(f"  Episodi per target {target_score}: {episodes_target}")
        print(f"  Stabilità (std ultimi {stability_window}): {stability:.2f}")
    
    return results


def plot_cartpole_comparison(scores_no_baseline, scores_with_baseline):
    """
    Funzione specifica per confrontare REINFORCE con/senza baseline su CartPole
    """
    scores_dict = {
        'Senza Baseline': scores_no_baseline,
        'Con Baseline': scores_with_baseline
    }
    
    plot_comparison_scores(scores_dict, title='REINFORCE Comparison: With vs Without Baseline')
    
    # Detailed analysis
    results = analyze_training_results(scores_dict, target_score=180)
    
    print("\n=== COMPARATIVE STATISTICS ===")
    for name, stats in results.items():
        print(f"{name}: {stats['mean_final_100']:.2f} ± {stats['std_final_100']:.2f}")
    
    return results

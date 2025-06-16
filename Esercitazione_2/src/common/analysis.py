"""
Modulo di analisi per funzioni di plotting e analisi dei risultati di training
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def moving_average(data, window=100):
    """
    Calcola la media mobile di una serie temporale
    
    Args:
        data: Lista/array di valori
        window: Dimensione finestra per media mobile
    
    Returns:
        Array numpy con media mobile
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_scores(scores, title="Training Progress", window=100):
    """
    Plotta i risultati del training con media mobile
    
    Args:
        scores: Lista dei punteggi per episodio
        title: Titolo del grafico
        window: Dimensione finestra per media mobile
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
    Plotta confronto tra diversi algoritmi/configurazioni
    
    Args:
        scores_dict: Dict con nome_algoritmo -> lista_scores
        window: Dimensione finestra per media mobile
        title: Titolo del grafico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Reward per episodio
    for i, (name, scores) in enumerate(scores_dict.items()):
        color = colors[i % len(colors)]
        ax1.plot(scores, alpha=0.6, label=name, color=color)
        
        # Media mobile
        if len(scores) >= window:
            ma = moving_average(scores, window)
            ax1.plot(range(window - 1, len(scores)), ma, 
                    label=f'MA {name}', color=color, linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Istogramma degli score finali (ultimi 100 episodi)
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
    Trova il numero di episodi necessari per raggiungere un target score
    
    Args:
        scores: Lista dei punteggi
        target: Score target da raggiungere
    
    Returns:
        Numero di episodi per raggiungere il target
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
    
    print("=== ANALISI DEI RISULTATI ===")
    
    for name, scores in scores_dict.items():
        # Statistiche di base
        final_score = scores[-1] if scores else 0
        final_scores = scores[-100:] if len(scores) >= 100 else scores
        mean_final = np.mean(final_scores)
        std_final = np.std(final_scores)
        
        # Convergenza
        episodes_target = episodes_to_target(scores, target_score)
        
        # Stabilità (std degli ultimi episodi)
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
    
    plot_comparison_scores(scores_dict, title='Confronto REINFORCE: Con vs Senza Baseline')
    
    # Analisi dettagliata
    results = analyze_training_results(scores_dict, target_score=180)
    
    print("\n=== STATISTICHE COMPARATIVE ===")
    for name, stats in results.items():
        print(f"{name}: {stats['mean_final_100']:.2f} ± {stats['std_final_100']:.2f}")
    
    return results

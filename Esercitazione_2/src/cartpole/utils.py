"""
Funzioni di utilità per valutazione e registrazione video
"""
import numpy as np
import imageio
from IPython.display import Video, display


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Valuta l'agente per `n_eval_episodes` episodi e restituisce media e deviazione standard dei reward.

    :param env: Environment di valutazione (gym.Env)
    :param max_steps: Numero massimo di step per episodio
    :param n_eval_episodes: Numero di episodi per la valutazione
    :param policy: Policy REINFORCE con metodo `act(state)`
    :return: Tupla di (mean_reward, std_reward)
    """
    episode_rewards = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset()  # Gymnasium reset restituisce (obs, info)
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward
            done = terminated or truncated

            if done:
                break
            state = next_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, policy, out_path, fps=30):
    """
    Registra un video dell'agente che agisce nell'environment.
    
    :param env: environment con render_mode='rgb_array'
    :param policy: agente, deve implementare policy.act(state)
    :param out_path: percorso completo per output .mp4 o .gif
    :param fps: frames per secondo
    """
    images = []
    state, _ = env.reset()
    done = False

    img = env.render()
    images.append(img)

    while not done:
        action, _ = policy.act(state)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        img = env.render()
        images.append(img)
        state = obs

    imageio.mimsave(out_path, [np.array(img) for img in images], fps=fps)
    print(f"Video salvato in {out_path}")
    display(Video(out_path, embed=True))

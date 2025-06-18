"""
Utilities per CarRacing simili a quelle del cartpole
"""
import cv2
import numpy as np
import torch
import gymnasium as gym
from IPython.display import Video, display


def record_carracing_video(agent, env, filename="carracing_video.mp4", max_steps=500, fps=30):
    """
    Registra un video dell'agente CarRacing in azione
    Versione semplificata che evita problemi di sincronizzazione
    
    Args:
        agent: L'agente addestrato
        env: L'environment CarRacing personalizzato
        filename: Nome del file video
        max_steps: Numero massimo di step
        fps: Frame per secondo
    """
    print(f"🎬 Registrazione video semplificata per {max_steps} step...")
    
    try:
        # Usa l'environment personalizzato per la logica
        frames = []
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]
        
        # Crea environment standard solo per il rendering
        video_env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
        video_env.reset()
        
        for step in range(max_steps):
            # Seleziona azione dall'agente
            with torch.no_grad():
                action, _ = agent.select_action(state)
            
            # Execute step in your custom environment
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            # Simulate action in standard environment for frame
            try:
                video_obs, _, _, _, _ = video_env.step(action)
                frame = video_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                if step < 5:  # Log solo i primi errori
                    print(f"Warning frame {step}: {e}")
            
            state = next_state
            action_history = [action_history[-1], action]
            
            if done:
                print(f"Episodio terminato al step {step}: {death_reason}")
                break
        
        video_env.close()
        
        # Salva il video se abbiamo frames
        if len(frames) > 10:  # Almeno 10 frames per un video significativo
            save_video_from_frames(frames, filename, fps)
            print(f"✅ Video salvato: {filename} ({len(frames)} frames)")
            return True
        else:
            print(f"❌ Troppo pochi frames catturati: {len(frames)}")
            return False
            
    except Exception as e:
        print(f"❌ Errore nella registrazione video: {e}")
        return False


def record_carracing_video_simple(agent, env, filename="carracing_simple.mp4", max_steps=200, fps=30):
    """
    Versione ultra-semplificata che registra solo usando l'environment standard
    """
    print(f"🎬 Registrazione video ultra-semplificata...")
    
    try:
        # Usa solo environment standard
        video_env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
        
        frames = []
        obs, _ = video_env.reset()
        
        for step in range(max_steps):
            # Azione casuale per semplicità (puoi sostituire con l'agente se necessario)
            action = video_env.action_space.sample()
            
            # Render frame
            frame = video_env.render()
            if frame is not None:
                frames.append(frame)
            
            # Step
            obs, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            
            if done:
                print(f"Episodio standard terminato al step {step}")
                break
        
        video_env.close()
        
        if frames:
            save_video_from_frames(frames, filename, fps)
            print(f"✅ Video semplice salvato: {filename}")
            return True
        else:
            print("❌ Nessun frame catturato")
            return False
            
    except Exception as e:
        print(f"❌ Errore video semplice: {e}")
        return False


def save_video_from_frames(frames, filename, fps=30):
    """Saves a list of frames as MP4 video"""
    if not frames:
        return
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # OpenCV usa BGR invece di RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def evaluate_carracing_agent(agent, env, n_eval_episodes=5, max_steps=500):
    """
    Valuta le performance dell'agente CarRacing
    
    Args:
        agent: L'agente da valutare
        env: L'environment CarRacing
        n_eval_episodes: Numero di episodi di valutazione
        max_steps: Step massimi per episodio
    
    Returns:
        Tupla (media_score, std_score, episodi_completati)
    """
    episode_scores = []
    completed_episodes = 0
    
    print(f"Valutazione agente per {n_eval_episodes} episodi...")
    
    for episode in range(n_eval_episodes):
        score = 0
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]
        
        for step in range(max_steps):
            with torch.no_grad():
                action, _ = agent.select_action(state)
            
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            score += reward
            state = next_state
            action_history = [action_history[-1], action]
            
            if done:
                if step > 100:  # Considera completato se dura almeno 100 step
                    completed_episodes += 1
                print(f"Episodio {episode + 1}: Score={score:.2f}, Step={step}, Reason={death_reason}")
                break
        
        episode_scores.append(score)
    
    mean_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    
    print(f"\n📊 Risultati valutazione:")
    print(f"Score medio: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Episodi completati: {completed_episodes}/{n_eval_episodes}")
    
    return mean_score, std_score, completed_episodes


def display_video(filename):
    """Mostra il video nel notebook"""
    try:
        display(Video(filename, width=800, height=600))
    except Exception as e:
        print(f"Errore nella visualizzazione del video: {e}")

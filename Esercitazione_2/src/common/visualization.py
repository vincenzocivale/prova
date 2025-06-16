"""
Modulo per visualizzazione di video e display
"""
import os
from IPython.display import Video, display


def display_video(filename):
    """
    Mostra il video nel notebook
    
    Args:
        filename: Percorso al file video
    """
    if os.path.exists(filename):
        try:
            display(Video(filename, width=800, height=600))
            print(f"📹 Video visualizzato: {filename}")
        except Exception as e:
            print(f"Errore nella visualizzazione di {filename}: {e}")
    else:
        print(f"❌ File non trovato: {filename}")


def find_and_display_video(possible_paths):
    """
    Cerca e visualizza il primo video trovato tra una lista di percorsi
    
    Args:
        possible_paths: Lista di percorsi possibili per il video
    
    Returns:
        True se un video è stato trovato e visualizzato, False altrimenti
    """
    for video_path in possible_paths:
        if os.path.exists(video_path):
            print(f"📹 Video trovato: {video_path}")
            display_video(video_path)
            return True
    
    print("❌ Nessun video trovato tra i percorsi:")
    for path in possible_paths:
        print(f"  - {path}")
    return False


def display_carracing_video():
    """
    Cerca e visualizza il video CarRacing nei percorsi standard
    """
    video_files = [
        "record/carracing_agent_test.mp4",
        "carracing_agent_test.mp4", 
        "record/carracing_fallback.mp4",
        "record/carracing_simple.mp4"
    ]
    
    print("🎬 Ricerca video dell'agente CarRacing addestrato:")
    return find_and_display_video(video_files)


def display_cartpole_video():
    """
    Cerca e visualizza il video CartPole
    """
    video_files = [
        "record/Cart_Pole.mp4",
        "Cart_Pole.mp4"
    ]
    
    print("🎬 Ricerca video dell'agente CartPole:")
    return find_and_display_video(video_files)

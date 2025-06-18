"""
Module for video visualization and display
"""
import os
from IPython.display import Video, display


def display_video(filename):
    """
    Shows video in notebook
    
    Args:
        filename: Path to video file
    """
    if os.path.exists(filename):
        try:
            display(Video(filename, width=800, height=600))
            print(f"📹 Video displayed: {filename}")
        except Exception as e:
            print(f"Error displaying {filename}: {e}")
    else:
        print(f"❌ File not found: {filename}")


def find_and_display_video(possible_paths):
    """
    Searches for and displays the first video found among a list of paths
    
    Args:
        possible_paths: List of possible paths for video
    
    Returns:
        True if a video was found and displayed, False otherwise
    """
    for video_path in possible_paths:
        if os.path.exists(video_path):
            print(f"📹 Video found: {video_path}")
            display_video(video_path)
            return True
    
    print("❌ No video found among paths:")
    for path in possible_paths:
        print(f"  - {path}")
    return False


def display_carracing_video():
    """
    Searches for and displays CarRacing video in standard paths
    """
    video_files = [
        "record/carracing_agent_test.mp4",
        "carracing_agent_test.mp4", 
        "record/carracing_fallback.mp4",
        "record/carracing_simple.mp4"
    ]
    
    print("🎬 Searching for trained CarRacing agent video:")
    return find_and_display_video(video_files)


def display_cartpole_video():
    """
    Searches for and displays CartPole video
    """
    video_files = [
        "record/Cart_Pole.mp4",
        "Cart_Pole.mp4"
    ]
    
    print("🎬 Searching for CartPole agent video:")
    return find_and_display_video(video_files)

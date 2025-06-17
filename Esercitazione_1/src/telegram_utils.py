"""
Utilities for Telegram notifications configuration and testing.
"""
import os
from typing import Optional, Dict, Any
from .telegram_notifier import TelegramNotifier, create_telegram_notifier_from_env


def setup_telegram_from_config(
    bot_token: str,
    chat_id: str,
    save_env: bool = True
) -> TelegramNotifier:
    """
    Configure a Telegram notifier and optionally save credentials to environment variables.
    
    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID
        save_env: Whether to save credentials to current process environment variables
    
    Returns:
        Configured TelegramNotifier
    """
    if save_env:
        os.environ['TELEGRAM_BOT_TOKEN'] = bot_token
        os.environ['TELEGRAM_CHAT_ID'] = chat_id
    
    return TelegramNotifier(bot_token=bot_token, chat_id=chat_id)


def test_telegram_notification(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None
) -> bool:
    """
    Test sending a Telegram notification.
    
    Args:
        bot_token: Bot token (optional, will be taken from environment variables if not provided)
        chat_id: Chat ID (optional, will be taken from environment variables if not provided)
    
    Returns:
        True if test was successful, False otherwise
    """
    if bot_token and chat_id:
        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    else:
        notifier = create_telegram_notifier_from_env()
    
    if not notifier:
        print("❌ Unable to create Telegram notifier")
        return False
    
    # Test data
    test_metrics = {
        "accuracy": 0.9543,
        "loss": 0.1234,
        "precision": 0.9567,
        "recall": 0.9321
    }
    
    test_config = {
        "epochs": 25,
        "batch_size": 64,
        "learning_rate": 0.001,
        "model": "TestCNN"
    }
    
    success = notifier.send_training_completion_sync(
        model_name="TestModel",
        training_duration=1847,  # ~30 minutes
        final_metrics=test_metrics,
        config_summary=test_config,
        additional_info="🧪 This is a test message to verify Telegram notifications."
    )
    
    if success:
        print("✅ Test Telegram notification sent successfully!")
    else:
        print("❌ Test Telegram notification failed")
    
    return success


def get_telegram_setup_instructions() -> str:
    """
    Return instructions for setting up a Telegram bot.
    
    Returns:
        String with setup instructions
    """
    instructions = """
🤖 TELEGRAM BOT SETUP - Instructions

1. 📱 Open Telegram and search for @BotFather
2. 🆕 Send the command /newbot
3. 📝 Choose a name for your bot (e.g. "ML Training Notifier")
4. 🔖 Choose a username ending with "bot" (e.g. "ml_training_bot")
5. 🎉 You'll receive the bot TOKEN (save it!)

6. 💬 To get the CHAT_ID:
   - Start your bot by clicking the link provided by BotFather
   - Send any message to the bot
   - Go to: https://api.telegram.org/bot<TOKEN>/getUpdates
   - Find "chat":{"id": NUMBER} in the JSON response
   - This NUMBER is your CHAT_ID

7. 🔧 Configuration in code:

   Option A - Environment variables (recommended):
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   export TELEGRAM_CHAT_ID="your_chat_id_here"

   Option B - In TrainingConfig:
   config = TrainingConfig(
       use_telegram_notifications=True,
       telegram_bot_token="your_bot_token_here",
       telegram_chat_id="your_chat_id_here",
       telegram_additional_info="Training model XYZ"
   )

8. 🧪 Test:
   python -c "from src.telegram_utils import test_telegram_notification; test_telegram_notification()"

⚠️  SECURITY: Never share your bot token and keep it secret!
"""
    return instructions


def print_telegram_setup_instructions():
    """Print instructions for setting up Telegram."""
    print(get_telegram_setup_instructions())


if __name__ == "__main__":
    print_telegram_setup_instructions()
    print("\n" + "="*50)
    print("🧪 Want to test Telegram connection? (y/n): ", end="")
    
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', 'si', 's']:
            test_telegram_notification()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")

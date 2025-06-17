"""
Module for sending Telegram notifications upon training completion.
"""
import asyncio
import os
from typing import Dict, Optional, Any
from datetime import datetime
import traceback
import threading
import time

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramNotifier:
    """
    Class for sending training completion notifications via Telegram Bot.
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID where to send messages
            enabled: Whether to enable notifications
        """
        self.enabled = enabled and TELEGRAM_AVAILABLE
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        
        if self.enabled:
            try:
                self.bot = Bot(token=bot_token)
            except Exception as e:
                print(f"⚠️  Error initializing Telegram bot: {e}")
                self.enabled = False
        elif not TELEGRAM_AVAILABLE:
            print("⚠️  python-telegram-bot not installed. Telegram notifications disabled.")
    
    async def send_training_completion_notification(
        self, 
        model_name: str,
        training_duration: float,
        final_metrics: Dict[str, float],
        config_summary: Dict[str, Any],
        additional_info: Optional[str] = None
    ) -> bool:
        """
        Send a training completion notification.
        
        Args:
            model_name: Name of the trained model
            training_duration: Training duration in seconds
            final_metrics: Final test metrics
            config_summary: Configuration summary
            additional_info: Optional additional information
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            message = self._format_training_message(
                model_name, training_duration, final_metrics, config_summary, additional_info
            )
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            print("✅ Telegram notification sent successfully!")
            return True
            
        except TelegramError as e:
            print(f"❌ Error sending Telegram notification: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending notification: {e}")
            return False
    
    def _format_training_message(
        self,
        model_name: str,
        training_duration: float,
        final_metrics: Dict[str, float],
        config_summary: Dict[str, Any],
        additional_info: Optional[str] = None
    ) -> str:
        """
        Format the training completion message.
        """
        # Convert duration to readable format
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        seconds = int(training_duration % 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        # Format metrics
        metrics_str = "\n".join([f"• *{k.capitalize()}*: {v:.4f}" for k, v in final_metrics.items()])
        
        # Format config summary
        config_str = "\n".join([f"• *{k}*: `{v}`" for k, v in config_summary.items()])
        
        message = f"""🎉 *Training Completed!*

🤖 *Model*: `{model_name}`
⏱️ *Duration*: {duration_str}
📅 *Completed*: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

📊 *Final Metrics*:
{metrics_str}

⚙️ *Configuration*:
{config_str}"""

        if additional_info:
            message += f"\n\n📝 *Additional Info*:\n{additional_info}"
            
        return message
    
    
    def send_training_completion_sync(
        self,
        model_name: str,
        training_duration: float,
        final_metrics: Dict[str, float],
        config_summary: Dict[str, Any],
        additional_info: Optional[str] = None
    ) -> bool:
        """
        Synchronous version to send notifications (works in Jupyter notebooks).
        Uses threading to avoid event loop conflicts.
        """
        if not self.enabled:
            return False
            
        def _send_in_thread():
            """Send notification in a separate thread with its own event loop."""
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        self.send_training_completion_notification(
                            model_name, training_duration, final_metrics, config_summary, additional_info
                        )
                    )
                    return result
                finally:
                    loop.close()
            except Exception as e:
                print(f"❌ Error in threaded notification sending: {e}")
                return False
        
        try:
            # Use threading to avoid event loop conflicts in Jupyter
            result_container = [False]
            
            def thread_target():
                result_container[0] = _send_in_thread()
            
            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if thread.is_alive():
                print("⚠️  Telegram notification timeout")
                return False
                
            return result_container[0]
            
        except Exception as e:
            print(f"❌ Error in synchronous notification sending: {e}")
            return False


def create_telegram_notifier_from_env() -> Optional[TelegramNotifier]:
    """
    Create a TelegramNotifier from environment variables.
    
    Required environment variables:
    - TELEGRAM_BOT_TOKEN: Telegram bot token
    - TELEGRAM_CHAT_ID: Chat ID
    
    Returns:
        TelegramNotifier if variables are present, None otherwise
    """
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        return TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    else:
        print("ℹ️  Environment variables TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not found.")
        print("   Telegram notifications are disabled.")
        return None


# Usage example
if __name__ == "__main__":
    # Test the notifier
    notifier = create_telegram_notifier_from_env()
    if notifier:
        test_metrics = {"accuracy": 0.9876, "loss": 0.0234}
        test_config = {"epochs": 50, "batch_size": 128, "lr": 0.001}
        
        success = notifier.send_training_completion_sync(
            model_name="TestCNN",
            training_duration=3665,  # ~1 hour
            final_metrics=test_metrics,
            config_summary=test_config,
            additional_info="Test completed successfully!"
        )
        
        if success:
            print("Test notification sent!")
        else:
            print("Test notification failed.")

import datetime

class Args:
    def __init__(self):
        # Environment
        self.seed = 42
        self.action_repeat = 4
        self.deathThreshold = 1000
        self.deathByGreeneryThreshold = 50
        self.maxDistance = 50
        self.numberOfLasers = 5
        self.valueStackSize = 4
        self.actionStack = 3
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.buffer_capacity = 1000
        self.batch_size = 32
        
        # Training
        self.start_episode = 0
        self.max_episodes = 2000
        self.log_interval = 10
        self.plot_interval = 100
        self.saveLocation = "models/"
        
        # Wandb configuration
        self.wandb_project = "car-racing-ppo"
        self.run_name = f"ppo-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Optional: Add tags based on hyperparameters
        self.wandb_tags = [
            "PPO", 
            "CarRacing", 
            f"gamma-{self.gamma}",
            f"clip-{self.clip_param}",
            f"buffer-{self.buffer_capacity}"
        ]
        
    def actionTransformation(self, action):
        """Transform normalized action [0,1] to environment action space"""
        return [
            action[0] * 2 - 1,  # steering: [0,1] -> [-1, 1]
            action[1],          # gas: [0,1] -> [0, 1]  
            action[2]           # brake: [0,1] -> [0, 1]
        ]
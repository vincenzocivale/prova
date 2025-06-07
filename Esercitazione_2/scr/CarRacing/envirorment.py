import gymnasium as gym

class CarRacingEnv:
    def __init__(self, seed=None):
        """
        Inizializza l'ambiente CarRacing.
        """
        self.env = gym.make("CarRacing-v2", continuous=True)
        if seed is not None:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

    def reset(self):
        """
        Resetta l'ambiente e restituisce l'osservazione iniziale.
        """
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        """
        Esegue un'azione nell'ambiente e restituisce la nuova osservazione,
        la ricompensa, lo stato di terminazione e troncamento.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        """
        Chiude l'ambiente.
        """
        self.env.close()

if __name__ == '__main__':
    # Esempio di utilizzo:
    env_wrapper = CarRacingEnv()
    obs = env_wrapper.reset()
    print(f"Dimensione osservazione iniziale: {obs.shape}") # (96, 96, 3)
    env_wrapper.close()
from src.CarRacing.agent import Agent
from src.CarRacing.environment import Env
from src.CarRacing.config import configure
from src.CarRacing.run import RunManager


def main():
    configs, use_cuda,  device = configure()

    agent = Agent(configs.checkpoint, configs, device)
    env = Env(configs)

    run = RunManager(
        project_name="CarRacing",
        run_name=f"run_{configs.checkpoint}",
        config=configs,
        device=device,
        model=agent,
        env=env,
        output_dir="runs",
        save_every=100
    )

    for episodeIndex in range(configs.checkpoint, 100000):
        score = 0
        prevState = env.reset()
        for t in range(10000):
            action, a_logp = agent.select_action(prevState)

            curState, reward, dead, reasonForDeath = env.step(action, t, agent)
            
            if not configs.test:
                loss, entropy = agent.update((prevState, action, a_logp, reward, curState), episodeIndex)
                # log ogni passo update
                run.log_training_step(episode=episodeIndex, step=t, reward=reward, loss=loss, entropy=entropy)
            
            score += reward
            prevState = curState

            if dead:
                print(f"Dead at score={score:.2f} || Timesteps={t} || Reason={reasonForDeath}")
                break

        # log reward totale episodio
        run.log_episode(episodeIndex, score)

        print(f"Ep {episodeIndex}\tLast score: {score:.2f}\n--------------------\n")

        run.maybe_save_checkpoint(episodeIndex)

        if configs.test:
            break

    # fine training
    if not configs.test:
        run.save_model()
        run.record_test_video()
        run.finish()

if __name__ == "__main__":
    main()
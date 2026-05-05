from gymnasium.envs.registration import register

register(
    id='Sailing-v0',
    entry_point='mcts_forest.envs.sailing:SailingEnv',
    max_episode_steps=500,
)

from gymnasium.envs.registration import register

# register(
#     id="create3_env/GridWorld-v0",
#     entry_point="create3_env.envs:GridWorldEnv",
# )

register(
    id="create3_env/CreateRedBall-v0",
    entry_point="create3_env.envs:CreateRedBall",
)

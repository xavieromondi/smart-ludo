from stable_baselines3 import PPO
from ludo_env import LudoEnv  # Make sure your import path matches your project structure

env = LudoEnv()
model = PPO.load("ludo_ai_model")

obs = env.reset()
done = False

while not done:
    env.render()
    print("Dice Roll:", "waiting...")

    if env.current_player == 0:
        # ✅ Human input with validation
        valid_action = False
        while not valid_action:
            try:
                action = int(input("Your turn! Choose a token (0–3): "))
                if 0 <= action < env.tokens_per_player and env.state[env.current_player][action] < 57:
                    valid_action = True
                else:
                    print("Invalid token. Already at goal or out of range.")
            except ValueError:
                print("Please enter a number (0–3).")
    else:
        # AI plays
        action, _ = model.predict(obs)

    obs, reward, done, info = env.step(action)
    print(f"Dice Roll: {info['dice_roll']}, Reward: {reward}")

print("\nGame Over!")
env.render()

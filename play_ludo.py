import pygame
from stable_baselines3 import PPO
from ludo_env import LudoEnv

env = LudoEnv()
model = PPO.load("ludo_ai_model")

obs = env.reset()
done = False

pygame.init()
screen = pygame.display.set_mode((240, 240))

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    env.render_pygame(screen)
    print("Dice Roll: waiting...")

    if env.current_player == 0:
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
        action, _ = model.predict(obs)

    obs, reward, done, info = env.step(action)
    print(f"Dice Roll: {info['dice_roll']}, Reward: {reward}")

pygame.quit()
print("\nGame Over!")

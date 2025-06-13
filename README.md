# 🎲 Smart Ludo — AI-Powered Ludo Game

Smart Ludo is a machine learning–powered 2D version of the classic Ludo board game. It demonstrates how reinforcement learning (specifically PPO via Stable-Baselines3) can be used to create intelligent AI opponents in a turn-based strategy game.

This project showcases my skills in Python, AI training, custom environment design using OpenAI Gym, and game visualization with Pygame.

---

## 🚀 Features

- 🤖 AI opponents trained with **Proximal Policy Optimization (PPO)**
- 🎮 Human vs AI gameplay with token selection
- 🧩 Custom **Gym-compatible** Ludo environment
- 🧠 Reward shaping, safe zones, and opponent captures
- 🖼️ Real-time 2D board rendering using **Pygame**

---

## 🧠 Tech Stack

- Python 3.x
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- OpenAI Gym
- Pygame
- NumPy

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/xavieromondi/smart-ludo.git
cd smart-ludo

# Install dependencies
pip install -r requirements.txt

# To play the trained AI model
python play_with_model.py

# Or train your own model
python train_agent.py

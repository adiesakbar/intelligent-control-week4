import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ujienvaronmentcar import DQNAgent

# Inisialisasi environment
env = gym.make("LunarLander-v2", render_mode="human")  # Tanpa RGB array
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)pip install gym[box2d]

agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

def visualize_episode():
    frames = []
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(1000):
        frame = env.render()
        frames.append(frame)  # Simpan frame untuk animasi
        
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Gabungkan status selesai
        state = np.reshape(next_state, [1, state_size])
        
        if done:
            print(f"Test Episode Score: {time}")
            break
    
    return frames

def create_animation(frames):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(frames[0])
    
    def update(frame):
        img.set_array(frame)
        return img,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    plt.show()

frames = visualize_episode()
create_animation(frames)
env.close()
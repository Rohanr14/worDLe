import gymnasium as gym
import numpy as np
from environment.wordle_env import WordleEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train(episodes, batch_size, update_target_every=100):
    env = WordleEnv("data/word_list.txt")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay()
        
        scores.append(score)
        
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, scores

def plot_scores(scores):
    plt.plot(scores)
    plt.title('DQN Training on Wordle')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

if __name__ == "__main__":
    episodes = 10000
    batch_size = 64
    
    agent, scores = train(episodes, batch_size)
    agent.save("models/dqn_wordle.pth")
    
    plot_scores(scores)
from flask import Flask, render_template, request, jsonify
from environment.wordle_env import WordleEnv
from agents.dqn_agent import DQNAgent
import numpy as np

app = Flask(__name__)

# Initialize environment and agent
env = WordleEnv("data/word_list.txt")
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load("models/dqn_wordle.pth")  # Load pre-trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    data = request.json
    action = data['action']
    
    state = np.array(data['state'])
    action = agent.act(state) if action == 'ai' else env.action_space.sample()
    
    next_state, reward, done, _, info = env.step(action)
    
    return jsonify({
        'state': next_state.tolist(),
        'reward': reward,
        'done': done,
        'word': env.word_list[action],
        'target_word': env.target_word if done else None
    })

@app.route('/reset', methods=['POST'])
def reset():
    state, _ = env.reset()
    return jsonify({
        'state': state.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
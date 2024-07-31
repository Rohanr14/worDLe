import optuna
from environment.wordle_env import WordleEnv
from agents.dqn_agent import DQNAgent
import numpy as np
import torch
import random
import gymnasium as gym

def objective(trial):
    # Define the hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.9999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    try:
        # Create environment with a smaller subset of words
        env = WordleEnv("data/word_list.txt")
        env.word_list = random.sample(env.word_list, min(1000, len(env.word_list)))  # Use at most 1000 words
        env.action_space = gym.spaces.Discrete(len(env.word_list))  # Update the action space
        
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=len(env.word_list),  # Use the new action space size
            hidden_size=hidden_size,
            learning_rate=lr,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size
        )
    
        # Reduced number of episodes for faster tuning
        episodes = 100
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
        
            # Optuna pruning: stop unpromising trials early
            if episode % 10 == 0:
                mean_score = np.mean(scores[-10:])
                trial.report(mean_score, episode)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
        return np.mean(scores)
    except Exception as e:
        print(f"An error occurred in trial {trial.number}: {str(e)}")
        return float('-inf')  # Return a very low score for failed trials

def run_optimization(n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print('\nOptimization finished.')
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    return study

if __name__ == "__main__":
    study = run_optimization()
    
    # Optionally, save the study results
    import joblib
    joblib.dump(study, 'optuna_study.pkl')
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WordleEnv(gym.Env):
    def __init__(self, word_list_path, max_attempts=6):
        super().__init__()
        
        with open(word_list_path, 'r') as f:
            self.word_list = [word.strip().lower() for word in f if len(word.strip()) == 5]
        
        if not self.word_list:
            raise ValueError("No valid 5-letter words found in the word list.")
        
        self.word_length = 5
        self.max_attempts = max_attempts
        
        self.action_space = spaces.Discrete(len(self.word_list))
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.word_length * 3 + self.max_attempts,),
            dtype=np.int8
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.target_word = self.np_random.choice(self.word_list)
        self.attempts_left = self.max_attempts
        self.current_state = np.zeros(self.observation_space.shape, dtype=np.int8)
        return self.current_state, {}
    
    def step(self, action):
        if self.attempts_left == 0:
            return self.current_state, 0, True, False, {}
        
        if action < 0 or action >= len(self.word_list):
            print(f"Invalid action: {action}. Action should be between 0 and {len(self.word_list) - 1}")
            return self.current_state, -10, False, False, {}
        
        guess = self.word_list[action]
        reward = self._calculate_reward(guess)
        self.attempts_left -= 1
        
        self._update_state(guess)
        
        done = (guess == self.target_word) or (self.attempts_left == 0)
        return self.current_state, reward, done, False, {}
    
    def _calculate_reward(self, guess):
        if guess == self.target_word:
            return 100
        
        reward = 0
        for i, letter in enumerate(guess):
            if letter == self.target_word[i]:
                reward += 3
            elif letter in self.target_word:
                reward += 1
        
        return reward - 5
    
    def _update_state(self, guess):
        state = np.zeros(self.word_length * 3, dtype=np.int8)
        for i, letter in enumerate(guess):
            if letter == self.target_word[i]:
                state[i*3 + 2] = 1  # Correct
            elif letter in self.target_word:
                state[i*3 + 1] = 1  # Present
            else:
                state[i*3] = 1  # Absent
        
        attempts_encoding = np.zeros(self.max_attempts, dtype=np.int8)
        attempts_left_index = max(0, min(self.max_attempts - 1, self.max_attempts - self.attempts_left))
        attempts_encoding[attempts_left_index] = 1
        
        self.current_state = np.concatenate([state, attempts_encoding])

    def render(self):
        # temporarily empty
        pass
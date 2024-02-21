import numpy as np


class HiddenMarkovModel:
    """
    Class for Hidden Markov Model
    """

    def __init__(
        self,
        observation_states: np.ndarray,
        hidden_states: np.ndarray,
        prior_p: np.ndarray,
        transition_p: np.ndarray,
        emission_p: np.ndarray,
    ):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states
            hidden_states (np.ndarray): hidden states
            prior_p (np.ndarray): prior probabities of hidden states
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states
        """
        if not isinstance(observation_states, np.ndarray):
            raise ValueError("Observation states must be a numpy array")
        self.observation_states = observation_states
        self.observation_states_dict = {
            state: index for index, state in enumerate(list(self.observation_states))
        }

        if not isinstance(hidden_states, np.ndarray):
            raise ValueError("Hidden states must be a numpy array")
        self.hidden_states = hidden_states
        self.hidden_states_dict = {
            index: state for index, state in enumerate(list(self.hidden_states))
        }

        if (
            not isinstance(prior_p, np.ndarray)
            and isinstance(transition_p, np.ndarray)
            and isinstance(emission_p, np.ndarray)
        ):
            raise ValueError("Probabilities must be a numpy array")
        if prior_p.shape[0] != len(hidden_states):
            raise ValueError(
                "Prior probabilities must have the same length as hidden states"
            )
        if transition_p.shape[0] != len(hidden_states) and transition_p.shape[1] != len(
            hidden_states
        ):
            raise ValueError(
                "Transition probabilities must have the same length as hidden states"
            )
        if emission_p.shape[0] != len(hidden_states) and emission_p.shape[1] != len(
            observation_states
        ):
            raise ValueError(
                "Emission probabilities must have the same length as hidden states and observation states"
            )

        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence
        """
        if len(input_observation_states) == 0:
            raise ValueError("Input observation sequence cannot be empty")
        if not isinstance(input_observation_states, np.ndarray):
            raise ValueError("Input observation sequence must be a numpy array")
        if not all(
            input_observation_states[i] in self.observation_states
            for i in range(len(input_observation_states))
        ):
            raise ValueError(
                "Input observation sequence contains invalid observation states"
            )

        # Step 1. Initialize variables
        O = len(input_observation_states)
        S = len(self.hidden_states)
        alpha = np.zeros((O, S))

        # Initialize first alphas
        for i in range(S):
            obs_index = self.observation_states_dict[input_observation_states[0]]
            alpha[0, i] = self.prior_p[i] * self.emission_p[i, obs_index]

        # Step 2. Calculate probabilities
        for t in range(1, O):
            for j in range(S):
                obs_index = self.observation_states_dict[input_observation_states[t]]
                alpha[t, j] = (
                    np.sum(alpha[t - 1, :] * self.transition_p[:, j])
                    * self.emission_p[j, obs_index]
                )

        # Step 3. Return final probability
        return np.sum(alpha[O - 1, :])

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """
        if len(decode_observation_states) == 0:
            raise ValueError("Observation sequence to decode cannot be empty")
        if not isinstance(decode_observation_states, np.ndarray):
            raise ValueError("Observation sequence to decode must be a numpy array")
        if not all(
            decode_observation_states[i] in self.observation_states
            for i in range(len(decode_observation_states))
        ):
            raise ValueError(
                "Observation sequence to decode contains invalid observation states"
            )
        # Step 1. Initialize variables
        O = len(decode_observation_states)
        S = len(self.hidden_states)

        # store probabilities of hidden state at each step
        viterbi_table = np.zeros((O, S))
        # store best path for traceback
        best_path = np.zeros(O)

        # Store backpointers
        backpointer_table = np.zeros((O, S), dtype=int)

        for i in range(S):
            obs_index = self.observation_states_dict[decode_observation_states[0]]
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, obs_index]

        # Step 2. Calculate Probabilities
        for t in range(1, O):
            for j in range(S):
                obs_index = self.observation_states_dict[decode_observation_states[t]]
                (prob, state) = max(
                    (
                        viterbi_table[t - 1, i]
                        * self.transition_p[i, j]
                        * self.emission_p[j, obs_index],
                        i,
                    )
                    for i in range(S)
                )  # max will return max based on first item in tuple
                viterbi_table[t, j] = prob
                backpointer_table[t, j] = state

        # Step 3. Traceback
        best_final_state = np.argmax(viterbi_table[O - 1, :])
        best_path = [best_final_state]
        for t in range(O - 1, 0, -1):
            best_path.insert(0, backpointer_table[t, best_path[0]])

        # Step 4. Return best hidden state sequence
        return [self.hidden_states[state] for state in best_path]

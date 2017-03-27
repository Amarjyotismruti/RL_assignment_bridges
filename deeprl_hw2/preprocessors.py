"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.


    """

    def __init__(self, memory, history_length=4):
        pass

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        pass

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        pass

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    memory: Sequencialmemory object to access the replay memory.
    """


    def __init__(self, new_size=(84,84)):
        self.new_size=new_size

    def process_state_for_memory(self, obs):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        im=Image.fromarray(np.uint8(obs))
        obs=im.convert('L')
        obs=obs.resize((84,84))
        obs=np.array(obs)
        return obs


    def process_state_for_network(self, state_l):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        # batchsize = 1 here

        # s1,s2,s3 = np.float32(state_l[0]).shape
        # state = np.empty((1, s1, s2, s3*len(state_l)))
        # for i in range(len(state_l)):
        #     state[0,:s1,:s2,i*3:(i+1)*s3] = np.float32(state_l[i])
        state_l=np.expand_dims(state_l,axis=0)
        state_l=np.transpose(state_l,(0,2,3,1))
        state=np.float32(state_l)

        return state


    def process_batch(self, batch):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        # batch_size = len(batch)
        # s1,s2,s3 = np.float32(batch[0][0]).shape
        # batch_state = np.empty((batch_size, s1, s2, s3*len(batch[0])))

        # for i in range(len(batch)):
        #   state = self.process_state_for_network(batch[i])
        #   batch_state[i,:s1,:s2,:s3*len(batch[0])] = state
        batch_state=np.float32(batch)
        batch_state=np.transpose(batch_state,(0,2,3,1))

        return batch_state


    def process_reward(self, reward):
        """Clip reward between -1 and 1."""

        return np.clip(reward,-1,1)


#**HistoryPreprocessor isn't necessary as batches can be sampled using memory.sample() method from core.py.**#

# class PreprocessorSequence(Preprocessor):
#     """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

#     You can easily do this by just having a class that calls each preprocessor in succession.

#     For example, if you call the process_state_for_network and you
#     have a sequence of AtariPreproccessor followed by
#     HistoryPreprocessor. This this class could implement a
#     process_state_for_network that does something like the following:

#     state = atari.process_state_for_network(state)
#     return history.process_state_for_network(state)
#     """
#     def __init__(self, preprocessors):
#         pass
import pickle
import random

class ShapePool:
    """_summary_
    """
    def __init__(self,path_to_pkl):
        """_summary_

        Args:
            path_to_pkl (_type_): _description_
        """
        with open(path_to_pkl,'rb') as f:
            self.shape_pool = pickle.load(f)

    def get_n_random_shapes(self,n):
        """_summary_

        Args:
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        return random.sample(self.shape_pool,n)
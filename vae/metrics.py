import numpy as np

class Metrics:
    """_summary_
    """
    def __init__(self,points):
        """_summary_

        Args:
            points (_type_): _description_
        """
        self.points = points

    @staticmethod
    def _get_perimeter(points):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        distances = np.linalg.norm(np.diff(points,axis = 0),axis = 1)
        distances = np.append(distances,np.linalg.norm(points[-1] - points[0]))
        return np.sum(distances)
    
    @staticmethod
    def _get_area(points):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        x,y = points[:,0],points[:,1]
        return 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))
    
    def compute_compactness(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._get_perimeter(self.points)**2/self._get_area(self.points)
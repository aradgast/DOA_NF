import numpy as np
import scipy as sp


class Module:
    def __init__(self, array_geometry: str, wavelength: int, num_sensors: int, is_2d: bool = False,
                 angle_low: float = -90, angle_high: float = 90, angle_min_gap: int = 2, angle_max_gap: int = 180,
                 distance_min_gap: int = 1, distance_max_gap: int = 100):
        self.array_geometry = array_geometry
        self.wavelength = wavelength
        self.num_sensors = num_sensors
        self.is_2d = is_2d
        self.angle_low = angle_low
        self.angle_high = angle_high
        self.angle_min_gap = angle_min_gap
        self.angle_max_gap = angle_max_gap
        self.distance_min_gap = distance_min_gap
        self.distance_max_gap = distance_max_gap

    def compute_steering_vector(self, theta: float, dist: float = None):
        """
        Compute the steering vector for a given array geometry and wavelength
        :param dist: float
        :param theta: float
        :return: 1D array of shape (num_sensors, )
        """
        if self.is_2d:
            if dist is None:
                raise ValueError('Distance must be provided for 2D steering vector')
            return self._compute_steering_vector_2D(theta, dist)
        else:
            return self._compute_steering_vector_1D(theta)

    def _compute_steering_vector_1D(self, theta: float):
        """
        Compute the steering vector for a given array geometry and wavelength
        :param theta: float
        :return: 1D array of shape (num_sensors, )
        """
        if self.array_geometry == 'ULA':
            return np.exp(-1j * np.pi * np.linspace(0, self.num_sensors, self.num_sensors, endpoint=False)
                          * np.sin(theta) / self.wavelength)
        else:
            raise ValueError('Invalid array geometry')

    def _compute_steering_vector_2D(self, theta: float, dist: float):
        """
        Compute the steering vector for a given array geometry and wavelength
        :param theta: float
        :param dist: float
        :return: 1D array of shape (num_sensors, )
        """
        if self.array_geometry == 'ULA':
            array = np.linspace(-self.num_sensors // 2, self.num_sensors // 2, self.num_sensors)
            first_order = array * np.sin(theta)
            second_order = -0.5 * (array * np.cos(theta)) ** 2 / dist

            return np.exp(-1j * 2 * np.pi * (first_order + second_order) / self.wavelength)
        else:
            raise ValueError('Invalid array geometry')

    def calculate_fraunhofer_distance(self):
        if self.array_geometry == "ULA":
            D = (self.num_sensors - 1) * self.wavelength / 2
            d_f = 2 * (D ** 2) / self.wavelength
            return d_f, D
        else:
            raise TypeError(f"{self.array_geometry} not supported")

    def choose_angles(self, num_angles: int):
        """
        Choose random angles
        :param num_angles: int
        :return: list of angles
        """
        angles = []
        while len(angles) < num_angles:
            angle = np.random.randint(self.angle_low, self.angle_high)
            if len(angles) == 0:
                angles.append(angle)
            else:
                if np.min(np.abs(np.array(angles) - angle)) >= self.angle_min_gap and \
                        np.max(np.abs(np.array(angles) - angle)) <= self.angle_max_gap:
                    angles.append(angle)
        return np.deg2rad(angles)

    def choose_distances(self, num_distances: int):
        """
        Choose random distances
        :param num_distances:
        :return: distances list
        """
        if self.array_geometry == "ULA":
            distance_high, distance_low = self.calculate_fraunhofer_distance()
        else:
            raise ValueError(f"The array geometry, {self.array_geometry}, is not recognized")
        distances = []
        while len(distances) < num_distances:
            distance = np.random.randint(distance_low, distance_high)
            if len(distances) == 0:
                distances.append(distance)
            else:
                if np.min(np.abs(np.array(distances) - distance)) >= self.distance_min_gap and \
                        np.max(np.abs(np.array(distances) - distance)) <= self.distance_max_gap:
                    distances.append(distance)

        return distances

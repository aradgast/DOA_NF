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

    def _compute_steering_vector_2D(self, theta: np.ndarray, dist: np.ndarray):
        """
        Compute the steering vector for a given array geometry and wavelength
        :param theta: float
        :param dist: float
        :return: 1D array of shape (num_sensors, )
        """
        if self.array_geometry == 'ULA':
            theta = np.atleast_1d(theta)
            dist = np.atleast_1d(dist)
            theta = theta[:, np.newaxis]
            dist = dist[:, np.newaxis]
            limit = np.floor(self.num_sensors / 2)
            array = np.linspace(-limit, limit, self.num_sensors, endpoint=True)
            array = array[:, np.newaxis]
            array = np.tile(array, (1, self.num_sensors))
            array_square = np.power(array, 2)

            first_order = np.sin(theta)
            first_order = np.tile(first_order, (1, self.num_sensors))
            first_order = array @ first_order.T
            first_order = np.tile(first_order[:, :, np.newaxis], (1, 1, len(dist)))
            second_order = -0.5 * np.divide(np.power(np.cos(theta), 2), dist.T)
            second_order = np.tile(second_order[:, :, np.newaxis], (1, 1, self.num_sensors))
            # second_order = array_square * np.transpose(second_order, (2, 1, 0))
            second_order = np.einsum('ij,jkl->ilk', array_square, np.transpose(second_order, (2, 1, 0)))

            time_delay = first_order + second_order

            return np.exp(-1j * 2 * np.pi * time_delay / self.wavelength)


        else:
            raise ValueError('Invalid array geometry')

    def _compute_steering_vector_full_phase(self, theta: np.ndarray, dist: np.ndarray):
        """
        
        :param theta: 
        :param dist: 
        :return: 
        """''
        if self.array_geometry == "ULA":
            theta, dist = np.meshgrid(theta, dist)
            limit = np.floor(self.num_sensors / 2)
            array = np.linspace(-limit, limit, self.num_sensors, endpoint=True)
            # array, _ = np.meshgrid(array, array)
            mul_1 = dist / self.wavelength
            sqrt_1 = (array / np.tile(dist[:, :, np.newaxis], (1, 1, self.num_sensors))) ** 2
            sqrt_2 = - 2 * (array / np.tile(dist[:, :, np.newaxis], (1, 1, self.num_sensors))) * np.sin(
                np.tile(theta[:, :, np.newaxis], (1, 1, self.num_sensors)))
            mul_2 = 1 - np.sqrt(1 + sqrt_1 + sqrt_2)
            exp = np.einsum('ij,ijk->ijk', mul_1, mul_2)
            return np.exp(-2 * np.pi * 1j * exp.T)
        else:
            raise ValueError("Invalid array geometry")
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
            distance = np.random.randint(1, distance_high)
            if len(distances) == 0:
                distances.append(distance)
            else:
                if np.min(np.abs(np.array(distances) - distance)) >= self.distance_min_gap and \
                        np.max(np.abs(np.array(distances) - distance)) <= self.distance_max_gap:
                    distances.append(distance)

        return np.array(distances)

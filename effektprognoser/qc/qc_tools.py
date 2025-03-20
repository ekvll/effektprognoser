import numpy as np


TRANSPORT_CATEGORIES = {"LL", "PB", "TT"}


class QCTools:
    def __init__(self, table) -> None:
        self.table = table

    def count_zero(self, arr: np.ndarray) -> bool:
        """
        Check if any value in an array is zero, unless the table is a transport category.

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            bool: True is zeros in array, False otherwise.
        """
        if self.is_transport_category():
            return False
        return np.any(np.isclose(arr, 0.0))

    def boundary_values(self, arr: np.ndarray) -> str | bool:
        """
        Check if any value in array is outside given threshold boundaries.

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            str | bool: String of exceeded boundary, or None if within boundaries.

        """
        min_val, max_val = self.get_boundary_values()
        if min_val is not None and np.any(arr < min_val):
            return f"< {min_val}"
        if max_val is not None and np.any(arr > max_val):
            return f"> {max_val}"
        return None

    def is_transport_category(self) -> bool:
        """
        Check if the table belongs to category.

        Args:
            None

        Returns:
            bool: True is table belongs to transport categories, False otherwise.
        """
        return any(category in self.table for category in TRANSPORT_CATEGORIES)

    def get_boundary_values(self) -> tuple[int | None, int | None]:
        """
        Define boundary values based on table

        Args:
            None

        Returns:
            tuple: (int, int) if table belongs to a category, (None, None) otherwise.

        """
        if self.is_transport_category():
            return 0, 10**10
        return None, None  # Default boundaries

from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes the ChessDataset with the provided data.

        Args:
            X (list): A list of chess positions.
            y (list): A list of corresponding labels.
        """
        self.X = X
        self.y = y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the sample and label at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the sample and its corresponding label.
        """

        return self.X[idx], self.y[idx]
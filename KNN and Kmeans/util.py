import numpy as np


def normalize(data):
    """Normalizes a given array. Edits the array in place, aka does not return."""

    max_values = np.max(data, 0)
    min_values = np.min(data, 0)

    for row in data:
        for col in range(data.shape[1]):
            row[col] = (row[col] - min_values[col]) / (
                max_values[col] - min_values[col]
            )


def import_weather_dataset(filepath, year):
    """
    Returns:
        data, labels
    """
    year *= 10000

    data = np.genfromtxt(
        filepath,
        delimiter=";",
        usecols=[1, 2, 3, 4, 5, 6, 7],
        converters={
            5: lambda s: 0 if s == b"-1" else float(s),
            7: lambda s: 0 if s == b"-1" else float(s),
        },
    )

    dates = np.genfromtxt(filepath, delimiter=";", usecols=[0])
    labels = []
    for label in dates:
        if label < year + 301:
            labels.append("winter")
        elif year + 301 <= label < year + 601:
            labels.append("lente")
        elif year + 601 <= label < year + 901:
            labels.append("zomer")
        elif year + 901 <= label < year + 1201:
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")

    return data, labels
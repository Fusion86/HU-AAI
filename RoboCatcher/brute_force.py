import math
import time
from robocatcher import fitness, deg_to_rad

if __name__ == "__main__":
    start = time.time()
    best_solution = None
    best_score = math.inf

    for s1 in range(180):
        for s2 in range(180):
            for s3 in range(180):
                err = fitness([deg_to_rad(s1), deg_to_rad(s2), deg_to_rad(s3)], 0)
                if err < best_score:
                    best_score = err
                    best_solution = [s1, s2, s3]

    duration = time.time() - start
    print("That took {:.2f} seconds".format(duration))

    print("Best solution:", best_solution)
    print("Which has score: {:.6f}".format(best_score))

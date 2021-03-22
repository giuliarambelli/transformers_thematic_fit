import numpy as np
import copy
from scipy.spatial import distance


def centroid(vecs, n):

    if len(vecs) == 0:
        #print (list_of_words)
        return None
    else:
        centroid = None
        num_vecs = 0
        for v in vecs:
            if num_vecs > n:
                break
            else:
                try:
                    vector = vecs[num_vecs]
                    if centroid is None:
                        centroid = copy.deepcopy(vector)
                    else:
                        centroid += vector
                    num_vecs += 1
                except KeyError:
                    pass

        # if centroid is not None:
        res = centroid / n

        return res


def cosine(vec1, vec2):

    score = 1-distance.cosine(vec1, vec2)
    return score
def create_one_hot_vector(R, candidate_list):
    one_hot_vector = []
    for word in R:
        if word in candidate_list:
            one_hot_vector.append(1)
        else:
            one_hot_vector.append(0)
    return one_hot_vector


def get_cutoff_freq(c, threshold):
    total_items = sum(c.values())
    # Calculate frequency of each item
    item_frequencies = {k: v / total_items for k, v in c.items()}

    # Sort items by frequency
    sorted_items = c.most_common()

    # Iterate through sorted items and find cut-off point
    n = None
    for item, count in sorted_items:
        if item_frequencies[item] < threshold:
            n = count
            break

    return n
def parse_index_list(point_list: list, index_list: list):
    res = []
    limit = len(point_list)
    for i in index_list:
        if i >= limit:
            a="index: %d beyond the size of point_list: %d" % (i, limit)
            raise Exception(a)
        else:
            res.append(point_list[i])
    return res

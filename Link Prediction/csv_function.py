__author__ = 'sym44'

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header:
            f.readline()
        data = []
        line_num = 0
        for line in f:
            line_num += 1
            print(line_num)
            line = line.strip().split(",")
            data.append([float(x) for x in line])
    return data


def write_csv(file_path, list):
    with open(file_path, "wb") as f:
        for a in list:
            f.write(str(a)+"\n")

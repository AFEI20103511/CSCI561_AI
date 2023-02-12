import math


def readInput(input):
    case_path = './' + str(input[0])
    print(case_path)
    label_path = './' + str(input[1])
    print(label_path)
    test_path = './' + str(input[2])
    print(test_path)
    case = []
    label = []
    test = []
    with open(case_path, 'r') as f:
        points = f.readlines()
        for p in points:
            x = float(p.split(',')[0])
            y = float(p.split(',')[1])
            x_sq = x * x
            y_sq = y * y
            xy = x * y
            sinx = math.sin(x)
            siny = math.sin(y)
            case.append([x,y,x_sq,y_sq,xy,sinx,siny])

    with open(label_path, 'r') as f:
        labels = f.readlines()
        for l in labels:
            label.append(int(l))

    with open(test_path, 'r') as f:
        points = f.readlines()
        for p in points:
            x = float(p.split(',')[0])
            y = float(p.split(',')[1])
            x_sq = x * x
            y_sq = y * y
            xy = x * y
            sinx = math.sin(x)
            siny = math.sin(y)
            test.append([x,y,x_sq,y_sq,xy,sinx,siny])

    return case, label, test

def readTestLabel(path):
    with open(path, 'r') as f:
        label = []
        labels = f.readlines()
        for l in labels:
            label.append(int(l))
        return label
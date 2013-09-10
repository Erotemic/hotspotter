from numba import autojit

@autojit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

def sum2d2(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

if __name__ == '__main__':
    arr = np.random.rand(1000,1000)
    res = sum2d2(arr)
    print res

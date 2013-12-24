import pylru


class lru_cache_(object):
    def __init__(self, max_size):
        self.cache = pylru.lrucache(max_size)

    def __call__(self, func):
        def wrapped(*args):  # XXX What about kwargs
            try:
                return self.cache[args]
            except KeyError:
                pass

            value = func(*args)
            self.cache[args] = value
            return value
        wrapped.func_name = func.func_name
        wrapped.cache_clear = self.cache.clear
        return wrapped


@lru_cache(2)
def myfunc(arg):
    print('computing arg=%r' % arg)
    return arg

print(myfunc(1))
print(myfunc(1))
print(myfunc(2))
print(myfunc(2))
print(myfunc(3))
print(myfunc(2))
print(myfunc(2))
print(myfunc(4))
print(myfunc(4))
print(myfunc(4))
print(myfunc(3))
print(myfunc(3))
print(myfunc(1))
print(myfunc(1))
print(myfunc(1))

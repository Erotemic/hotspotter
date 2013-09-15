y = 4

def foo(x):
    print("Global scope variables")
    for (key,val) in globals().iteritems():
        print("  * %r = %r" % (key, val))
    print("---------------------")
    print("Local scope variables")
    for (key,val) in locals().iteritems():
        print("  * %r = %r" % (key, val))


foo(None)


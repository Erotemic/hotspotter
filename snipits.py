# ### SNIPIT: Namespace Dict 
namespace_dict = freak_params
for key, val in namespace_dict.iteritems():
    exec(key+' = '+repr(val))
# ### ----

import cv2 
import types

# See what types exist in cv2
type_dict = {}
for attr_name in dir(cv2):
    attr = cv2.__getattribute__(attr_name)
    attr_type = type(attr)
    if not attr_type in type_dict.keys():
        type_dict[attr_type] = []
    type_dict[attr_type].append(attr_name)

# Print them out in their corresponding category
key_list = type_dict.keys()
for key in key_list:
    print '\n\n==================\n____'+ str(key) + '____\n\n'
    for attr_name in type_dict[key]:
        attr = cv2.__getattribute__(attr_name)
        if type(attr) is types.BuiltinFunctionType:
            typestr = str(attr.__doc__)
        else:
            typestr =  repr(attr)
        attr_name += ' '*(42-len(attr_name)) #padding
        typestr = typestr.replace('->','\n  ->')
        print '%s = %s' % (attr_name, typestr)

print key_list

import cv2 
import sys
import types

# See what types exist in cv2
type_dict = {}
for attr_name in dir(cv2):
    attr = cv2.__getattribute__(attr_name)
    attr_type = type(attr)
    if not attr_type in type_dict.keys():
        type_dict[attr_type] = []
    type_dict[attr_type].append(attr_name)
key_list = type_dict.keys()

def find_in_list(str_, tofind_list, agg_fn=all):
    if len(tofind_list) == 0:
        return True
    str_ = str_.lower()
    tofind_list = [tofind.lower() for tofind in tofind_list]
    found_list = [str_.find(tofind) > -1 for tofind in tofind_list]
    return agg_fn(found_list)

def get_print_list(tofind_list=[]):
    # Print them out in their corresponding category
    print_list = []
    for key in key_list:
        print_list.append('\n\n==================\n____'+ str(key) + '____\n\n')
        for attr_name in type_dict[key]:
            attr = cv2.__getattribute__(attr_name)
            if type(attr) is types.BuiltinFunctionType:
                typestr = str(attr.__doc__)
            else:
                typestr =  repr(attr)
            attr_name += ' '*(42-len(attr_name)) #padding
            #typestr = typestr.replace('->','\n  ->')
            line = '%s = %s' % (attr_name, typestr)
            if find_in_list(line, tofind_list):
                print_list.append(line)
    return print_list

print_list = get_print_list()
print('\n'.join(print_list))
print(key_list)


print('\n\n__CV2 SEARCH__')
tofind_list = sys.argv[1:] if len(sys.argv) > 1 else []
print('Searching for: %r ' % tofind_list)

print_list = get_print_list(tofind_list)
print('\n'.join(print_list))


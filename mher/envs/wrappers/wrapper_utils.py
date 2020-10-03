
def recurse_attribute(obj, attr, max_depth=3):
    '''find env's attribution'''
    tmp_obj = obj
    depth = 0
    while depth < max_depth and not hasattr(tmp_obj, attr):
        tmp_obj = tmp_obj.env
        depth += 1
    if hasattr(tmp_obj, attr):
        return getattr(tmp_obj, attr)
    else:
        return None
    

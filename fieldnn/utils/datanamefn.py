   
def parse_full_recfldgrn(full_recfldgrn):
    # suffix = full_recfldgrn.split('_')[-1]
    recfld = [i for i in full_recfldgrn.split('-') if '@' in i][0]
    rec, fld = recfld.split('@')
    grn_suffix = [i for i in full_recfldgrn.split('-') if 'Grn' in i][0]
    grn, suffix = grn_suffix.split('_')
    prefix_ids = [i for i in full_recfldgrn.split('-') if 'Grn' not in i and '@' not in i]
    return prefix_ids, rec, fld, grn, suffix

# fieldnn.utils.datanamefn.py

def get_curfld_recinfo(curfld):
    '''
        a helper function that get the rec information from the current recfldgrn name.
        the input is: prefix_rec@fld_grn
        we want to check whether @ is here. 
        If there is a @ in the current level, we want to put the rec as the potential merger target. 
    '''
    if '@' in curfld:
        fld_list = curfld.split('@')
        return '@'.join(fld_list[:-1]), fld_list[-1]
    else:
        return None, curfld
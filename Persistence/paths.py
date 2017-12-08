def get_path_hmm_json():
    path={'single_variate':'Models/HMM/JSON'}
    return path

def

def create_path(user, activity):
    path = 'Data/citizen_id_' + str(user) + '/'
    file_name='citizen_id_'+str(user)+'_activity_'+activity
    extension='.pkl'
    whole_path=path+file_name+extension
    return whole_path

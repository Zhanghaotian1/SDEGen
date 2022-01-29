import pickle

def split_pickle(pkl_path,path_with_output_name,split_num):
    '''
    Args:
        pkl_path: e.g. /C:/user/hp/alearn/train.pkl
        path_with_output_name:e.g. /C:/user/hp/alearn/mini_train.pkl
        split_num: new_data = old_data[:split_num]

    Returns: None

    '''
    data = []
    with open(pkl_path, 'rb') as fin:
        data = pickle.load(fin)
    print('the number of data', len(data))
    new_data = data[:split_num]
    with open(path_with_output_name, 'wb') as file:
        pickle.dump(new_data,file)
    print('new_data has been saved, the number of the new dataset is', len(new_data))



#if we want used_sigmas, which shape is (num_edges), to be converted as (num_edges,1), we can write
used_sigmas = torch.random_like.randn(100)
#method 1
used_sigmas = used_sigmas.unsequeeze(-1)
#method 2
used_sigmas = used_sigmas[:,None]
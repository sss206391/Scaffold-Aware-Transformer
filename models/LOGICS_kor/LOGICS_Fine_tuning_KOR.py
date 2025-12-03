import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import json
import torch
from package import global_settings, logics, predictor, chemistry, analysis, smiles_vocab, smiles_lstm

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LOGICS fine-tuning config
    config = global_settings.Object()
    
    config.ablation = None  # we will use full LOGICS model
    
    config.tokens_path ="../package/logics_tokens.txt"
    config.pretrain_setting_path = "../package/pretrain_setting.json"
    config.pretrained_model_path = "../package/logics_prior_e10.ckpt"
    config.featurizer = predictor.featurizer
    config.predictor_path = "../package/logics_kor_rfr_cv3.pkl"
    
    config.max_epoch = 31
    config.save_period = 2
    config.save_ckpt_fmt = 'kor_logics_e%d.ckpt'
    config.sample_fmt = 'kor_logics_e%d.txt'
    config.memory_fmt ='kor_logics_mem_e%d.csv'
    config.memory_size = 50000
    config.save_size = 20000
    config.gen_size = 20000
    config.exp_size = 20000
    config.finetune_lr = 0.0001
    config.finetune_bs = 32
    config.sampling_bs = 256
    
    config.device_name = device
    # perform fine-tuning
    logics.LOGICS_training(config)

    vocab_obj = smiles_vocab.Vocabulary(init_from_file=config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab_obj)
    
    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)
        
    agent_ckpt = torch.load(config.save_ckpt_fmt%30, map_location='cpu')
    lstm_agent = smiles_lstm.SmilesLSTMGenerator(vocab_obj, model_setting['emb_size'], model_setting['hidden_units'], device_name='cpu')
    lstm_agent.lstm.load_state_dict(agent_ckpt['model_state_dict'])

    # sampling
    ssplr = analysis.SafeSampler(lstm_agent, batch_size=16)
    generated_smiles = ssplr.sample_clean(50, maxlen=150)
    display(generated_smiles)

    config.vc_fmt = 'kor_logics_vc_e%d.smi'  # save valid & canonical smiles
    config.npfps_fmt = 'kor_logics_npfps_e%d.npy'  # save fingerprint in npy
    config.fcvec_fmt ='kor_logics_fcvec_e%d.npy'  # save Frechet ChemNet vectors
    
    epochs = list(range(0, config.max_epoch+1, config.save_period))

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use tensorflow cpu
    
    import fcd
    from package import frechet_chemnet
    fc_ref_model = fcd.load_ref_model()

    for epo in epochs:
        print(epo)
        with open(config.sample_fmt%epo, 'r') as f:
            gens = [line.strip() for line in f.readlines()]
        vcs, invids = chemistry.get_valid_canons(gens)
        print("- count invalids: ", len(invids))
        with open(config.vc_fmt%epo, 'w') as f:
            f.writelines([line+'\n' for line in vcs])
        fps = chemistry.get_fps_from_smilist(vcs)
        np.save(config.npfps_fmt%epo, chemistry.rdk2npfps(fps))
        fcvecs = fcd.get_predictions(fc_ref_model, vcs)  # ChemNet vectors
        np.save(config.fcvec_fmt%epo, fcvecs)

    # loading validation dataset
    with open("../../dataset/kor/kor_fold_splits.json", 'r') as f:
        pik3_folds = json.load(f)
    
    # data_npfps = np.load(project_paths['PIK3CA_DATA_FP'])
    # data_fcvecs = np.load(project_paths['PIK3CA_DATA_FCVEC'])
    
    data_npfps = np.load("../../dataset/kor/kor_aff_npfps.npy")
    data_fcvecs = np.load("../../dataset/kor/kor_aff_fcvec.npy")
    
    val_fold_id = "3"
    val_npfps = data_npfps[pik3_folds[val_fold_id]]
    val_rdkfps = chemistry.np2rdkfps(val_npfps)
    val_fcvecs = data_fcvecs[pik3_folds[val_fold_id]]
    
    dsize = len(val_rdkfps)  # demand size for OT
    ssize = dsize*10


    epochs = list(range(0, 31, config.save_period))
    
    val_fcd_list = []
    val_otd_list = []
    for epo in epochs:
        print(epo)
        # load fc vectors of generation
        gen_fcvecs = np.load(config.fcvec_fmt%epo)
        fcdval = frechet_chemnet.fcd_calculation(val_fcvecs, gen_fcvecs)
        val_fcd_list.append(fcdval)
        
        gen_npfps = np.load(config.npfps_fmt%epo)[:ssize]  # only need this amount
        gen_rdkfps = chemistry.np2rdkfps(gen_npfps)
        simmat = analysis.calculate_simmat(gen_rdkfps, val_rdkfps)  # row:gen, col:data
        distmat = analysis.transport_distmat(analysis.tansim_to_dist, simmat, 10)
        _, _, motds = analysis.repeated_optimal_transport(distmat, repeat=10)
        val_otd_list.append(np.mean(motds))
    
    import sys
    sys.path.append('..')
    
    import pandas as pd
    import numpy as np
    import pickle
    import json
    from package import global_settings, analysis, chemistry, evaluation, frechet_chemnet
    import fcd
    
    ## run this cell in case CuDNN library error occurs 
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    fc_ref_model = fcd.load_ref_model()
    
    SAMPLE_SIZE = 20000  
    INTDIV_SIZE = 1000 
    
    scales = list(range(2, 31, 2))  # [2, 4, 6, 8, ..., 28, 30]
    
    model_names = [f'logics{scale}' for scale in scales]
    
    perf_table = pd.DataFrame(index=['validity','uniqueness','novelty','diversity','PredAct','PwSim','FCD','OTD'], 
                            columns=model_names)
    metrics = perf_table.index.tolist()
    
    paths_vc = {}
    paths_npfps = {}
    paths_fc_vecs = {}
    
    for scale in scales:
        model_name = f'logics{scale}'
        paths_vc[model_name] = f"kor_logics_vc_e{scale}.smi"
        paths_npfps[model_name] = f"kor_logics_npfps_e{scale}.npy"
        paths_fc_vecs[model_name] = f"kor_logics_fcvec_e{scale}.npy"
    
    # pre-training dataset loading
    with open("../../dataset/guacamol_raw.smi", 'r') as f:
        pret_smis = [line.strip() for line in f.readlines()]
    len(pret_smis)
    
    # loading predictor 
    pred_path = "../package/logics_kor_rfr_cv3.pkl"
    with open(pred_path, 'rb') as f:
        predictor = pickle.load(f)
    predictor
    
    # loading test set actives (tsa)
    affinity_data = pd.read_csv("../../dataset/kor/kor_affinity.csv")
    with open("../../dataset/kor/kor_fold_splits.json", 'r') as f:
        folds = json.load(f)
    test_ids = folds[str(5)]
    test_data = affinity_data.iloc[test_ids]
    
    tsa_data = test_data[test_data['affinity']>7.0]  # active among test set
    tsa_smis = tsa_data['smiles'].tolist()
    tsa_rdkfps = chemistry.get_fps_from_smilist(tsa_smis)
    tsa_fc_vecs = fcd.get_predictions(fc_ref_model, tsa_smis)
    
    # evaluation config objects
    evcons = {}
    for mn in model_names:
        with open(paths_vc[mn], 'r') as f:
            vc_smis = [line.strip() for line in f.readlines()]
        npfps = np.load(paths_npfps[mn])
        fc_vecs = np.load(paths_fc_vecs[mn])
        evc = evaluation.EvalConfig(
                ssize=SAMPLE_SIZE, vc_smis=vc_smis, npfps=npfps, simmat_size=INTDIV_SIZE, fc_vecs=fc_vecs,
                data_smis=tsa_data, data_rdkfps=tsa_rdkfps, data_fc_vecs=tsa_fc_vecs, ot_repeats=10
        )
        evcons[mn] = evc
    
    # evaluate and fillout the performance table
    for mn in model_names:
        print(mn)
        va, uni, nov, div = evaluation.eval_standard(evcons[mn], pret_smis)
        predact, pwsim, fcdval, otdval = evaluation.eval_optimization(evcons[mn], predictor)
        perf_table[mn]['validity'] = va
        perf_table[mn]['uniqueness'] = uni
        perf_table[mn]['novelty'] = nov
        perf_table[mn]['diversity'] = div
        perf_table[mn]['PredAct'] = predact
        perf_table[mn]['PwSim'] = pwsim
        perf_table[mn]['FCD'] = fcdval
        perf_table[mn]['OTD'] = otdval
    perf_table.to_csv(f'logics_kor_performance.csv')
    perf_table
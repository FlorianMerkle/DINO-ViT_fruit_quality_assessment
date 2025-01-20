import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
from adversarial_attacks.my_utils import _fit, _evaluate_model, evaluate_advs,get_adversarial_robustness,get_adversarial_robustness_l0, get_original_classifier, get_torch_classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def pipeline():
    
    m,_ = get_torch_classifier()
    clf,_ = get_original_classifier()
    transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', verbose=False).to(device)
    for attack in ['FGSM', 'PGD']:#, 'CW', 'DF', 'BB0']:
        if attack in ['FGSM', 'PGD']:epsilons=[1/255,2/255,4/255,8/255]
        if attack == 'BB0': epsilons = [0,50,100,200,500,1000]
        if attack in ['CW', 'DF']: epsilons = [1,2,5,10,20,50,100]
        for eps in epsilons:
            if attack == 'CW': confidences = [0,1,2,4,6]
            if attack == 'DF': confidences = [.2,.4,.8,]
            if attack == 'BB0': confidences = [1.1, 2,4,8]
            if attack == 'PGD' or attack == 'FGSM': confidences = [None]
            for confidence in confidences:
                df = pd.DataFrame(
                    data=[[None, None, None]],
                    index=[f'{attack}-{eps}-{confidence}'],
                    columns=["rob","transfer-rob", "duration"],
                    dtype=float,
                )
                robustness, advs, duration = get_adversarial_robustness(m, attack , eps, confidence)
                advs = torch.from_numpy(advs).float()
                torch.save(advs, f'./adversarial_attacks/adversarial-examples/{attack}-{eps}-{confidence}.pkl')
                transfer_robustness = evaluate_advs(transformer, advs, clf)
                df['rob'] = robustness.item()
                df['transfer-rob'] = transfer_robustness
                df['duration'] = duration
                print(df)
                df.to_csv(f'./evaluation/{attack}-{eps}-{confidence}.csv')


def pipeline_l0():
    
    m,_ = get_torch_classifier()
    clf,_ = get_original_classifier()
    transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', verbose=False).to(device)
    for attack in ['BB0']:
        for lr in [0.01]:
            if attack == 'BB0': confidences = [2.0,3.0,5.0]
            for confidence in confidences:
                df = pd.DataFrame(
                    data=[[None, None, None]],
                    index=[f'{attack}-{lr}-{confidence}'],
                    columns=["rob","transfer-rob", "duration"],
                    dtype=float,
                )
                robustness, advs, duration = get_adversarial_robustness_l0(m, attack , lr, confidence)
                advs = torch.from_numpy(advs).float()
                torch.save(advs, f'./adversarial_attacks/adversarial-examples/l0_only{attack}-{lr}-{confidence}.pkl')
                transfer_robustness = evaluate_advs(transformer, advs, clf)
                df['rob'] = robustness.item()
                df['transfer-rob'] = transfer_robustness
                df['duration'] = duration
                print(df)
                df.to_csv(f'./evaluation/l0_only-{attack}-{lr}-{confidence}.csv')






from __future__ import division, unicode_literals
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
import torch
from rdkit import Chem, RDLogger

from models import KGCL, BeamSearch

lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def canonicalize_p(smi):
    p = canonicalize(smi)
    p_mol = Chem.MolFromSmiles(p)
    [a.SetAtomMapNum(a.GetIdx() + 1) for a in p_mol.GetAtoms()]
    p_smi = Chem.MolToSmiles(p_mol)
    return p_smi


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k', help='dataset: USPTO_50k or USPTO_full')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='BEST',
                        help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int, default=50, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9, help='maximum number of edit steps')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)
    if args.use_rxn_class:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
    else:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')

    checkpoint = torch.load(os.path.join(exp_dir, 'epoch_132.pt'))
    config = checkpoint['saveables']

    model = KGCL(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    rt_top_k = np.zeros(args.beam_size)
    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))

    # Load the forward model prediction results
    pred_text = os.path.join(exp_dir, 'forward_predictions_50k_top50.txt')
    with open(pred_text, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    current_target_idx = 0

    for idx in p_bar:
        rxn_data = test_data[idx]
        rxn_smi = rxn_data.rxn_smi
        rxn_class = rxn_data.rxn_class

        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
        r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
        r_smi = Chem.MolToSmiles(r_mol)
        r_set = set(r_smi.split('.'))

        pred_text = os.path.join(exp_dir, 'pred_text1', f'{idx}.txt')

        with torch.no_grad():
            top_k_results = beam_model.run_search(prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

        beam_matched = False
        counter = 0
        with open(pred_text, 'a') as fp:
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                if pred_smi != 'final_smi_unmapped' and prob>0.0:
                    pred_result = smi_tokenizer(pred_smi)
                    fp.write(f'{pred_result}\n')
                    counter +=1
                pred_set = set(pred_smi.split('.'))
                if pred_set == r_set and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                    true_idx = beam_idx


        p_mol = Chem.MolFromSmiles(p)
        [a.ClearProp('molAtomMapNumber') for a in p_mol.GetAtoms()]
        p_mol = Chem.MolFromSmiles(Chem.MolToSmiles(p_mol))
        p_smi = Chem.MolToSmiles(p_mol)

        end_idx = current_target_idx + counter

        all_predictions = list(targets[current_target_idx:end_idx])

        rt_matched = False
        for rt_idx, predictions in enumerate(all_predictions):
            pred = (''.join(predictions.strip().split(' ')))
            mol = Chem.MolFromSmiles(pred)
            if mol is not None:
                pred = Chem.MolToSmiles(mol, isomericSmiles=True)
            else:
                pred = ''

            if beam_matched and not rt_matched:
                if true_idx <= rt_idx:
                    rt_top_k[true_idx] += 1
                    rt_matched = True
                else:
                    if pred == p_smi:
                        rt_top_k[rt_idx] += 1
                        rt_matched = True
            if pred == p_smi and not beam_matched and not rt_matched:
                rt_top_k[rt_idx] += 1
                rt_matched = True

        # update current_target_idx
        current_target_idx = end_idx

        msg = ''
        for beam_idx in [1, 3, 5, 10, 50]:
            match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
            Rt_acc = np.sum(rt_top_k[:beam_idx]) / (idx + 1)
            msg += 'Exact accuracy, t%d: %.3f' % (beam_idx, match_acc)
            msg += ' Round-trip accuracy, t%d: %.3f ' % (beam_idx, Rt_acc)
        p_bar.set_description(msg)


if __name__ == '__main__':
    main()
